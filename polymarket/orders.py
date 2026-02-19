"""
Order sizing and execution.

Bet sizing strategy
-------------------
All sizing is proportional to the **live USDC wallet balance**, fetched at
startup and refreshed every config.RISK.balance_refresh_secs seconds.

Smart dynamic order sizing (_compute_smart_order_size)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Order size is calculated in three stages:

1. **Wallet-tier lookup** (config.DYNAMIC_SIZING.tiers)
   Smaller wallets need a higher fraction to stay above min_order_usdc;
   larger wallets use a lower fraction to bound absolute risk.

   Example tiers (defaults):
       $2 000 + wallet → 3 % of available, cap $50
       $500  + wallet → 4 % of available, cap $40
       $200  + wallet → 5 % of available, cap $25
       $50   + wallet → 7 % of available, cap $15
       $20   + wallet → 10% of available, cap $5
       < $20   wallet → 15% of available, cap $3

2. **Available-balance base** (wallet minus already-deployed USDC)
   Sizes against what is *actually free* rather than the headline balance,
   so the bot never over-commits when positions are already open.

3. **Exposure-utilisation scale-down**
   When the exposure budget is more than `exposure_scale_threshold` full
   (default 60 %), order size is reduced linearly toward `min_exposure_scale`
   × base size (default 40 %) as utilisation approaches 100 %.  This makes
   each successive bet smaller as the book fills up.

Per-order size (momentum):
    order_usdc = _compute_smart_order_size()   # see above

Per-order size (arb, Kelly-weighted):
    smart_cap  = _compute_smart_order_size()
    kelly_usdc = kelly_f * kelly_fraction * wallet_balance
    order_usdc = max(min_order_usdc, min(kelly_usdc, smart_cap))

Total exposure cap:
    max_deployed = wallet_balance * max_exposure_fraction   # e.g. 20 %

Example with $500 wallet, 0 % utilisation:
    tier        → 4 %, cap $40
    available   → $500 (nothing deployed yet)
    base_order  → $500 * 0.04 = $20
    scale       → 1.0 (utilisation below 60 % threshold)
    order_usdc  → min($20, $40) = $20

Same wallet at 80 % utilisation ($80 of $100 budget deployed):
    available   → $500 − $80 = $420
    base_order  → $420 * 0.04 = $16.80
    scale       → 1.0 − (1.0 − 0.40) × (0.80 − 0.60) / 0.40 ≈ 0.70
    order_usdc  → $16.80 × 0.70 = $11.76
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from polymarket.client import PolymarketClient
from polymarket.markets import MarketCache, MarketInfo
import config

if TYPE_CHECKING:
    from strategy.arbitrage import ArbSignal

log = logging.getLogger(__name__)


class OrderManager:
    """
    Manages order sizing, placement, and position tracking.

    Wallet balance is fetched at startup and kept fresh via
    run_balance_refresh_loop().
    """

    def __init__(self, client: PolymarketClient, cache: MarketCache) -> None:
        self._client = client
        self._cache = cache
        self._wallet_balance: float = 0.0          # live USDC balance
        self._last_balance_log: float = 0.0
        # condition_id → last trade timestamp (monotonic)
        self._last_trade: dict[str, float] = {}
        # condition_id → USDC currently deployed
        self._open_exposure: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Wallet balance management
    # ------------------------------------------------------------------

    async def refresh_balance(self) -> None:
        """Fetch live USDC balance and cache it."""
        balance = await self._client.get_usdc_balance()
        changed = abs(balance - self._wallet_balance) > 0.01
        self._wallet_balance = balance
        # Log balance changes and periodically as a heartbeat
        now = time.monotonic()
        if changed or now - self._last_balance_log > 300:
            log.info(
                "Wallet balance: $%.2f USDC  |  deployed: $%.2f  |  "
                "available: $%.2f  |  max_order: $%.2f  |  max_exposure: $%.2f",
                self._wallet_balance,
                self.total_exposure,
                self._available_balance,
                self._max_order_usdc,
                self._max_exposure_usdc,
            )
            self._last_balance_log = now

    async def run_balance_refresh_loop(self) -> None:
        """Background task: keep wallet balance fresh."""
        await self.refresh_balance()          # immediate fetch on start
        while True:
            await asyncio.sleep(config.RISK.balance_refresh_secs)
            await self.refresh_balance()

    # ------------------------------------------------------------------
    # Derived sizing limits
    # ------------------------------------------------------------------

    @property
    def _available_balance(self) -> float:
        """USDC balance minus what's already deployed."""
        return max(0.0, self._wallet_balance - self.total_exposure)

    @property
    def _max_exposure_usdc(self) -> float:
        return self._wallet_balance * config.RISK.max_exposure_fraction

    @property
    def _max_order_usdc(self) -> float:
        """Per-order cap: fraction of wallet, hard ceiling applied."""
        return min(
            self._wallet_balance * config.RISK.max_order_fraction,
            config.RISK.max_order_usdc_hard,
        )

    def _remaining_budget(self) -> float:
        """How much more USDC we can deploy before hitting the exposure cap."""
        return max(0.0, self._max_exposure_usdc - self.total_exposure)

    def _compute_smart_order_size(self, source: str = "") -> float:
        """
        Dynamic order size that adapts to current wallet size and exposure state.

        Steps
        -----
        1. **Wallet-tier lookup** – selects a (fraction, cap) pair from
           config.DYNAMIC_SIZING.tiers based on current wallet balance.
           Smaller wallets use a larger fraction so orders stay above the
           minimum; larger wallets use a smaller fraction to bound absolute
           risk.

        2. **Available-balance sizing** – applies the tier fraction to
           *available* USDC (wallet minus already-deployed capital) rather
           than the raw wallet total, so the bot never over-commits when
           positions are already open.

        3. **Exposure-utilisation scale-down** – when the exposure budget is
           more than `exposure_scale_threshold` full, order size is reduced
           linearly down to `min_exposure_scale` × base size.  This makes
           each successive bet smaller as the book fills up.

        4. **Clamp** – the result is clamped to
           [0, min(tier_cap, remaining_budget)].

        Returns 0.0 if nothing can safely be deployed (the caller should
        check against config.RISK.min_order_usdc before placing an order).
        """
        wallet    = self._wallet_balance
        available = self._available_balance
        remaining = self._remaining_budget()

        # --- Step 1: pick wallet tier ---
        tier_fraction = config.RISK.max_order_fraction     # fallback defaults
        tier_cap      = config.RISK.max_order_usdc_hard

        for min_wallet, fraction, cap in config.DYNAMIC_SIZING.tiers:
            if wallet >= min_wallet:
                tier_fraction = fraction
                tier_cap      = cap
                break

        # --- Step 2: base order from available balance ---
        base_order = available * tier_fraction

        # --- Step 3: exposure-utilisation scale-down ---
        util      = (self.total_exposure / self._max_exposure_usdc
                     if self._max_exposure_usdc > 0 else 0.0)
        threshold = config.DYNAMIC_SIZING.exposure_scale_threshold
        min_scale = config.DYNAMIC_SIZING.min_exposure_scale

        if util > threshold:
            # Linear interpolation: 1.0 at threshold → min_scale at 1.0
            scale = 1.0 - (1.0 - min_scale) * (util - threshold) / (1.0 - threshold)
        else:
            scale = 1.0

        order_usdc = min(base_order * scale, tier_cap, remaining)
        order_usdc = max(order_usdc, 0.0)

        log.debug(
            "SmartSize[%s]  wallet=$%.2f  avail=$%.2f  tier=%.0f%%  "
            "tier_cap=$%.2f  util=%.0f%%  scale=%.2f  → $%.2f",
            source, wallet, available,
            tier_fraction * 100, tier_cap,
            util * 100, scale, order_usdc,
        )

        return order_usdc

    # ------------------------------------------------------------------
    # Risk helpers
    # ------------------------------------------------------------------

    @property
    def total_exposure(self) -> float:
        return sum(self._open_exposure.values())

    def _on_cooldown(self, condition_id: str) -> bool:
        last = self._last_trade.get(condition_id, 0.0)
        return (time.monotonic() - last) < config.STRATEGY.trade_cooldown_secs

    def _too_many_positions(self, symbol: str) -> bool:
        count = sum(
            1 for m in self._cache.all_markets()
            if m.symbol == symbol and m.condition_id in self._open_exposure
        )
        return count >= config.STRATEGY.max_open_positions

    def _risk_ok(self, symbol: str) -> bool:
        """Combined pre-trade risk gate."""
        if self._wallet_balance < config.RISK.min_order_usdc:
            log.warning("Wallet balance $%.2f too low to trade.", self._wallet_balance)
            return False
        if self.total_exposure >= self._max_exposure_usdc:
            log.warning(
                "Exposure cap reached: $%.2f / $%.2f",
                self.total_exposure, self._max_exposure_usdc,
            )
            return False
        if self._too_many_positions(symbol):
            log.warning("Too many open positions for %s.", symbol)
            return False
        return True

    # ------------------------------------------------------------------
    # Momentum signal handler
    # ------------------------------------------------------------------

    async def execute_signal(
        self,
        symbol: str,
        direction: str,     # "UP" or "DOWN"
        price_move_pct: float,
    ) -> None:
        """
        Momentum signal: price moved fast, find matching markets and bet.
        direction="UP"   → buy Up token
        direction="DOWN" → buy Down token
        """
        log.info("MOMENTUM  %-3s  %s  %.3f%%", symbol, direction, price_move_pct)

        if not self._risk_ok(symbol):
            return

        markets = self._cache.get_markets_for_symbol(symbol)
        if not markets:
            log.debug("No markets found for %s.", symbol)
            return

        for market in markets:
            await self._trade_momentum(market, direction)

    async def _trade_momentum(self, market: MarketInfo, direction: str) -> None:
        cid = market.condition_id
        if self._on_cooldown(cid):
            return

        # Pick the token matching the direction
        if direction == "UP":
            token = market.up_token
            token_label = "Up"
            # Fetch fresh mid for up token
            try:
                mid = await self._client.get_midpoint(token.token_id)
            except Exception as exc:
                log.warning("Midpoint fetch failed %s: %s", cid[:8], exc)
                return
        else:
            token = market.down_token
            token_label = "Down"
            try:
                up_mid = await self._client.get_midpoint(market.up_token.token_id)
                mid = 1.0 - up_mid
            except Exception as exc:
                log.warning("Midpoint fetch failed %s: %s", cid[:8], exc)
                return

        # Don't chase already-extreme prices
        if not (config.STRATEGY.min_yes_prob <= mid <= config.STRATEGY.max_yes_prob):
            log.debug("Market %s: %s mid=%.3f outside range.", cid[:8], token_label, mid)
            return

        order_usdc = self._compute_smart_order_size(source="MOMENTUM")
        if order_usdc < config.RISK.min_order_usdc:
            return

        await self._place_order(
            cid=cid,
            token_id=token.token_id,
            token_label=token_label,
            mid=mid,
            order_usdc=order_usdc,
            question=market.question,
            source="MOMENTUM",
        )

    # ------------------------------------------------------------------
    # Arbitrage signal handler
    # ------------------------------------------------------------------

    async def execute_arb_signal(self, sig: "ArbSignal") -> None:
        """
        Arb signal: execute using fractional Kelly sizing vs wallet balance.

            kelly_usdc = sig.kelly_f × config.STRATEGY.kelly_fraction × wallet_balance
            order_usdc = clamp(kelly_usdc, min_order_usdc, max_order_usdc)
        """
        cid = sig.market.condition_id

        if self._on_cooldown(cid):
            log.debug("Market %s on cooldown (arb).", cid[:8])
            return

        if not self._risk_ok(sig.symbol):
            return

        # Dynamic cap: smart sizing accounts for wallet tier + exposure utilisation.
        # Kelly sizing is preserved but cannot exceed the dynamic cap.
        smart_cap  = self._compute_smart_order_size(source="ARB")
        kelly_usdc = sig.kelly_f * config.STRATEGY.kelly_fraction * self._wallet_balance
        order_usdc = max(
            config.RISK.min_order_usdc,
            min(kelly_usdc, smart_cap),
        )
        if order_usdc < config.RISK.min_order_usdc:
            return

        await self._place_order(
            cid=cid,
            token_id=sig.token_id,
            token_label=sig.bet,
            mid=sig.market_prob,
            order_usdc=order_usdc,
            question=sig.market.question,
            source=f"ARB edge={sig.edge:.3f} kelly={sig.kelly_f:.3f} wallet=${self._wallet_balance:.0f}",
        )

    # ------------------------------------------------------------------
    # Shared placement helper
    # ------------------------------------------------------------------

    async def _place_order(
        self,
        cid: str,
        token_id: str,
        token_label: str,
        mid: float,
        order_usdc: float,
        question: str,
        source: str,
    ) -> None:
        limit_price = round(min(mid + config.RISK.slippage_tolerance, 0.99), 4)
        shares = round(order_usdc / limit_price, 2)

        log.info(
            "[%s]  Buy %-4s  %s  @ %.4f  (%.2f shares / $%.2f USDC)"
            "  wallet=$%.2f  deployed=$%.2f",
            source, token_label, question[:50],
            limit_price, shares, order_usdc,
            self._wallet_balance, self.total_exposure,
        )

        resp = await self._client.create_limit_order(
            token_id=token_id,
            side="BUY",
            price=limit_price,
            size=shares,
        )

        if resp is not None:
            self._last_trade[cid] = time.monotonic()
            self._open_exposure[cid] = (
                self._open_exposure.get(cid, 0.0) + order_usdc
            )
            log.info(
                "Order recorded.  cid=%s  total_exposure=$%.2f / $%.2f (%.0f%%)",
                cid[:8], self.total_exposure,
                self._max_exposure_usdc,
                (self.total_exposure / self._max_exposure_usdc * 100)
                if self._max_exposure_usdc else 0,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def record_resolution(self, condition_id: str) -> None:
        """Call when a market resolves to free up tracked exposure."""
        self._open_exposure.pop(condition_id, None)
        self._last_trade.pop(condition_id, None)
