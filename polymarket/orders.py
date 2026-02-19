"""
Order sizing, execution, and early-exit management.

Bet sizing strategy
-------------------
All sizing is proportional to the **live USDC wallet balance**, fetched at
startup and refreshed every config.RISK.balance_refresh_secs seconds.

Per-order size (flat momentum bets):
    order_usdc = min(
        wallet_balance * max_order_fraction,   # e.g. 5% of wallet
        max_order_usdc_hard,                   # hard cap regardless of size
        remaining_budget,                      # what's still undeployed
    )

Per-order size (arb bets with Kelly sizing):
    kelly_usdc = kelly_f * kelly_fraction * wallet_balance
    order_usdc = max(min_order_usdc, min(kelly_usdc, flat_order_usdc))

Total exposure cap:
    max_deployed = wallet_balance * max_exposure_fraction   # e.g. 20%

Early exit (take-profit)
------------------------
After buying, the bot continuously monitors each open position.
When Polymarket's current market price has risen by exit_take_profit
(default 7¢) above the price we paid, it places a sell order:

    sell if: current_mid >= entry_price + exit_take_profit

This lets the bot:
  - Lock in gains from short-lived mispricings (buy at 0.50, sell at 0.57)
  - Redeploy capital into new arb opportunities faster
  - Avoid 10-minute lock-ups on positions that have already repriced

Example with $500 wallet:
    max_order  = min($500*0.05, $50) = $25 per bet
    max_deploy = $500*0.20 = $100 total across all open positions
    Kelly arb  = kelly_f * 0.25 * $500 — capped at $25
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polymarket.client import PolymarketClient
from polymarket.markets import MarketCache, MarketInfo
import config

if TYPE_CHECKING:
    from strategy.arbitrage import ArbSignal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Open position tracking
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    """Tracks an open bet for early-exit monitoring."""
    condition_id: str
    token_id: str
    bet: str            # "Up" or "Down"
    entry_price: float  # price we paid (limit price)
    shares: float       # shares bought
    cost_usdc: float    # USDC spent
    question: str
    symbol: str
    is_snipe: bool
    entered_at: float   # time.monotonic()


class OrderManager:
    """
    Manages order sizing, placement, position tracking, and early exits.

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
        # condition_id → OpenPosition (one position per market at a time)
        self._positions: dict[str, OpenPosition] = {}

    # ------------------------------------------------------------------
    # Wallet balance management
    # ------------------------------------------------------------------

    async def refresh_balance(self) -> None:
        """Fetch live USDC balance and cache it."""
        balance = await self._client.get_usdc_balance()
        changed = abs(balance - self._wallet_balance) > 0.01
        self._wallet_balance = balance
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
        await self.refresh_balance()
        while True:
            await asyncio.sleep(config.RISK.balance_refresh_secs)
            await self.refresh_balance()

    # ------------------------------------------------------------------
    # Derived sizing limits
    # ------------------------------------------------------------------

    @property
    def _available_balance(self) -> float:
        return max(0.0, self._wallet_balance - self.total_exposure)

    @property
    def _max_exposure_usdc(self) -> float:
        return self._wallet_balance * config.RISK.max_exposure_fraction

    @property
    def _max_order_usdc(self) -> float:
        return min(
            self._wallet_balance * config.RISK.max_order_fraction,
            config.RISK.max_order_usdc_hard,
        )

    def _remaining_budget(self) -> float:
        return max(0.0, self._max_exposure_usdc - self.total_exposure)

    # ------------------------------------------------------------------
    # Risk helpers
    # ------------------------------------------------------------------

    @property
    def total_exposure(self) -> float:
        return sum(p.cost_usdc for p in self._positions.values())

    def _on_cooldown(self, condition_id: str) -> bool:
        last = self._last_trade.get(condition_id, 0.0)
        return (time.monotonic() - last) < config.STRATEGY.trade_cooldown_secs

    def _too_many_positions(self, symbol: str) -> bool:
        count = sum(
            1 for p in self._positions.values() if p.symbol == symbol
        )
        return count >= config.STRATEGY.max_open_positions

    def _risk_ok(self, symbol: str) -> bool:
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
        direction: str,
        price_move_pct: float,
    ) -> None:
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
        if self._on_cooldown(cid) or cid in self._positions:
            return

        if direction == "UP":
            token = market.up_token
            token_label = "Up"
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

        if not (config.STRATEGY.min_yes_prob <= mid <= config.STRATEGY.max_yes_prob):
            log.debug("Market %s: %s mid=%.3f outside range.", cid[:8], token_label, mid)
            return

        order_usdc = min(self._max_order_usdc, self._remaining_budget())
        if order_usdc < config.RISK.min_order_usdc:
            return

        await self._place_order(
            cid=cid,
            token_id=token.token_id,
            token_label=token_label,
            mid=mid,
            order_usdc=order_usdc,
            question=market.question,
            symbol=market.symbol,
            source="MOMENTUM",
            is_snipe=False,
        )

    # ------------------------------------------------------------------
    # Arbitrage signal handler
    # ------------------------------------------------------------------

    async def execute_arb_signal(self, sig: "ArbSignal") -> None:
        cid = sig.market.condition_id

        if self._on_cooldown(cid):
            log.debug("Market %s on cooldown (%s).", cid[:8], "snipe" if sig.is_snipe else "arb")
            return

        if cid in self._positions:
            log.debug("Market %s already has an open position.", cid[:8])
            return

        if not self._risk_ok(sig.symbol):
            return

        kelly_mult = (
            config.STRATEGY.snipe_kelly_fraction
            if sig.is_snipe
            else config.STRATEGY.kelly_fraction
        )
        kelly_usdc = sig.kelly_f * kelly_mult * self._wallet_balance
        order_usdc = max(
            config.RISK.min_order_usdc,
            min(kelly_usdc, self._max_order_usdc, self._remaining_budget()),
        )
        if order_usdc < config.RISK.min_order_usdc:
            return

        mode = "SNIPE" if sig.is_snipe else "ARB"
        await self._place_order(
            cid=cid,
            token_id=sig.token_id,
            token_label=sig.bet,
            mid=sig.market_prob,
            order_usdc=order_usdc,
            question=sig.market.question,
            symbol=sig.symbol,
            source=(
                f"{mode} fair={sig.fair_prob:.3f} edge={sig.edge:.3f}"
                f" kelly={sig.kelly_f:.3f}×{kelly_mult} wallet=${self._wallet_balance:.0f}"
            ),
            is_snipe=sig.is_snipe,
        )

    # ------------------------------------------------------------------
    # Early exit scan
    # ------------------------------------------------------------------

    async def check_exits(self) -> None:
        """
        Scan all open positions and sell any that have repriced by
        config.STRATEGY.exit_take_profit cents above the entry price.

        Called every arb scan cycle (every ~2s).

        Skip exits if:
          - exit_take_profit == 0.0 (disabled)
          - Position is a snipe with very little time left (let it resolve)
          - Market is within exit_min_t_rem seconds of resolution
        """
        if config.STRATEGY.exit_take_profit == 0.0:
            return
        if not self._positions:
            return

        for cid, pos in list(self._positions.items()):
            # Find market to check time remaining
            market = self._cache.get_market(cid)
            t_rem = market.time_remaining_secs if market else 0.0

            # Don't sell within exit_min_t_rem of expiry — just let it resolve
            if t_rem < config.STRATEGY.exit_min_t_rem:
                continue

            # Snipe positions near the end of the window are near-locks;
            # holding to resolution collects the full $1 per share.
            if pos.is_snipe and t_rem < config.STRATEGY.snipe_window_secs:
                continue

            # Fetch current market price for the token we hold
            try:
                current_mid = await self._client.get_midpoint(pos.token_id)
            except Exception as exc:
                log.debug("Exit check: midpoint fetch failed %s: %s", cid[:8], exc)
                continue

            profit_per_share = current_mid - pos.entry_price

            if profit_per_share < config.STRATEGY.exit_take_profit:
                continue

            # Take-profit threshold reached — sell
            total_profit_usdc = profit_per_share * pos.shares
            log.info(
                "EXIT  %-3s  %s  %s  entry=%.4f  now=%.4f  "
                "profit=+$%.2f (%.1f%%)  t_rem=%.0fs",
                pos.symbol, cid[:8], pos.bet,
                pos.entry_price, current_mid,
                total_profit_usdc,
                profit_per_share / pos.entry_price * 100,
                t_rem,
            )
            await self._sell_position(pos, current_mid)

    async def _sell_position(self, pos: OpenPosition, current_mid: float) -> None:
        """Place a sell limit order slightly below mid to ensure quick fill."""
        sell_price = round(max(current_mid - config.RISK.slippage_tolerance, 0.01), 4)

        log.info(
            "[EXIT]  Sell %-4s  %s  @ %.4f  (%.2f shares / est. $%.2f USDC)"
            "  wallet=$%.2f",
            pos.bet, pos.question[:50],
            sell_price, pos.shares, sell_price * pos.shares,
            self._wallet_balance,
        )

        resp = await self._client.create_limit_order(
            token_id=pos.token_id,
            side="SELL",
            price=sell_price,
            size=pos.shares,
        )

        if resp is not None:
            self._positions.pop(pos.condition_id, None)
            self._last_trade[pos.condition_id] = time.monotonic()
            log.info(
                "Position exited early.  cid=%s  total_exposure=$%.2f",
                pos.condition_id[:8], self.total_exposure,
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
        symbol: str,
        source: str,
        is_snipe: bool,
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
            self._positions[cid] = OpenPosition(
                condition_id=cid,
                token_id=token_id,
                bet=token_label,
                entry_price=limit_price,
                shares=shares,
                cost_usdc=order_usdc,
                question=question,
                symbol=symbol,
                is_snipe=is_snipe,
                entered_at=time.monotonic(),
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
        """Call when a market resolves to free up tracked position."""
        self._positions.pop(condition_id, None)
        self._last_trade.pop(condition_id, None)
