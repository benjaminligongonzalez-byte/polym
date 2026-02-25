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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from feeds.aggregator import PriceAggregator
from polymarket.client import PolymarketClient
from polymarket.markets import MarketCache, MarketInfo
import config

if TYPE_CHECKING:
    from strategy.arbitrage import ArbSignal

log = logging.getLogger(__name__)

# ── ANSI colour constants (terminal display) ──────────────────────────────────
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_GREEN   = "\033[92m"
_RED     = "\033[91m"
_YELLOW  = "\033[93m"
_CYAN    = "\033[96m"
_MAGENTA = "\033[95m"
_ORANGE = "\033[33m"


# ---------------------------------------------------------------------------
# Open position tracking
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    """Tracks an open bet for early-exit monitoring."""
    condition_id: str
    token_id: str
    bet: str            # "Up" or "Down"
    entry_price: float  # limit price we paid = mid + slippage_tolerance
    entry_mid: float    # raw market mid at entry (no slippage) — breakeven reference
    shares: float       # shares bought
    cost_usdc: float    # USDC spent
    question: str
    symbol: str
    strike_price: float # strike K used in the fair-value model
    is_snipe: bool
    entered_at: float   # time.monotonic()
    source: str = ""    # "MOMENTUM", "ARB", "SNIPE"


@dataclass
class ClosedTrade:
    """Historical record of a completed bet."""
    condition_id: str
    symbol: str
    bet: str                    # "Up" or "Down"
    question: str
    entry_price: float
    exit_price: Optional[float] # None if resolved (we don't yet know outcome)
    shares: float
    cost_usdc: float
    pnl_usdc: Optional[float]   # None if resolved without early exit
    source: str                 # "MOMENTUM", "ARB", "SNIPE"
    close_type: str             # "early-exit" | "stop-loss" | "resolved"
    opened_at: float            # time.monotonic()
    closed_at: float            # time.monotonic()

    @property
    def is_win(self) -> Optional[bool]:
        if self.pnl_usdc is None:
            return None
        return self.pnl_usdc > 0


class OrderManager:
    """
    Manages order sizing, placement, position tracking, and early exits.

    Wallet balance is fetched at startup and kept fresh via
    run_balance_refresh_loop().
    """

    def __init__(
        self,
        client: PolymarketClient,
        cache: MarketCache,
        aggregator: PriceAggregator,
    ) -> None:
        self._client = client
        self._cache = cache
        self._aggregator = aggregator
        self._wallet_balance: float = 0.0          # live USDC balance
        self._last_balance_log: float = 0.0
        # condition_id → last trade timestamp (monotonic)
        self._last_trade: dict[str, float] = {}
        # condition_id → OpenPosition (one position per market at a time)
        self._positions: dict[str, OpenPosition] = {}
        # Trade history
        self._closed_trades: list[ClosedTrade] = []
        self._orders_placed: int = 0        # total orders sent (momentum + arb)
        self._blind_copy_count: int = 0     # blind copy orders placed
        self._blind_positions: list[OpenPosition] = []  # tracked separately from _positions
                                            # so check_exits doesn't prematurely resolve them
        self._mid_cache: dict[str, float] = {}  # token_id → last fetched mid (for unrealized P&L)
        self._session_start: float = time.monotonic()
        # Paper-trade running P&L (applied to wallet balance so sizing stays accurate)
        self._paper_pnl: float = 0.0
        # Pause / drain control
        self._paused: bool = False
        # Optional reference to MomentumStrategy for TA buffer access
        self._momentum: object | None = None

    def set_momentum(self, strategy: object) -> None:
        """Wire the MomentumStrategy so check_exits can read price buffers for TA."""
        self._momentum = strategy

    # ------------------------------------------------------------------
    # Wallet balance management
    # ------------------------------------------------------------------

    async def refresh_balance(self) -> None:
        """Fetch live USDC balance and cache it.

        In paper trade mode a virtual balance is used so sizing works even
        when the wallet holds no real USDC.
        """
        if config.PAPER_TRADE:
            # Start from the configured virtual balance and apply all realized P&L
            # so the wallet shrinks on losses and grows on wins — just like real trading.
            balance = config.PAPER_BALANCE_USDC + self._paper_pnl
        else:
            balance = await self._client.get_usdc_balance()
        changed = abs(balance - self._wallet_balance) > 0.01
        self._wallet_balance = balance
        now = time.monotonic()
        if changed or now - self._last_balance_log > 300:
            base_cap  = self._wallet_balance * config.RISK.per_symbol_base_fraction
            surge_cap = self._wallet_balance * config.RISK.per_symbol_surge_fraction
            log.info(
                "Wallet balance: $%.2f USDC  |  deployed: $%.2f  |  "
                "available: $%.2f  |  max_order: $%.2f  |  "
                "sym_cap: $%.0f–$%.0f (conviction-scaled)%s",
                self._wallet_balance,
                self.total_exposure,
                self._available_balance,
                self._max_order_usdc,
                base_cap, surge_cap,
                "  [PAUSED]" if self._paused else "",
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
        """Global safety ceiling (high; real throttle is per-symbol)."""
        return self._wallet_balance * config.RISK.max_exposure_fraction

    @staticmethod
    def _conviction_score(edge: float, fair_prob: float) -> float:
        """
        Map (edge, fair_prob) → [0.0, 1.0] conviction score.

        Conviction is the average of two independent subscores:
          edge_conv  = min(1, edge / conviction_edge_scale)
          prob_conv  = (fair_prob - prob_floor) / (prob_ceil - prob_floor), clipped

        0.0 → weak/unscored (momentum default)
        1.0 → near-certain (strong snipe)
        """
        cfg = config.RISK
        edge_conv = min(1.0, edge / cfg.conviction_edge_scale) if cfg.conviction_edge_scale > 0 else 0.0
        prob_range = cfg.conviction_prob_ceil - cfg.conviction_prob_floor
        prob_conv  = max(0.0, min(1.0, (fair_prob - cfg.conviction_prob_floor) / prob_range)) if prob_range > 0 else 0.0
        return (edge_conv + prob_conv) / 2.0

    def _dynamic_sym_cap(self, edge: float = 0.0, fair_prob: float = 0.5) -> float:
        """
        Per-symbol USDC cap, scaled by conviction.

        Weak signal  (conviction=0.0): per_symbol_base_fraction  × wallet
        Near-certain (conviction=1.0): per_symbol_surge_fraction × wallet
        """
        conviction = self._conviction_score(edge, fair_prob)
        base  = config.RISK.per_symbol_base_fraction
        surge = config.RISK.per_symbol_surge_fraction
        fraction = base + (surge - base) * conviction
        return self._wallet_balance * fraction

    def _symbol_exposure(self, symbol: str) -> float:
        """Total USDC currently deployed in open positions for one symbol."""
        return sum(
            p.cost_usdc for p in self._positions.values() if p.symbol == symbol
        )

    @property
    def _max_order_usdc(self) -> float:
        return min(
            self._wallet_balance * config.RISK.max_order_fraction,
            config.RISK.max_order_usdc_hard,
        )

    def _remaining_budget(
        self, symbol: str = "", edge: float = 0.0, fair_prob: float = 0.5
    ) -> float:
        """
        Available budget for the next order.

        Constrained by:
          1. Per-symbol dynamic cap  (primary, scales with edge + fair_prob)
          2. Global safety cap       (backstop for extreme edge cases)
        """
        global_room = max(0.0, self._max_exposure_usdc - self.total_exposure)
        if symbol:
            sym_cap  = self._dynamic_sym_cap(edge, fair_prob)
            sym_room = max(0.0, sym_cap - self._symbol_exposure(symbol))
            return min(global_room, sym_room)
        return global_room

    # ------------------------------------------------------------------
    # Risk helpers
    # ------------------------------------------------------------------

    @property
    def total_exposure(self) -> float:
        return (
            sum(p.cost_usdc for p in self._positions.values())
            + sum(p.cost_usdc for p in self._blind_positions)
        )

    def _on_cooldown(self, condition_id: str) -> bool:
        last = self._last_trade.get(condition_id, 0.0)
        return (time.monotonic() - last) < config.STRATEGY.trade_cooldown_secs

    def _too_many_positions(self, symbol: str) -> bool:
        count = sum(
            1 for p in self._positions.values() if p.symbol == symbol
        )
        return count >= config.STRATEGY.max_open_positions

    def _risk_ok(self, symbol: str, edge: float = 0.0, fair_prob: float = 0.5) -> bool:
        if self._paused:
            log.debug("Bot is paused — skipping new order for %s.", symbol)
            return False
        if self._wallet_balance < config.RISK.min_order_usdc:
            log.warning("Wallet balance $%.2f too low to trade.", self._wallet_balance)
            return False
        # Per-symbol dynamic cap
        sym_exp = self._symbol_exposure(symbol)
        sym_cap = self._dynamic_sym_cap(edge, fair_prob)
        conviction = self._conviction_score(edge, fair_prob)
        if sym_exp >= sym_cap:
            log.warning(
                "%s per-symbol cap reached: $%.2f / $%.2f  (conviction=%.2f)",
                symbol, sym_exp, sym_cap, conviction,
            )
            return False
        # Global safety net
        if self.total_exposure >= self._max_exposure_usdc:
            log.warning(
                "Global exposure cap reached: $%.2f / $%.2f",
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
        dir_arrow = f"{_GREEN}▲ UP{_RESET}" if direction == "UP" else f"{_RED}▼ DOWN{_RESET}"
        log.info(
            "%sMOMENTUM%s  %s%s%s  %s  %.3f%%",
            _YELLOW, _RESET,
            _CYAN, symbol, _RESET,
            dir_arrow, price_move_pct,
        )

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

        # Skip markets that have expired or won't open for 15+ minutes
        t_rem = market.time_remaining_secs
        if t_rem <= 0:
            log.debug("Skipping expired market %s (%s).", cid[:8], market.question[:40])
            return
        if t_rem > 930:
            log.debug("Skipping far-future market %s (t_rem=%.0fs).", cid[:8], t_rem)
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

        # ── Fair-value gate ──────────────────────────────────────────────────
        # Require model edge ≥ momentum_min_edge before entering.
        # This prevents buying tokens the model considers overpriced and
        # entering in the opposite direction from our own fair-value signal.
        # If no consensus price or strike is available, skip — never trade blind.
        from strategy.arbitrage import fair_prob_up  # local import avoids circular
        consensus = self._aggregator.consensus_price(market.symbol)
        if consensus is None or not market.strike_price:
            log.debug(
                "MOMENTUM gate: no consensus/strike for %s — skip.",
                market.symbol,
            )
            return
        vol  = config.VOLATILITY.get(market.symbol, config.DEFAULT_ANNUAL_VOL)
        p_up = fair_prob_up(consensus, market.strike_price, max(t_rem, 5), vol)
        fair = p_up if direction == "UP" else (1.0 - p_up)
        edge = fair - mid
        if edge < config.STRATEGY.momentum_min_edge:
            log.debug(
                "MOMENTUM gate: %s %s  fair=%.3f  mkt=%.3f  edge=%+.3f < min=%.3f — skip",
                market.symbol, token_label, fair, mid, edge,
                config.STRATEGY.momentum_min_edge,
            )
            return
        # ── TA confirmation ──────────────────────────────────────────────────
        # Use live technical analysis to filter out momentum signals that fire
        # in unfavourable conditions (overbought Up, oversold Down, counter-trend).
        # Skips gracefully if the price buffer doesn't have enough data yet
        # (first 4-5 minutes of a session) — FLAT signals don't block trades.
        if self._momentum is not None:
            from strategy.tech_analysis import compute_ta
            ta = compute_ta(self._momentum.get_buffer(market.symbol))  # type: ignore[union-attr]
            if direction == "UP":
                if ta.rsi is not None and ta.rsi > config.STRATEGY.ta_rsi_overbought:
                    log.debug(
                        "MOMENTUM TA: %s UP RSI=%.0f > %.0f (overbought) — skip",
                        market.symbol, ta.rsi, config.STRATEGY.ta_rsi_overbought,
                    )
                    return
                if ta.trend == "DOWN":
                    log.debug(
                        "MOMENTUM TA: %s UP but trend=DOWN — skip", market.symbol,
                    )
                    return
            else:  # direction == "DOWN"
                if ta.rsi is not None and ta.rsi < config.STRATEGY.ta_rsi_oversold:
                    log.debug(
                        "MOMENTUM TA: %s DOWN RSI=%.0f < %.0f (oversold) — skip",
                        market.symbol, ta.rsi, config.STRATEGY.ta_rsi_oversold,
                    )
                    return
                if ta.trend == "UP":
                    log.debug(
                        "MOMENTUM TA: %s DOWN but trend=UP — skip", market.symbol,
                    )
                    return
        # ────────────────────────────────────────────────────────────────────

        order_usdc = min(self._max_order_usdc, self._remaining_budget(market.symbol))
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
            strike_price=market.strike_price or 0.0,
            source=f"MOMENTUM fair={fair:.3f} edge={edge:+.3f}",
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

        if not self._risk_ok(sig.symbol, edge=sig.edge, fair_prob=sig.fair_prob):
            return

        kelly_mult = (
            config.STRATEGY.snipe_kelly_fraction
            if sig.is_snipe
            else config.STRATEGY.kelly_fraction
        )
        kelly_usdc = sig.kelly_f * kelly_mult * self._wallet_balance
        order_usdc = max(
            config.RISK.min_order_usdc,
            min(
                kelly_usdc,
                self._max_order_usdc,
                self._remaining_budget(sig.symbol, edge=sig.edge, fair_prob=sig.fair_prob),
            ),
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
            strike_price=sig.strike_price,
            source=(
                f"{mode} fair={sig.fair_prob:.3f} edge={sig.edge:.3f}"
                f" kelly={sig.kelly_f:.3f}×{kelly_mult} wallet=${self._wallet_balance:.0f}"
            ),
            is_snipe=sig.is_snipe,
        )

    # ------------------------------------------------------------------
    # Blind copy-trade handler
    # ------------------------------------------------------------------

    async def execute_blind_copy(
        self,
        trade: "TrackedTrade",  # type: ignore[name-defined]
        target_wallet_value: float | None,
    ) -> None:
        """
        Copy a trade from the tracked wallet WITHOUT running through any
        strategy gates (no fair-value, no TA, no edge requirement).

        Order sizing is proportional: we put the same fraction of our wallet
        that the target used of theirs.  Falls back to COPY_BLIND_USDC if
        the target's wallet value is unknown.

        Only hard guards applied:
          - bot must not be paused
          - we must have sufficient balance
          - order size is clamped to [min_order_usdc, max_order_usdc_hard]
        """
        from tracking.tracker import TrackedTrade  # local import

        if not isinstance(trade, TrackedTrade):
            return
        if self._paused:
            log.debug("Blind copy skipped — bot is paused.")
            return

        # ── Proportional sizing ────────────────────────────────────────
        if target_wallet_value and target_wallet_value > 0 and trade.amount > 0:
            fraction  = trade.amount / target_wallet_value
            order_usdc = fraction * self._wallet_balance
            log.debug(
                "Blind copy sizing: target used %.2f / %.2f = %.2f%% → our order $%.2f",
                trade.amount, target_wallet_value, fraction * 100, order_usdc,
            )
        else:
            order_usdc = config.COPY_BLIND_USDC
            log.debug(
                "Blind copy sizing: no target wallet value — using fallback $%.2f",
                order_usdc,
            )

        order_usdc = max(
            config.RISK.min_order_usdc,
            min(order_usdc, config.RISK.max_order_usdc_hard),
        )

        # ── Balance check ──────────────────────────────────────────────
        min_reserve = getattr(config.RISK, "min_wallet_balance", 0.0)
        if self._wallet_balance - order_usdc < min_reserve:
            log.warning(
                "Blind copy: insufficient balance $%.2f for $%.2f order (reserve $%.2f) — skip",
                self._wallet_balance, order_usdc, min_reserve,
            )
            return

        # ── Token ID guard ─────────────────────────────────────────────
        if not trade.token_id:
            log.warning(
                "Blind copy: no token_id for trade %s — cannot place order. "
                "Run --loglevel DEBUG to see raw API fields.",
                trade.trade_id[:16],
            )
            return

        # ── Place order ────────────────────────────────────────────────
        limit_price = round(min(trade.price + config.RISK.slippage_tolerance, 0.99), 4)
        shares      = round(order_usdc / limit_price, 2)

        outcome_col = _GREEN if trade.outcome in ("Up", "Yes") else _RED
        log.info(
            "%s[BLIND-COPY]%s  %s%s%s  %s  @ %.4f  "
            "(%.2f shares / $%.2f USDC)  %swallet=$%.2f%s",
            _MAGENTA, _RESET,
            outcome_col, trade.outcome, _RESET,
            trade.question[:50],
            limit_price, shares, order_usdc,
            _BOLD, self._wallet_balance, _RESET,
        )

        resp = await self._client.create_limit_order(
            token_id=trade.token_id,
            side="BUY",
            price=limit_price,
            size=shares,
        )

        if resp is not None:
            self._blind_copy_count += 1
            self._blind_positions.append(OpenPosition(
                condition_id=trade.condition_id,
                token_id=trade.token_id,
                bet=trade.outcome or "?",
                entry_price=limit_price,
                entry_mid=trade.price,
                shares=shares,
                cost_usdc=order_usdc,
                question=trade.question,
                symbol=trade.symbol or "?",
                strike_price=0.0,
                is_snipe=False,
                entered_at=time.monotonic(),
                source="BLIND-COPY",
            ))
            log.info(
                "%s[BLIND-COPY]%s  order placed — %s%s%s  token=%s  cid=%s",
                _MAGENTA, _RESET,
                outcome_col, trade.outcome, _RESET,
                trade.token_id[:16] + "…",
                trade.condition_id[:12] + "…",
            )

    # ------------------------------------------------------------------
    # Blind copy sell mirror
    # ------------------------------------------------------------------

    async def execute_blind_sell(
        self,
        trade: "TrackedTrade",  # type: ignore[name-defined]
    ) -> None:
        """
        Mirror a SELL from the target wallet: close every blind copy position
        we hold in the same token.

        We sell ALL matching positions when the target sells any amount —
        the simplest safe behaviour for a scalper who exits fully before
        re-entering.
        """
        if self._paused:
            return

        token_id = trade.token_id
        if not token_id:
            log.debug("Blind sell: no token_id on SELL trade — skip")
            return

        matching = [p for p in self._blind_positions if p.token_id == token_id]
        if not matching:
            # Target sold a position we never copied (e.g. pre-session entry)
            log.debug(
                "Blind sell: no matching position for token %s — target sold pre-session hold",
                token_id[:16],
            )
            return

        # Get current mid — use cache first, fetch fresh if missing
        mid = self._mid_cache.get(token_id)
        if mid is None:
            try:
                mid = await self._client.get_midpoint(token_id)
                self._mid_cache[token_id] = mid
            except Exception as exc:
                log.warning("Blind sell: midpoint fetch failed for %s: %s", token_id[:16], exc)
                return

        sell_price = round(max(mid - config.RISK.slippage_tolerance, 0.01), 4)

        for pos in matching:
            pnl = round((sell_price - pos.entry_price) * pos.shares, 4)
            pnl_col  = _GREEN if pnl >= 0 else _RED
            pnl_sign = "+" if pnl >= 0 else ""
            log.info(
                "%s[BLIND-SELL]%s  %s%s%s  %s  entry=%.4f → %.4f  "
                "%s%s$%s%.4f%s  (%.2f shares)",
                _MAGENTA, _RESET,
                _GREEN if pos.bet in ("Up", "Yes") else _RED, pos.bet, _RESET,
                pos.question[:45],
                pos.entry_price, sell_price,
                pnl_col, _BOLD, pnl_sign, pnl, _RESET,
                pos.shares,
            )
            resp = await self._client.create_limit_order(
                token_id=pos.token_id,
                side="SELL",
                price=sell_price,
                size=pos.shares,
            )
            if resp is not None:
                self._blind_positions.remove(pos)
                if config.PAPER_TRADE:
                    self._paper_pnl += pnl
                    self._wallet_balance = config.PAPER_BALANCE_USDC + self._paper_pnl
                self._closed_trades.append(ClosedTrade(
                    condition_id=pos.condition_id,
                    symbol=pos.symbol,
                    bet=pos.bet,
                    question=pos.question,
                    entry_price=pos.entry_price,
                    exit_price=sell_price,
                    shares=pos.shares,
                    cost_usdc=pos.cost_usdc,
                    pnl_usdc=pnl,
                    source="BLIND-COPY",
                    close_type="blind-sell",
                    opened_at=pos.entered_at,
                    closed_at=time.monotonic(),
                ))

        remaining = len([p for p in self._blind_positions if p.token_id == token_id])
        sold = len(matching) - remaining
        log.info(
            "%s[BLIND-SELL]%s  closed %d position(s) for token %s…  "
            "(%d remaining in other markets)",
            _MAGENTA, _RESET, sold, token_id[:16],
            len(self._blind_positions),
        )

    # ------------------------------------------------------------------
    # Early exit scan
    # ------------------------------------------------------------------

    async def check_exits(self) -> None:
        """
        Scan all open positions and exit any that have sufficiently repriced.

        Two exit triggers (either is enough to sell):

        1. Fair-value alignment (primary):
               current_mid >= live_fair_prob - exit_residual_edge
           i.e. Polymarket has caught up to within exit_residual_edge (default 2¢)
           of our fair-value estimate.  This captures the full reprice, not just
           a flat bump.  Requires a live consensus price from the aggregator.

        2. Flat take-profit (fallback / safety):
               current_mid - entry_price >= exit_take_profit
           Used when we have no live price (stale aggregator) or as an extra
           safety net.  Disabled when exit_take_profit == 0.0.

        Skip exits if:
          - Market is within exit_min_t_rem seconds of resolution (nearly over)
          - Position is a snipe with very little time left (let it resolve)
        """
        from strategy.arbitrage import fair_prob_up  # local import to avoid circular

        if not self._positions and not self._blind_positions:
            return

        for cid, pos in list(self._positions.items()):
            market = self._cache.get_market(cid)
            t_rem = market.time_remaining_secs if market else 0.0

            # Market fully expired — free capital; outcome unknown until oracle settles.
            if t_rem <= 0:
                log.info(
                    "Market expired: releasing %s %s/%s position  "
                    "(cost=$%.2f — recorded as 'resolved', P&L pending oracle).",
                    pos.symbol, pos.bet, cid[:8], pos.cost_usdc,
                )
                self.record_resolution(cid)
                continue

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
                self._mid_cache[pos.token_id] = current_mid
            except Exception as exc:
                log.debug("Exit check: midpoint fetch failed %s: %s", cid[:8], exc)
                continue

            # Pre-compute current fair probability (needed for guard + triggers).
            consensus = self._aggregator.consensus_price(pos.symbol)
            current_fair: float | None = None
            if consensus is not None:
                vol = config.VOLATILITY.get(pos.symbol, config.DEFAULT_ANNUAL_VOL)
                p_up = fair_prob_up(consensus, pos.strike_price, t_rem, vol)
                current_fair = p_up if pos.bet == "Up" else (1.0 - p_up)

            # Per-strategy stop-loss threshold:
            #   MOMENTUM entered at market price (~0.50-0.55) — use a looser gate.
            #   ARB/SNIPE entered only because fair ≥ 0.65 — apply the tighter gate.
            is_momentum = pos.source == "MOMENTUM"
            stop_loss_threshold = (
                config.STRATEGY.momentum_stop_loss_fair
                if is_momentum
                else config.STRATEGY.arb_min_fair_prob
            )

            # Guard: hold through temporary dips.
            #
            # CRITICAL BUG FIX: entry_price = mid_at_entry + slippage, so comparing
            # (current_mid - slip) <= entry_price is ALWAYS true immediately after entry
            # (since mid hasn't moved yet).  This caused every trade to show 0s held.
            #
            # Fix: compare current_mid to pos.entry_mid (the raw mid with NO slippage).
            # The guard only fires when the market has genuinely moved against us.
            if current_mid >= pos.entry_mid:
                # Price is at or above our entry midpoint — no loss to protect against.
                # Fall through to take-profit triggers only; stop-loss does not apply.
                pass
            else:
                # Price has fallen below our entry midpoint — check conviction.
                if current_fair is None or current_fair >= stop_loss_threshold:
                    continue  # model still confident — hold through the dip
                # conviction gone → fall through to stop-loss trigger

            # ---- Trigger 1: fair-value alignment (profit capture) ----
            should_exit = False
            exit_reason = ""
            if config.STRATEGY.exit_residual_edge > 0.0 and current_fair is not None:
                residual = current_fair - current_mid
                if residual <= config.STRATEGY.exit_residual_edge:
                    should_exit = True
                    exit_reason = (
                        f"fair-val fair={current_fair:.3f} mkt={current_mid:.3f} "
                        f"residual={residual:.3f}"
                    )

            # ---- Trigger 1b: stop-loss (conviction lost + TA confirmation) ----
            # Exit at a loss only when the model's fair probability has dropped
            # below the threshold AND live TA confirms the trade has turned against us.
            # This prevents paper-hands exits on momentary mid-price wobbles where
            # the underlying crypto price is still moving in our favor.
            if not should_exit and current_fair is not None:
                if current_fair < stop_loss_threshold:
                    from strategy.tech_analysis import compute_ta
                    ta = (
                        compute_ta(self._momentum.get_buffer(pos.symbol))  # type: ignore[union-attr]
                        if self._momentum is not None
                        else None
                    )
                    ta_supports = (
                        (ta.supports_up() if pos.bet == "Up" else ta.supports_down())
                        if ta is not None
                        else False
                    )
                    # Exit immediately when conviction is very low (no TA rescue)
                    conviction_very_low = current_fair < stop_loss_threshold * 0.75
                    if conviction_very_low or not ta_supports:
                        strat_label = "momentum" if is_momentum else "arb"
                        ta_desc = ta.describe() if ta is not None else "no-ta"
                        should_exit = True
                        pos_age = time.monotonic() - pos.entered_at
                        exit_reason = (
                            f"stop-loss fair={current_fair:.3f} < "
                            f"{strat_label}_thresh={stop_loss_threshold:.2f} "
                            f"{ta_desc} held={pos_age:.0f}s"
                        )

            # ---- Trigger 2: flat take-profit fallback ----
            if not should_exit and config.STRATEGY.exit_take_profit > 0.0:
                profit_per_share = current_mid - pos.entry_price
                if profit_per_share >= config.STRATEGY.exit_take_profit:
                    should_exit = True
                    exit_reason = (
                        f"flat-tp entry={pos.entry_price:.4f} now={current_mid:.4f} "
                        f"profit={profit_per_share:+.4f}"
                    )

            if not should_exit:
                continue

            profit_per_share = current_mid - pos.entry_price
            total_profit_usdc = profit_per_share * pos.shares
            pnl_col = _GREEN if total_profit_usdc >= 0 else _RED
            close_type = "stop-loss" if exit_reason.startswith("stop-loss") else "early-exit"
            exit_tag = f"{_RED}[STOP-LOSS]{_RESET}" if close_type == "stop-loss" else f"{_GREEN}[EXIT]{_RESET}"
            log.info(
                "%s  %-3s  %s  %s  entry=%.4f → now=%.4f  "
                "%sprofit=$%+.2f (%.1f%%)%s  t_rem=%.0fs  [%s]",
                exit_tag, pos.symbol, cid[:8], pos.bet,
                pos.entry_price, current_mid,
                pnl_col, total_profit_usdc,
                profit_per_share / pos.entry_price * 100, _RESET,
                t_rem,
                exit_reason,
            )
            await self._sell_position(pos, current_mid, close_type=close_type)

        # Refresh midpoint cache for blind copy positions (no exit logic — just pricing)
        for pos in self._blind_positions:
            if not pos.token_id:
                continue
            try:
                mid = await self._client.get_midpoint(pos.token_id)
                self._mid_cache[pos.token_id] = mid
            except Exception:
                pass

    async def _sell_position(
        self, pos: OpenPosition, current_mid: float, close_type: str = "early-exit"
    ) -> None:
        """Place a sell limit order slightly below mid to ensure quick fill."""
        sell_price = round(max(current_mid - config.RISK.slippage_tolerance, 0.01), 4)

        dir_col = _GREEN if pos.bet == "Up" else _RED
        log.info(
            "%s[SELL]%s  %s%-4s%s  %s  @ %.4f  (%.2f shares / est. $%.2f USDC)"
            "  %swallet=$%.2f%s",
            _YELLOW, _RESET,
            dir_col, pos.bet, _RESET,
            pos.question[:50],
            sell_price, pos.shares, sell_price * pos.shares,
            _BOLD, self._wallet_balance, _RESET,
        )

        resp = await self._client.create_limit_order(
            token_id=pos.token_id,
            side="SELL",
            price=sell_price,
            size=pos.shares,
        )

        if resp is not None:
            pnl = round((sell_price - pos.entry_price) * pos.shares, 4)
            # Update paper balance immediately so the next trade sizes correctly.
            if config.PAPER_TRADE:
                self._paper_pnl += pnl
                self._wallet_balance = config.PAPER_BALANCE_USDC + self._paper_pnl
            self._closed_trades.append(ClosedTrade(
                condition_id=pos.condition_id,
                symbol=pos.symbol,
                bet=pos.bet,
                question=pos.question,
                entry_price=pos.entry_price,
                exit_price=sell_price,
                shares=pos.shares,
                cost_usdc=pos.cost_usdc,
                pnl_usdc=pnl,
                source=pos.source,
                close_type=close_type,
                opened_at=pos.entered_at,
                closed_at=time.monotonic(),
            ))
            self._positions.pop(pos.condition_id, None)
            self._last_trade[pos.condition_id] = time.monotonic()
            pnl_col2 = _GREEN if pnl >= 0 else _RED
            log.info(
                "Position closed.  cid=%s  %spnl=$%+.4f%s  total_exposure=$%.2f",
                pos.condition_id[:8], pnl_col2, pnl, _RESET, self.total_exposure,
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
        strike_price: float,
        source: str,
        is_snipe: bool,
    ) -> None:
        limit_price = round(min(mid + config.RISK.slippage_tolerance, 0.99), 4)
        shares = round(order_usdc / limit_price, 2)

        dir_col = _GREEN if token_label == "Up" else _RED
        log.info(
            "%s[BUY]%s  %s%-4s%s  %s  @ %.4f  "
            "(%.2f shares / $%.2f USDC)  "
            "%swallet=$%.2f%s  deployed=$%.2f",
            _CYAN, _RESET,
            dir_col, token_label, _RESET,
            question[:50],
            limit_price, shares, order_usdc,
            _BOLD, self._wallet_balance, _RESET,
            self.total_exposure,
        )

        resp = await self._client.create_limit_order(
            token_id=token_id,
            side="BUY",
            price=limit_price,
            size=shares,
        )

        if resp is not None:
            self._orders_placed += 1
            self._last_trade[cid] = time.monotonic()
            # First word of source is the clean strategy tag ("MOMENTUM"/"ARB"/"SNIPE")
            strategy_tag = source.split()[0]
            self._positions[cid] = OpenPosition(
                condition_id=cid,
                token_id=token_id,
                bet=token_label,
                entry_price=limit_price,
                entry_mid=mid,          # raw mid at entry, no slippage
                shares=shares,
                cost_usdc=order_usdc,
                question=question,
                symbol=symbol,
                strike_price=strike_price,
                is_snipe=is_snipe,
                entered_at=time.monotonic(),
                source=strategy_tag,
            )
            log.info(
                "%sOrder recorded%s  cid=%s  total_exposure=$%.2f / $%.2f (%.0f%%)",
                _DIM, _RESET,
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
        pos = self._positions.pop(condition_id, None)
        self._last_trade.pop(condition_id, None)
        if pos is not None:
            # P&L unknown until the oracle sets the final price; mark as resolved.
            self._closed_trades.append(ClosedTrade(
                condition_id=condition_id,
                symbol=pos.symbol,
                bet=pos.bet,
                question=pos.question,
                entry_price=pos.entry_price,
                exit_price=None,
                shares=pos.shares,
                cost_usdc=pos.cost_usdc,
                pnl_usdc=None,
                source=pos.source,
                close_type="resolved",
                opened_at=pos.entered_at,
                closed_at=time.monotonic(),
            ))

    # ------------------------------------------------------------------
    # Pause / drain control
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Stop accepting new BUY orders. Existing positions keep running."""
        self._paused = True
        log.info(
            "Bot PAUSED — no new orders will be placed.  "
            "Open positions: %d  |  deployed: $%.2f",
            len(self._positions), self.total_exposure,
        )

    def resume(self) -> None:
        """Resume accepting new BUY orders."""
        self._paused = False
        log.info("Bot RESUMED — new orders enabled.")

    # ------------------------------------------------------------------
    # Live stats / console reporting
    # ------------------------------------------------------------------

    def _unrealized_pnl_lines(self) -> list[str]:
        """
        Compute unrealized P&L from cached midpoints for all open positions
        (own-strategy + blind copies).  Returns formatted lines for stats_report.
        Positions with no cached mid are shown as '? (no price yet)'.
        """
        all_positions = list(self._positions.values()) + self._blind_positions
        if not all_positions:
            return []

        lines: list[str] = []
        total_unrealized = 0.0
        any_missing = False
        per_pos: list[tuple[str, float, float]] = []  # (label, cost, unrealized)

        for pos in all_positions:
            mid = self._mid_cache.get(pos.token_id)
            if mid is None:
                any_missing = True
                continue
            unrealized = (mid - pos.entry_price) * pos.shares
            total_unrealized += unrealized
            label = f"{pos.symbol}/{pos.bet}" if pos.symbol != "?" else pos.bet
            per_pos.append((label, pos.cost_usdc, unrealized))

        if not per_pos and any_missing:
            lines.append(f"  Unrealized P&L  : {_DIM}waiting for first price update…{_RESET}")
            return lines

        u_col  = _GREEN if total_unrealized >= 0 else _RED
        u_sign = "+" if total_unrealized >= 0 else ""
        stale  = f"  {_DIM}(partial — {sum(1 for p in all_positions if p.token_id not in self._mid_cache)} pos. no price yet){_RESET}" if any_missing else ""
        lines.append(
            f"  Unrealized P&L  : {u_col}{_BOLD}${u_sign}{total_unrealized:.4f}{_RESET} USDC"
            f"  across {len(per_pos)} position(s){stale}"
        )

        # Per-position breakdown (collapsed to keep the display tidy)
        for label, cost, upnl in sorted(per_pos, key=lambda x: x[2]):
            col  = _GREEN if upnl >= 0 else _RED
            sign = "+" if upnl >= 0 else ""
            pct  = upnl / cost * 100 if cost else 0.0
            lines.append(
                f"    {_DIM}├{_RESET} {label:<12}  "
                f"{col}{sign}${upnl:.4f}  ({sign}{pct:.1f}%){_RESET}"
            )

        return lines

    def stats_report(self) -> str:
        """Return a multi-line formatted stats summary for the console."""
        uptime_secs = time.monotonic() - self._session_start
        h, rem = divmod(int(uptime_secs), 3600)
        m, s   = divmod(rem, 60)

        # Closed trades with known P&L (early exits and stop-losses)
        priced = [t for t in self._closed_trades if t.pnl_usdc is not None]
        wins   = [t for t in priced if t.pnl_usdc > 0]
        losses = [t for t in priced if t.pnl_usdc <= 0]
        resolved = [t for t in self._closed_trades if t.close_type == "resolved"]

        realized_pnl = sum(t.pnl_usdc for t in priced)
        win_rate = len(wins) / len(priced) * 100 if priced else 0.0

        # Unrealised P&L from open positions (mark-to-market not available here,
        # so show cost basis only)
        deployed = self.total_exposure

        base_cap  = self._wallet_balance * config.RISK.per_symbol_base_fraction
        surge_cap = self._wallet_balance * config.RISK.per_symbol_surge_fraction
        status = "⏸  PAUSED  (no new orders)" if self._paused else "▶  RUNNING"

        pnl_col    = _GREEN if realized_pnl >= 0 else _RED
        pnl_sign   = "+" if realized_pnl >= 0 else ""
        wr_col     = _GREEN if win_rate >= 50 else _RED
        depl_pct   = f" ({deployed / self._wallet_balance * 100:.1f}%)" if self._wallet_balance else ""

        lines = [
            f"{_BOLD}{'═' * 60}{_RESET}",
            f"  {_BOLD}POLYMARKET BOT{_RESET}  —  uptime {h:02d}h {m:02d}m {s:02d}s  |  {status}",
            f"{'─' * 60}",
            f"  {_BOLD}Wallet balance{_RESET}  : {_BOLD}${self._wallet_balance:>10.4f}{_RESET} USDC",
            f"  Deployed        : ${deployed:>10.4f} USDC{depl_pct}",
            f"  Available       : ${self._wallet_balance - deployed:>10.4f} USDC",
            f"  {_DIM}Sym cap (base)  : ${base_cap:>10.2f}   (conviction=0.0 / momentum){_RESET}",
            f"  {_DIM}Sym cap (surge) : ${surge_cap:>10.2f}   (conviction=1.0 / near-certain snipe){_RESET}",
            f"{'─' * 60}",
            f"  Orders placed   : {_CYAN}{self._orders_placed}{_RESET}"
            + (f"  {_MAGENTA}(+ {self._blind_copy_count} blind copies){_RESET}" if self._blind_copy_count else ""),
            f"  Open positions  : {_CYAN}{len(self._positions)}{_RESET}"
            + (f"  {_MAGENTA}(+ {len(self._blind_positions)} blind){_RESET}" if self._blind_positions else ""),
            f"  Closed trades   : {len(self._closed_trades)}",
            f"    ├ Early exits : {len(priced)}  ({_GREEN}wins={len(wins)}{_RESET}  {_RED}losses={len(losses)}{_RESET})",
            f"    └ Resolved    : {len(resolved)}  (P&L pending market resolution)",
            f"{'─' * 60}",
            f"  Realized P&L    : {pnl_col}{_BOLD}${pnl_sign}{realized_pnl:.4f}{_RESET} USDC",
            f"  Win rate        : {wr_col}{_BOLD}{win_rate:.1f}%{_RESET}  ({len(wins)}/{len(priced)} priced trades)",
            *self._unrealized_pnl_lines(),
        ]

        # Per-symbol breakdown
        symbols = sorted({t.symbol for t in self._closed_trades} | {p.symbol for p in self._positions.values()})
        if symbols:
            lines.append(f"{'─' * 60}")
            lines.append("  Per-symbol breakdown:")
            for sym in symbols:
                sym_priced = [t for t in priced if t.symbol == sym]
                sym_open   = sum(1 for p in self._positions.values() if p.symbol == sym)
                sym_pnl    = sum(t.pnl_usdc for t in sym_priced)
                sym_wins   = sum(1 for t in sym_priced if t.pnl_usdc > 0)
                sym_col    = _GREEN if sym_pnl >= 0 else _RED
                lines.append(
                    f"  {_CYAN}{sym:<4}{_RESET}  open={sym_open}  closed={len(sym_priced)}"
                    f"  {_GREEN}wins={sym_wins}{_RESET}  {sym_col}pnl=${sym_pnl:+.4f}{_RESET}"
                )

        lines.append(f"{_BOLD}{'═' * 60}{_RESET}")
        return "\n".join(lines)

    def positions_report(self) -> str:
        """Return details of all currently open positions."""
        if not self._positions and not self._blind_positions:
            return "  No open positions currently open."

        now = time.monotonic()
        lines = [f"{_BOLD}{'═' * 60}{_RESET}"]

        # ── Own-strategy positions ────────────────────────────────────
        if self._positions:
            total_cost = sum(p.cost_usdc for p in self._positions.values())
            lines.append(
                f"  {_BOLD}OWN POSITIONS{_RESET}  "
                f"({_CYAN}{len(self._positions)} bets{_RESET}  /  ${total_cost:.2f} at risk)"
            )
            lines.append(f"{'─' * 60}")
            for i, pos in enumerate(self._positions.values(), 1):
                age_s = int(now - pos.entered_at)
                age_m, age_s2 = divmod(age_s, 60)
                market = self._cache.get_market(pos.condition_id)
                t_rem = market.time_remaining_secs if market else None
                if t_rem is not None and t_rem > 0:
                    tr_m, tr_s = divmod(int(t_rem), 60)
                    t_col = _RED if t_rem < 60 else (_YELLOW if t_rem < 180 else _GREEN)
                    t_rem_str = f"{t_col}{tr_m}m{tr_s:02d}s left{_RESET}"
                elif t_rem is not None:
                    t_rem_str = f"{_RED}EXPIRED{_RESET}"
                else:
                    t_rem_str = f"{_DIM}t_rem unknown{_RESET}"

                dir_col = _GREEN if pos.bet == "Up" else _RED
                src_col = _YELLOW if pos.source == "SNIPE" else _CYAN if pos.source == "ARB" else _DIM
                lines.append(
                    f"  [{i}] {_BOLD}{_CYAN}{pos.symbol}{_RESET} "
                    f"{dir_col}{pos.bet.upper():<4}{_RESET}  "
                    f"{src_col}{pos.source:<9}{_RESET}  "
                    f"entry={_BOLD}${pos.entry_price:.4f}{_RESET}  "
                    f"shares={pos.shares:.2f}  cost=${pos.cost_usdc:.2f}"
                )
                lines.append(
                    f"       age={age_m}m{age_s2:02d}s  {t_rem_str}  "
                    f"{_DIM}cid={pos.condition_id[:10]}…{_RESET}"
                )
                lines.append(f"       {_DIM}{pos.question}{_RESET}")
                if i < len(self._positions):
                    lines.append(f"  {'·' * 56}")

        # ── Blind copy positions ──────────────────────────────────────
        if self._blind_positions:
            blind_cost = sum(p.cost_usdc for p in self._blind_positions)
            if self._positions:
                lines.append(f"{'─' * 60}")
            lines.append(
                f"  {_BOLD}{_MAGENTA}BLIND COPIES{_RESET}  "
                f"({_MAGENTA}{len(self._blind_positions)} bets{_RESET}  /  ${blind_cost:.2f} at risk)"
            )
            lines.append(f"{'─' * 60}")
            for i, pos in enumerate(self._blind_positions, 1):
                age_s = int(now - pos.entered_at)
                age_m, age_s2 = divmod(age_s, 60)
                dir_col = _GREEN if pos.bet in ("Up", "Yes") else _RED
                lines.append(
                    f"  [{i}] {_MAGENTA}COPY{_RESET}  "
                    f"{dir_col}{pos.bet.upper():<4}{_RESET}  "
                    f"entry={_BOLD}${pos.entry_price:.4f}{_RESET}  "
                    f"shares={pos.shares:.2f}  cost=${pos.cost_usdc:.2f}"
                )
                lines.append(f"       age={age_m}m{age_s2:02d}s  {_DIM}cid={pos.condition_id[:10]}…{_RESET}")
                lines.append(f"       {_DIM}{pos.question[:65]}{_RESET}")
                if i < len(self._blind_positions):
                    lines.append(f"  {'·' * 56}")

        lines.append(f"{_BOLD}{'═' * 60}{_RESET}")
        return "\n".join(lines)

    def trades_report(self) -> str:
        """Full session trade log — plain-text dump suitable for copy-paste analysis."""
        uptime_secs = time.monotonic() - self._session_start
        h, rem = divmod(int(uptime_secs), 3600)
        m, s   = divmod(rem, 60)

        priced   = [t for t in self._closed_trades if t.pnl_usdc is not None]
        resolved = [t for t in self._closed_trades if t.close_type == "resolved"]
        wins     = [t for t in priced if t.pnl_usdc > 0]
        losses   = [t for t in priced if t.pnl_usdc <= 0]
        realized = sum(t.pnl_usdc for t in priced)

        W = 74  # report width
        if priced:
            summary = (
                f"  realized_pnl={'+' if realized >= 0 else ''}${realized:.4f}  "
                f"wins={len(wins)}  losses={len(losses)}  resolved={len(resolved)}  "
                f"win_rate={len(wins)/len(priced)*100:.1f}%"
            )
        else:
            summary = "  realized_pnl=$0.0000  no closed trades yet"

        lines = [
            "=" * W,
            "  POLYMARKET BOT  —  FULL SESSION TRADE LOG",
            f"  uptime {h:02d}h {m:02d}m {s:02d}s  |  "
            f"orders={self._orders_placed}  open={len(self._positions)}  "
            f"closed={len(self._closed_trades)}",
            summary,
            "=" * W,
            f"  {'#':<3}  {'Sym':<4}  {'Dir':<5}  {'Strat':<10}  "
            f"{'Entry':>7}  {'Exit':>7}  {'P&L':>10}  {'%':>7}  Close",
            "-" * W,
        ]

        for i, t in enumerate(self._closed_trades, 1):
            if t.pnl_usdc is not None and t.exit_price is not None:
                pnl_str  = f"${t.pnl_usdc:+.4f}"
                pct      = (t.exit_price - t.entry_price) / t.entry_price * 100
                pct_str  = f"{pct:+.1f}%"
                exit_str = f"{t.exit_price:.4f}"
            else:
                pnl_str  = "pending"
                pct_str  = "─"
                exit_str = "─"

            strategy_tag  = t.source.split()[0]
            duration_s    = int(t.closed_at - t.opened_at)
            dur_m, dur_s2 = divmod(duration_s, 60)

            lines.append(
                f"  {i:<3}  {t.symbol:<4}  {t.bet:<5}  {strategy_tag:<10}  "
                f"{t.entry_price:>7.4f}  {exit_str:>7}  {pnl_str:>10}  {pct_str:>7}  "
                f"{t.close_type}  held={dur_m}m{dur_s2:02d}s"
            )
            lines.append(f"       {t.question[:68]}")
            if i < len(self._closed_trades):
                lines.append(f"  {'·' * (W - 4)}")

        if not self._closed_trades:
            lines.append("  (no trades recorded yet)")

        lines += [
            "=" * W,
            f"  OPEN POSITIONS ({len(self._positions)}):",
        ]
        if self._positions:
            now = time.monotonic()
            for pos in self._positions.values():
                age_s = int(now - pos.entered_at)
                age_m2, age_s3 = divmod(age_s, 60)
                lines.append(
                    f"  {pos.symbol} {pos.bet:<5}  {pos.source.split()[0]:<10}  "
                    f"entry={pos.entry_price:.4f}  cost=${pos.cost_usdc:.2f}  "
                    f"held={age_m2}m{age_s3:02d}s"
                )
                lines.append(f"    {pos.question[:68]}")
        else:
            lines.append("  (none)")

        lines.append("=" * W)
        return "\n".join(lines)
