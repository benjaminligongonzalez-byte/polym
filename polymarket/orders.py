"""
Order sizing and execution logic.

Handles two kinds of signals:

1. Momentum signals  (execute_signal)
   Simple directional momentum: price moved X%, bet accordingly.
   Uses flat order sizing (max_order_usdc).

2. Arbitrage signals  (execute_arb_signal)
   Fair-value edge detected: Polymarket price differs from our model by
   more than the threshold.  Uses fractional Kelly sizing based on the
   computed edge and full-Kelly fraction stored in the ArbSignal.
"""

from __future__ import annotations

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
    Handles order sizing and tracks open positions for risk control.
    """

    def __init__(self, client: PolymarketClient, cache: MarketCache) -> None:
        self._client = client
        self._cache = cache
        # condition_id → last trade timestamp (monotonic)
        self._last_trade: dict[str, float] = {}
        # condition_id → USDC currently deployed
        self._open_exposure: dict[str, float] = {}

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

    def _remaining_budget(self) -> float:
        return config.RISK.max_total_exposure_usdc - self.total_exposure

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
        Given a confirmed momentum signal, find matching markets and place orders.

        direction="UP"   → buy YES on "higher" markets, NO on "lower" markets
        direction="DOWN" → buy YES on "lower" markets,  NO on "higher" markets
        """
        log.info("MOMENTUM  %s  %s  %.3f%%", symbol, direction, price_move_pct)

        if self.total_exposure >= config.RISK.max_total_exposure_usdc:
            log.warning("Total exposure cap reached ($%.2f). Skipping.", self.total_exposure)
            return

        if self._too_many_positions(symbol):
            log.warning("Too many open positions for %s. Skipping.", symbol)
            return

        target_markets = self._cache.get_markets_for(symbol, direction)
        if not target_markets:
            log.debug("No matching markets for %s %s.", symbol, direction)
            return

        for market in target_markets:
            await self._trade_market(market, direction)

    async def _trade_market(self, market: MarketInfo, signal_direction: str) -> None:
        cid = market.condition_id

        if self._on_cooldown(cid):
            log.debug("Market %s on cooldown.", cid[:8])
            return

        # Fetch live mid price
        try:
            yes_mid = await self._client.get_midpoint(market.yes_token.token_id)
        except Exception as exc:
            log.warning("Could not fetch mid for %s: %s", cid[:8], exc)
            return

        no_mid = 1.0 - yes_mid

        if signal_direction == market.direction:
            token_id = market.yes_token.token_id
            mid = yes_mid
            token_label = "YES"
        else:
            token_id = market.no_token.token_id
            mid = no_mid
            token_label = "NO"

        # Probability bounds guard
        if not (config.STRATEGY.min_yes_prob <= mid <= config.STRATEGY.max_yes_prob):
            log.debug(
                "Market %s: %s mid=%.3f outside tradeable range.",
                cid[:8], token_label, mid,
            )
            return

        order_usdc = min(config.RISK.max_order_usdc, self._remaining_budget())
        if order_usdc < config.RISK.min_order_usdc:
            return

        await self._place_order(
            cid=cid,
            token_id=token_id,
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
        Execute an arbitrage trade based on a pre-computed ArbSignal.

        Uses fractional Kelly sizing scaled to our risk limits:
            bet = kelly_f * kelly_fraction * total_budget
        capped at max_order_usdc and floored at min_order_usdc.
        """
        cid = sig.market.condition_id

        if self._on_cooldown(cid):
            log.debug("Market %s on cooldown (arb).", cid[:8])
            return

        if self.total_exposure >= config.RISK.max_total_exposure_usdc:
            log.warning("Total exposure cap reached. Skipping arb signal.")
            return

        if self._too_many_positions(sig.symbol):
            log.warning("Too many positions for %s. Skipping arb signal.", sig.symbol)
            return

        # Kelly bet: f* × kelly_fraction × bankroll
        bankroll = config.RISK.max_total_exposure_usdc
        kelly_usdc = sig.kelly_f * config.STRATEGY.kelly_fraction * bankroll
        order_usdc = max(
            config.RISK.min_order_usdc,
            min(kelly_usdc, config.RISK.max_order_usdc, self._remaining_budget()),
        )
        if order_usdc < config.RISK.min_order_usdc:
            return

        await self._place_order(
            cid=cid,
            token_id=sig.token_id,
            token_label=sig.side,
            mid=sig.market_prob,
            order_usdc=order_usdc,
            question=sig.market.question,
            source=f"ARB edge={sig.edge:.3f} kelly={sig.kelly_f:.3f}",
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
        # Aggressive limit: bid slightly above current mid
        limit_price = round(min(mid + config.RISK.slippage_tolerance, 0.99), 4)
        shares = round(order_usdc / limit_price, 2)

        log.info(
            "[%s]  Buy %-3s  %s  @ %.4f  (%.2f shares / $%.2f USDC)",
            source, token_label, question[:50], limit_price, shares, order_usdc,
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
                "Order recorded. cid=%s  total_exposure=$%.2f",
                cid[:8], self.total_exposure,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def record_resolution(self, condition_id: str) -> None:
        """Call when a market resolves to free up tracked exposure."""
        self._open_exposure.pop(condition_id, None)
        self._last_trade.pop(condition_id, None)
