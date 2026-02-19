"""
Order sizing and execution logic.

Given a trade signal (symbol, direction), this module:
1. Looks up the matching Polymarket market(s)
2. Fetches the current order book mid price
3. Calculates a safe order size within risk limits
4. Places the order via PolymarketClient
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from polymarket.client import PolymarketClient
from polymarket.markets import MarketCache, MarketInfo
import config

log = logging.getLogger(__name__)


class OrderManager:
    """
    Handles order sizing and tracks open positions for risk control.
    """

    def __init__(self, client: PolymarketClient, cache: MarketCache) -> None:
        self._client = client
        self._cache = cache
        # condition_id → last trade timestamp
        self._last_trade: dict[str, float] = {}
        # condition_id → USDC currently deployed
        self._open_exposure: dict[str, float] = {}

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

    async def execute_signal(
        self,
        symbol: str,
        direction: str,     # "UP" or "DOWN"
        price_move_pct: float,
    ) -> None:
        """
        Given a confirmed price signal, find matching markets and place orders.

        direction="UP"  → buy YES on "higher" markets  (or NO on "lower" markets)
        direction="DOWN"→ buy YES on "lower" markets   (or NO on "higher" markets)
        """
        log.info(
            "Signal: %s moved %.3f%% → betting %s",
            symbol, price_move_pct, direction,
        )

        # Risk guard: total exposure cap
        if self.total_exposure >= config.RISK.max_total_exposure_usdc:
            log.warning("Total exposure cap reached ($%.2f). Skipping.", self.total_exposure)
            return

        # Per symbol position cap
        if self._too_many_positions(symbol):
            log.warning("Too many open positions for %s. Skipping.", symbol)
            return

        # We look for markets whose *direction matches* our bet:
        # If price is going UP → we want markets where outcome is "higher" (direction=UP)
        # and we buy the YES token of those markets.
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

        # Fetch live mid price for the YES token
        try:
            yes_mid = await self._client.get_midpoint(market.yes_token.token_id)
        except Exception as exc:
            log.warning("Could not fetch mid for %s: %s", cid[:8], exc)
            return

        no_mid = 1.0 - yes_mid

        # The market direction tells us what event it's asking about:
        # e.g. direction="UP" means market asks "Will price be HIGHER?"
        # Our signal_direction also says "UP" → buy YES.
        # signal_direction="DOWN" on a direction="UP" market → buy NO.
        if signal_direction == market.direction:
            token_id = market.yes_token.token_id
            mid = yes_mid
            token_label = "YES"
        else:
            token_id = market.no_token.token_id
            mid = no_mid
            token_label = "NO"

        # Probability guard — don't buy if already too extreme
        strategy = config.STRATEGY
        if not (strategy.min_yes_prob <= mid <= strategy.max_yes_prob):
            log.debug(
                "Market %s: %s mid=%.3f outside tradeable range. Skipping.",
                cid[:8], token_label, mid,
            )
            return

        # Size the order
        remaining_budget = config.RISK.max_total_exposure_usdc - self.total_exposure
        order_usdc = min(config.RISK.max_order_usdc, remaining_budget)
        if order_usdc < config.RISK.min_order_usdc:
            log.debug("Insufficient budget for order (%.2f USDC).", order_usdc)
            return

        # Place a limit order slightly above mid (aggressive taker)
        limit_price = round(min(mid + config.RISK.slippage_tolerance, 0.99), 4)
        shares = round(order_usdc / limit_price, 2)

        log.info(
            "Placing order: %s %s @ %.4f (%.2f shares / $%.2f USDC)",
            token_label, market.question[:50], limit_price, shares, order_usdc,
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
            log.info("Order accepted. cid=%s exposure=$%.2f", cid[:8], self._open_exposure[cid])

    def record_resolution(self, condition_id: str) -> None:
        """Call when a market resolves to free up tracked exposure."""
        self._open_exposure.pop(condition_id, None)
        self._last_trade.pop(condition_id, None)
