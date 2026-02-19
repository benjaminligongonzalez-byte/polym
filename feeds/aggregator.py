"""
Multi-source price aggregator.

Collects ticks from multiple exchanges (Binance, Coinbase) and provides:
  - A consensus (median) price for each symbol
  - Per-source latest prices for transparency / logging
  - A divergence flag when sources disagree beyond a threshold

The aggregator also routes each incoming tick to any registered subscribers
(e.g. MomentumStrategy, ArbStrategy) so both can share a single feed
callback without coupling the feed objects to strategy objects.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional

log = logging.getLogger(__name__)

# If two sources differ by more than this %, treat prices as diverging
DIVERGENCE_PCT_THRESHOLD = 0.15   # 0.15%

# How old a price can be (seconds) before we consider it stale
MAX_STALE_SECS = 5.0

PriceCallback = Callable[[str, float, float], Awaitable[None]]


@dataclass
class SourcePrice:
    source: str
    price: float
    ts: float       # UNIX timestamp (seconds)


class PriceAggregator:
    """
    Thread-safe (asyncio) multi-source price store.

    Feed each exchange tick into update().
    Read the consensus (median of fresh sources) via consensus_price().
    Subscribe additional async handlers via add_subscriber().
    """

    def __init__(self) -> None:
        # symbol → {source_name → SourcePrice}
        self._prices: dict[str, dict[str, SourcePrice]] = {}
        # Downstream subscribers that also want every tick
        self._subscribers: list[PriceCallback] = []

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def add_subscriber(self, cb: PriceCallback) -> None:
        """Register a callback that receives every tick after aggregation."""
        self._subscribers.append(cb)

    # ------------------------------------------------------------------
    # Ingestion — called by each feed's callback
    # ------------------------------------------------------------------

    async def on_tick(self, source: str, symbol: str, price: float, ts: float) -> None:
        """
        Record an incoming price tick from *source* and forward to subscribers.
        This is the method passed (partially applied) as the feed callback.
        """
        if symbol not in self._prices:
            self._prices[symbol] = {}

        self._prices[symbol][source] = SourcePrice(source=source, price=price, ts=ts)

        # Forward to all downstream subscribers
        for cb in self._subscribers:
            try:
                await cb(symbol, price, ts)
            except Exception as exc:
                log.warning("Subscriber error for %s tick: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def consensus_price(self, symbol: str) -> Optional[float]:
        """
        Median of all fresh source prices for *symbol*.
        Returns None if no fresh data is available.
        """
        fresh = self._fresh_prices(symbol)
        if not fresh:
            return None
        sorted_prices = sorted(fresh)
        mid = len(sorted_prices) // 2
        # True median: average of two middle values for even-length lists
        if len(sorted_prices) % 2 == 0:
            return (sorted_prices[mid - 1] + sorted_prices[mid]) / 2
        return sorted_prices[mid]

    def source_prices(self, symbol: str) -> dict[str, Optional[float]]:
        """Return the latest price per source (None if stale)."""
        result: dict[str, Optional[float]] = {}
        now = time.time()
        for src, sp in self._prices.get(symbol, {}).items():
            result[src] = sp.price if now - sp.ts <= MAX_STALE_SECS else None
        return result

    def is_diverging(self, symbol: str) -> bool:
        """
        True if at least two sources have fresh prices that disagree by more
        than DIVERGENCE_PCT_THRESHOLD.  A divergence usually indicates a
        feed glitch or extreme illiquidity — skip trading when this fires.
        """
        fresh = self._fresh_prices(symbol)
        if len(fresh) < 2:
            return False
        mn, mx = min(fresh), max(fresh)
        return (mx - mn) / mn * 100.0 > DIVERGENCE_PCT_THRESHOLD

    def source_count(self, symbol: str) -> int:
        """Number of sources with fresh prices for *symbol*."""
        return len(self._fresh_prices(symbol))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fresh_prices(self, symbol: str) -> list[float]:
        now = time.time()
        return [
            sp.price
            for sp in self._prices.get(symbol, {}).values()
            if now - sp.ts <= MAX_STALE_SECS
        ]


def make_feed_callback(aggregator: PriceAggregator, source: str) -> PriceCallback:
    """
    Return an async callback suitable for passing to BinanceFeed / CoinbaseFeed
    that routes ticks into the aggregator under the given source name.

    Usage:
        binance_cb = make_feed_callback(aggregator, "binance")
        feed = BinanceFeed(callback=binance_cb)
    """
    async def _cb(symbol: str, price: float, ts: float) -> None:
        await aggregator.on_tick(source, symbol, price, ts)

    return _cb
