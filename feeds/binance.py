"""
Real-time price feed from Binance combined stream WebSocket.

Connects to wss://stream.binance.com:9443 and subscribes to individual
trade streams for BTC/USDT and ETH/USDT.  Each trade is the fastest
possible public price signal — sub-100 ms latency from Binance match
engine to here.

Usage
-----
feed = BinanceFeed(callback=my_handler)
await feed.run()          # runs until cancelled

The callback receives (symbol: str, price: float, ts: float) where
ts is the event time as a UNIX timestamp (seconds, float).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Callable, Awaitable

import websockets

from config import BINANCE_WS_BASE, BINANCE_STREAMS, STREAM_SYMBOL_MAP

log = logging.getLogger(__name__)

# Maximum reconnect attempts before giving up (0 = infinite)
MAX_RECONNECT_ATTEMPTS = 0
RECONNECT_DELAY_BASE = 1.0   # seconds
RECONNECT_DELAY_MAX = 30.0


PriceCallback = Callable[[str, float, float], Awaitable[None]]


class BinanceFeed:
    """
    Async WebSocket price feed.  Calls *callback* for every trade tick.

    Parameters
    ----------
    callback:
        Async function(symbol, price, timestamp_unix_sec).
    """

    def __init__(self, callback: PriceCallback) -> None:
        self._callback = callback
        self._url = BINANCE_WS_BASE + "/".join(BINANCE_STREAMS)
        self._running = False

    async def run(self) -> None:
        self._running = True
        attempt = 0
        delay = RECONNECT_DELAY_BASE

        while self._running:
            try:
                log.info("Connecting to Binance stream: %s", self._url)
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    attempt = 0
                    delay = RECONNECT_DELAY_BASE
                    log.info("Binance WebSocket connected.")
                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle(raw)

            except (websockets.ConnectionClosed, OSError) as exc:
                if not self._running:
                    break
                attempt += 1
                log.warning(
                    "Binance WebSocket disconnected (%s). Reconnecting in %.1fs …",
                    exc, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_DELAY_MAX)

            except asyncio.CancelledError:
                break

        log.info("Binance feed stopped.")

    async def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Combined stream wraps events: {"stream": "btcusdt@trade", "data": {...}}
        stream = msg.get("stream", "")
        data = msg.get("data", msg)

        if data.get("e") != "trade":
            return

        stream_sym = stream.split("@")[0]                 # e.g. "btcusdt"
        symbol = STREAM_SYMBOL_MAP.get(stream_sym)
        if symbol is None:
            return

        price = float(data["p"])
        ts = data["T"] / 1000.0    # Binance gives ms, convert to seconds

        await self._callback(symbol, price, ts)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# In-memory price buffer — a rolling window for momentum detection
# ---------------------------------------------------------------------------

class PriceBuffer:
    """
    Stores (timestamp, price) pairs for a single symbol.
    Efficiently answers: what was the price N seconds ago?
    """

    def __init__(self, max_age_secs: float = 60.0) -> None:
        self._data: deque[tuple[float, float]] = deque()
        self._max_age = max_age_secs

    def add(self, ts: float, price: float) -> None:
        self._data.append((ts, price))
        self._evict(ts)

    def _evict(self, now: float) -> None:
        cutoff = now - self._max_age
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def price_n_secs_ago(self, secs: float) -> float | None:
        """
        Return the oldest price still within *secs* seconds of the newest tick.
        Returns None if the buffer is empty or doesn't span *secs* yet.
        """
        if not self._data:
            return None
        now_ts = self._data[-1][0]
        target = now_ts - secs
        # Walk from oldest until we exceed target
        result = None
        for ts, price in self._data:
            if ts >= target:
                result = price
                break
        return result

    @property
    def latest_price(self) -> float | None:
        return self._data[-1][1] if self._data else None

    @property
    def latest_ts(self) -> float | None:
        return self._data[-1][0] if self._data else None

    def prices_list(self) -> list[float]:
        """Return all buffered prices, oldest first."""
        return [price for _, price in self._data]
