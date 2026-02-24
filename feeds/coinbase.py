"""
Real-time price feed from the Coinbase Exchange (legacy) WebSocket API.

Endpoint: wss://ws-feed.exchange.coinbase.com
Channel:  ticker — emits on every best-bid/ask change and match event.

No authentication is required for public ticker data.

Usage is identical to BinanceFeed: pass an async callback(symbol, price, ts)
and call await feed.run().
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Awaitable

import websockets

import config

log = logging.getLogger(__name__)

COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"

# Coinbase product → our canonical symbol (defined centrally in config)
PRODUCT_SYMBOL_MAP: dict[str, str] = config.COINBASE_SYMBOL_MAP

RECONNECT_DELAY_BASE = 1.0
RECONNECT_DELAY_MAX = 30.0

PriceCallback = Callable[[str, float, float], Awaitable[None]]


class CoinbaseFeed:
    """
    Async WebSocket price feed from Coinbase Exchange.
    Calls *callback* for every ticker update that contains a price.

    Parameters
    ----------
    callback:
        Async function(symbol, price, timestamp_unix_sec).
    """

    def __init__(self, callback: PriceCallback) -> None:
        self._callback = callback
        self._running = False

    async def run(self) -> None:
        self._running = True
        delay = RECONNECT_DELAY_BASE

        while self._running:
            try:
                log.info("Connecting to Coinbase Exchange WebSocket …")
                async with websockets.connect(
                    COINBASE_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    delay = RECONNECT_DELAY_BASE

                    # Subscribe to ticker channel for all target products
                    sub = {
                        "type": "subscribe",
                        "product_ids": config.COINBASE_PRODUCTS,
                        "channels": ["ticker"],
                    }
                    await ws.send(json.dumps(sub))
                    log.info("Coinbase WebSocket connected and subscribed.")

                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle(raw)

            except (websockets.ConnectionClosed, OSError) as exc:
                if not self._running:
                    break
                log.warning(
                    "Coinbase WebSocket disconnected (%s). Reconnecting in %.1fs …",
                    exc, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_DELAY_MAX)

            except asyncio.CancelledError:
                break

        log.info("Coinbase feed stopped.")

    async def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")
        if msg_type != "ticker":
            return

        product_id: str = msg.get("product_id", "")
        symbol = PRODUCT_SYMBOL_MAP.get(product_id)
        if symbol is None:
            return

        price_str = msg.get("price", "")
        if not price_str:
            return

        price = float(price_str)
        # Coinbase ticker includes an ISO timestamp; parse it or fall back to now
        time_str: str = msg.get("time", "")
        if time_str:
            try:
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(
                    time_str.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                ts = time.time()
        else:
            ts = time.time()

        await self._callback(symbol, price, ts)

    def stop(self) -> None:
        self._running = False
