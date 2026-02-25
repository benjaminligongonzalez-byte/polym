"""
Live Polymarket CLOB price feed via WebSocket.

Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market and
subscribes to order-book events for all active prediction market tokens.
When a price update arrives it:

  1. Updates the internal mid-price cache (token_id → float)
  2. Fires the async on_update(token_id, mid) callback so the arbitrage
     strategy can re-evaluate edge immediately — no polling wait.

This replaces HTTP midpoint polling (which had 1.5 s cache TTL) with
sub-100 ms event-driven price updates, matching the latency of the
Binance / Coinbase price feeds on the other side of the arbitrage.

Subscription management
-----------------------
Token IDs to watch are supplied externally:

  feed.update_subscriptions(token_ids)   # called by bot.py on market refresh

The feed re-subscribes automatically on reconnect so new tokens added
between reconnects are not lost.

WebSocket event format (Polymarket CLOB)
-----------------------------------------
Events arrive as a JSON array; each element is one of:

  {"event_type": "book",
   "asset_id": "<token_id>",
   "bids": [{"price": "0.51", "size": "100"}, ...],
   "asks": [{"price": "0.53", "size": "80"},  ...]}

  {"event_type": "price_change",
   "asset_id": "<token_id>",
   "price": "0.52"}

  {"event_type": "last_trade_price",
   "asset_id": "<token_id>",
   "price": "0.52"}

Mid = (best_bid + best_ask) / 2  for book events.
Mid = price field                for price_change / last_trade_price.

All prices are on the 0–1 scale (Polymarket's cent convention).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Awaitable

import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)

CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Seconds between reconnect attempts (exponential back-off up to MAX)
_RECONNECT_BASE = 1.0
_RECONNECT_MAX  = 30.0

# Type alias: callback fired on each price update
MidCallback = Callable[[str, float], Awaitable[None]]


class ClobFeed:
    """
    Async WebSocket client for the Polymarket CLOB market data channel.

    Parameters
    ----------
    on_update:
        Coroutine called with (token_id, mid_price) on every price change.
        Runs in the event loop; keep it fast (no blocking I/O).
    """

    def __init__(self, on_update: MidCallback) -> None:
        self._on_update = on_update
        # token_id → latest mid (0–1 scale); updated on every event
        self._mid_cache: dict[str, float] = {}
        # token IDs we want to be subscribed to at all times
        self._desired: set[str] = set()
        # WebSocket connection reference (held while connected)
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._running = False
        # Optional sync callback fired on last_trade_price events (token_id → None)
        # Used to wake the tracker immediately when market activity is detected.
        self._on_trade_activity: Callable[[str], None] | None = None
        # Stats
        self._events_received = 0
        self._connected_at: float = 0.0

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def set_trade_activity_callback(self, cb: Callable[[str], None]) -> None:
        """
        Register a sync callback fired on every last_trade_price event.
        The callback receives the token_id of the market that just traded.
        Used to wake the copy-trade tracker immediately on market activity.
        """
        self._on_trade_activity = cb

    def update_subscriptions(self, token_ids: list[str]) -> None:
        """
        Tell the feed which token IDs to watch.

        New IDs are subscribed immediately if the socket is open;
        they are queued and sent on the next (re)connect otherwise.
        """
        new = set(token_ids) - self._desired
        self._desired |= set(token_ids)
        if new and self._ws is not None:
            asyncio.create_task(self._send_subscribe(list(new)))

    def get_mid(self, token_id: str) -> float | None:
        """Return the latest cached mid, or None if not yet received."""
        return self._mid_cache.get(token_id)

    @property
    def is_connected(self) -> bool:
        return self._ws is not None

    @property
    def events_received(self) -> int:
        return self._events_received

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect, subscribe, receive events — reconnect on any failure."""
        self._running = True
        delay = _RECONNECT_BASE

        while self._running:
            try:
                log.info("ClobFeed: connecting to %s", CLOB_WS_URL)
                async with websockets.connect(
                    CLOB_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**22,         # 4 MB — large order books
                ) as ws:
                    self._ws = ws
                    self._connected_at = time.monotonic()
                    delay = _RECONNECT_BASE  # reset back-off on success

                    # Re-subscribe to all desired tokens after (re)connect
                    if self._desired:
                        await self._send_subscribe(list(self._desired))
                    log.info(
                        "ClobFeed: connected, subscribed to %d token(s).",
                        len(self._desired),
                    )

                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle(raw)

            except asyncio.CancelledError:
                break
            except (ConnectionClosed, OSError, Exception) as exc:
                if not self._running:
                    break
                log.warning(
                    "ClobFeed: disconnected (%s) — reconnecting in %.0fs",
                    exc, delay,
                )
            finally:
                self._ws = None

            await asyncio.sleep(delay)
            delay = min(delay * 2, _RECONNECT_MAX)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _send_subscribe(self, token_ids: list[str]) -> None:
        if self._ws is None or not token_ids:
            return
        msg = json.dumps({"assets_ids": token_ids, "type": "subscribe"})
        try:
            await self._ws.send(msg)
            log.debug("ClobFeed: subscribed to %d token(s)", len(token_ids))
        except Exception as exc:
            log.debug("ClobFeed: subscribe send failed: %s", exc)

    async def _handle(self, raw: str | bytes) -> None:
        """Parse a raw WebSocket frame and update the mid-price cache."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Events arrive as a JSON array or a single object
        events: list[dict] = data if isinstance(data, list) else [data]

        for event in events:
            if not isinstance(event, dict):
                continue

            token_id = event.get("asset_id") or event.get("assetId")
            if not token_id:
                continue

            etype = (event.get("event_type") or event.get("type") or "").lower()

            mid: float | None = None

            if etype in ("price_change", "last_trade_price", "tick"):
                # Scalar price field
                raw_price = event.get("price")
                if raw_price is not None:
                    try:
                        mid = float(raw_price)
                    except (ValueError, TypeError):
                        pass

            elif etype in ("book", "orderbook"):
                # Order book snapshot → compute mid from best bid/ask
                bids = event.get("bids") or []
                asks = event.get("asks") or []
                if bids and asks:
                    try:
                        best_bid = max(float(b["price"]) for b in bids if b.get("size", "1") != "0")
                        best_ask = min(float(a["price"]) for a in asks if a.get("size", "1") != "0")
                        if best_bid > 0 and best_ask > 0:
                            mid = (best_bid + best_ask) / 2.0
                    except (KeyError, ValueError, TypeError):
                        pass

            if mid is not None and 0.0 < mid < 1.0:
                self._mid_cache[token_id] = mid
                self._events_received += 1
                try:
                    await self._on_update(token_id, mid)
                except Exception as exc:
                    log.debug("ClobFeed: on_update error: %s", exc)

            # Fire the trade-activity callback on last_trade_price events so the
            # tracker can interrupt its sleep and poll the Data API immediately.
            if etype == "last_trade_price" and self._on_trade_activity is not None:
                try:
                    self._on_trade_activity(token_id)
                except Exception as exc:
                    log.debug("ClobFeed: on_trade_activity error: %s", exc)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> str:
        uptime = time.monotonic() - self._connected_at if self._connected_at else 0
        return (
            f"ClobFeed  connected={'yes' if self._ws else 'no'}  "
            f"subscribed={len(self._desired)}  "
            f"events={self._events_received}  "
            f"uptime={uptime:.0f}s"
        )
