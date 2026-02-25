"""
Momentum signal generator.

Consumes a stream of (symbol, price, timestamp) ticks and fires trade
signals when the price has moved more than *trigger_pct* over the last
*lookback_secs* seconds.

Duplicate signals are suppressed by a per-symbol cooldown so we don't
flood the order manager with the same directional bet.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Awaitable

from feeds.binance import PriceBuffer
import config

log = logging.getLogger(__name__)

# Type alias for the callback that receives a confirmed signal
SignalCallback = Callable[[str, str, float], Awaitable[None]]
#                          symbol  direction  pct_move


class MomentumStrategy:
    """
    Tracks rolling price windows for each symbol and emits directional
    signals when threshold is crossed.

    Parameters
    ----------
    on_signal:
        Async callback(symbol, direction, pct_move) invoked on each signal.
    """

    def __init__(self, on_signal: SignalCallback) -> None:
        self._on_signal = on_signal
        self._cfg = config.STRATEGY
        self._buffers: dict[str, PriceBuffer] = {}
        # symbol → monotonic timestamp of last signal emitted (any direction)
        self._last_signal: dict[str, float] = {}

    def get_buffer(self, symbol: str) -> PriceBuffer:
        if symbol not in self._buffers:
            # Keep 300 seconds (5 min) for TA indicators (RSI needs ~280s of history).
            # The momentum lookback (default 5s) is a tiny subset of this window.
            self._buffers[symbol] = PriceBuffer(max_age_secs=300.0)
        return self._buffers[symbol]

    async def on_tick(self, symbol: str, price: float, ts: float) -> None:
        """
        Called for every price tick.  Records the tick and checks for signals.
        """
        buf = self.get_buffer(symbol)
        buf.add(ts, price)
        await self._check_signal(symbol, buf)

    async def _check_signal(self, symbol: str, buf: PriceBuffer) -> None:
        current = buf.latest_price
        if current is None:
            return

        past = buf.price_n_secs_ago(self._cfg.lookback_secs)
        if past is None or past == 0:
            return

        pct_move = (current - past) / past * 100.0

        if abs(pct_move) < self._cfg.trigger_pct:
            return

        # Signal detected — check per-symbol cooldown
        last = self._last_signal.get(symbol, 0.0)
        if (time.monotonic() - last) < self._cfg.trade_cooldown_secs:
            return

        direction = "UP" if pct_move > 0 else "DOWN"
        self._last_signal[symbol] = time.monotonic()

        log.info(
            "SIGNAL  %s  %s  %.4f%%  (curr=%.2f  prev=%.2f)",
            symbol, direction, pct_move, current, past,
        )

        # Fire without awaiting to keep the tick handler fast
        asyncio.create_task(self._on_signal(symbol, direction, abs(pct_move)))
