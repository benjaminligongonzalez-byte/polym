"""
Technical analysis signals computed from a rolling Binance price buffer.

Used by both the momentum entry filter and the exit conviction check.

Signals
-------
- RSI(14) sampled at 20-second intervals (needs ~300s of history)
- EMA trend: short (12×10s ≈ 2min) vs long (30×10s ≈ 5min)
- Momentum: % price change over 30s and 120s

Design note
-----------
The Binance feed fires ~10-20 ticks/second, so PriceBuffer can easily hold
300 seconds of data.  TA is computed on-demand (not on every tick) so CPU
cost is paid only when an entry or exit decision is being made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feeds.binance import PriceBuffer


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class TASignals:
    rsi: float | None       # 0–100; None = not enough history
    trend: str              # "UP" | "DOWN" | "FLAT"
    momentum_30s: float     # % price change over last 30 s (+ = up)
    momentum_120s: float    # % price change over last 120 s (+ = up)
    ema_short: float | None # ~2-min EMA price level
    ema_long: float | None  # ~5-min EMA price level

    def supports_up(self) -> bool:
        """Return True when TA conditions still favour a long (Up) position."""
        rsi_ok = self.rsi is None or self.rsi < 68
        trend_ok = self.trend != "DOWN"
        not_falling = self.momentum_30s > -0.25  # price not collapsing
        return rsi_ok and trend_ok and not_falling

    def supports_down(self) -> bool:
        """Return True when TA conditions still favour a short (Down) position."""
        rsi_ok = self.rsi is None or self.rsi > 32
        trend_ok = self.trend != "UP"
        not_rising = self.momentum_30s < 0.25  # price not surging
        return rsi_ok and trend_ok and not_rising

    def opposes(self, bet: str) -> bool:
        """Return True when TA is clearly opposing our bet direction."""
        if bet == "Up":
            return not self.supports_up()
        return not self.supports_down()

    def describe(self) -> str:
        rsi_str = f"{self.rsi:.0f}" if self.rsi is not None else "?"
        return (
            f"trend={self.trend} rsi={rsi_str} "
            f"m30={self.momentum_30s:+.2f}% m120={self.momentum_120s:+.2f}%"
        )


# Sentinel returned when the buffer doesn't have enough data
FLAT = TASignals(
    rsi=None,
    trend="FLAT",
    momentum_30s=0.0,
    momentum_120s=0.0,
    ema_short=None,
    ema_long=None,
)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_ta(buf: "PriceBuffer") -> TASignals:
    """
    Compute TA signals from a PriceBuffer.  Returns FLAT if the buffer
    doesn't yet have enough data (first few minutes of the session).
    """
    latest = buf.latest_price
    if latest is None:
        return FLAT

    # ── Momentum ─────────────────────────────────────────────────────────────
    def pct_change(past_secs: float) -> float:
        past = buf.price_n_secs_ago(past_secs)
        if past is None or past <= 0:
            return 0.0
        return (latest - past) / past * 100.0

    m30 = pct_change(30)
    m120 = pct_change(120)

    # ── EMA trend ────────────────────────────────────────────────────────────
    # Sample prices at 10-second intervals for EMA computation.
    # short = 12 samples (≈ 2 min), long = 30 samples (≈ 5 min).
    SHORT_TAPS = 12
    LONG_TAPS  = 30
    INTERVAL   = 10.0   # seconds between samples

    samples: list[float] = []
    for i in range(LONG_TAPS):
        p = buf.price_n_secs_ago(i * INTERVAL) if i > 0 else latest
        if p is None:
            break
        samples.append(p)

    # samples[0] = newest, samples[-1] = oldest (reversed for EMA)
    oldest_first = list(reversed(samples))

    ema_short = _ema(oldest_first, SHORT_TAPS) if len(oldest_first) >= SHORT_TAPS else None
    ema_long  = _ema(oldest_first, LONG_TAPS)  if len(oldest_first) >= LONG_TAPS  else None

    if ema_short is not None and ema_long is not None and ema_long > 0:
        gap_pct = (ema_short - ema_long) / ema_long * 100
        trend = "UP" if gap_pct > 0.05 else ("DOWN" if gap_pct < -0.05 else "FLAT")
    elif abs(m30) >= 0.15:
        trend = "UP" if m30 > 0 else "DOWN"
    else:
        trend = "FLAT"

    # ── RSI(14) at 20-second intervals ───────────────────────────────────────
    rsi = _rsi(buf, period=14, interval_secs=20.0)

    return TASignals(
        rsi=rsi,
        trend=trend,
        momentum_30s=m30,
        momentum_120s=m120,
        ema_short=ema_short,
        ema_long=ema_long,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(values: list[float], span: int) -> float | None:
    """Compute EMA on an oldest-first price list with the given span."""
    if len(values) < span:
        return None
    k = 2.0 / (span + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _rsi(buf: "PriceBuffer", period: int = 14, interval_secs: float = 20.0) -> float | None:
    """
    RSI using prices sampled at *interval_secs* intervals going backwards
    from the latest tick.  Needs (period + 1) × interval_secs of history.

    Returns None if the buffer doesn't span the required window.
    """
    latest = buf.latest_price
    if latest is None:
        return None

    # Collect (period + 1) price samples, newest first
    samples: list[float] = [latest]
    for i in range(1, period + 1):
        p = buf.price_n_secs_ago(i * interval_secs)
        if p is None:
            return None   # not enough history yet
        samples.append(p)

    # Compute gains/losses from oldest → newest
    gains: list[float] = []
    losses: list[float] = []
    for i in range(len(samples) - 1, 0, -1):
        delta = samples[i - 1] - samples[i]   # newer − older
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-delta)

    if not gains:
        return None

    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
