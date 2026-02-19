"""
Arbitrage strategy: compare fair value to live Polymarket prices.

Core idea
---------
A Polymarket "Will BTC be higher in 15 minutes?" market should have a YES
price equal to the true probability that BTC ends above its reference
(strike) price.

We compute that fair probability using a log-normal model:

    P(S_T > K) = N(d₂)

where:
    d₂ = [ln(S / K) - ½σ²T] / (σ√T)
    S  = current live price (consensus of Binance + Coinbase)
    K  = strike price (price at market open, stored when first seen)
    T  = time remaining (years)
    σ  = annualised historical volatility for the symbol
    N  = standard normal CDF

When the Polymarket YES price deviates from fair value by more than the
configured `arb_edge_threshold`, we emit an ArbSignal.

The ArbStrategy scans all cached markets every `arb_scan_interval` seconds.

Kelly sizing
------------
Optimal bet fraction for a binary market:
    f* = (p_fair - p_market) / (1 - p_market)   [buying YES]
    f* = (p_fair_no - p_no)  / (1 - p_no)       [buying NO]

We apply a fractional Kelly multiplier (default 0.25) for safety, then
cap between min_order_usdc and max_order_usdc.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional

from feeds.aggregator import PriceAggregator
from polymarket.markets import MarketCache, MarketInfo
from polymarket.client import PolymarketClient
import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fair_prob_up(
    current: float,
    strike: float,
    time_remaining_secs: float,
    annual_vol: float,
) -> float:
    """
    Probability that *current* price exceeds *strike* after *time_remaining_secs*
    seconds, assuming log-normal price dynamics with zero drift.

    Returns a value in (0, 1).
    """
    T = max(time_remaining_secs, 1.0) / (365.25 * 24.0 * 3600.0)
    sigma_sqrt_T = annual_vol * math.sqrt(T)
    if sigma_sqrt_T == 0:
        return 1.0 if current > strike else 0.0
    d2 = (math.log(current / strike) - 0.5 * annual_vol ** 2 * T) / sigma_sqrt_T
    return _norm_cdf(d2)


def kelly_fraction(fair: float, market: float) -> float:
    """
    Full Kelly fraction for buying a binary outcome priced at *market*
    when true probability is *fair*.

    f* = (fair - market) / (1 - market)
    """
    denom = 1.0 - market
    if denom <= 0:
        return 0.0
    return max(0.0, (fair - market) / denom)


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArbSignal:
    """A confirmed mispricing signal ready for order execution."""
    market: MarketInfo
    token_id: str       # CLOB token ID to buy
    side: str           # "YES" or "NO"
    fair_prob: float    # our estimate of true probability
    market_prob: float  # current Polymarket mid price for this outcome
    edge: float         # fair_prob - market_prob  (always positive)
    kelly_f: float      # full Kelly fraction (pre-scaling)
    symbol: str
    consensus_price: float   # live price used for this calc
    strike_price: float      # reference price the market was opened at


ArbCallback = Callable[["ArbSignal"], Awaitable[None]]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ArbStrategy:
    """
    Periodically scans all cached Polymarket markets, computes fair value,
    and emits ArbSignals when the edge exceeds the configured threshold.

    Parameters
    ----------
    aggregator:
        The multi-source price aggregator (provides consensus prices).
    cache:
        The live market cache (provides MarketInfo objects + strike prices).
    pm_client:
        Used to fetch live Polymarket mid prices.
    on_signal:
        Async callback invoked with each confirmed ArbSignal.
    """

    def __init__(
        self,
        aggregator: PriceAggregator,
        cache: MarketCache,
        pm_client: PolymarketClient,
        on_signal: ArbCallback,
    ) -> None:
        self._aggregator = aggregator
        self._cache = cache
        self._pm_client = pm_client
        self._on_signal = on_signal
        # condition_id → monotonic timestamp of last emitted signal
        self._last_signal: dict[str, float] = {}

    async def run(self) -> None:
        """Continuously scan all markets. Run as a background asyncio task."""
        while True:
            await asyncio.sleep(config.STRATEGY.arb_scan_interval)
            await self._scan_all()

    async def _scan_all(self) -> None:
        markets = self._cache.all_markets()
        # Gather mid prices for all markets concurrently to minimise latency
        tasks = [self._evaluate(m) for m in markets]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _evaluate(self, market: MarketInfo) -> None:
        symbol = market.symbol
        cid = market.condition_id

        # --- 1. Get consensus live price ---
        consensus = self._aggregator.consensus_price(symbol)
        if consensus is None:
            return

        # Abort if sources are diverging (feed reliability issue)
        if self._aggregator.is_diverging(symbol):
            log.debug("Prices diverging for %s — skipping arb scan.", symbol)
            return

        # Require at least 2 independent sources for confidence
        if self._aggregator.source_count(symbol) < 2:
            log.debug("Only 1 price source for %s — waiting for second feed.", symbol)
            return

        # --- 2. Validate strike price ---
        # Strike is parsed from the question text at market discovery time
        # (e.g. "$84,500" from "Will BTC be above $84,500 at 2:30 PM?").
        # If it's missing the market was skipped at parse time — bail out.
        if market.strike_price is None:
            return

        # --- 3. Check time remaining ---
        t_rem = market.time_remaining_secs
        if t_rem <= 0:
            return          # market has closed
        if t_rem > 930:     # > 15m 30s: not yet open or wrong market
            return

        # --- 4. Fetch live Polymarket mid prices ---
        try:
            yes_mid = await self._pm_client.get_midpoint(market.yes_token.token_id)
        except Exception as exc:
            log.debug("Midpoint fetch failed for %s: %s", cid[:8], exc)
            return
        no_mid = 1.0 - yes_mid

        # --- 5. Compute fair probability ---
        vol = config.VOLATILITY.get(symbol, config.DEFAULT_ANNUAL_VOL)
        p_up = fair_prob_up(consensus, market.strike_price, t_rem, vol)

        # YES token means "price ends higher" for UP markets, "lower" for DOWN
        if market.direction == "UP":
            fair_yes = p_up
        else:
            fair_yes = 1.0 - p_up
        fair_no = 1.0 - fair_yes

        # --- 6. Compute edges ---
        edge_yes = fair_yes - yes_mid
        edge_no  = fair_no  - no_mid

        threshold = config.STRATEGY.arb_edge_threshold

        best_edge = max(edge_yes, edge_no)
        if best_edge < threshold:
            log.debug(
                "%s %s | fair_yes=%.3f mkt_yes=%.3f edge_yes=%.3f (below threshold)",
                symbol, cid[:8], fair_yes, yes_mid, edge_yes,
            )
            return

        if edge_yes >= edge_no:
            sig = ArbSignal(
                market=market,
                token_id=market.yes_token.token_id,
                side="YES",
                fair_prob=fair_yes,
                market_prob=yes_mid,
                edge=edge_yes,
                kelly_f=kelly_fraction(fair_yes, yes_mid),
                symbol=symbol,
                consensus_price=consensus,
                strike_price=market.strike_price,
            )
        else:
            sig = ArbSignal(
                market=market,
                token_id=market.no_token.token_id,
                side="NO",
                fair_prob=fair_no,
                market_prob=no_mid,
                edge=edge_no,
                kelly_f=kelly_fraction(fair_no, no_mid),
                symbol=symbol,
                consensus_price=consensus,
                strike_price=market.strike_price,
            )

        await self._maybe_emit(sig)

    async def _maybe_emit(self, sig: ArbSignal) -> None:
        cid = sig.market.condition_id
        last = self._last_signal.get(cid, 0.0)
        if time.monotonic() - last < config.STRATEGY.trade_cooldown_secs:
            return

        self._last_signal[cid] = time.monotonic()

        src_prices = self._aggregator.source_prices(sig.symbol)
        src_str = "  ".join(f"{s}={p:.2f}" if p else f"{s}=stale"
                            for s, p in src_prices.items())

        # e.g.: ARB  BTC  live=85200.00  strike=85000.00  buy=YES(above)
        #             fair=0.581  mkt=0.500  edge=0.081  kelly=0.193  t_rem=487s
        above_below = "above" if sig.market.direction == "UP" else "below"
        log.info(
            "ARB  %s  live=%.2f  strike=%.2f (%s)  buy=%s"
            "  fair=%.3f  mkt=%.3f  edge=%.3f  kelly=%.3f  t_rem=%.0fs  [%s]",
            sig.symbol, sig.consensus_price, sig.strike_price, above_below,
            sig.side, sig.fair_prob, sig.market_prob, sig.edge, sig.kelly_f,
            sig.market.time_remaining_secs, src_str,
        )

        asyncio.create_task(self._on_signal(sig))
