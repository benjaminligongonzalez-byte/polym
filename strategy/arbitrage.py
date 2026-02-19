"""
Arbitrage strategy: compare fair value to live Polymarket prices.

Core idea
---------
Each 15-min window asks: "Will [SYMBOL] end ABOVE the Price to Beat?"
  Up   token → resolves $1 if final price > strike
  Down token → resolves $1 if final price < strike

We compute the fair probability using a log-normal model:

    P(S_T > K) = N(d₂)

where:
    d₂ = [ln(S / K) - ½σ²T] / (σ√T)
    S  = live consensus price (Binance + Coinbase median)
    K  = "Price to Beat" (parsed from market description)
    T  = time remaining (converted to years)
    σ  = annualised historical vol for the symbol (config.VOLATILITY)
    N  = standard normal CDF

We buy the Up token if:   fair_prob_up  - up_market_price  > edge_threshold
We buy the Down token if: fair_prob_down - down_market_price > edge_threshold

The house spread (Up + Down = ~$1.02) means we need edge > HOUSE_SPREAD
just to break even.  The default threshold (0.05) already accounts for this.

Example (from the screenshot)
------------------------------
  Symbol: XRP    Live: $1.422    Strike: $1.4208    t_rem: 12m53s
  σ = 120%  →  fair_prob_up  ≈ 0.515  (barely above strike)
  Polymarket:   up_mid = 0.62   down_mid = 0.40
  edge_up   = 0.515 - 0.62  = -0.105  (Up is OVERPRICED — skip)
  edge_down = 0.485 - 0.40  =  0.085  (Down is underpriced — BUY DOWN)

Kelly sizing
------------
    f* = (fair - market) / (1 - market)

We apply a fractional Kelly multiplier (config.STRATEGY.kelly_fraction,
default 0.25) and cap between min/max_order_usdc.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Optional

from feeds.aggregator import PriceAggregator
from polymarket.markets import MarketCache, MarketInfo
from polymarket.client import PolymarketClient
import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf — no scipy needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fair_prob_up(
    current: float,
    strike: float,
    time_remaining_secs: float,
    annual_vol: float,
) -> float:
    """
    Probability that *current* exceeds *strike* after *time_remaining_secs*,
    assuming log-normal dynamics with zero drift.
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
    when the true probability is *fair*:
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
    """A confirmed mispricing ready for execution."""
    market: MarketInfo
    token_id: str       # CLOB token ID to buy
    bet: str            # "Up" or "Down"
    fair_prob: float    # our estimated true probability
    market_prob: float  # current Polymarket price for this token
    edge: float         # fair_prob - market_prob  (always > 0)
    kelly_f: float      # full Kelly fraction (caller applies scaling)
    symbol: str
    consensus_price: float
    strike_price: float
    is_snipe: bool = False  # True when fired in late-window sniper mode


ArbCallback = Callable[["ArbSignal"], Awaitable[None]]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ArbStrategy:
    """
    Periodically scans all cached markets, computes fair value for Up and Down
    tokens, and emits ArbSignals when edge > arb_edge_threshold.

    Requires at least 2 independent price sources (Binance + Coinbase) to
    trade — a single stale source is not enough.
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
        self._order_manager: Any = None

    def set_order_manager(self, order_manager: Any) -> None:
        """Wire in the OrderManager so we can trigger exit scans."""
        self._order_manager = order_manager

    async def run(self) -> None:
        while True:
            await asyncio.sleep(config.STRATEGY.arb_scan_interval)
            # Check early exits first (free up capital before scanning for new entries)
            if self._order_manager is not None:
                await self._order_manager.check_exits()
            await self._scan_all()

    async def _scan_all(self) -> None:
        markets = self._cache.all_markets()
        await asyncio.gather(*[self._evaluate(m) for m in markets],
                             return_exceptions=True)

    async def _evaluate(self, market: MarketInfo) -> None:
        symbol = market.symbol
        cid = market.condition_id

        # --- 1. Live price checks ---
        consensus = self._aggregator.consensus_price(symbol)
        if consensus is None:
            return
        if self._aggregator.is_diverging(symbol):
            log.debug("Prices diverging for %s — skipping.", symbol)
            return
        if self._aggregator.source_count(symbol) < 2:
            log.debug("Only 1 source for %s — waiting for second feed.", symbol)
            return

        # --- 2. Strike check ---
        if market.strike_price is None:
            return

        # --- 3. Time check ---
        t_rem = market.time_remaining_secs
        if t_rem <= 0 or t_rem > 930:
            return

        # --- 3b. Select mode: sniper (final minutes) vs standard arb ---
        is_snipe = t_rem < config.STRATEGY.snipe_window_secs
        threshold = (
            config.STRATEGY.snipe_edge_threshold
            if is_snipe
            else config.STRATEGY.arb_edge_threshold
        )

        # --- 4. Fetch live Polymarket prices for both tokens ---
        try:
            up_mid = await self._pm_client.get_midpoint(market.up_token.token_id)
        except Exception as exc:
            log.debug("Midpoint fetch failed for %s: %s", cid[:8], exc)
            return
        down_mid = 1.0 - up_mid   # Up + Down must sum to 1 per contract

        # --- 5. Fair probability ---
        vol = config.VOLATILITY.get(symbol, config.DEFAULT_ANNUAL_VOL)
        p_up   = fair_prob_up(consensus, market.strike_price, t_rem, vol)
        p_down = 1.0 - p_up

        # --- 5b. Sniper confidence gate ---
        # In sniper mode we only act when the outcome is near-certain.
        # This is the core of the "all winners" strategy: only bet when
        # fair probability is already very high (price far past the strike).
        if is_snipe and max(p_up, p_down) < config.STRATEGY.snipe_min_fair_prob:
            log.debug(
                "%s %s sniper: max_fair=%.3f < min_fair=%.3f (not certain enough)",
                symbol, cid[:8], max(p_up, p_down), config.STRATEGY.snipe_min_fair_prob,
            )
            return

        # --- 5c. Sniper distance-from-strike gate ---
        # Even with high fair_prob, a price only 0.3–0.65% past the strike can
        # flip in the final minutes of a volatile market.  Require a meaningful
        # clearance before committing to a snipe bet.
        if is_snipe:
            price_dist_pct = abs(consensus - market.strike_price) / market.strike_price
            if price_dist_pct < config.STRATEGY.snipe_min_price_dist_pct:
                log.debug(
                    "%s %s sniper: price dist %.2f%% < min %.2f%% — too close to strike",
                    symbol, cid[:8],
                    price_dist_pct * 100, config.STRATEGY.snipe_min_price_dist_pct * 100,
                )
                return

        # --- 6. Edge calculation ---
        edge_up   = p_up   - up_mid
        edge_down = p_down - down_mid

        best_edge = max(edge_up, edge_down)

        if best_edge < threshold:
            log.debug(
                "%s %s | p_up=%.3f up_mkt=%.3f e_up=%.3f"
                "  p_dn=%.3f dn_mkt=%.3f e_dn=%.3f (no edge, mode=%s)",
                symbol, cid[:8],
                p_up, up_mid, edge_up,
                p_down, down_mid, edge_down,
                "snipe" if is_snipe else "arb",
            )
            return

        # Pick which side to bet.
        #
        # Sniper mode: ONLY bet the high-probability (near-certain) side.
        # The whole point is to ride the information lag — BTC moved up, so we
        # buy Up before Polymarket reprices it.  Betting the cheap low-prob side
        # (Down at 2¢ when Up should win 90%) is pure stat-arb and almost always
        # loses.  If the near-certain side no longer has edge, skip entirely.
        #
        # Standard ARB mode: pick whichever side has more edge (normal arbitrage).
        if is_snipe:
            if p_up >= p_down:
                if edge_up < threshold:
                    log.debug(
                        "%s %s sniper: near-certain side (Up) has no edge (%.3f < %.3f) — skip",
                        symbol, cid[:8], edge_up, threshold,
                    )
                    return
                sig = ArbSignal(
                    market=market,
                    token_id=market.up_token.token_id,
                    bet="Up",
                    fair_prob=p_up,
                    market_prob=up_mid,
                    edge=edge_up,
                    kelly_f=kelly_fraction(p_up, up_mid),
                    symbol=symbol,
                    consensus_price=consensus,
                    strike_price=market.strike_price,
                    is_snipe=True,
                )
            else:
                if edge_down < threshold:
                    log.debug(
                        "%s %s sniper: near-certain side (Down) has no edge (%.3f < %.3f) — skip",
                        symbol, cid[:8], edge_down, threshold,
                    )
                    return
                sig = ArbSignal(
                    market=market,
                    token_id=market.down_token.token_id,
                    bet="Down",
                    fair_prob=p_down,
                    market_prob=down_mid,
                    edge=edge_down,
                    kelly_f=kelly_fraction(p_down, down_mid),
                    symbol=symbol,
                    consensus_price=consensus,
                    strike_price=market.strike_price,
                    is_snipe=True,
                )
        elif edge_up >= edge_down:
            # ARB conviction gate: require minimum fair probability.
            # Rejects coin-flip bets (~55%) and cheap-option longshots (~7%)
            # that have statistical edge but lack real directional conviction.
            if p_up < config.STRATEGY.arb_min_fair_prob:
                log.debug(
                    "%s %s arb: Up fair=%.3f < arb_min=%.3f — skip (low conviction)",
                    symbol, cid[:8], p_up, config.STRATEGY.arb_min_fair_prob,
                )
                return
            sig = ArbSignal(
                market=market,
                token_id=market.up_token.token_id,
                bet="Up",
                fair_prob=p_up,
                market_prob=up_mid,
                edge=edge_up,
                kelly_f=kelly_fraction(p_up, up_mid),
                symbol=symbol,
                consensus_price=consensus,
                strike_price=market.strike_price,
                is_snipe=False,
            )
        else:
            if p_down < config.STRATEGY.arb_min_fair_prob:
                log.debug(
                    "%s %s arb: Down fair=%.3f < arb_min=%.3f — skip (low conviction)",
                    symbol, cid[:8], p_down, config.STRATEGY.arb_min_fair_prob,
                )
                return
            sig = ArbSignal(
                market=market,
                token_id=market.down_token.token_id,
                bet="Down",
                fair_prob=p_down,
                market_prob=down_mid,
                edge=edge_down,
                kelly_f=kelly_fraction(p_down, down_mid),
                symbol=symbol,
                consensus_price=consensus,
                strike_price=market.strike_price,
                is_snipe=False,
            )

        await self._maybe_emit(sig)

    async def _maybe_emit(self, sig: ArbSignal) -> None:
        cid = sig.market.condition_id
        if time.monotonic() - self._last_signal.get(cid, 0.0) < config.STRATEGY.trade_cooldown_secs:
            return

        self._last_signal[cid] = time.monotonic()

        src = self._aggregator.source_prices(sig.symbol)
        src_str = "  ".join(
            f"{s}={p:.4f}" if p else f"{s}=stale"
            for s, p in src.items()
        )

        mode = "SNIPE" if sig.is_snipe else "ARB"
        log.info(
            "%s %-3s  live=%.4f  strike=%.4f  buy=%-4s"
            "  fair=%.3f  mkt=%.3f  edge=%.3f  kelly=%.3f  t_rem=%.0fs  [%s]",
            mode,
            sig.symbol, sig.consensus_price, sig.strike_price, sig.bet,
            sig.fair_prob, sig.market_prob, sig.edge, sig.kelly_f,
            sig.market.time_remaining_secs, src_str,
        )

        asyncio.create_task(self._on_signal(sig))
