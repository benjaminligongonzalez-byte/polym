"""
Central configuration for the Polymarket momentum + arbitrage trading bot.

All tuneable parameters live here so nothing is scattered across files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Polymarket CLOB endpoints
# ---------------------------------------------------------------------------
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API  = "https://data-api.polymarket.com"   # positions, trades, open interest

# Chain ID: 137 = Polygon mainnet
CHAIN_ID = 137

# ---------------------------------------------------------------------------
# Credentials (loaded from .env)
# ---------------------------------------------------------------------------
PK: str = os.environ.get("PK", "")
CLOB_API_KEY: str = os.environ.get("CLOB_API_KEY", "")
CLOB_SECRET: str = os.environ.get("CLOB_SECRET", "")
CLOB_PASS_PHRASE: str = os.environ.get("CLOB_PASS_PHRASE", "")
FUNDER_ADDRESS: str = os.environ.get("FUNDER_ADDRESS", "")
PAPER_TRADE: bool = os.environ.get("PAPER_TRADE", "false").lower() == "true"
# Virtual USDC balance used when PAPER_TRADE=true (real wallet not needed)
PAPER_BALANCE_USDC: float = float(os.environ.get("PAPER_BALANCE_USDC", "1000"))

# ---------------------------------------------------------------------------
# Market filter — which Polymarket markets to trade
# ---------------------------------------------------------------------------
# Keywords that identify 15-min crypto Up/Down bet markets.
# The primary live format is "XRP Up or Down - 15 Minutes".
# Legacy phrasings are kept for backward compatibility.
MARKET_KEYWORDS: list[str] = [
    "up or down",           # new format: "Bitcoin Up or Down - Jan 7, 10:45AM-11:00AM ET"
    "up or down - 15",      # old format: "XRP Up or Down - 15 Minutes"
    "up or down - 15 min",  # variant
    "higher 15 minutes",    # legacy
    "lower 15 minutes",     # legacy
    "up in 15",             # legacy
    "down in 15",           # legacy
    "above $",              # legacy "Will BTC be above $84,500 …"
    "below $",              # legacy
]

# Crypto symbols we trade (must match text in Polymarket market titles)
TARGET_SYMBOLS: list[str] = ["BTC", "ETH", "XRP", "SOL"]

# ---------------------------------------------------------------------------
# Binance WebSocket price feed
# ---------------------------------------------------------------------------
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="
BINANCE_STREAMS: list[str] = [
    "btcusdt@trade",
    "ethusdt@trade",
    "xrpusdt@trade",
    "solusdt@trade",
]

# Mapping from Binance stream symbol → our canonical symbol
STREAM_SYMBOL_MAP: dict[str, str] = {
    "btcusdt": "BTC",
    "ethusdt": "ETH",
    "xrpusdt": "XRP",
    "solusdt": "SOL",
}

# ---------------------------------------------------------------------------
# Coinbase Exchange WebSocket price feed
# ---------------------------------------------------------------------------
# Uses the legacy Coinbase Exchange WS (no auth required for ticker)
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
COINBASE_PRODUCTS: list[str] = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD"]

# Mapping from Coinbase product → our canonical symbol
COINBASE_SYMBOL_MAP: dict[str, str] = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "XRP-USD": "XRP",
    "SOL-USD": "SOL",
}

# ---------------------------------------------------------------------------
# Volatility model
# ---------------------------------------------------------------------------
# Annualised historical volatility used in the log-normal fair-value model.
# These are rough estimates; tune them to recent realised vol.
# BTC: ~80–100% annualised; ETH: ~90–110% annualised (as of 2025 levels)
VOLATILITY: dict[str, float] = {
    "BTC": 0.90,   # ~90% annualised vol
    "ETH": 1.00,   # ~100%
    "XRP": 1.20,   # ~120% — higher vol than BTC/ETH
    "SOL": 1.30,   # ~130%
}
DEFAULT_ANNUAL_VOL: float = 1.00   # fallback for unlisted symbols

# House spread baked into Polymarket prices (Up + Down sum to ~$1.02).
# Our edge threshold must exceed this to be profitable after the spread.
HOUSE_SPREAD: float = 0.02

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    # ---- Momentum sub-strategy (speed layer) ----
    # % price move that triggers a momentum trade signal (e.g. 0.10 = 0.10%)
    trigger_pct: float = 0.10

    # Lookback window (seconds) over which we measure the move
    lookback_secs: float = 5.0

    # Probability bounds — don't bet if market price is already extreme.
    # Tightened from 0.10/0.90: prices near the extremes mean the market has
    # already fully priced the outcome; momentum adds no edge there.
    min_yes_prob: float = 0.20
    max_yes_prob: float = 0.80

    # Minimum edge (fair_prob − market_price) required before placing a momentum
    # order.  0.0 = require at least break-even (fair ≥ market).  Prevents
    # buying tokens the model already considers overpriced.
    # Example: Down at 0.895, model fair=0.494 → edge=−0.401 → blocked.
    momentum_min_edge: float = 0.0

    # Stop-loss conviction threshold for MOMENTUM positions.
    # Lower than arb_min_fair_prob (0.65) because momentum entries are made at
    # market prices (~0.50–0.55), not because the model is already 65% confident.
    # Exiting a profitable momentum trade just because fair=0.640 < 0.65 is wrong.
    # 0.50 = only stop-loss when our model thinks the bet is a coin-flip or worse.
    momentum_stop_loss_fair: float = 0.50

    # Minimum seconds to hold any position before a stop-loss can fire.
    # Prevents exiting immediately on entry-tick noise or mid-price wobble.
    # Take-profit exits are unaffected (they only fire when price moves in our favour).
    min_hold_secs: float = 30.0

    # ---- Arbitrage sub-strategy (edge layer) ----
    # Minimum fair-value edge (in probability points) before we act.
    # e.g. 0.05 = we only trade when Polymarket is >5 cents wrong.
    arb_edge_threshold: float = 0.05

    # Minimum fair probability on the chosen bet side for ARB mode.
    # e.g. 0.65 = only trade when our model says ≥65% chance of winning.
    # Filters out cheap-option longshots (7% win) and coin-flip arb (55% win).
    # Only enter when price has moved enough to give real directional conviction.
    # Sniper mode has its own separate gate (snipe_min_fair_prob, default 0.75).
    arb_min_fair_prob: float = 0.65

    # Minimum |d2| (vol-adjusted standard deviations from strike) for ARB entries.
    # d2 = [ln(S/K) - ½σ²T] / (σ√T) — the same value used in the fair_prob model.
    # When |d2| < 1.1 the price is less than 1.1σ from the strike, meaning a small
    # adverse move collapses fair probability by 20–30 points and forces a stop-loss
    # exit.  This scales automatically with asset volatility and time remaining, unlike
    # a flat %-distance check.  Set to 0.0 to disable.
    arb_min_d2: float = 1.1

    # How often (seconds) to run the polling scan loop (exit checks + midpoint refresh).
    # Entry re-evaluation is also triggered on every price tick (event-driven),
    # so this only needs to be fast enough to catch midpoint updates between ticks.
    arb_scan_interval: float = 0.1

    # How long (seconds) to reuse a cached Polymarket midpoint before re-fetching.
    # Prevents hammering the CLOB API when price ticks arrive faster than the API
    # can respond.  Entry re-evaluations on each tick use the cache; fresh fetches
    # happen at most once per midpoint_cache_ttl per market.
    midpoint_cache_ttl: float = 1.5

    # Fractional Kelly multiplier applied to the full-Kelly bet size.
    # 0.25 = quarter Kelly (conservative; good starting point).
    kelly_fraction: float = 0.25

    # ---- Early exit (take-profit / sell before resolution) ----
    # Sell a position early when Polymarket has repriced by this many cents.
    # e.g. 0.07 = if we bought Up at 0.50, exit when it's now trading at ≥0.57.
    # This frees capital for the next arb trade instead of waiting 10 minutes.
    # Set to 0.0 to disable early exits entirely (hold all positions to resolution).
    exit_take_profit: float = 0.07

    # Never exit early if fewer than this many seconds remain in the window.
    # At <90s the market is nearly resolved — selling early just wastes fees.
    exit_min_t_rem: float = 90.0

    # Fair-value exit: sell when current market price is within this many cents
    # of the live fair probability.  e.g. 0.02 = exit when market has caught up
    # to within 2¢ of our fair value estimate (captures the full reprice rather
    # than a flat +7¢ bump).  Set to 0.0 to use only the flat exit_take_profit.
    exit_residual_edge: float = 0.02

    # ---- Late-window sniper mode ----
    # The target strategy: only bet in the final minutes when the outcome
    # is nearly certain but Polymarket prices haven't fully caught up yet.
    #
    # Example: 2 min left, BTC clearly past strike → fair_prob = 0.93,
    # Polymarket still shows 0.75 → edge = 0.18, near-certain win.
    #
    # Activate sniper mode when time remaining < snipe_window_secs.
    snipe_window_secs: float = 300.0     # final 5 minutes of the window

    # Only fire in sniper mode when our fair probability is this high.
    # Below this threshold the outcome is still too uncertain — skip it.
    # 0.92 removes marginal snipes where price is barely past strike (≤0.5%
    # away) and a small reversal in the final minutes causes a full loss.
    snipe_min_fair_prob: float = 0.92    # ≥92% confident the bet wins

    # Minimum distance (as a fraction of strike) between live price and strike
    # before firing a sniper bet.  Even with high fair_prob, a price only 0.3%
    # past the strike can flip in the final minutes of a volatile market.
    # e.g. 0.010 = live price must be ≥1% above/below the strike to snipe.
    snipe_min_price_dist_pct: float = 0.010

    # Minimum edge in sniper mode (lower than arb_edge_threshold because
    # a near-certain bet needs only a small margin over the house spread).
    snipe_edge_threshold: float = 0.03

    # More aggressive Kelly multiplier in sniper mode — we're near-certain,
    # so it's correct to size up relative to the early-window arb mode.
    snipe_kelly_fraction: float = 0.50   # half-Kelly on near-locks

    # ---- Shared ----
    # Maximum open positions at once per symbol
    max_open_positions: int = 3

    # Cooldown (seconds) after firing a trade on the same market
    trade_cooldown_secs: float = 30.0


@dataclass
class RiskConfig:
    # ------------------------------------------------------------------
    # Wallet-proportional sizing
    # The bot fetches the real USDC balance at startup and every
    # balance_refresh_secs thereafter.  All limits scale with the wallet.
    # ------------------------------------------------------------------

    # Max USDC to risk on a single order as a fraction of wallet balance.
    # e.g. 0.05 = never bet more than 5% of your wallet on one trade.
    max_order_fraction: float = 0.05

    # Per-symbol exposure cap — scales dynamically with signal conviction.
    #
    # conviction_score = f(edge, fair_prob) → 0.0 (weak) … 1.0 (near-certain)
    #
    # Dynamic cap = base + (surge - base) × conviction_score
    #
    # Weak / unscored signals (momentum default):  cap = per_symbol_base_fraction
    # Near-certain snipes (edge≥scale, prob≥ceil):  cap = per_symbol_surge_fraction
    #
    # Example with $1 000 wallet:
    #   Momentum  conviction=0.0  → cap $150  (15%)
    #   Arb       conviction=0.5  → cap $325  (32.5%)
    #   Snipe     conviction=1.0  → cap $500  (50%)
    per_symbol_base_fraction: float = 0.15   # floor  (weak / momentum signals)
    per_symbol_surge_fraction: float = 0.50  # ceiling (near-certain snipes)

    # Edge value (probability points) that maps to 100% edge-conviction.
    # Anything ≥ this is treated as full edge conviction.
    conviction_edge_scale: float = 0.12   # 12¢ edge → full edge conviction

    # Fair-probability range for conviction scoring.
    # Below floor → 0% prob-conviction; above ceil → 100% prob-conviction.
    conviction_prob_floor: float = 0.65
    conviction_prob_ceil: float = 0.90

    # Global safety ceiling: total deployed across ALL symbols.
    # Set high (default 0.95) so it only catches runaway edge cases.
    # The real throttle is the per-symbol dynamic cap above.
    max_exposure_fraction: float = 0.95

    # Hard ceiling on a single order regardless of wallet size.
    # Prevents runaway bets on large wallets.
    max_order_usdc_hard: float = 50.0

    # Minimum order (below this skip — not worth the fee).
    min_order_usdc: float = 2.0

    # How far above mid-price we'll bid (aggressive taker).
    # Reduced from 0.03 to 0.01: the old 6¢ roundtrip (buy+sell) was eating the
    # full profit on BTC trades (94 shares × 0.06 = $5.69).  At 0.01 roundtrip
    # cost is 2¢/share which still ensures fills while keeping friction low.
    slippage_tolerance: float = 0.01

    # How often (seconds) to re-fetch the wallet USDC balance.
    balance_refresh_secs: float = 30.0


STRATEGY = StrategyConfig()
RISK = RiskConfig()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
