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

# ---------------------------------------------------------------------------
# Market filter — which Polymarket markets to trade
# ---------------------------------------------------------------------------
# Keywords that identify 15-min crypto Up/Down bet markets.
# The primary live format is "XRP Up or Down - 15 Minutes".
# Legacy phrasings are kept for backward compatibility.
MARKET_KEYWORDS: list[str] = [
    "up or down - 15",      # "XRP Up or Down - 15 Minutes"  ← primary format
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

    # Probability bounds — don't bet if market price is already extreme
    min_yes_prob: float = 0.10
    max_yes_prob: float = 0.90

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

    # How often (seconds) to scan all markets for arb opportunities
    arb_scan_interval: float = 2.0

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
    snipe_min_fair_prob: float = 0.75    # ≥75% confident the bet wins

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

    # Maximum total USDC deployed across all open positions at once,
    # as a fraction of wallet balance.
    # e.g. 0.20 = never have more than 20% of wallet exposed simultaneously.
    max_exposure_fraction: float = 0.20

    # Hard ceiling on a single order regardless of wallet size.
    # Prevents runaway bets on large wallets.
    max_order_usdc_hard: float = 50.0

    # Minimum order (below this skip — not worth the fee).
    min_order_usdc: float = 2.0

    # How far above mid-price we'll bid (aggressive taker).
    # e.g. 0.03 = pay up to 3 cents more than current best ask.
    slippage_tolerance: float = 0.03

    # How often (seconds) to re-fetch the wallet USDC balance.
    balance_refresh_secs: float = 30.0


STRATEGY = StrategyConfig()
RISK = RiskConfig()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
