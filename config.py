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
# Keywords that identify 15-min crypto up/down bet markets.
# Polymarket uses multiple phrasings — cover them all.
# Each entry must appear in the question text (case-insensitive).
MARKET_KEYWORDS: list[str] = [
    "higher 15 minutes",
    "lower 15 minutes",
    "up in 15",
    "down in 15",
    "above $",          # "Will BTC be above $84,500 at ..."
    "below $",          # "Will BTC be below $3,200 at ..."
]

# Crypto symbols we care about (must match text found in Polymarket questions)
TARGET_SYMBOLS: list[str] = ["BTC", "ETH"]

# ---------------------------------------------------------------------------
# Binance WebSocket price feed
# ---------------------------------------------------------------------------
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="
BINANCE_STREAMS: list[str] = ["btcusdt@trade", "ethusdt@trade"]

# Mapping from Binance stream symbol → our canonical symbol
STREAM_SYMBOL_MAP: dict[str, str] = {
    "btcusdt": "BTC",
    "ethusdt": "ETH",
}

# ---------------------------------------------------------------------------
# Coinbase Exchange WebSocket price feed
# ---------------------------------------------------------------------------
# Uses the legacy Coinbase Exchange WS (no auth required for ticker)
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
COINBASE_PRODUCTS: list[str] = ["BTC-USD", "ETH-USD"]

# Mapping from Coinbase product → our canonical symbol
COINBASE_SYMBOL_MAP: dict[str, str] = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
}

# ---------------------------------------------------------------------------
# Volatility model
# ---------------------------------------------------------------------------
# Annualised historical volatility used in the log-normal fair-value model.
# These are rough estimates; tune them to recent realised vol.
# BTC: ~80–100% annualised; ETH: ~90–110% annualised (as of 2025 levels)
VOLATILITY: dict[str, float] = {
    "BTC": 0.90,   # 90% annualised vol
    "ETH": 1.00,   # 100% annualised vol
}
DEFAULT_ANNUAL_VOL: float = 0.90   # fallback for unlisted symbols

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

    # How often (seconds) to scan all markets for arb opportunities
    arb_scan_interval: float = 2.0

    # Fractional Kelly multiplier applied to the full-Kelly bet size.
    # 0.25 = quarter Kelly (conservative; good starting point).
    kelly_fraction: float = 0.25

    # ---- Shared ----
    # Maximum open positions at once per symbol
    max_open_positions: int = 3

    # Cooldown (seconds) after firing a trade on the same market
    trade_cooldown_secs: float = 30.0


@dataclass
class RiskConfig:
    # Maximum USDC to spend on a single order
    max_order_usdc: float = 20.0

    # Minimum order (below this we skip — not worth fees / slippage)
    min_order_usdc: float = 2.0

    # Hard cap on total USDC deployed across all open positions
    max_total_exposure_usdc: float = 200.0

    # How far above mid-price we'll bid (aggressive taker)
    # e.g. 0.03 = pay up to 3 cents above current best ask
    slippage_tolerance: float = 0.03


STRATEGY = StrategyConfig()
RISK = RiskConfig()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
