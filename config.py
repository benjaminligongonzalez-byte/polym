"""
Central configuration for the Polymarket momentum trading bot.

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
# Keywords found in the market question for 15-min crypto up/down bets.
# Polymarket typically phrases them as "Will BTC be higher 15 minutes from now?"
MARKET_KEYWORDS: list[str] = [
    "higher 15 minutes",
    "lower 15 minutes",
    "up in 15",
    "down in 15",
]

# Crypto symbols we care about (Polymarket uses these in question text)
TARGET_SYMBOLS: list[str] = ["BTC", "ETH"]

# ---------------------------------------------------------------------------
# Binance WebSocket price feed
# ---------------------------------------------------------------------------
# Streams: <symbol>@trade  gives every trade in real time (fastest feed)
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="
BINANCE_STREAMS: list[str] = ["btcusdt@trade", "ethusdt@trade"]

# Mapping from Binance stream symbol → our canonical symbol
STREAM_SYMBOL_MAP: dict[str, str] = {
    "btcusdt": "BTC",
    "ethusdt": "ETH",
}

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    # % price move on Binance that triggers a trade signal (e.g. 0.10 = 0.10%)
    trigger_pct: float = 0.10

    # Lookback window (seconds) over which we measure the price move
    lookback_secs: float = 5.0

    # Minimum Polymarket probability for a YES token before we'll buy YES
    # (avoid buying when market already fully priced in the move)
    min_yes_prob: float = 0.10
    max_yes_prob: float = 0.90

    # Same bounds for NO
    min_no_prob: float = 0.10
    max_no_prob: float = 0.90

    # Maximum number of open (unresolved) positions at once per symbol
    max_open_positions: int = 3

    # Cooldown (seconds) after firing a trade before we can fire again
    # on the same market — prevents hammering on a single window
    trade_cooldown_secs: float = 30.0


@dataclass
class RiskConfig:
    # Maximum USDC to spend per order
    max_order_usdc: float = 20.0

    # Minimum order size (below this we skip — not worth fees)
    min_order_usdc: float = 2.0

    # Maximum total USDC deployed across all open positions
    max_total_exposure_usdc: float = 200.0

    # Slippage tolerance: how far from mid we'll accept (fraction of price)
    # e.g. 0.03 means we'll pay up to 3 cents more than current best ask
    slippage_tolerance: float = 0.03


STRATEGY = StrategyConfig()
RISK = RiskConfig()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
