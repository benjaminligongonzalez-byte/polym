"""
Market discovery: find active Polymarket 15-min crypto up/down bet markets.

Polymarket exposes markets via both:
  - The Gamma REST API  (gamma-api.polymarket.com) — rich metadata, fast search
  - The CLOB REST API   (clob.polymarket.com)       — live order book data

We use Gamma for discovery (it has better filtering) then cross-reference
with the CLOB for token IDs needed to place orders.

Each market we track is stored as a MarketInfo dataclass.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TokenInfo:
    token_id: str       # CLOB token ID (used for order placement)
    outcome: str        # "Yes" or "No"


@dataclass
class MarketInfo:
    condition_id: str
    question: str
    symbol: str                     # "BTC" or "ETH"
    direction: str                  # "UP" or "DOWN"
    end_date_iso: str               # ISO 8601 close time string
    yes_token: TokenInfo
    no_token: TokenInfo
    # Cached mid prices (updated on each refresh)
    yes_mid: float = 0.5
    no_mid: float = 0.5
    last_refresh: float = field(default_factory=time.monotonic)
    # Strike price: the fixed dollar price embedded in the market question
    # (e.g. $84,500 in "Will BTC be above $84,500 at 2:30 PM?").
    # Parsed once at discovery; never changes for the life of the window.
    strike_price: Optional[float] = None

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self.last_refresh) > 10.0

    @property
    def time_remaining_secs(self) -> float:
        """
        Seconds until the market closes (negative if already closed).
        Parses end_date_iso as UTC; falls back to 0 on parse error.
        """
        if not self.end_date_iso:
            return 0.0
        try:
            end = datetime.fromisoformat(
                self.end_date_iso.replace("Z", "+00:00")
            )
            return (end - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError):
            return 0.0


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

# Matches the fixed dollar strike in questions like "Will BTC be above $84,500.00?"
# Handles optional commas and decimal places.
_STRIKE_RE = re.compile(r'\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)')

# "above" and "higher/up" both mean the YES outcome pays on a price increase
_DIRECTION_PATTERNS = [
    (re.compile(r"\babove\b|\bhigher\b|\bup\b", re.I), "UP"),
    (re.compile(r"\bbelow\b|\blower\b|\bdown\b", re.I), "DOWN"),
]


def _parse_direction(question: str) -> Optional[str]:
    for pattern, direction in _DIRECTION_PATTERNS:
        if pattern.search(question):
            return direction
    return None


def _parse_symbol(question: str) -> Optional[str]:
    for sym in config.TARGET_SYMBOLS:
        if sym in question.upper():
            return sym
    return None


def _parse_strike(question: str) -> Optional[float]:
    """
    Extract the fixed dollar strike price from the market question.

    Examples:
      "Will BTC be above $84,500 at 2:30 PM?"  → 84500.0
      "Will ETH be below $3,200.50 in 15 min?" → 3200.5
    """
    match = _STRIKE_RE.search(question)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _is_15min_market(question: str) -> bool:
    return any(kw.lower() in question.lower() for kw in config.MARKET_KEYWORDS)


# ---------------------------------------------------------------------------
# MarketCache — periodically refreshes the list of tradeable markets
# ---------------------------------------------------------------------------

class MarketCache:
    """
    Maintains a live list of MarketInfo objects for active 15-min crypto markets.

    Call start() once at bot startup (fetches initial market list).
    Markets are keyed by condition_id.
    """

    REFRESH_INTERVAL = 60.0    # seconds between full refreshes

    def __init__(self) -> None:
        self._markets: dict[str, MarketInfo] = {}
        self._last_full_refresh: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        await self.refresh()

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    async def refresh(self) -> None:
        """Fetch fresh market list from Gamma API."""
        log.info("Refreshing market cache from Gamma API …")
        try:
            markets = await self._fetch_gamma_markets()
            new_cache: dict[str, MarketInfo] = {}
            for m in markets:
                if m.condition_id in self._markets:
                    # Preserve mutable runtime state from existing entry.
                    # strike_price is parsed from the question so it stays
                    # the same — no need to overwrite, but keep existing as
                    # a fallback in case parse fails on re-fetch.
                    existing = self._markets[m.condition_id]
                    m.yes_mid = existing.yes_mid
                    m.no_mid = existing.no_mid
                    if m.strike_price is None:
                        m.strike_price = existing.strike_price
                new_cache[m.condition_id] = m
            self._markets = new_cache
            self._last_full_refresh = time.monotonic()
            log.info("Market cache: %d tradeable markets found.", len(self._markets))
        except Exception as exc:
            log.error("Market refresh failed: %s", exc)

    async def _fetch_gamma_markets(self) -> list[MarketInfo]:
        """
        Query Gamma API for active markets, paginating through results.
        Filters to 15-min crypto up/down markets only.
        """
        results: list[MarketInfo] = []
        offset = 0
        limit = 100

        while True:
            url = (
                f"{config.GAMMA_API}/markets"
                f"?active=true&closed=false&limit={limit}&offset={offset}"
            )
            async with self._session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()

            markets = data if isinstance(data, list) else data.get("markets", [])
            if not markets:
                break

            for raw in markets:
                m = self._parse_gamma_market(raw)
                if m is not None:
                    results.append(m)

            if len(markets) < limit:
                break
            offset += limit

        return results

    @staticmethod
    def _parse_gamma_market(raw: dict) -> Optional[MarketInfo]:
        question: str = raw.get("question", "")
        if not question:
            return None
        if not _is_15min_market(question):
            return None

        symbol = _parse_symbol(question)
        direction = _parse_direction(question)
        if not symbol or not direction:
            return None

        condition_id: str = raw.get("conditionId", "")
        end_date: str = raw.get("endDate", "")

        # Tokens — Gamma returns an array of outcome token dicts
        tokens: list[dict] = raw.get("tokens", []) or raw.get("clobTokenIds", [])

        yes_token_id = ""
        no_token_id = ""

        # Gamma v2 format: tokens is list of {"token_id": "...", "outcome": "Yes/No"}
        for t in tokens:
            outcome = t.get("outcome", "")
            tid = t.get("token_id", t.get("tokenId", ""))
            if outcome.lower() == "yes":
                yes_token_id = tid
            elif outcome.lower() == "no":
                no_token_id = tid

        if not yes_token_id or not no_token_id:
            return None

        strike = _parse_strike(question)
        if strike is None:
            log.debug("Could not parse strike price from: %r — skipping.", question)
            return None

        return MarketInfo(
            condition_id=condition_id,
            question=question,
            symbol=symbol,
            direction=direction,
            end_date_iso=end_date,
            yes_token=TokenInfo(token_id=yes_token_id, outcome="Yes"),
            no_token=TokenInfo(token_id=no_token_id, outcome="No"),
            strike_price=strike,
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_markets_for(self, symbol: str, direction: str) -> list[MarketInfo]:
        """Return all active markets for the given symbol + direction."""
        return [
            m for m in self._markets.values()
            if m.symbol == symbol and m.direction == direction
        ]

    def all_markets(self) -> list[MarketInfo]:
        return list(self._markets.values())

    async def run_refresh_loop(self) -> None:
        """Periodically refresh the market cache. Run as a background task."""
        while True:
            await asyncio.sleep(self.REFRESH_INTERVAL)
            await self.refresh()
