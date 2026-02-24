"""
Market discovery: find active Polymarket 15-min crypto Up/Down bet markets.

Market structure (as shown in the Polymarket UI)
-------------------------------------------------
Title:          "XRP Up or Down - 15 Minutes"
Price to Beat:  $1.4208   (fixed strike, set when the window opens)
Current Price:  $1.422    (live reference, shown for context only)
Tokens:
  Up   token  → resolves $1 if price ends ABOVE the strike
  Down token  → resolves $1 if price ends BELOW the strike
Pricing:        Up 62¢ + Down 40¢ = $1.02  (2¢ house spread baked in)

The strike ("Price to Beat") appears in the market description, not the title.
The token outcomes are literally "Up" and "Down" in the Gamma API response.
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
    outcome: str        # "Up" or "Down"


@dataclass
class MarketInfo:
    condition_id: str
    question: str           # e.g. "XRP Up or Down - 15 Minutes"
    symbol: str             # "BTC", "ETH", "XRP", "SOL" …
    end_date_iso: str       # ISO 8601 close time string
    up_token: TokenInfo     # token that pays $1 if price ends ABOVE strike
    down_token: TokenInfo   # token that pays $1 if price ends BELOW strike
    # Cached market mid prices (0–1 scale)
    up_mid: float = 0.5
    down_mid: float = 0.5
    last_refresh: float = field(default_factory=time.monotonic)
    # The fixed "Price to Beat" parsed from the market description.
    # Set once at discovery; never changes for the life of the window.
    strike_price: Optional[float] = None

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self.last_refresh) > 10.0

    @property
    def time_remaining_secs(self) -> float:
        """Seconds until the window closes (negative when expired)."""
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
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches dollar amounts like $1.4208  $84,500  $3,200.50
_STRIKE_RE = re.compile(
    r'\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)'
)


def _parse_symbol(text: str) -> Optional[str]:
    upper = text.upper()
    for sym in config.TARGET_SYMBOLS:
        if sym in upper:
            return sym
    return None


def _parse_strike(question: str, description: str = "") -> Optional[float]:
    """
    Extract the fixed dollar strike from the title or description.

    The title ("XRP Up or Down - 15 Minutes") never contains the price.
    It appears in the description, e.g.:
      "If XRP price is above $1.4208 at 2:00 AM ET … Up resolves to $1."
    Also handles old-style markets where it's in the question text.
    """
    for text in (description, question):
        match = _STRIKE_RE.search(text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _is_15min_market(question: str) -> bool:
    return any(kw.lower() in question.lower() for kw in config.MARKET_KEYWORDS)


# ---------------------------------------------------------------------------
# MarketCache
# ---------------------------------------------------------------------------

class MarketCache:
    """
    Maintains a live list of MarketInfo objects for active 15-min windows.
    Refreshes from the Gamma API every REFRESH_INTERVAL seconds.
    """

    REFRESH_INTERVAL = 60.0

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
        log.info("Refreshing market cache from Gamma API …")
        try:
            markets = await self._fetch_gamma_markets()
            new_cache: dict[str, MarketInfo] = {}
            for m in markets:
                if m.condition_id in self._markets:
                    existing = self._markets[m.condition_id]
                    m.up_mid = existing.up_mid
                    m.down_mid = existing.down_mid
                    # Preserve parsed strike (never overwrite with None)
                    if m.strike_price is None:
                        m.strike_price = existing.strike_price
                new_cache[m.condition_id] = m
            self._markets = new_cache
            self._last_full_refresh = time.monotonic()
            log.info("Market cache: %d tradeable markets found.", len(self._markets))
        except Exception as exc:
            log.error("Market refresh failed: %s", exc)

    async def _fetch_gamma_markets(self) -> list[MarketInfo]:
        """Fetch current 15-min window markets by slug.

        Slugs follow the pattern: {sym}-updown-15m-{epoch}
        where epoch is the unix timestamp of the 15-min window start.
        Boundaries fall at :00, :15, :30, :45 of every hour (every 900s).
        We check the current and previous boundary to cover overlap.
        """
        now = int(time.time())
        current_boundary = (now // 900) * 900
        boundaries = [current_boundary, current_boundary - 900]

        sym_slugs = {"BTC": "btc", "ETH": "eth", "XRP": "xrp", "SOL": "sol"}
        raw_markets: list[dict] = []
        seen_cids: set[str] = set()

        for boundary in boundaries:
            for sym, slug_sym in sym_slugs.items():
                slug = f"{slug_sym}-updown-15m-{boundary}"
                url = f"{config.GAMMA_API}/events/slug/{slug}"
                try:
                    async with self._session.get(url) as resp:
                        if resp.status == 404:
                            log.debug("Slug 404: %s", slug)
                            continue
                        resp.raise_for_status()
                        event = await resp.json()
                    if isinstance(event, list):
                        event = event[0] if event else {}
                    mkts = event.get("markets", [])
                    log.info("Slug hit: %s → %d market(s)", slug, len(mkts))
                    for m in mkts:
                        cid = m.get("conditionId", "")
                        if cid not in seen_cids:
                            seen_cids.add(cid)
                            raw_markets.append(m)
                except Exception as exc:
                    log.warning("Slug fetch error for %s: %s", slug, exc)

        if not raw_markets:
            log.warning("Slug lookup: all slugs returned 404 — no active 15-min markets found.")
        else:
            log.info("Slug lookup: %d raw markets before filtering.", len(raw_markets))
            for r in raw_markets[:8]:
                log.info("  question: %r", r.get("question", r.get("title", "(no question)")))

        results: list[MarketInfo] = []
        for raw in raw_markets:
            m = self._parse_gamma_market(raw)
            if m is not None:
                results.append(m)
        return results

    @staticmethod
    def _parse_gamma_market(raw: dict) -> Optional[MarketInfo]:
        question: str = raw.get("question", "")
        if not question:
            return None
        if not _is_15min_market(question):
            return None

        symbol = _parse_symbol(question)
        if not symbol:
            return None

        condition_id: str = raw.get("conditionId", "")
        end_date: str = raw.get("endDate", "")

        # Strike ("Price to Beat") — check description first, then question
        description: str = raw.get("description", "") or ""
        strike = _parse_strike(question, description)
        if strike is None:
            log.debug("No strike price found for: %r — skipping.", question)
            return None

        # Token IDs — outcomes are "Up" / "Down" for these markets.
        # Fall back to "Yes" / "No" for old-style markets.
        tokens: list[dict] = raw.get("tokens", []) or raw.get("clobTokenIds", [])
        up_token_id = ""
        down_token_id = ""

        for t in tokens:
            outcome = t.get("outcome", "").lower()
            tid = t.get("token_id", t.get("tokenId", ""))
            if outcome in ("up", "yes"):
                up_token_id = tid
            elif outcome in ("down", "no"):
                down_token_id = tid

        if not up_token_id or not down_token_id:
            log.debug("Missing Up/Down token IDs for: %r — skipping.", question)
            return None

        return MarketInfo(
            condition_id=condition_id,
            question=question,
            symbol=symbol,
            end_date_iso=end_date,
            up_token=TokenInfo(token_id=up_token_id, outcome="Up"),
            down_token=TokenInfo(token_id=down_token_id, outcome="Down"),
            strike_price=strike,
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_market(self, condition_id: str) -> MarketInfo | None:
        """Return a single market by condition ID, or None if not found."""
        return self._markets.get(condition_id)

    def get_markets_for_symbol(self, symbol: str) -> list[MarketInfo]:
        """Return all active markets for the given symbol."""
        return [m for m in self._markets.values() if m.symbol == symbol]

    def all_markets(self) -> list[MarketInfo]:
        return list(self._markets.values())

    async def run_refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.REFRESH_INTERVAL)
            await self.refresh()
