"""
Live trade tracker for a target Polymarket wallet address.

Polls the Polymarket Data API and CLOB API for new trades from the
target address and surfaces them in real-time.  Optionally fires copy-trade
signals through the bot's normal order-manager (which applies all standard
gates — fair-value, TA, risk — so we never blindly copy a bad trade).

Endpoints tried (in order):
  1. data-api.polymarket.com/activity?user=ADDRESS  (comprehensive)
  2. clob.polymarket.com/trades?maker_address=ADDRESS  (maker fills)
     clob.polymarket.com/trades?taker_address=ADDRESS  (taker fills)

Market names are resolved from gamma-api.polymarket.com and cached
for the lifetime of the tracker.

Console commands added by bot.py:
  w   — print current tracker status + tracked positions
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import aiohttp

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI colours (self-contained so tracker.py is standalone)
# ---------------------------------------------------------------------------
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_GREEN   = "\033[92m"
_RED     = "\033[91m"
_YELLOW  = "\033[93m"
_CYAN    = "\033[96m"
_MAGENTA = "\033[95m"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrackedTrade:
    trade_id:     str
    condition_id: str
    token_id:     str
    side:         str    # "BUY" | "SELL"
    price:        float  # 0–1 scale
    size:         float  # shares
    amount:       float  # USDC
    question:     str    # market question text
    outcome:      str    # "Up" | "Down" | "Yes" | "No" | "?"
    timestamp:    float  # unix epoch seconds

    @property
    def is_crypto_updown(self) -> bool:
        """True if this trade is in one of our tracked 15-min Up/Down markets."""
        q = self.question.lower()
        return any(kw in q for kw in config.MARKET_KEYWORDS)

    @property
    def symbol(self) -> str | None:
        """Return the crypto symbol if this is a tracked market, else None."""
        q = self.question.upper()
        for sym in config.TARGET_SYMBOLS:
            if sym in q:
                return sym
        return None

    @property
    def direction(self) -> str | None:
        """'UP' or 'DOWN' if outcome maps to a direction, else None."""
        if self.outcome in ("Up", "Yes"):
            return "UP"
        if self.outcome in ("Down", "No"):
            return "DOWN"
        return None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TraderTracker:
    """
    Tracks all trades from a single Polymarket wallet address.

    Parameters
    ----------
    address:
        Ethereum wallet address to watch (checksummed or lowercase).
    on_trade:
        Optional async callback(TrackedTrade) fired for every new trade.
        Used for copy-trade signals.
    """

    def __init__(
        self,
        address: str,
        on_trade: Callable[[TrackedTrade], Awaitable[None]] | None = None,
    ) -> None:
        self._address   = address.lower()
        self._on_trade  = on_trade
        self._session: aiohttp.ClientSession | None = None

        # Deduplication: trade IDs / tx hashes we've already processed
        self._seen_ids: set[str] = set()

        # Enrichment caches
        self._question_cache: dict[str, str]   = {}  # condition_id → question
        self._token_labels:   dict[str, str]   = {}  # token_id     → "Up"/"Down"

        # Aggregated position view: token_id → net shares (+ = long, − = sold)
        self._net_shares: dict[str, float] = {}
        # token_id → question (for display in status_report)
        self._token_questions: dict[str, str] = {}

        # Stats
        self._total_seen:  int = 0
        self._started_at:  float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main async loop.  Runs forever; cancel the task to stop.

        On startup takes an initial snapshot so we only alert on NEW
        trades that arrive after the bot starts — no historical spam.
        """
        self._started_at = time.monotonic()
        timeout = aiohttp.ClientTimeout(total=15)
        headers = {"User-Agent": "polymarket-bot-tracker/1.0", "Accept": "application/json"}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            self._session = session

            log.info(
                "%s👁 TRACKER%s  watching %s%s%s  (poll every %gs)",
                _MAGENTA, _RESET,
                _CYAN, self._address[:12] + "…", _RESET,
                config.TRACKER_POLL_SECS,
            )

            # Initial snapshot — load seen IDs silently
            await self._poll(initial=True)
            log.info(
                "%s👁 TRACKER%s  snapshot loaded — %d existing trades suppressed, "
                "will alert on new activity.",
                _MAGENTA, _RESET, len(self._seen_ids),
            )

            while True:
                await asyncio.sleep(config.TRACKER_POLL_SECS)
                try:
                    await self._poll()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.debug("TraderTracker poll error: %s", exc)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll(self, initial: bool = False) -> None:
        raw_trades = await self._fetch_all()

        # Sort oldest-first so log lines come out in chronological order
        new = [t for t in raw_trades if self._trade_id(t) not in self._seen_ids]
        new.sort(key=lambda t: float(t.get("timestamp") or t.get("matchTime") or 0))

        for raw in new:
            tid = self._trade_id(raw)
            if not tid:
                continue
            self._seen_ids.add(tid)
            self._total_seen += 1

            if initial:
                continue  # suppress historical trades at startup

            try:
                trade = await self._build_trade(raw, tid)
                if trade:
                    self._log_trade(trade)
                    self._update_positions(trade)
                    if self._on_trade:
                        asyncio.create_task(self._on_trade(trade))
            except Exception as exc:
                log.debug("TraderTracker: failed to process %s: %s", tid[:12], exc)

    # ------------------------------------------------------------------
    # API calls  (Data API primary, CLOB API fallback)
    # ------------------------------------------------------------------

    async def _fetch_all(self) -> list[dict]:
        """Fetch recent trades; try Data API first, CLOB as fallback."""
        try:
            result = await self._fetch_data_api()
            if result is not None:
                return result
        except Exception as exc:
            log.debug("Tracker: Data API error (%s) — trying CLOB API", exc)

        try:
            return await self._fetch_clob_api()
        except Exception as exc:
            log.debug("Tracker: CLOB API also failed: %s", exc)
            return []

    async def _fetch_data_api(self) -> list[dict] | None:
        """
        GET data-api.polymarket.com/activity?user=ADDRESS&limit=100

        Returns a list of trade dicts, or None if the endpoint is
        unavailable (so we fall back to the CLOB API).
        """
        assert self._session is not None
        url    = f"{config.DATA_API}/activity"
        params = {"user": self._address, "limit": 100}

        async with self._session.get(url, params=params) as resp:
            if resp.status == 404:
                return None          # endpoint doesn't exist → use fallback
            if resp.status != 200:
                log.debug("Tracker: Data API returned %d", resp.status)
                return None

            body = await resp.json(content_type=None)

        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            # Could be {"data": [...]} or {"activities": [...]}
            for key in ("data", "activities", "trades"):
                if key in body and isinstance(body[key], list):
                    return body[key]
        return None

    async def _fetch_clob_api(self) -> list[dict]:
        """
        GET clob.polymarket.com/trades?maker_address=ADDR  (our fills as maker)
        GET clob.polymarket.com/trades?taker_address=ADDR  (our fills as taker)

        Merges both result sets and deduplicates by trade ID / tx hash.
        """
        assert self._session is not None
        url     = f"{config.CLOB_HOST}/trades"
        merged: list[dict] = []
        seen:   set[str]   = set()

        for role_param in ("maker_address", "taker_address"):
            params = {role_param: self._address, "limit": 100}
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        continue
                    body = await resp.json(content_type=None)
                    trades = body.get("data", body) if isinstance(body, dict) else body
                    for t in (trades or []):
                        tid = self._trade_id(t)
                        if tid and tid not in seen:
                            seen.add(tid)
                            merged.append(t)
            except Exception as exc:
                log.debug("Tracker: CLOB %s query failed: %s", role_param, exc)

        return merged

    # ------------------------------------------------------------------
    # Trade normalisation + enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_id(raw: dict) -> str:
        """Extract a unique identifier from a raw trade dict."""
        return (
            raw.get("id")
            or raw.get("tradeId")
            or raw.get("trade_id")
            or raw.get("transactionHash")
            or raw.get("transaction_hash")
            or raw.get("txHash")
            or ""
        )

    async def _build_trade(self, raw: dict, trade_id: str) -> TrackedTrade | None:
        """Normalise a raw API response dict into a typed TrackedTrade."""

        # ── Condition / token IDs ─────────────────────────────────────────
        condition_id = (
            raw.get("conditionId")
            or raw.get("condition_id")
            or raw.get("market")
            or ""
        )
        token_id = (
            raw.get("tokenId")
            or raw.get("token_id")
            or raw.get("asset_id")
            or raw.get("assetId")
            or ""
        )

        # ── Side ─────────────────────────────────────────────────────────
        side_raw = (raw.get("side") or raw.get("type") or "BUY").upper()
        side = "BUY" if side_raw in ("BUY", "B") else "SELL"

        # ── Numeric fields ────────────────────────────────────────────────
        try:
            price = float(raw.get("price", 0))
        except (ValueError, TypeError):
            price = 0.0
        try:
            size = float(raw.get("size") or raw.get("shares") or 0)
        except (ValueError, TypeError):
            size = 0.0
        try:
            amount = float(raw.get("amount") or raw.get("usdcSize") or (price * size))
        except (ValueError, TypeError):
            amount = price * size

        # ── Timestamp ────────────────────────────────────────────────────
        ts_raw = (
            raw.get("timestamp")
            or raw.get("matchTime")
            or raw.get("block_time")
            or raw.get("createdAt")
        )
        try:
            timestamp = float(ts_raw) if ts_raw else time.time()
            # If timestamp looks like milliseconds, convert to seconds
            if timestamp > 1e12:
                timestamp /= 1000.0
        except (ValueError, TypeError):
            timestamp = time.time()

        # ── Market enrichment ─────────────────────────────────────────────
        question = await self._get_question(condition_id)
        outcome  = await self._get_outcome(token_id, condition_id, raw)

        return TrackedTrade(
            trade_id=trade_id,
            condition_id=condition_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            amount=amount,
            question=question,
            outcome=outcome,
            timestamp=timestamp,
        )

    async def _get_question(self, condition_id: str) -> str:
        if not condition_id:
            return "(unknown market)"
        if condition_id in self._question_cache:
            return self._question_cache[condition_id]

        try:
            assert self._session is not None
            url    = f"{config.GAMMA_API}/markets"
            params = {"id": condition_id}
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    body    = await resp.json(content_type=None)
                    markets = body if isinstance(body, list) else body.get("markets", [])
                    if markets:
                        m = markets[0]
                        q = m.get("question") or m.get("title") or condition_id[:20]
                        self._question_cache[condition_id] = q
                        # Cache token → outcome while we're here
                        for tok in m.get("tokens") or m.get("outcomes") or []:
                            tok_id   = tok.get("token_id") or tok.get("tokenId") or ""
                            tok_out  = tok.get("outcome") or ""
                            if tok_id:
                                self._token_labels[tok_id] = tok_out
                        return q
        except Exception as exc:
            log.debug("Tracker: gamma lookup failed for %s: %s", condition_id[:12], exc)

        fallback = condition_id[:16] + "…"
        self._question_cache[condition_id] = fallback
        return fallback

    async def _get_outcome(
        self, token_id: str, condition_id: str, raw: dict
    ) -> str:
        # Some APIs include outcome directly
        for key in ("outcome", "outcomeName", "tokenName"):
            val = raw.get(key)
            if val:
                return str(val)

        if token_id in self._token_labels:
            return self._token_labels[token_id]

        # Side effect of _get_question populates self._token_labels
        if condition_id:
            await self._get_question(condition_id)

        return self._token_labels.get(token_id, "?")

    # ------------------------------------------------------------------
    # Output + position tracking
    # ------------------------------------------------------------------

    def _log_trade(self, trade: TrackedTrade) -> None:
        side_col = _GREEN if trade.side == "BUY" else _RED
        side_tag = f"{side_col}{'[BUY] ' if trade.side == 'BUY' else '[SELL]'}{_RESET}"

        out_col = (
            _GREEN if trade.outcome in ("Up", "Yes")
            else _RED if trade.outcome in ("Down", "No")
            else _CYAN
        )

        ts_str = datetime.datetime.fromtimestamp(trade.timestamp).strftime("%H:%M:%S")

        log.info(
            "%s👁 COPY-WATCH%s  %s  %s%s %s%s  "
            "@ %s%.2f¢%s  %.1f shares / %s$%.2f%s  |  %s",
            _MAGENTA, _RESET,
            side_tag,
            ts_str, _DIM, "", _RESET,
            _BOLD, trade.price * 100, _RESET,
            trade.size,
            _BOLD, trade.amount, _RESET,
            trade.question[:65],
        )

        # Secondary line with market context
        sym_tag = f"  {_CYAN}[{trade.symbol}]{_RESET}" if trade.symbol else ""
        dir_tag = (
            f"  {_GREEN}▲ {trade.direction}{_RESET}" if trade.direction == "UP"
            else f"  {_RED}▼ {trade.direction}{_RESET}" if trade.direction == "DOWN"
            else ""
        )
        out_tag = f"  outcome={out_col}{trade.outcome}{_RESET}"
        if sym_tag or dir_tag:
            log.info("            %s%s%s", sym_tag, dir_tag, out_tag)

    def _update_positions(self, trade: TrackedTrade) -> None:
        key = trade.token_id or trade.condition_id
        if not key:
            return
        delta = trade.size if trade.side == "BUY" else -trade.size
        current = self._net_shares.get(key, 0.0)
        new_val = current + delta
        if abs(new_val) < 0.01:
            self._net_shares.pop(key, None)
            self._token_questions.pop(key, None)
        else:
            self._net_shares[key] = new_val
            if trade.question:
                self._token_questions[key] = trade.question

    # ------------------------------------------------------------------
    # Status report (for 'w' console command)
    # ------------------------------------------------------------------

    def status_report(self) -> str:
        uptime = time.monotonic() - self._started_at
        h, m = divmod(int(uptime) // 60, 60)
        s = int(uptime) % 60

        lines = [
            "",
            f"{_MAGENTA}{_BOLD}👁  COPY-WATCH — {self._address}{_RESET}",
            f"  uptime     : {h:02d}h {m:02d}m {s:02d}s",
            f"  trades seen: {self._total_seen}  (new since start)",
            f"  positions  : {len(self._net_shares)} open tokens tracked",
        ]

        if self._net_shares:
            lines.append("")
            lines.append(f"  {'Outcome / Market':<45}  {'Net Shares':>10}")
            lines.append("  " + "─" * 58)
            # Sort by abs position size descending
            sorted_pos = sorted(
                self._net_shares.items(), key=lambda kv: -abs(kv[1])
            )
            for tok, shares in sorted_pos:
                label   = self._token_labels.get(tok, "?")
                q       = self._token_questions.get(tok, tok[:20])
                col     = _GREEN if shares > 0 else _RED
                dir_sym = "▲" if shares > 0 else "▼"
                lines.append(
                    f"  {dir_sym} {label:<6} {q[:38]:<38}  "
                    f"{col}{shares:>+10.1f}{_RESET}"
                )

        lines.append("")
        return "\n".join(lines)
