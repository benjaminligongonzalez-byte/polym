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
import collections
import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import aiohttp
from web3 import Web3

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
        # Polymarket titles use full names ("Bitcoin Up or Down") not tickers
        _FULL_NAMES = {"BITCOIN": "BTC", "ETHEREUM": "ETH", "SOLANA": "SOL", "RIPPLE": "XRP"}
        for name, sym in _FULL_NAMES.items():
            if name in q:
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


@dataclass
class LoggedTrade:
    """Enriched record of a single observed trade, stored for strategy analysis."""
    trade:        TrackedTrade
    seq:          int           # trade number within this session (1-based)
    pos_before:   float         # net shares in this token BEFORE this trade
    pos_after:    float         # net shares AFTER this trade
    trade_type:   str           # OPEN / ADD / TRIM / CLOSE / FLIP / SHORT / ADD_S / COVER
    spot_price:   float | None  # consensus crypto spot price at time of trade (USD)
    avg_cost:     float         # avg $/share paid for current long position
    realized_pnl: float | None  # P&L in USDC (SELL trades only; None for BUY)

    @property
    def price_cents(self) -> float:
        return self.trade.price * 100

    def _infer_type(side: str, pos_before: float, pos_after: float) -> str:
        if side == "BUY":
            if pos_before == 0:       return "OPEN"
            if pos_before > 0:        return "ADD"
            if pos_after >= 0:        return "FLIP"
            return "COVER"
        else:  # SELL
            if pos_before <= 0:       return "SHORT"
            if pos_after <= 0:        return "CLOSE"
            return "TRIM"


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
        on_trade:   Callable[[TrackedTrade], Awaitable[None]] | None = None,
        price_feed: Callable[[str, float], float | None] | None = None,
    ) -> None:
        self._address    = address.lower()
        self._on_trade   = on_trade
        self._price_feed = price_feed  # price_at(symbol, timestamp) → spot USD price
        self._session: aiohttp.ClientSession | None = None

        # Deduplication: trade IDs / tx hashes we've already processed
        self._seen_ids: set[str] = set()

        # Enrichment caches
        self._question_cache: dict[str, str]   = {}  # condition_id → question
        self._token_labels:   dict[str, str]   = {}  # token_id     → "Up"/"Down"

        # Adaptive CLOB bypass: if CLOB returns 0 for _CLOB_SKIP_AFTER consecutive
        # polls, skip CLOB for _CLOB_SKIP_POLLS polls before retrying.
        # Proxy-wallet users always get 0 from CLOB — no point hammering it.
        _CLOB_SKIP_AFTER = 10     # give up after 10 empty results (~20 s)
        _CLOB_SKIP_POLLS = 300    # skip for 300 polls (~10 min) then retry
        self._clob_empty_streak: int = 0
        self._clob_skip_remaining: int = 0
        self._CLOB_SKIP_AFTER = _CLOB_SKIP_AFTER
        self._CLOB_SKIP_POLLS = _CLOB_SKIP_POLLS

        # Aggregated position view: token_id → net shares (+ = long, − = sold)
        self._net_shares: dict[str, float] = {}
        # token_id → weighted average cost per share for current long position
        self._avg_cost: dict[str, float] = {}
        # token_id → question (for display in status_report)
        self._token_questions: dict[str, str] = {}

        # Target wallet value (USDC) — fetched periodically for proportional sizing
        self._target_wallet_value: float | None = None
        self._wallet_value_ts: float = 0.0  # monotonic time of last fetch

        # Copy-trade warmup: suppress signals for this many seconds after startup
        # to avoid acting on the initial high-lag backlog of trades.
        self._COPY_WARMUP_SECS: float = 60.0

        # Strategy-analysis log: every trade this session, fully enriched
        self._logged_trades: list[LoggedTrade] = []
        self._session_seq:   int = 0   # monotonically increments per trade

        # Stats
        self._total_seen:  int = 0
        self._started_at:  float = 0.0
        # Reactive polling: set by signal_activity() when WS detects market activity.
        # Interrupts the poll sleep so we hit the Data API without waiting the full interval.
        self._activity_event: asyncio.Event = asyncio.Event()
        # Minimum gap (seconds) between activity-triggered polls to avoid Data API spam.
        _REACTIVE_MIN_INTERVAL = 3.0
        self._REACTIVE_MIN_INTERVAL = _REACTIVE_MIN_INTERVAL
        self._last_poll_time: float = 0.0

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

            # Try to find the target's CLOB proxy wallet address.
            # If found, swapping TRACK_ADDRESS to that address reduces detection
            # lag from ~16s (Data API) to ~2-5s (CLOB API).
            await self.detect_proxy_address()

            # Fetch target wallet value for proportional copy sizing
            await self.refresh_wallet_value()

            _wallet_refresh_countdown = 0
            while True:
                # Wait for either: normal poll interval or a WS activity signal.
                # whichever comes first wakes us; the event is cleared after.
                try:
                    await asyncio.wait_for(
                        self._activity_event.wait(),
                        timeout=config.TRACKER_POLL_SECS,
                    )
                    self._activity_event.clear()
                    # Debounce: don't react faster than REACTIVE_MIN_INTERVAL
                    now = time.monotonic()
                    if now - self._last_poll_time < self._REACTIVE_MIN_INTERVAL:
                        continue
                    log.debug("Tracker: WS activity signal — polling immediately")
                except asyncio.TimeoutError:
                    pass  # normal poll interval expired

                self._last_poll_time = time.monotonic()
                try:
                    await self._poll()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.debug("TraderTracker poll error: %s", exc)

                # Refresh target wallet value every ~5 minutes
                _wallet_refresh_countdown += 1
                if _wallet_refresh_countdown >= max(1, int(300 / config.TRACKER_POLL_SECS)):
                    _wallet_refresh_countdown = 0
                    await self.refresh_wallet_value()

    def seed_known_markets(
        self,
        condition_map: dict[str, str],
        token_map: dict[str, str],
    ) -> None:
        """
        Pre-populate question and outcome caches from the bot's market cache
        so we don't need gamma API calls for markets we already know about.
        condition_map: {condition_id → question}
        token_map:     {token_id    → outcome label (e.g. "Up"/"Down")}
        """
        self._question_cache.update(condition_map)
        self._token_labels.update(token_map)
        log.debug("Tracker: seeded %d markets, %d token labels from market cache",
                  len(condition_map), len(token_map))

    def signal_activity(self, token_id: str) -> None:
        """
        Called by the ClobFeed when a last_trade_price event fires on a watched
        token — meaning SOMEONE just traded that market.  Sets the activity event
        so the poll loop wakes up immediately instead of waiting for the next timer.
        """
        self._activity_event.set()

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
                        uptime = time.monotonic() - self._started_at
                        if uptime < self._COPY_WARMUP_SECS:
                            log.info(
                                "Tracker: warmup (%ds remaining) — copy suppressed",
                                int(self._COPY_WARMUP_SECS - uptime),
                            )
                        else:
                            asyncio.create_task(self._on_trade(trade))
            except Exception as exc:
                log.debug("TraderTracker: failed to process %s: %s", tid[:12], exc)

    # ------------------------------------------------------------------
    # API calls  (CLOB primary — lowest latency; Data API fallback)
    # ------------------------------------------------------------------

    async def _fetch_all(self) -> list[dict]:
        """
        Fetch recent trades; try CLOB API first (lowest indexing latency —
        trades appear here the moment they match), Data API as fallback.

        Adaptive bypass: if CLOB returns 0 for _CLOB_SKIP_AFTER consecutive
        polls it is skipped for _CLOB_SKIP_POLLS polls before being retried.
        This handles proxy-wallet users where CLOB never returns the target's
        fills, eliminating a wasted request pair every poll cycle.
        """
        use_clob = True
        if self._clob_skip_remaining > 0:
            self._clob_skip_remaining -= 1
            use_clob = False

        if use_clob:
            try:
                result = await self._fetch_clob_api()
                if result:
                    log.debug("Tracker: CLOB API returned %d trades", len(result))
                    self._clob_empty_streak = 0  # reset — CLOB is working
                    return result
                # CLOB returned nothing
                self._clob_empty_streak += 1
                if self._clob_empty_streak >= self._CLOB_SKIP_AFTER:
                    self._clob_skip_remaining = self._CLOB_SKIP_POLLS
                    self._clob_empty_streak   = 0
                    log.info(
                        "Tracker: CLOB empty for %d polls — target likely uses proxy wallet. "
                        "Skipping CLOB for next %d polls (~%g min) to reduce wasted requests.",
                        self._CLOB_SKIP_AFTER,
                        self._CLOB_SKIP_POLLS,
                        round(self._CLOB_SKIP_POLLS * config.TRACKER_POLL_SECS / 60, 1),
                    )
                else:
                    log.debug("Tracker: CLOB API returned 0 trades — falling back to Data API")
            except Exception as exc:
                log.debug("Tracker: CLOB API error (%s) — trying Data API", exc)

        try:
            result = await self._fetch_data_api()
            if result is not None:
                log.debug("Tracker: Data API returned %d trades", len(result))
                return result
        except Exception as exc:
            log.debug("Tracker: Data API also failed: %s", exc)

        return []

    async def _fetch_data_api(self) -> list[dict] | None:
        """
        GET data-api.polymarket.com/activity?user=ADDRESS&limit=100

        Returns a list of trade dicts, or None if the endpoint is
        unavailable (so we fall back to the CLOB API).
        """
        assert self._session is not None
        url    = f"{config.DATA_API}/activity"
        params = {"user": self._address, "limit": 20}  # only need recent; smaller = faster

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
        GET clob.polymarket.com/trades?maker_address=ADDR  (fills as maker)
        GET clob.polymarket.com/trades?taker_address=ADDR  (fills as taker)

        Both requests fire concurrently to halve round-trip time.
        Results are merged and deduplicated by trade ID.
        """
        assert self._session is not None
        url = f"{config.CLOB_HOST}/trades"

        checksum_addr = Web3.to_checksum_address(self._address)

        async def _fetch_role(role_param: str) -> list[dict]:
            try:
                params = {role_param: checksum_addr, "limit": 100}
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        return []
                    body = await resp.json(content_type=None)
                    trades = body.get("data", body) if isinstance(body, dict) else body
                    return trades or []
            except Exception as exc:
                log.debug("Tracker: CLOB %s query failed: %s", role_param, exc)
                return []

        maker_trades, taker_trades = await asyncio.gather(
            _fetch_role("maker_address"),
            _fetch_role("taker_address"),
        )

        merged: list[dict] = []
        seen:   set[str]   = set()
        for t in maker_trades + taker_trades:
            tid = self._trade_id(t)
            if tid and tid not in seen:
                seen.add(tid)
                merged.append(t)
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
            or raw.get("asset")          # CLOB API fill format
            or raw.get("outcome_id")
            or raw.get("outcomeId")
            or ""
        )

        # Log unknown field layout once so we can diagnose missing token_ids
        if not token_id:
            log.debug("Tracker: token_id not found in raw trade — keys: %s", list(raw.keys()))

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
            raw.get("matchTime")    # actual fill/match time — most accurate
            or raw.get("block_time")
            or raw.get("timestamp")
            or raw.get("createdAt") # order creation time — least accurate, last resort
        )
        try:
            timestamp = float(ts_raw) if ts_raw else time.time()
            # If timestamp looks like milliseconds, convert to seconds
            if timestamp > 1e12:
                timestamp /= 1000.0
        except (ValueError, TypeError):
            timestamp = time.time()

        # ── Market enrichment ─────────────────────────────────────────────
        question = await self._get_question(condition_id, token_id)
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

    async def _get_question(self, condition_id: str, token_id: str = "") -> str:
        if not condition_id and not token_id:
            return "(unknown market)"
        cache_key = condition_id or token_id
        if cache_key in self._question_cache:
            return self._question_cache[cache_key]

        assert self._session is not None
        url = f"{config.GAMMA_API}/markets"

        # Try up to three query strategies in order
        attempts = []
        if condition_id:
            # 1. conditionId with 0x prefix as-is
            attempts.append({"conditionId": condition_id})
            # 2. conditionId without 0x prefix (some API versions prefer this)
            if condition_id.startswith("0x"):
                attempts.append({"conditionId": condition_id[2:]})
        if token_id:
            # 3. clob_token_ids — most reliable when we have the token ID
            attempts.append({"clob_token_ids": token_id})

        for params in attempts:
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        log.debug("Tracker: gamma %s → HTTP %d", params, resp.status)
                        continue
                    body    = await resp.json(content_type=None)
                    markets = body if isinstance(body, list) else body.get("markets", [])
                    if not markets:
                        log.info("Tracker: gamma %s → empty response", params)
                        continue
                    m = markets[0]
                    q = m.get("question") or m.get("title") or ""
                    if not q:
                        continue
                    self._question_cache[cache_key] = q
                    # Cache token → outcome labels while we're here
                    for tok in m.get("tokens") or m.get("outcomes") or []:
                        tid = tok.get("token_id") or tok.get("tokenId") or ""
                        out = tok.get("outcome") or ""
                        if tid:
                            self._token_labels[tid] = out
                    log.info("Tracker: resolved market → %r", q)
                    return q
            except Exception as exc:
                log.info("Tracker: gamma lookup failed (%s): %s", params, exc)

        fallback = (condition_id or token_id)[:16] + "…"
        self._question_cache[cache_key] = fallback
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
        if condition_id or token_id:
            await self._get_question(condition_id, token_id)

        return self._token_labels.get(token_id, "?")

    # ------------------------------------------------------------------
    # Output + position tracking
    # ------------------------------------------------------------------

    def _log_trade(self, trade: TrackedTrade) -> None:
        # ── Strategy-analysis record ──────────────────────────────────────────
        key        = trade.token_id or trade.condition_id
        pos_before  = self._net_shares.get(key, 0.0)
        delta       = trade.size if trade.side == "BUY" else -trade.size
        pos_after   = pos_before + delta
        spot        = self._price_feed(trade.symbol, trade.timestamp) if self._price_feed and trade.symbol else None

        # ── Cost basis + realized P&L ─────────────────────────────────────────
        avg_cost     = self._avg_cost.get(key, 0.0)
        realized_pnl: float | None = None

        if trade.side == "BUY":
            # Update weighted average entry price for this long position.
            # If we were short (or flat), reset cost basis to this entry price.
            if pos_before <= 0:
                new_avg = trade.price        # fresh long: avg cost = entry price
            else:
                total   = pos_before + trade.size
                new_avg = (avg_cost * pos_before + trade.price * trade.size) / total
            self._avg_cost[key] = new_avg
        else:  # SELL
            if pos_before > 0 and avg_cost > 0:
                shares_closed = min(trade.size, pos_before)
                realized_pnl  = (trade.price - avg_cost) * shares_closed
            # Average cost of remaining shares doesn't change on partial sell;
            # clear cost basis when the position fully closes.
            if pos_after <= 0:
                self._avg_cost.pop(key, None)

        self._session_seq += 1
        self._logged_trades.append(LoggedTrade(
            trade        = trade,
            seq          = self._session_seq,
            pos_before   = pos_before,
            pos_after    = pos_after,
            trade_type   = LoggedTrade._infer_type(trade.side, pos_before, pos_after),
            spot_price   = spot,
            avg_cost     = avg_cost,
            realized_pnl = realized_pnl,
        ))
        # ─────────────────────────────────────────────────────────────────────
        side_col = _GREEN if trade.side == "BUY" else _RED
        side_tag = f"{side_col}{'[BUY] ' if trade.side == 'BUY' else '[SELL]'}{_RESET}"

        out_col = (
            _GREEN if trade.outcome in ("Up", "Yes")
            else _RED if trade.outcome in ("Down", "No")
            else _CYAN
        )

        ts_str = datetime.datetime.fromtimestamp(trade.timestamp).strftime("%H:%M:%S")

        # Detection lag: how old was the trade when we first saw it
        lag_secs = time.time() - trade.timestamp
        lag_col  = _RED if lag_secs > 5 else _YELLOW if lag_secs > 2 else _GREEN
        lag_tag  = f"  {lag_col}+{lag_secs:.1f}s lag{_RESET}"

        log.info(
            "%s👁 COPY-WATCH%s  %s  %s%s %s%s  "
            "@ %s%.2f¢%s  %.1f shares / %s$%.2f%s%s  |  %s",
            _MAGENTA, _RESET,
            side_tag,
            ts_str, _DIM, "", _RESET,
            _BOLD, trade.price * 100, _RESET,
            trade.size,
            _BOLD, trade.amount, _RESET,
            lag_tag,
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
    # Target wallet value (for proportional copy-trade sizing)
    # ------------------------------------------------------------------

    async def detect_proxy_address(self) -> str | None:
        """
        Attempt to find the target's CLOB proxy wallet address from:
          1. Known field names in raw Data API trade records
          2. The /profile endpoint on the Data API

        Logs the result clearly. If found, the user can set TRACK_ADDRESS
        to the proxy address for direct CLOB polling (~2-5s lag vs ~16s).
        """
        if self._session is None:
            return None

        proxy: str | None = None

        # ── Check raw trade fields ─────────────────────────────────────
        try:
            url    = f"{config.DATA_API}/activity"
            params = {"user": self._address, "limit": 5}
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    body = await resp.json(content_type=None)
                    trades = body if isinstance(body, list) else body.get("data", [])
                    for trade in (trades or []):
                        log.debug("Tracker: raw trade keys: %s", list(trade.keys()))
                        for field in (
                            "proxyWallet", "proxy_wallet", "makerAddress", "maker_address",
                            "takerAddress", "taker_address", "signer", "trader",
                            "owner", "funder", "safeAddress", "safe_address",
                        ):
                            val = trade.get(field, "")
                            if (
                                isinstance(val, str)
                                and val.startswith("0x")
                                and len(val) >= 40
                                and val.lower() != self._address
                            ):
                                proxy = val.lower()
                                log.debug("Tracker: found proxy candidate in field '%s': %s", field, proxy)
                                break
                        if proxy:
                            break
        except Exception as exc:
            log.debug("Tracker: proxy detection trade scan failed: %s", exc)

        # ── Try /profile endpoint ──────────────────────────────────────
        if not proxy:
            try:
                for endpoint in ("profile", "user"):
                    url = f"{config.DATA_API}/{endpoint}"
                    async with self._session.get(url, params={"address": self._address}) as resp:
                        if resp.status == 200:
                            body = await resp.json(content_type=None)
                            if isinstance(body, dict):
                                log.debug("Tracker: profile keys: %s", list(body.keys()))
                                for field in (
                                    "proxyWallet", "proxy_wallet", "clobAddress",
                                    "clob_address", "safeAddress", "funder",
                                ):
                                    val = body.get(field, "")
                                    if (
                                        isinstance(val, str)
                                        and val.startswith("0x")
                                        and len(val) >= 40
                                        and val.lower() != self._address
                                    ):
                                        proxy = val.lower()
                                        break
                            if proxy:
                                break
            except Exception as exc:
                log.debug("Tracker: proxy detection profile lookup failed: %s", exc)

        if proxy:
            log.info(
                "%s👁 TRACKER%s  proxy wallet found: %s%s%s\n"
                "  → Set TRACK_ADDRESS=%s in .env to use CLOB directly "
                "and cut detection lag from ~16s to ~2-5s.",
                _MAGENTA, _RESET, _CYAN, proxy, _RESET, proxy,
            )
        else:
            log.info(
                "%s👁 TRACKER%s  proxy wallet not found in API data. "
                "To find it manually: polygonscan.com/address/%s → Internal Txns → "
                "look for 'From' address on Polymarket CLOB transactions.",
                _MAGENTA, _RESET, self._address,
            )

        return proxy

    @property
    def target_wallet_value(self) -> float | None:
        """Last known USDC value of the target's Polymarket portfolio."""
        return self._target_wallet_value

    async def refresh_wallet_value(self) -> float | None:
        """
        Query the Polymarket Data API for the target address's portfolio value.

        Tries several known endpoints in order; returns the value in USDC or
        None if unavailable.  Result is cached internally and exposed via the
        target_wallet_value property.
        """
        if self._session is None:
            return self._target_wallet_value

        val: float | None = None

        # Attempt 1: /value endpoint (returns {"value": N} or {"portfolio": N})
        try:
            url = f"{config.DATA_API}/value"
            async with self._session.get(url, params={"user": self._address}) as resp:
                if resp.status == 200:
                    body = await resp.json(content_type=None)
                    for key in ("value", "portfolio", "usdcBalance", "portfolioValue", "balance"):
                        if isinstance(body, dict) and key in body:
                            val = float(body[key])
                            break
        except Exception as exc:
            log.debug("Tracker: wallet value /value fetch failed: %s", exc)

        # Attempt 2: sum open positions currentValue if /value didn't work
        if val is None:
            try:
                url = f"{config.DATA_API}/positions"
                params = {"user": self._address, "sizeThreshold": "0", "limit": "500"}
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        body = await resp.json(content_type=None)
                        positions = body if isinstance(body, list) else body.get("data", [])
                        total = 0.0
                        for pos in positions:
                            for key in ("currentValue", "value", "usdcValue"):
                                if key in pos:
                                    total += float(pos[key])
                                    break
                        if total > 0:
                            val = total
            except Exception as exc:
                log.debug("Tracker: wallet value /positions sum failed: %s", exc)

        if val is not None and val > 0:
            prev = self._target_wallet_value
            self._target_wallet_value = val
            self._wallet_value_ts = time.monotonic()
            if prev is None or abs(val - prev) > 1.0:
                log.info(
                    "👁 TRACKER  target wallet value: $%.2f USDC%s",
                    val,
                    f"  (was ${prev:.2f})" if prev is not None else "",
                )
        else:
            log.debug("Tracker: could not determine target wallet value — will use fallback sizing")

        return self._target_wallet_value

    # ------------------------------------------------------------------
    # Status report (for 'w' console command)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helpers for status_report
    # ------------------------------------------------------------------

    @staticmethod
    def _shorten_market(q: str, maxlen: int = 32) -> str:
        q = (q.replace("Up or Down - ", "").replace("Up or Down — ", "")
               .replace("Bitcoin", "BTC").replace("Ethereum", "ETH")
               .replace("Solana", "SOL").replace("Ripple", "XRP"))
        return q[:maxlen]

    @staticmethod
    def _median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def status_report(self) -> str:
        uptime = time.monotonic() - self._started_at
        h, m   = divmod(int(uptime) // 60, 60)
        s_rem  = int(uptime) % 60

        W = 88  # table width
        SEP  = "═" * W
        sep  = "─" * W

        lines = ["", SEP]
        lines.append(
            f"👁  TRADE LOG — {self._address}"
        )
        lines.append(
            f"   Session {h:02d}h {m:02d}m {s_rem:02d}s  |  "
            f"{len(self._logged_trades)} trades recorded  |  "
            f"{len(set(lt.trade.token_id for lt in self._logged_trades))} unique tokens  |  "
            f"{len(self._net_shares)} open positions"
        )
        lines.append(SEP)

        # ── Full trade log ────────────────────────────────────────────────────
        if not self._logged_trades:
            lines.append("  (no trades yet)")
        else:
            hdr = (
                f"  {'#':>4}  {'TIME':8}  {'TYPE':5}  {'SIDE':4}  "
                f"{'DIR':4}  {'BET':>6}  {'SHARES':>7}  {'USD':>7}  "
                f"{'BEFORE':>7}  {'AFTER':>7}  {'P&L':>8}  {'SPOT':>10}  MARKET"
            )
            lines += ["", hdr, "  " + sep]

            for lt in self._logged_trades:
                t      = lt.trade
                ts_str = datetime.datetime.fromtimestamp(t.timestamp).strftime("%H:%M:%S")
                side_s = "BUY " if t.side == "BUY" else "SELL"
                dir_s  = (t.direction or "?").ljust(4)
                mkt    = self._shorten_market(t.question, 28)
                spot_s = f"${lt.spot_price:,.2f}" if lt.spot_price else "     n/a"
                if lt.realized_pnl is not None:
                    pnl_s = f"{lt.realized_pnl:>+7.2f}"
                else:
                    pnl_s = "       "
                lines.append(
                    f"  {lt.seq:>4}  {ts_str}  {lt.trade_type:<5}  {side_s}  "
                    f"{dir_s}  {lt.price_cents:>5.1f}¢  {t.size:>7.1f}  ${t.amount:>6.2f}  "
                    f"{lt.pos_before:>+7.1f}  {lt.pos_after:>+7.1f}  {pnl_s}  {spot_s:>10}  {mkt}"
                )

        # ── Summary ───────────────────────────────────────────────────────────
        if self._logged_trades:
            buys  = [lt for lt in self._logged_trades if lt.trade.side == "BUY"]
            sells = [lt for lt in self._logged_trades if lt.trade.side == "SELL"]

            type_counts: dict[str, int] = collections.Counter(
                lt.trade_type for lt in self._logged_trades
            )
            type_str = "  ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))

            buy_p  = [lt.price_cents for lt in buys]
            sell_p = [lt.price_cents for lt in sells]
            all_u  = [lt.trade.amount for lt in self._logged_trades]

            realized = [lt.realized_pnl for lt in self._logged_trades if lt.realized_pnl is not None]
            total_pnl = sum(realized) if realized else None

            lines += ["", SEP, "  SUMMARY", "  " + sep]
            lines.append(
                f"  Buys: {len(buys)}  Sells: {len(sells)}  Total: {len(self._logged_trades)}"
            )
            lines.append(f"  Trade types: {type_str}")
            if total_pnl is not None:
                pnl_sign = "+" if total_pnl >= 0 else ""
                lines.append(
                    f"  Realized P&L: {pnl_sign}${total_pnl:.2f} USDC "
                    f"({len(realized)} closed trades)"
                )
            if buy_p:
                lines.append(
                    f"  BUY  prices: {min(buy_p):.1f}–{max(buy_p):.1f}¢  "
                    f"avg {sum(buy_p)/len(buy_p):.1f}¢  median {self._median(buy_p):.1f}¢"
                )
            if sell_p:
                lines.append(
                    f"  SELL prices: {min(sell_p):.1f}–{max(sell_p):.1f}¢  "
                    f"avg {sum(sell_p)/len(sell_p):.1f}¢  median {self._median(sell_p):.1f}¢"
                )
            if all_u:
                lines.append(
                    f"  Trade $ size: min ${min(all_u):.2f}  max ${max(all_u):.2f}  "
                    f"avg ${sum(all_u)/len(all_u):.2f}"
                )

            # ── Per-token breakdown ───────────────────────────────────────────
            by_token: dict[str, list[LoggedTrade]] = collections.defaultdict(list)
            for lt in self._logged_trades:
                by_token[lt.trade.token_id].append(lt)

            lines += ["", SEP, "  TOKEN BREAKDOWN  (sorted by trade count)", "  " + sep]
            hdr2 = (
                f"  {'TOKEN':18}  {'MARKET':32}  "
                f"{'B':>3} {'S':>3}  {'PRICE RANGE':14}  {'AVG':>5}  DIRS"
            )
            lines.append(hdr2)
            lines.append("  " + sep)
            for tok, trades in sorted(by_token.items(), key=lambda kv: -len(kv[1])):
                t0     = trades[0].trade
                mkt    = self._shorten_market(t0.question, 32)
                n_b    = sum(1 for lt in trades if lt.trade.side == "BUY")
                n_s    = sum(1 for lt in trades if lt.trade.side == "SELL")
                prices = [lt.price_cents for lt in trades]
                dir_ct: dict[str, int] = collections.Counter(
                    (lt.trade.direction or "?") for lt in trades
                )
                dir_str = " ".join(f"{d}:{n}" for d, n in sorted(dir_ct.items()))
                lines.append(
                    f"  {tok[:16]+'…':18}  {mkt:<32}  "
                    f"{n_b:>3} {n_s:>3}  "
                    f"{min(prices):>5.1f}–{max(prices):>5.1f}¢  "
                    f"avg {sum(prices)/len(prices):>4.1f}¢  {dir_str}"
                )

            # ── Hold duration analysis ────────────────────────────────────────
            hold_times: list[float] = []
            for tok, trades in by_token.items():
                sorted_t = sorted(trades, key=lambda lt: lt.trade.timestamp)
                pending: list[float] = []
                for lt in sorted_t:
                    if lt.trade.side == "BUY":
                        pending.append(lt.trade.timestamp)
                    elif lt.trade.side == "SELL" and pending:
                        dt = lt.trade.timestamp - pending.pop(0)
                        if 0 < dt < 7200:
                            hold_times.append(dt)

            if hold_times:
                buckets = {"<5s": 0, "5-15s": 0, "15-30s": 0, "30-60s": 0, "60-120s": 0, ">120s": 0}
                for dt in hold_times:
                    if   dt <  5:  buckets["<5s"]    += 1
                    elif dt < 15:  buckets["5-15s"]  += 1
                    elif dt < 30:  buckets["15-30s"] += 1
                    elif dt < 60:  buckets["30-60s"] += 1
                    elif dt < 120: buckets["60-120s"] += 1
                    else:          buckets[">120s"]  += 1
                bkt_str = "  ".join(f"{k}:{v}" for k, v in buckets.items() if v)
                lines += ["", SEP, "  HOLD DURATIONS  (inferred BUY→SELL pairs)", "  " + sep]
                lines.append(
                    f"  Pairs found: {len(hold_times)}  "
                    f"min {min(hold_times):.0f}s  max {max(hold_times):.0f}s  "
                    f"avg {sum(hold_times)/len(hold_times):.0f}s  "
                    f"median {self._median(hold_times):.0f}s"
                )
                lines.append(f"  Distribution: {bkt_str}")

            # ── Scale-in pattern analysis ─────────────────────────────────────
            # For each OPEN trade, count how many ADD trades follow before CLOSE
            scale_counts: list[int] = []
            for tok, trades in by_token.items():
                sorted_t = sorted(trades, key=lambda lt: lt.trade.timestamp)
                in_position = False
                adds = 0
                for lt in sorted_t:
                    if lt.trade_type == "OPEN":
                        in_position = True
                        adds = 0
                    elif lt.trade_type == "ADD" and in_position:
                        adds += 1
                    elif lt.trade_type in ("CLOSE", "FLIP") and in_position:
                        scale_counts.append(adds)
                        in_position = False
                        adds = 0
            if scale_counts:
                avg_adds = sum(scale_counts) / len(scale_counts)
                max_adds = max(scale_counts)
                lines += ["", SEP, "  SCALE-IN PATTERN", "  " + sep]
                lines.append(
                    f"  Positions tracked: {len(scale_counts)}  "
                    f"avg ADD trades before close: {avg_adds:.1f}  "
                    f"max: {max_adds}"
                )
                dist = collections.Counter(scale_counts)
                dist_str = "  ".join(f"{k}adds:{v}x" for k, v in sorted(dist.items()))
                lines.append(f"  Distribution: {dist_str}")

        # ── Open positions ────────────────────────────────────────────────────
        if self._net_shares:
            lines += ["", SEP, "  OPEN POSITIONS NOW", "  " + sep]
            for tok, shares in sorted(self._net_shares.items(), key=lambda kv: -abs(kv[1])):
                label   = self._token_labels.get(tok, "?")
                q       = self._shorten_market(self._token_questions.get(tok, tok[:20]), 40)
                dir_sym = "▲" if shares > 0 else "▼"
                lines.append(
                    f"  {dir_sym} {label:<4}  {tok[:16]}…  {shares:>+8.1f} shares  {q}"
                )

        lines += [SEP, ""]
        return "\n".join(lines)
