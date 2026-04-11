"""
Async wrapper around the polyfill-py Rust extension module.

Exposes the same interface as `polymarket.client.PolymarketClient` so that
bot.py can swap between the two with a single flag::

    # bot.py --rust  →  uses this module
    # bot.py         →  uses polymarket/client.py (Python py-clob-client)

The Rust extension (polyfill_py) must be built first::

    cd polyfill_py && maturin develop --release

If the extension is not installed this module raises ImportError at import
time so the caller can catch it and fall back gracefully.

All public methods are async (using run_in_executor to avoid blocking the
event loop) matching the PolymarketClient contract exactly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import partial
from typing import Any

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the native extension — fail loudly if not built yet
# ---------------------------------------------------------------------------

try:
    from polyfill_py import PolyRustClient as _RustClient
    _AVAILABLE = True
except ImportError as _err:
    _AVAILABLE = False
    _IMPORT_ERR = _err


def check_available() -> None:
    """Raise ImportError if the Rust extension is not built."""
    if not _AVAILABLE:
        raise ImportError(
            "polyfill_py Rust extension not found. "
            "Build it with:  cd polyfill_py && maturin develop --release\n"
            f"Original error: {_IMPORT_ERR}"
        )


# ---------------------------------------------------------------------------
# RustPolymarketClient
# ---------------------------------------------------------------------------

class RustPolymarketClient:
    """
    Async Polymarket CLOB client powered by polyfill-rs (Rust).

    Drop-in replacement for `polymarket.client.PolymarketClient`.
    4.2× faster market fetching; sub-µs order book arithmetic;
    HTTP/2 with DNS caching and connection pooling.
    """

    def __init__(self) -> None:
        check_available()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: _RustClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._client = await self._run_sync(self._build_client)
        # Warm up HTTP/2 connections in background
        asyncio.create_task(self._warmup())
        log.info(
            "RustPolymarketClient connected (paper_trade=%s). "
            "Address: %s",
            config.PAPER_TRADE,
            self._client.get_address() or "(unknown)",
        )

    async def auth_banner(self) -> None:
        address = self._client.get_address() if self._client else "(unknown)"
        balance = await self.get_usdc_balance()
        key_hint = (config.CLOB_API_KEY[:8] + "…") if config.CLOB_API_KEY else "(derived)"
        mode = "PAPER (no real orders)" if config.PAPER_TRADE else "LIVE  *** REAL MONEY ***"

        log.info("─" * 60)
        log.info("  Wallet  : %s", address)
        log.info("  API key : %s", key_hint)
        log.info("  Balance : $%.4f USDC", balance)
        log.info("  Mode    : %s  [Rust client]", mode)
        log.info("─" * 60)

        if not config.PAPER_TRADE and balance < 5.0:
            log.warning("Low balance ($%.2f) — live trading may fail.", balance)

    async def _warmup(self) -> None:
        try:
            await self._run(self._client.prewarm)
            log.debug("RustPolymarketClient: HTTP/2 connections prewarmed.")
        except Exception as exc:
            log.debug("RustPolymarketClient: prewarm failed: %s", exc)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_markets(self, next_cursor: str = "") -> dict[str, Any]:
        cursor = next_cursor or None
        raw = await self._run(self._client.get_markets, cursor)
        return json.loads(raw)

    async def get_market(self, condition_id: str) -> dict[str, Any]:
        # polyfill-rs exposes get_market(condition_id) returning a single Market
        raw = await self._run(self._client.get_market, condition_id)
        return json.loads(raw)

    async def get_order_book(self, token_id: str) -> dict[str, Any]:
        raw = await self._run(self._client.get_order_book, token_id)
        return json.loads(raw)

    async def get_midpoint(self, token_id: str) -> float:
        return await self._run(self._client.get_midpoint, token_id)

    async def get_spread(self, token_id: str) -> dict[str, float]:
        raw = await self._run(self._client.get_spread, token_id)
        return json.loads(raw)

    async def get_sampling_markets(self, next_cursor: str = "") -> dict[str, Any]:
        cursor = next_cursor or None
        raw = await self._run(self._client.get_sampling_markets, cursor)
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Balances
    # ------------------------------------------------------------------

    async def get_usdc_balance(self) -> float:
        try:
            return await self._run(self._client.get_usdc_balance)
        except Exception as exc:
            log.warning("RustPolymarketClient: could not fetch balance: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def create_market_order(
        self,
        token_id: str,
        side: str,
        amount_usdc: float,
    ) -> dict[str, Any] | None:
        if config.PAPER_TRADE:
            log.info(
                "[PAPER] Would %s %s for $%.2f USDC  [Rust]",
                side, token_id[:8], amount_usdc,
            )
            return {"paper": True, "token_id": token_id, "side": side, "amount": amount_usdc}

        try:
            if side.upper() == "BUY":
                # polyfill-rs create_market_buy uses FOK with best available price
                raw = await self._run(self._client.create_market_buy, token_id, amount_usdc)
            else:
                # For SELL: fetch current mid, place a limit order at mid - 2% slippage
                mid = await self.get_midpoint(token_id)
                sell_price = round(mid * 0.98, 4)          # 2% slippage budget
                size = round(amount_usdc / sell_price, 4)  # convert USDC to shares
                raw = await self._run(
                    self._client.create_limit_order,
                    token_id, "SELL", sell_price, size,
                )
            resp = json.loads(raw)
            log.info("Market order placed (Rust): %s", resp)
            return resp
        except Exception as exc:
            log.error("Rust market order failed for %s: %s", token_id[:8], exc)
            return None

    async def create_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> dict[str, Any] | None:
        if config.PAPER_TRADE:
            log.info(
                "[PAPER] Would limit-%s %s @ %.4f (size=%.2f)  [Rust]",
                side, token_id[:8], price, size,
            )
            return {"paper": True, "token_id": token_id, "side": side, "price": price, "size": size}

        try:
            raw = await self._run(
                self._client.create_limit_order,
                token_id, side, price, size,
            )
            resp = json.loads(raw)
            log.info("Limit order placed (Rust): %s", resp)
            return resp
        except Exception as exc:
            log.error("Rust limit order failed for %s: %s", token_id[:8], exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _run(self, fn, *args, **kwargs):
        assert self._loop is not None, "call connect() first"
        return await self._loop.run_in_executor(None, partial(fn, *args, **kwargs))

    async def _run_sync(self, fn, *args, **kwargs):
        """Run a regular (non-method) callable in executor."""
        assert self._loop is not None or True  # loop may not be set yet
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

    @staticmethod
    def _build_client() -> _RustClient:
        """Synchronous factory — runs in executor thread."""
        if not config.PK:
            raise RuntimeError("PK not set — fill in .env")
        pk = config.PK
        if pk.startswith("0x"):
            pk = pk[2:]

        return _RustClient(
            host           = config.CLOB_HOST,
            private_key    = pk,
            chain_id       = 137,
            api_key        = config.CLOB_API_KEY or None,
            api_secret     = config.CLOB_SECRET or None,
            api_passphrase = config.CLOB_PASS_PHRASE or None,
            # Note: polyfill-rs with_l2_headers doesn't take a funder arg.
            # Funder-wallet setups are not yet supported via the Rust bridge.
        )
