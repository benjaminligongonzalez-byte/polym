"""
Thin async-friendly wrapper around py-clob-client.

py-clob-client is synchronous, so we run blocking calls in an executor
to avoid blocking the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    OrderArgs,
    OrderType,
    MarketOrderArgs,
)
from py_clob_client.constants import POLYGON

import config

log = logging.getLogger(__name__)


def _build_sync_client() -> ClobClient:
    """
    Construct and authenticate a ClobClient using credentials from config.
    Call once at startup; re-use for the session lifetime.
    """
    if not config.PK:
        raise RuntimeError(
            "PK not set — copy .env.example to .env and fill in your private key."
        )

    creds = None
    if config.CLOB_API_KEY:
        creds = ApiCreds(
            api_key=config.CLOB_API_KEY,
            api_secret=config.CLOB_SECRET,
            api_passphrase=config.CLOB_PASS_PHRASE,
        )

    client = ClobClient(
        host=config.CLOB_HOST,
        chain_id=POLYGON,
        key=config.PK,
        creds=creds,
        funder=config.FUNDER_ADDRESS or None,
    )

    # If no API creds supplied, derive them from the private key on first run
    if creds is None:
        log.info("No API creds found — deriving from private key …")
        derived = client.create_or_derive_api_creds()
        log.info(
            "API creds derived. Add these to your .env:\n"
            "  CLOB_API_KEY=%s\n  CLOB_SECRET=%s\n  CLOB_PASS_PHRASE=%s",
            derived.api_key,
            derived.api_secret,
            derived.api_passphrase,
        )
        client.set_api_creds(derived)

    return client


class PolymarketClient:
    """
    Async wrapper around the sync ClobClient.

    All public methods are async and safe to call from an asyncio task.
    Blocking calls are dispatched to the default thread pool executor.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: ClobClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._client = await self._loop.run_in_executor(None, _build_sync_client)
        log.info("PolymarketClient connected (paper_trade=%s).", config.PAPER_TRADE)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_markets(self, next_cursor: str = "") -> dict[str, Any]:
        """Return a page of markets from the CLOB."""
        return await self._run(self._client.get_markets, next_cursor)

    async def get_market(self, condition_id: str) -> dict[str, Any]:
        return await self._run(self._client.get_market, condition_id)

    async def get_order_book(self, token_id: str) -> dict[str, Any]:
        return await self._run(self._client.get_order_book, token_id)

    async def get_midpoint(self, token_id: str) -> float:
        """Return current mid price for a token (0–1 scale)."""
        resp = await self._run(self._client.get_midpoint, token_id)
        # Response: {"mid": "0.52"}
        return float(resp.get("mid", 0.5))

    async def get_spread(self, token_id: str) -> dict[str, float]:
        """Return {"ask": ..., "bid": ..., "spread": ...}."""
        resp = await self._run(self._client.get_spread, token_id)
        return {k: float(v) for k, v in resp.items()}

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def create_market_order(
        self,
        token_id: str,
        side: str,          # "BUY" or "SELL"
        amount_usdc: float,
    ) -> dict[str, Any] | None:
        """
        Place a market (FOK) order for *amount_usdc* USDC worth of *token_id*.

        Returns the order response dict, or None if paper trading.
        """
        if config.PAPER_TRADE:
            log.info(
                "[PAPER] Would %s %s for $%.2f USDC",
                side, token_id[:8], amount_usdc,
            )
            return {"paper": True, "token_id": token_id, "side": side, "amount": amount_usdc}

        args = MarketOrderArgs(
            token_id=token_id,
            amount=amount_usdc,
        )
        try:
            resp = await self._run(self._client.create_market_order, args)
            log.info("Market order placed: %s", resp)
            return resp
        except Exception as exc:
            log.error("Order failed for token %s: %s", token_id[:8], exc)
            return None

    async def create_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,       # 0–1 scale (e.g. 0.55 = 55 cents)
        size: float,        # number of shares (= USDC / price for buys)
    ) -> dict[str, Any] | None:
        """
        Place a GTC limit order.

        Returns the order response dict, or None on failure / paper trade.
        """
        if config.PAPER_TRADE:
            log.info(
                "[PAPER] Would limit-%s %s @ %.4f (size=%.2f)",
                side, token_id[:8], price, size,
            )
            return {"paper": True, "token_id": token_id, "side": side, "price": price, "size": size}

        args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
        )
        try:
            resp = await self._run(
                self._client.create_and_post_order, args
            )
            log.info("Limit order placed: %s", resp)
            return resp
        except Exception as exc:
            log.error("Limit order failed for token %s: %s", token_id[:8], exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _run(self, fn, *args, **kwargs) -> Any:
        assert self._loop is not None, "call connect() first"
        return await self._loop.run_in_executor(None, partial(fn, *args, **kwargs))
