"""
Polymarket 15-min crypto momentum trading bot.

Architecture
------------

  Binance WebSocket (BTC/ETH trade stream)
        │  tick (symbol, price, ts)
        ▼
  MomentumStrategy  ──── signal (symbol, direction, pct_move)
        │
        ▼
  OrderManager  ──── place limit order on Polymarket CLOB
        │
        ├─ PolymarketClient  (async wrapper → py-clob-client)
        └─ MarketCache       (Gamma API, refreshed every 60 s)

How to run
----------
1. Copy .env.example → .env and fill in your private key + API creds.
2. pip install -r requirements.txt
3. python bot.py [--paper]   (--paper forces paper-trade mode)

Flags
-----
--paper         Override PAPER_TRADE=true regardless of .env
--loglevel      DEBUG / INFO / WARNING (default INFO)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

import config
from feeds.binance import BinanceFeed
from polymarket.client import PolymarketClient
from polymarket.markets import MarketCache
from polymarket.orders import OrderManager
from strategy.momentum import MomentumStrategy


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    fmt = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
    # Reduce noise from websockets / aiohttp internals
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


log = logging.getLogger("bot")


# ---------------------------------------------------------------------------
# Main bot
# ---------------------------------------------------------------------------

class Bot:
    def __init__(self) -> None:
        self._pm_client = PolymarketClient()
        self._market_cache = MarketCache()
        self._order_manager: OrderManager | None = None
        self._strategy: MomentumStrategy | None = None
        self._feed: BinanceFeed | None = None
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        log.info("=== Polymarket Momentum Bot starting ===")
        log.info("Paper trade: %s", config.PAPER_TRADE)
        log.info(
            "Trigger: %.2f%% over %.0fs | Order size: $%.0f–$%.0f | "
            "Max exposure: $%.0f",
            config.STRATEGY.trigger_pct,
            config.STRATEGY.lookback_secs,
            config.RISK.min_order_usdc,
            config.RISK.max_order_usdc,
            config.RISK.max_total_exposure_usdc,
        )

        # 1. Connect to Polymarket
        await self._pm_client.connect()

        # 2. Load market list
        await self._market_cache.start()

        # 3. Wire up components
        self._order_manager = OrderManager(self._pm_client, self._market_cache)
        self._strategy = MomentumStrategy(on_signal=self._order_manager.execute_signal)
        self._feed = BinanceFeed(callback=self._strategy.on_tick)

        # 4. Background task: keep market list fresh
        refresh_task = asyncio.create_task(
            self._market_cache.run_refresh_loop(), name="market-refresh"
        )
        self._tasks.append(refresh_task)

        # 5. Run price feed (blocks until cancelled)
        feed_task = asyncio.create_task(self._feed.run(), name="binance-feed")
        self._tasks.append(feed_task)

        log.info("Bot running. Press Ctrl+C to stop.")
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        log.info("Shutting down …")
        if self._feed:
            self._feed.stop()
        for task in self._tasks:
            task.cancel()
        await self._market_cache.stop()
        log.info("Bot stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    if args.paper:
        config.PAPER_TRADE = True

    _setup_logging(args.loglevel)

    bot = Bot()

    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
        log.info("Received %s — initiating shutdown.", sig.name)
        loop.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    try:
        await bot.start()
    except asyncio.CancelledError:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket 15-min momentum bot")
    parser.add_argument(
        "--paper",
        action="store_true",
        default=False,
        help="Run in paper-trade mode (no real orders placed)",
    )
    parser.add_argument(
        "--loglevel",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parsed = parser.parse_args()

    try:
        asyncio.run(main(parsed))
    except KeyboardInterrupt:
        sys.exit(0)
