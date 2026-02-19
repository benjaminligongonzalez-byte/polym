"""
Polymarket 15-min crypto trading bot — dual strategy (momentum + arbitrage).

Architecture
------------

  Binance WS ──┐
               ├─► PriceAggregator ──► MomentumStrategy ──► OrderManager
  Coinbase WS ─┘         │                                        │
                          └──────────► ArbStrategy ───────────────┘
                                            │
                                     PolymarketClient (CLOB)
                                     MarketCache      (Gamma API)

Two complementary strategies run concurrently:

  1. MomentumStrategy (speed layer)
     Triggers on fast price moves (default 0.10% over 5 s).
     Fires immediately on the next tick, before Polymarket makers can react.
     Uses flat order sizing.

  2. ArbStrategy (edge layer)
     Scans all active markets every 2 s.
     Computes fair probability via a log-normal model:
         P(S_T > K) = N(d₂),  d₂ = [ln(S/K) - ½σ²T] / (σ√T)
     Trades when Polymarket price is >5 cents (default) from fair value.
     Uses fractional Kelly sizing (default ¼ Kelly).

Both strategies share a single OrderManager that enforces:
  - Per-market cooldowns
  - Per-symbol position caps
  - Total USDC exposure cap

How to run
----------
1. cp .env.example .env  →  fill in PK (private key) and CLOB API creds
2. pip install -r requirements.txt
3. python bot.py --paper          # dry run — no real orders placed
4. python bot.py                  # live trading

Flags
-----
--paper         Force paper-trade mode regardless of .env
--no-momentum   Disable momentum strategy (arb only)
--no-arb        Disable arb strategy (momentum only)
--loglevel      DEBUG / INFO / WARNING / ERROR  (default INFO)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

import config
from feeds.binance import BinanceFeed
from feeds.coinbase import CoinbaseFeed
from feeds.aggregator import PriceAggregator, make_feed_callback
from polymarket.client import PolymarketClient
from polymarket.markets import MarketCache
from polymarket.orders import OrderManager
from strategy.momentum import MomentumStrategy
from strategy.arbitrage import ArbStrategy


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    fmt = "%(asctime)s %(levelname)-8s %(name)-20s %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


log = logging.getLogger("bot")


# ---------------------------------------------------------------------------
# Main bot
# ---------------------------------------------------------------------------

class Bot:
    def __init__(self, use_momentum: bool = True, use_arb: bool = True) -> None:
        self._use_momentum = use_momentum
        self._use_arb = use_arb

        self._pm_client = PolymarketClient()
        self._market_cache = MarketCache()
        self._aggregator = PriceAggregator()

        self._order_manager: OrderManager | None = None
        self._momentum: MomentumStrategy | None = None
        self._arb: ArbStrategy | None = None

        self._binance_feed: BinanceFeed | None = None
        self._coinbase_feed: CoinbaseFeed | None = None

        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        log.info("=== Polymarket Trading Bot starting ===")
        log.info(
            "Mode: paper=%s  momentum=%s  arb=%s",
            config.PAPER_TRADE, self._use_momentum, self._use_arb,
        )
        log.info(
            "Risk: order=$%.0f–$%.0f  max_exposure=$%.0f  kelly=%.2f",
            config.RISK.min_order_usdc,
            config.RISK.max_order_usdc,
            config.RISK.max_total_exposure_usdc,
            config.STRATEGY.kelly_fraction,
        )
        log.info(
            "Arb:  edge_threshold=%.3f  scan_interval=%.1fs",
            config.STRATEGY.arb_edge_threshold,
            config.STRATEGY.arb_scan_interval,
        )
        log.info(
            "Momentum: trigger=%.2f%%  lookback=%.0fs",
            config.STRATEGY.trigger_pct,
            config.STRATEGY.lookback_secs,
        )

        # 1. Connect to Polymarket CLOB
        await self._pm_client.connect()

        # 2. Load initial market list
        await self._market_cache.start()

        # 3. Build order manager (aggregator wired in for fair-value exit)
        self._order_manager = OrderManager(
            self._pm_client, self._market_cache, self._aggregator
        )

        # 4. Build strategies and wire into aggregator
        if self._use_momentum:
            self._momentum = MomentumStrategy(
                on_signal=self._order_manager.execute_signal
            )
            self._aggregator.add_subscriber(self._momentum.on_tick)

        if self._use_arb:
            self._arb = ArbStrategy(
                aggregator=self._aggregator,
                cache=self._market_cache,
                pm_client=self._pm_client,
                on_signal=self._order_manager.execute_arb_signal,
            )
            # Give arb strategy access to OrderManager so it can run exit scans
            self._arb.set_order_manager(self._order_manager)
            # Subscribe to price ticks so the arb strategy re-evaluates markets
            # immediately on every incoming Binance/Coinbase trade event, rather
            # than waiting for the next polling interval.  Midpoints are cached
            # inside ArbStrategy so no extra CLOB API calls are made per tick.
            self._aggregator.add_subscriber(self._arb.on_price_tick)

        # 5. Wire feeds → aggregator
        binance_cb = make_feed_callback(self._aggregator, "binance")
        coinbase_cb = make_feed_callback(self._aggregator, "coinbase")

        self._binance_feed = BinanceFeed(callback=binance_cb)
        self._coinbase_feed = CoinbaseFeed(callback=coinbase_cb)

        # 6. Launch background tasks
        self._tasks = [
            asyncio.create_task(
                self._order_manager.run_balance_refresh_loop(), name="balance-refresh"
            ),
            asyncio.create_task(
                self._market_cache.run_refresh_loop(), name="market-refresh"
            ),
            asyncio.create_task(
                self._binance_feed.run(), name="binance-feed"
            ),
            asyncio.create_task(
                self._coinbase_feed.run(), name="coinbase-feed"
            ),
        ]
        if self._use_arb and self._arb:
            self._tasks.append(
                asyncio.create_task(self._arb.run(), name="arb-scanner")
            )

        log.info(
            "Bot running with %d tasks. Feeds: Binance+Coinbase (BTC/ETH/XRP/SOL). "
            "Press Ctrl+C to stop.",
            len(self._tasks),
        )
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        log.info("Shutting down …")
        if self._binance_feed:
            self._binance_feed.stop()
        if self._coinbase_feed:
            self._coinbase_feed.stop()
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

    bot = Bot(
        use_momentum=not args.no_momentum,
        use_arb=not args.no_arb,
    )

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
    parser = argparse.ArgumentParser(
        description="Polymarket 15-min crypto trading bot (momentum + arb)"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=False,
        help="Paper-trade mode: log orders but don't submit them",
    )
    parser.add_argument(
        "--no-momentum",
        action="store_true",
        default=False,
        help="Disable the momentum strategy",
    )
    parser.add_argument(
        "--no-arb",
        action="store_true",
        default=False,
        help="Disable the arbitrage strategy",
    )
    parser.add_argument(
        "--loglevel",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parsed = parser.parse_args()

    try:
        asyncio.run(main(parsed))
    except KeyboardInterrupt:
        sys.exit(0)
