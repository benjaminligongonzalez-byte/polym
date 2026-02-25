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
import threading
import time

import config
from feeds.binance import BinanceFeed

# ── ANSI colours (terminal display) ──────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


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
        self._drain_task: asyncio.Task | None = None

    async def start(self) -> None:
        log.info("=== Polymarket Trading Bot starting ===")
        log.info(
            "Mode: paper=%s  momentum=%s  arb=%s",
            config.PAPER_TRADE, self._use_momentum, self._use_arb,
        )
        log.info(
            "Risk: order=$%.0f–$%.0f  max_exposure=%.0f%%  kelly=%.2f",
            config.RISK.min_order_usdc,
            config.RISK.max_order_usdc_hard,
            config.RISK.max_exposure_fraction * 100,
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

        # 1. Connect to Polymarket CLOB and print auth summary
        await self._pm_client.connect()
        await self._pm_client.auth_banner()

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

        # 7. Start interactive console (stdin command reader)
        self._tasks.append(
            asyncio.create_task(self._console_loop(), name="console")
        )

        mode_label = f"{_YELLOW}PAPER TRADE{_RESET}" if config.PAPER_TRADE else f"{_GREEN}LIVE{_RESET}"
        strats = "  ".join(filter(None, [
            f"{_CYAN}Momentum{_RESET}" if self._use_momentum else "",
            f"{_CYAN}ARB+Snipe{_RESET}" if self._use_arb else "",
        ]))
        print(
            f"\n{_BOLD}{'═'*66}{_RESET}\n"
            f"  {_BOLD}POLYMARKET TRADING BOT{_RESET}  —  {mode_label}  |  {strats}\n"
            f"  Strategies: {strats}  |  Tasks: {len(self._tasks)}\n"
            f"  Commands: {_BOLD}s{_RESET}=stats  {_BOLD}p{_RESET}=positions  "
            f"{_BOLD}m{_RESET}=markets  {_BOLD}t{_RESET}=trades  "
            f"{_BOLD}pause{_RESET}  {_BOLD}drain{_RESET}  "
            f"{_BOLD}h{_RESET}=help  {_BOLD}q{_RESET}=quit\n"
            f"{_BOLD}{'═'*66}{_RESET}\n",
            flush=True,
        )
        log.info(
            "Bot running with %d tasks. Feeds: Binance+Coinbase (BTC/ETH/XRP/SOL).",
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

    # ------------------------------------------------------------------
    # Interactive console
    # ------------------------------------------------------------------

    _CONSOLE_HELP = """\
Commands (type and press Enter):
  s  /  status     — full dashboard: balance, P&L, win rate, trade counts
  p  /  positions  — open positions (symbol, direction, entry price, age)
  m  /  markets    — active markets in cache with time remaining
  t  /  trades     — full session trade log (copy-paste for analysis)
  pause            — stop new orders; let existing positions settle naturally
  resume           — resume trading after a pause (also cancels drain)
  drain            — pause + auto-shutdown once all open positions close
  h  /  help       — this message
  q  /  quit       — immediate graceful shutdown (Ctrl+C equivalent)
"""

    async def _console_loop(self) -> None:
        """
        Reads lines from stdin without blocking the event loop.

        A daemon thread blocks continuously on sys.stdin so Enter is always
        registered — even when log output scrolls the terminal mid-type.
        Commands are pushed into an asyncio Queue and dispatched on the
        event loop, keeping all bot state single-threaded.
        """
        loop  = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()

        def _reader() -> None:
            try:
                for line in sys.stdin:
                    loop.call_soon_threadsafe(queue.put_nowait, line)
            except Exception:
                pass

        threading.Thread(target=_reader, daemon=True, name="stdin-reader").start()

        print(
            f"\n{_DIM}[console] Ready — type a command + Enter.{_RESET}\n",
            flush=True,
        )
        while True:
            try:
                raw = await queue.get()
            except Exception:
                break
            cmd = raw.strip().lower()
            if not cmd:
                continue

            om = self._order_manager

            if cmd in ("s", "status", "stats"):
                if om:
                    print(om.stats_report(), flush=True)
                else:
                    print("  OrderManager not ready yet.", flush=True)

            elif cmd in ("p", "positions"):
                if om:
                    print(om.positions_report(), flush=True)
                else:
                    print("  OrderManager not ready yet.", flush=True)

            elif cmd in ("m", "markets"):
                mkts = self._market_cache.all_markets()
                if mkts:
                    lines = [
                        f"{_BOLD}{'─'*66}{_RESET}",
                        f"  {_BOLD}ACTIVE MARKETS{_RESET}  ({_CYAN}{len(mkts)}{_RESET} windows)",
                        f"{_BOLD}{'─'*66}{_RESET}",
                    ]
                    for m in sorted(mkts, key=lambda x: x.symbol):
                        t = m.time_remaining_secs
                        mi, se = divmod(int(max(t, 0)), 60)
                        # Time bar — fill based on fraction of 900s window remaining
                        frac = max(0.0, min(1.0, t / 900.0))
                        bar_fill = round(frac * 12)
                        bar = "█" * bar_fill + "░" * (12 - bar_fill)
                        # Colour by urgency
                        t_col = _RED if t < 60 else (_YELLOW if t < 180 else _GREEN)
                        strike_str = (
                            f"${m.strike_price:,.4f}"
                            if m.strike_price else f"{_DIM}unlocked{_RESET}"
                        )
                        up_mid = f"{m.up_mid:.2f}" if m.up_mid else " ─ "
                        lines.append(
                            f"  {_CYAN}{m.symbol:<4}{_RESET}"
                            f"  [{t_col}{bar}{_RESET}]"
                            f"  {t_col}{mi:02d}m{se:02d}s{_RESET}"
                            f"  strike={strike_str:<14}"
                            f"  up_mid={_BOLD}{up_mid}{_RESET}"
                            f"  {_DIM}{m.question[:42]}{_RESET}"
                        )
                    lines.append(f"{'─'*66}")
                    print("\n".join(lines), flush=True)
                else:
                    print("  No markets in cache.", flush=True)

            elif cmd in ("t", "trades"):
                if om:
                    print(om.trades_report(), flush=True)
                else:
                    print("  OrderManager not ready yet.", flush=True)

            elif cmd == "pause":
                if om:
                    om.pause()
                    print(
                        f"  Paused. {len(om._positions)} position(s) still open — "
                        "they will settle normally. Type 'resume' to restart trading, "
                        "or 'drain' to auto-shutdown when all positions close.",
                        flush=True,
                    )
                else:
                    print("  OrderManager not ready yet.", flush=True)

            elif cmd == "resume":
                if om:
                    if self._drain_task and not self._drain_task.done():
                        self._drain_task.cancel()
                        self._drain_task = None
                        print("  Drain cancelled.", flush=True)
                    om.resume()
                    print("  Resumed. New orders are enabled.", flush=True)
                else:
                    print("  OrderManager not ready yet.", flush=True)

            elif cmd == "drain":
                if om:
                    if self._drain_task and not self._drain_task.done():
                        print("  Already draining. Type 'resume' to cancel drain.", flush=True)
                    else:
                        om.pause()
                        n = len(om._positions)
                        if n == 0:
                            print("  No open positions — shutting down now.", flush=True)
                            loop.create_task(self.stop())
                            break
                        print(
                            f"  DRAIN MODE: {n} position(s) open. "
                            "No new orders. Auto-shutdown when all settle. "
                            "Type 'resume' to cancel drain.",
                            flush=True,
                        )
                        self._drain_task = loop.create_task(self._drain_loop())
                else:
                    print("  OrderManager not ready yet.", flush=True)

            elif cmd in ("h", "help"):
                print(self._CONSOLE_HELP, flush=True)

            elif cmd in ("q", "quit", "exit"):
                print("  Shutting down…", flush=True)
                loop.create_task(self.stop())
                break

            else:
                print(
                    f"  Unknown command '{cmd}'. Type 'h' for help.",
                    flush=True,
                )

    async def _drain_loop(self) -> None:
        """
        Drain mode: no new orders (already paused). Poll every 10 s.
        When all open positions have closed (via exit or resolution),
        trigger a graceful shutdown automatically.
        """
        check_interval = 10  # seconds between position checks
        while True:
            om = self._order_manager
            if om is None or len(om._positions) == 0:
                remaining = len(om._positions) if om else 0
                print(
                    f"\n  [drain] All positions settled ({remaining} open). "
                    "Shutting down…\n",
                    flush=True,
                )
                asyncio.get_running_loop().create_task(self.stop())
                break
            open_syms = ", ".join(
                f"{p.symbol}/{p.bet}" for p in om._positions.values()
            )
            print(
                f"  [drain] {len(om._positions)} position(s) still open: {open_syms}. "
                f"Checking again in {check_interval}s…",
                flush=True,
            )
            await asyncio.sleep(check_interval)


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

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown, sig)
    else:
        # Windows doesn't support add_signal_handler; use signal.signal for Ctrl+C
        signal.signal(signal.SIGINT, lambda s, f: loop.create_task(bot.stop()))

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
