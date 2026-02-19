"""
Analyze a Polymarket trader's 15-min crypto bet history.

Usage:
    python analyze_trader.py 0xe00740bce98a594e26861838885ab310ec3b548c
    python analyze_trader.py 0xe00740bce98a594e26861838885ab310ec3b548c --limit 1000

Fetches all activity from the Polymarket Data API and produces:
  - Win rate / PnL breakdown by symbol, by bet direction (Up/Down)
  - Average entry time within the 15-min window
  - Average price paid vs mid at entry (slippage / edge analysis)
  - Position sizing pattern
  - Trade frequency and timing distribution
  - Whether they appear to trade on momentum or mispricing
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

DATA_API = "https://data-api.polymarket.com"


def fetch_activity(address: str, limit: int = 500) -> list[dict]:
    """Fetch all trades for *address* from the Polymarket Data API."""
    url = f"{DATA_API}/activity"
    params = {"user": address.lower(), "limit": limit}
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; trader-analyzer/1.0)",
        "Accept": "application/json",
        "Referer": "https://polymarket.com/",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else data.get("data", data.get("activity", []))


def fetch_positions(address: str) -> list[dict]:
    """Fetch open/closed positions for richer PnL data."""
    url = f"{DATA_API}/positions"
    params = {"user": address.lower(), "sizeThreshold": "0"}
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_15min_market(title: str) -> bool:
    markers = ["up or down - 15", "15 minutes", "up in 15", "down in 15",
               "above $", "below $"]
    t = title.lower()
    return any(m in t for m in markers)


def parse_bet_side(title: str, outcome: str) -> str:
    """Return 'Up', 'Down', or 'Unknown'."""
    o = outcome.lower()
    if o in ("up", "yes"):
        return "Up"
    if o in ("down", "no"):
        # 'No' on a 'higher' market = Down bet
        if "higher" in title.lower() or "above" in title.lower():
            return "Down"
        return "Down"
    return "Unknown"


def parse_symbol(title: str) -> str:
    for sym in ["BTC", "ETH", "XRP", "SOL", "MATIC", "DOGE", "AVAX", "LINK"]:
        if sym in title.upper():
            return sym
    return "OTHER"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(trades: list[dict], positions: list[dict]) -> None:
    # Filter to 15-min crypto bets
    crypto_trades = []
    for t in trades:
        title = t.get("title", t.get("market", t.get("question", "")))
        if is_15min_market(title):
            crypto_trades.append(t)

    total = len(crypto_trades)
    if total == 0:
        print("No 15-min crypto trades found in the activity data.")
        print(f"(Total trades in dataset: {len(trades)})")
        print("\nRaw sample of first 3 records:")
        for t in trades[:3]:
            print(json.dumps(t, indent=2))
        return

    print(f"\n{'='*60}")
    print(f"  POLYMARKET 15-MIN CRYPTO TRADER ANALYSIS")
    print(f"{'='*60}")
    print(f"Total 15-min trades:  {total}")

    # -----------------------------------------------------------------------
    # 1. By symbol
    # -----------------------------------------------------------------------
    by_symbol: dict[str, list] = defaultdict(list)
    for t in crypto_trades:
        title = t.get("title", t.get("market", ""))
        sym = parse_symbol(title)
        by_symbol[sym].append(t)

    print(f"\n{'─'*40}")
    print("  TRADES BY SYMBOL")
    print(f"{'─'*40}")
    for sym, tlist in sorted(by_symbol.items(), key=lambda x: -len(x[1])):
        print(f"  {sym:6s}  {len(tlist):4d} trades")

    # -----------------------------------------------------------------------
    # 2. By bet side (Up vs Down)
    # -----------------------------------------------------------------------
    up_count = down_count = unknown_count = 0
    for t in crypto_trades:
        title = t.get("title", t.get("market", ""))
        outcome = t.get("outcome", t.get("side", t.get("tokenOutcome", "")))
        side = parse_bet_side(title, outcome)
        if side == "Up":
            up_count += 1
        elif side == "Down":
            down_count += 1
        else:
            unknown_count += 1

    print(f"\n{'─'*40}")
    print("  BET DIRECTION BIAS")
    print(f"{'─'*40}")
    print(f"  Up   bets: {up_count:4d}  ({up_count/total*100:.1f}%)")
    print(f"  Down bets: {down_count:4d}  ({down_count/total*100:.1f}%)")
    if unknown_count:
        print(f"  Unknown:   {unknown_count:4d}")

    # -----------------------------------------------------------------------
    # 3. Position sizing
    # -----------------------------------------------------------------------
    sizes = []
    prices_paid = []
    for t in crypto_trades:
        size = float(t.get("size", t.get("usdcSize", t.get("amount", 0)) or 0))
        price = float(t.get("price", t.get("avgPrice", 0)) or 0)
        if size > 0:
            sizes.append(size)
        if price > 0:
            prices_paid.append(price)

    if sizes:
        sizes.sort()
        mean_size = sum(sizes) / len(sizes)
        median_size = sizes[len(sizes) // 2]
        total_usdc = sum(sizes)
        print(f"\n{'─'*40}")
        print("  POSITION SIZING")
        print(f"{'─'*40}")
        print(f"  Total USDC bet:   ${total_usdc:,.2f}")
        print(f"  Avg bet size:     ${mean_size:.2f}")
        print(f"  Median bet size:  ${median_size:.2f}")
        print(f"  Min / Max:        ${sizes[0]:.2f}  /  ${sizes[-1]:.2f}")
        print(f"  # bets > $50:     {sum(1 for s in sizes if s > 50)}")
        print(f"  # bets > $100:    {sum(1 for s in sizes if s > 100)}")

    # -----------------------------------------------------------------------
    # 4. Price paid distribution — are they buying cheap or expensive?
    # -----------------------------------------------------------------------
    if prices_paid:
        prices_paid.sort()
        mean_price = sum(prices_paid) / len(prices_paid)
        bins = {"< 0.30": 0, "0.30–0.45": 0, "0.45–0.55": 0,
                "0.55–0.70": 0, "> 0.70": 0}
        for p in prices_paid:
            if p < 0.30:
                bins["< 0.30"] += 1
            elif p < 0.45:
                bins["0.30–0.45"] += 1
            elif p < 0.55:
                bins["0.45–0.55"] += 1
            elif p < 0.70:
                bins["0.55–0.70"] += 1
            else:
                bins["> 0.70"] += 1

        print(f"\n{'─'*40}")
        print("  ENTRY PRICE DISTRIBUTION  (implied probability paid)")
        print(f"{'─'*40}")
        print(f"  Mean price paid: {mean_price:.3f}  ({mean_price*100:.1f}¢)")
        for label, cnt in bins.items():
            bar = "█" * (cnt * 30 // max(bins.values(), default=1))
            print(f"  {label:12s}  {cnt:4d}  {bar}")

        # Are they a contrarian or momentum buyer?
        cheap_count = bins["< 0.30"] + bins["0.30–0.45"]
        expensive_count = bins["0.55–0.70"] + bins["> 0.70"]
        print(f"\n  Buying cheap (< 0.45):     {cheap_count:4d}  ({cheap_count/total*100:.0f}%)")
        print(f"  Buying near-fair (0.45–0.55): {bins['0.45–0.55']:4d}  ({bins['0.45–0.55']/total*100:.0f}%)")
        print(f"  Buying expensive (> 0.55): {expensive_count:4d}  ({expensive_count/total*100:.0f}%)")

        if cheap_count > expensive_count * 1.5:
            print("\n  ► PATTERN: Contrarian buyer — bets on long-shot underdogs")
        elif expensive_count > cheap_count * 1.5:
            print("\n  ► PATTERN: Momentum buyer — chases favorites / fast movers")
        else:
            print("\n  ► PATTERN: Mixed / arb — buying near-fair value")

    # -----------------------------------------------------------------------
    # 5. Win / loss (requires resolved positions)
    # -----------------------------------------------------------------------
    won = lost = pending = 0
    pnl_total = 0.0
    for t in crypto_trades:
        outcome_val = t.get("outcomeValue", t.get("redeemed", None))
        size = float(t.get("size", t.get("usdcSize", 0)) or 0)
        price = float(t.get("price", t.get("avgPrice", 0)) or 0)
        cost = size * price if size and price else size

        if outcome_val == 1 or t.get("winner") is True:
            won += 1
            pnl_total += (size - cost)
        elif outcome_val == 0 or t.get("winner") is False:
            lost += 1
            pnl_total -= cost
        else:
            pending += 1

    resolved = won + lost
    if resolved > 0:
        print(f"\n{'─'*40}")
        print("  WIN / LOSS RECORD")
        print(f"{'─'*40}")
        print(f"  Won:      {won:4d}  ({won/resolved*100:.1f}%)")
        print(f"  Lost:     {lost:4d}  ({lost/resolved*100:.1f}%)")
        print(f"  Pending:  {pending:4d}")
        print(f"  Est. PnL: ${pnl_total:+.2f}")

    # -----------------------------------------------------------------------
    # 6. Timing — when in the window do they buy?
    # -----------------------------------------------------------------------
    timestamps = []
    for t in crypto_trades:
        ts_raw = t.get("timestamp", t.get("createdAt", t.get("transactionTime")))
        if ts_raw:
            try:
                if isinstance(ts_raw, (int, float)):
                    ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                else:
                    ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                timestamps.append(ts)
            except (ValueError, OSError):
                pass

    if timestamps:
        # Minute within the 15-min window (0–14)
        window_minutes = [(ts.minute % 15) for ts in timestamps]
        minute_counts = defaultdict(int)
        for m in window_minutes:
            minute_counts[m] += 1

        print(f"\n{'─'*40}")
        print("  ENTRY TIMING (minute within the 15-min window)")
        print(f"{'─'*40}")
        peak_minute = max(minute_counts, key=minute_counts.__getitem__)
        for m in range(15):
            cnt = minute_counts.get(m, 0)
            bar = "█" * (cnt * 20 // max(minute_counts.values(), default=1))
            label = f"  min {m:2d}  {cnt:4d}  {bar}"
            if m == peak_minute:
                label += "  ◄ peak"
            print(label)

        # Hour-of-day distribution
        hours = defaultdict(int)
        for ts in timestamps:
            hours[ts.hour] += 1
        print(f"\n  Most active hour (UTC): {max(hours, key=hours.__getitem__):02d}:00"
              f"  ({max(hours.values())} trades)")

        if peak_minute <= 2:
            print("\n  ► TIMING PATTERN: Opens positions at window START (momentum)")
        elif peak_minute >= 12:
            print("\n  ► TIMING PATTERN: Opens positions near window END (last-second bet)")
        else:
            print(f"\n  ► TIMING PATTERN: Enters mid-window at ~{peak_minute} min mark")

    # -----------------------------------------------------------------------
    # 7. Strategy inference
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  STRATEGY INFERENCE")
    print(f"{'='*60}")

    clues = []
    if sizes and (max(sizes) / (sum(sizes) / len(sizes))) > 5:
        clues.append("Variable bet sizing → Kelly or edge-based sizing")
    else:
        clues.append("Uniform bet sizing → fixed stake system")

    if prices_paid and mean_price > 0.55:
        clues.append("Pays above 55¢ on average → momentum/trend follower")
    elif prices_paid and mean_price < 0.45:
        clues.append("Pays below 45¢ on average → contrarian / fade-the-move")
    else:
        clues.append("Pays near 50¢ → arb / fair-value seeker")

    if up_count > down_count * 1.5:
        clues.append("Strong Up bias → buys rallies / bullish bias")
    elif down_count > up_count * 1.5:
        clues.append("Strong Down bias → buys dips / bearish bias")
    else:
        clues.append("Balanced Up/Down → not directionally biased")

    if timestamps and peak_minute is not None and peak_minute <= 1:
        clues.append("Enters at minute 0–1 → very likely a momentum bot")
    elif timestamps and peak_minute is not None and peak_minute >= 7:
        clues.append("Enters mid-to-late window → waiting for price signal to develop")

    for c in clues:
        print(f"  • {c}")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Polymarket trader's 15-min bets")
    parser.add_argument("address", help="Ethereum wallet address")
    parser.add_argument("--limit", type=int, default=500, help="Max trades to fetch")
    parser.add_argument("--raw", action="store_true", help="Also dump raw JSON to stdout")
    args = parser.parse_args()

    print(f"Fetching activity for {args.address} …")
    try:
        trades = fetch_activity(args.address, args.limit)
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}")
        sys.exit(1)

    print(f"  → {len(trades)} total trades returned")

    positions = fetch_positions(args.address)
    print(f"  → {len(positions)} positions returned")

    if args.raw:
        print("\n--- RAW ACTIVITY (first 5) ---")
        for t in trades[:5]:
            print(json.dumps(t, indent=2))

    analyze(trades, positions)


if __name__ == "__main__":
    main()
