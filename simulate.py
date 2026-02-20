"""
Polymarket paper-trading simulator — no wallet credentials required.

Uses REAL Binance + Polymarket prices to evaluate strategy signals,
but fakes all orders against a virtual $1,000 wallet.

Modes
-----
python simulate.py              # live mode — needs internet
python simulate.py --demo       # offline demo with realistic fake data
python simulate.py --cycles 10  # run N scan cycles then exit (default 999)
python simulate.py --interval 5 # seconds between scans (default 5)
python simulate.py --wallet 500 # starting USDC balance (default 1000)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Inline strategy constants (mirrors config.py)
# ─────────────────────────────────────────────────────────────────────────────

GAMMA_API    = "https://gamma-api.polymarket.com"
CLOB_HOST    = "https://clob.polymarket.com"
BINANCE_API  = "https://api.binance.us"

MARKET_KEYWORDS = [
    "up or down",           # new format: "Bitcoin Up or Down - Jan 7, 10:45AM-11:00AM ET"
    "up or down - 15",      # old format: "XRP Up or Down - 15 Minutes"
    "up or down - 15 min",
    "higher 15 minutes",
    "lower 15 minutes",
    "up in 15",
    "down in 15",
    "above $",
    "below $",
]
TARGET_SYMBOLS = ["BTC", "ETH", "XRP", "SOL"]
VOLATILITY = {"BTC": 0.90, "ETH": 1.00, "XRP": 1.20, "SOL": 1.30}
DEFAULT_VOL    = 1.00
HOUSE_SPREAD   = 0.02

ARB_EDGE       = 0.05
ARB_MIN_PROB   = 0.65      # minimum fair probability for any ARB entry
ARB_MIN_D2     = 1.1       # minimum |d2| for ARB entries: price must be ≥1.1σ from strike
                            # Guards against near-the-money bets where a tiny adverse move
                            # collapses fair probability (e.g. XRP 0.6% from strike with 810s left)
SNIPE_WINDOW   = 300.0     # seconds — activate in final 5 min
SNIPE_MIN_PROB = 0.92      # minimum fair probability in sniper mode
SNIPE_MIN_PRICE_DIST_PCT = 0.010  # price must be ≥1% past strike to snipe
SNIPE_EDGE     = 0.03      # lower edge bar (near-certain outcome)
SNIPE_KELLY    = 0.50      # half-Kelly on near-locks
ARB_KELLY      = 0.25      # quarter-Kelly for standard arb

MAX_ORDER_FRAC   = 0.05    # 5% of wallet per order
MAX_EXPOSURE_FRAC= 0.20    # 20% of wallet total
MAX_ORDER_HARD   = 50.0    # hard cap per single order
MIN_ORDER        = 2.0
SLIPPAGE         = 0.03
COOLDOWN         = 30.0

# Early exit parameters (mirrors config.py)
EXIT_RESIDUAL_EDGE = 0.02  # sell when market is within 2¢ of fair value
EXIT_TAKE_PROFIT   = 0.07  # flat fallback: sell when up 7¢ from entry
EXIT_MIN_T_REM     = 90.0  # never exit within 90s of expiry

# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def fair_prob_up(current: float, strike: float, t_secs: float, vol: float) -> float:
    T = max(t_secs, 1.0) / (365.25 * 24.0 * 3600.0)
    sqt = vol * math.sqrt(T)
    if sqt == 0:
        return 1.0 if current > strike else 0.0
    d2 = (math.log(current / strike) - 0.5 * vol**2 * T) / sqt
    return norm_cdf(d2)

def full_kelly(fair: float, market: float) -> float:
    d = 1.0 - market
    if d <= 0:
        return 0.0
    return max(0.0, (fair - market) / d)

# ─────────────────────────────────────────────────────────────────────────────
# Market helpers
# ─────────────────────────────────────────────────────────────────────────────

_STRIKE_RE = re.compile(
    r'\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)'
)

def parse_strike(question: str, description: str = "") -> Optional[float]:
    for text in [description, question]:
        m = _STRIKE_RE.search(text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return None

def parse_symbol(title: str) -> Optional[str]:
    for sym in TARGET_SYMBOLS:
        if sym in title.upper():
            return sym
    return None

def is_15min(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in MARKET_KEYWORDS)

def secs_until(iso: str) -> float:
    try:
        end = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (end - datetime.now(timezone.utc)).total_seconds()
    except Exception:
        return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# Live data fetchers (real-mode only)
# ─────────────────────────────────────────────────────────────────────────────

def _headers():
    return {
        "User-Agent": "Mozilla/5.0 (compatible; poly-sim/1.0)",
        "Accept": "application/json",
        "Referer": "https://polymarket.com/",
    }

# Cache: conditionId → opening price so we don't re-query Coinbase every cycle.
_strike_cache: dict[str, float] = {}

def live_open_price(symbol: str, start_iso: str) -> Optional[float]:
    """Return the Coinbase price of `symbol` at the start of a market window.

    Polymarket's "BTC Up or Down - 15 min" markets use the price at
    `startDate` as the strike.  We fetch the 1-minute candle that opens at
    (or just after) that timestamp and take its open price.
    """
    import requests
    from datetime import timedelta
    product = {"BTC": "BTC-USD", "ETH": "ETH-USD",
               "XRP": "XRP-USD", "SOL": "SOL-USD"}.get(symbol)
    if not product or not start_iso:
        return None
    try:
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end_dt   = start_dt + timedelta(minutes=3)
        resp = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles",
            params={
                "start": start_dt.isoformat(),
                "end":   end_dt.isoformat(),
                "granularity": 60,
            },
            timeout=5,
        )
        resp.raise_for_status()
        candles = resp.json()   # [[time, low, high, open, close, vol], ...]
        if candles:
            return float(candles[-1][3])   # open of the earliest candle
    except Exception:
        pass
    return None

def live_prices() -> dict[str, float]:
    import requests
    products = {"BTC-USD": "BTC", "ETH-USD": "ETH", "XRP-USD": "XRP", "SOL-USD": "SOL"}
    prices = {}
    for product, sym in products.items():
        resp = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/ticker",
            timeout=5,
        )
        resp.raise_for_status()
        prices[sym] = float(resp.json()["price"])
    return prices

def live_markets() -> list[dict]:
    import requests
    resp = requests.get(
        f"{GAMMA_API}/markets",
        params={"active": "true", "closed": "false", "limit": 200},
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    mkts = data if isinstance(data, list) else data.get("markets", [])
    matched = [m for m in mkts if is_15min(m.get("question", m.get("title", "")))]
    if not matched:
        print(f"  {YELLOW}[debug] Gamma API returned {len(mkts)} markets, 0 matched 15-min filter.{RESET}")
        print(f"  {YELLOW}[debug] API keys in first market: {list(mkts[0].keys()) if mkts else 'NO MARKETS'}{RESET}")
        crypto_kw = ["btc", "eth", "xrp", "sol", "bitcoin", "ethereum", "solana", "ripple"]
        crypto_mkts = [m for m in mkts if any(k in (m.get("question","") + m.get("title","")).lower() for k in crypto_kw)]
        print(f"  {YELLOW}[debug] Crypto-related markets found: {len(crypto_mkts)}{RESET}")
        for m in crypto_mkts[:15]:
            q = m.get("question", m.get("title", ""))
            print(f"  {DIM}  → {q[:100]}{RESET}")
        if not crypto_mkts:
            print(f"  {YELLOW}[debug] First 10 markets (any topic):{RESET}")
            for m in mkts[:10]:
                q = m.get("question", m.get("title", ""))
                print(f"  {DIM}  → {q[:100]}{RESET}")
    return matched

def live_midpoint(token_id: str) -> Optional[float]:
    import requests
    try:
        resp = requests.get(
            f"{CLOB_HOST}/midpoint",
            params={"token_id": token_id},
            headers=_headers(),
            timeout=5,
        )
        resp.raise_for_status()
        return float(resp.json().get("mid", 0.5))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Demo data — realistic snapshot (no internet needed)
# ─────────────────────────────────────────────────────────────────────────────

# Approximate prices for Feb 2026 — prices shift slightly each cycle to mimic
# live movement.  Each market has (strike, up_mid, t_rem_secs, cid, up_token, down_token).

_BASE_PRICES = {"BTC": 97_450.0, "ETH": 2_830.0, "XRP": 2.485, "SOL": 172.50}

# Drift per cycle (+/- to simulate market moving)
_DRIFT_PCT = {"BTC": 0.0008, "ETH": 0.0005, "XRP": 0.0015, "SOL": 0.0010}

# Six demo markets, spread across different stages of their 15-min windows.
# Columns: question, description (strike), t_rem, up_token_id, up_mid
_DEMO_MARKETS_TEMPLATE = [
    {
        "cid":  "cid_xrp_snipe",
        "question": "XRP Up or Down - 15 Minutes",
        "desc": "Price to Beat: $2.4750",
        "strike": 2.4750,
        "t_rem_0": 185,       # 3m 5s left — sniper territory
        "up_token": "tok_xrp_up",
        "down_token": "tok_xrp_dn",
        "up_mid_0": 0.74,     # Polymarket lagging (fair ≈ 0.93)
        "symbol": "XRP",
    },
    {
        "cid":  "cid_btc_arb",
        "question": "BTC Up or Down - 15 Minutes",
        "desc": "Price to Beat: $97,300",
        "strike": 97_300.0,
        "t_rem_0": 510,       # 8m 30s left — standard arb window
        "up_token": "tok_btc_up",
        "down_token": "tok_btc_dn",
        "up_mid_0": 0.50,     # Polymarket flat; BTC has moved up
        "symbol": "BTC",
    },
    {
        "cid":  "cid_eth_skip",
        "question": "ETH Up or Down - 15 Minutes",
        "desc": "Price to Beat: $2,835",
        "strike": 2_835.0,
        "t_rem_0": 680,       # 11m 20s left
        "up_token": "tok_eth_up",
        "down_token": "tok_eth_dn",
        "up_mid_0": 0.48,     # Polymarket fair — no edge
        "symbol": "ETH",
    },
    {
        "cid":  "cid_sol_snipe",
        "question": "SOL Up or Down - 15 Minutes",
        "desc": "Price to Beat: $173.50",
        "strike": 173.50,
        "t_rem_0": 240,       # 4m left — sniper
        "up_token": "tok_sol_up",
        "down_token": "tok_sol_dn",
        "up_mid_0": 0.35,     # SOL is clearly BELOW strike; Down underpriced
        "symbol": "SOL",
    },
    {
        "cid":  "cid_btc2_arb",
        "question": "BTC Up or Down - 15 Minutes",
        "desc": "Price to Beat: $97,600",
        "strike": 97_600.0,
        "t_rem_0": 420,       # 7m left — arb
        "up_token": "tok_btc2_up",
        "down_token": "tok_btc2_dn",
        "up_mid_0": 0.43,
        "symbol": "BTC",
    },
    {
        "cid":  "cid_xrp2_early",
        "question": "XRP Up or Down - 15 Minutes",
        "desc": "Price to Beat: $2.510",
        "strike": 2.510,
        "t_rem_0": 810,       # 13m 30s left — early, just opened
        "up_token": "tok_xrp2_up",
        "down_token": "tok_xrp2_dn",
        "up_mid_0": 0.49,
        "symbol": "XRP",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Wallet + position tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    cid: str
    symbol: str
    bet: str            # "Up" or "Down"
    fair_prob: float
    entry_price: float  # price we paid (market_prob at entry)
    order_usdc: float
    shares: float
    strike: float
    end_secs: float     # sim_now / monotonic when market expires
    is_snipe: bool
    mode: str
    live_price_at_entry: float
    won: Optional[bool] = None
    pnl: float = 0.0
    exit_price: Optional[float] = None   # set on early exit
    exit_type: str = ""                  # "fair-val" | "flat-tp" | "resolution"

class Wallet:
    def __init__(self, starting: float):
        self.balance = starting
        self.starting = starting
        self._exposure: dict[str, float] = {}   # cid → usdc deployed
        self.wins = 0
        self.losses = 0

    @property
    def deployed(self) -> float:
        return sum(self._exposure.values())

    @property
    def max_order(self) -> float:
        return min(self.balance * MAX_ORDER_FRAC, MAX_ORDER_HARD)

    @property
    def max_exposure(self) -> float:
        return self.balance * MAX_EXPOSURE_FRAC

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.max_exposure - self.deployed)

    def open_order(self, cid: str, usdc: float) -> None:
        self._exposure[cid] = self._exposure.get(cid, 0.0) + usdc
        self.balance -= usdc        # cash leaves wallet when trade opens

    def close_order(self, cid: str) -> None:
        self._exposure.pop(cid, None)

    def settle(self, pos: Position, won: bool, final_price: float) -> None:
        pos.won = won
        pos.exit_type = "resolution"
        if won:
            pnl = pos.shares - pos.order_usdc   # net profit (shares × $1 − cost)
            self.balance += pos.shares           # collect $1 per winning share
            pos.pnl = pnl
            self.wins += 1
        else:
            pos.pnl = -pos.order_usdc            # total loss (already deducted at open)
            self.losses += 1
        self.close_order(pos.cid)

    def early_exit(self, pos: Position, sell_price: float, exit_type: str) -> None:
        """Simulate a take-profit sell before resolution."""
        proceeds = round(pos.shares * sell_price, 4)
        pos.pnl = proceeds - pos.order_usdc
        pos.won = pos.pnl >= 0
        pos.exit_price = sell_price
        pos.exit_type = exit_type
        self.balance += proceeds                 # cash back in wallet
        if pos.won:
            self.wins += 1
        else:
            self.losses += 1
        self.close_order(pos.cid)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    symbol: str,
    cid: str,
    strike: float,
    t_rem: float,
    live_price: float,
    up_mid: float,
    wallet: Wallet,
    cooldowns: dict[str, float],
    sim_now_ref: float = 0.0,
) -> Optional[dict]:
    """
    Returns a trade dict or None.  Pure function — no side effects.
    """
    down_mid = 1.0 - up_mid

    is_snipe = t_rem < SNIPE_WINDOW
    threshold = SNIPE_EDGE if is_snipe else ARB_EDGE
    kelly_mult = SNIPE_KELLY if is_snipe else ARB_KELLY

    vol = VOLATILITY.get(symbol, DEFAULT_VOL)
    p_up   = fair_prob_up(live_price, strike, t_rem, vol)
    p_down = 1.0 - p_up

    # Sniper confidence gate
    if is_snipe and max(p_up, p_down) < SNIPE_MIN_PROB:
        return None

    # Sniper distance gate: price must be ≥1% past strike.
    # A tiny margin (0.3–0.65%) can evaporate in the final minutes — this
    # blocks borderline snipes where a small reversal causes a full loss.
    if is_snipe and abs(live_price - strike) / strike < SNIPE_MIN_PRICE_DIST_PCT:
        return None

    # ARB mode: require |d2| ≥ ARB_MIN_D2.
    # d2 is the number of vol-adjusted standard deviations the price sits from
    # the strike.  When |d2| is small (< 1.1), a single-digit % adverse move
    # can shift fair probability by 20–30 points, invalidating the trade thesis
    # and forcing a stop-loss exit at a loss.  This scales correctly with both
    # asset volatility and time remaining — a flat %-distance check does not.
    # (Sniper mode uses its own distance gate above; this applies only to ARB.)
    if not is_snipe:
        T_arb   = max(t_rem, 1.0) / (365.25 * 24.0 * 3600.0)
        sqt_arb = vol * math.sqrt(T_arb)
        if sqt_arb > 0:
            d2_abs = abs((math.log(live_price / strike) - 0.5 * vol**2 * T_arb) / sqt_arb)
            if d2_abs < ARB_MIN_D2:
                return None

    edge_up   = p_up   - up_mid
    edge_down = p_down - down_mid
    best_edge = max(edge_up, edge_down)

    if best_edge < threshold:
        return None

    # Risk gates
    if wallet.remaining_budget < MIN_ORDER:
        return None
    # cooldown key encodes the sim_now at time of trade
    if (sim_now_ref - cooldowns.get(cid, 0.0)) < COOLDOWN:
        return None

    # Sniper mode: ONLY bet the high-probability (near-certain) side.
    # Standard ARB: pick whichever side has more edge.
    if is_snipe:
        if p_up >= p_down:
            if edge_up < threshold:
                return None   # near-certain side (Up) already fully priced — skip
            bet, token_side = "Up", "up"
            fair, mkt_p, edge = p_up, up_mid, edge_up
        else:
            if edge_down < threshold:
                return None   # near-certain side (Down) already fully priced — skip
            bet, token_side = "Down", "down"
            fair, mkt_p, edge = p_down, down_mid, edge_down
    elif edge_up >= edge_down:
        if p_up < ARB_MIN_PROB:
            return None   # low conviction — skip (fair prob < 65%)
        bet, token_side = "Up",   "up"
        fair, mkt_p, edge = p_up,   up_mid,   edge_up
    else:
        if p_down < ARB_MIN_PROB:
            return None   # low conviction — skip (fair prob < 65%)
        bet, token_side = "Down", "down"
        fair, mkt_p, edge = p_down, down_mid, edge_down

    kf = full_kelly(fair, mkt_p)
    kelly_usdc = kf * kelly_mult * wallet.balance
    order_usdc = max(MIN_ORDER, min(kelly_usdc, wallet.max_order, wallet.remaining_budget))

    limit_price = round(min(mkt_p + SLIPPAGE, 0.99), 4)
    shares = round(order_usdc / limit_price, 2)

    return dict(
        bet=bet,
        token_side=token_side,
        fair_prob=fair,
        market_prob=mkt_p,
        edge=edge,
        kelly_f=kf,
        kelly_mult=kelly_mult,
        order_usdc=order_usdc,
        limit_price=limit_price,
        shares=shares,
        is_snipe=is_snipe,
        mode="SNIPE" if is_snipe else "ARB",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
ORANGE = "\033[33m"

def _fmt_wallet(wallet: Wallet) -> str:
    pnl = wallet.balance - wallet.starting
    sign = "+" if pnl >= 0 else ""
    pnl_col = GREEN if pnl >= 0 else RED
    total = wallet.wins + wallet.losses
    wr = f"{wallet.wins/total*100:.0f}%" if total else "n/a"
    return (
        f"{BOLD}Wallet: ${wallet.balance:,.2f}{RESET}  "
        f"({pnl_col}{sign}${pnl:,.2f}{RESET})"
        f"  deployed=${wallet.deployed:.2f}/{wallet.max_exposure:.2f}"
        f"  W/L {GREEN}{wallet.wins}{RESET}/{RED}{wallet.losses}{RESET} ({wr})"
    )

def _bar(val: float, width: int = 20) -> str:
    filled = round(val * width)
    return "█" * filled + "░" * (width - filled)

def _fmt_market_scan(
    symbol, cid, strike, t_rem, live, up_mid, p_up, edge_up, edge_down, threshold, is_snipe
) -> str:
    mode_tag = f"{YELLOW}[SNIPE mode]{RESET}" if is_snipe else f"{DIM}[ARB mode]{RESET}"
    best = max(edge_up, edge_down)
    colour = GREEN if best >= threshold else DIM
    return (
        f"  {CYAN}{symbol:3s}{RESET}  strike=${strike:<10,.4f}  "
        f"live=${live:<10,.4f}  t_rem={t_rem:4.0f}s  "
        f"p_up={p_up:.3f} [{_bar(p_up,10)}]  "
        f"up_mkt={up_mid:.2f}  "
        f"{colour}edge_up={edge_up:+.3f}  edge_dn={edge_down:+.3f}{RESET}  "
        f"{mode_tag}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    wallet    = Wallet(args.wallet)
    positions: list[Position] = []
    cooldowns: dict[str, float] = {}
    cycle = 0

    # Seed randomness — fixed seed gives reproducible runs, None = different each time
    seed = args.seed if args.seed is not None else random.randint(0, 999_999)
    random.seed(seed)

    # In demo mode, start prices with a random ±2% offset from the base so each
    # run begins at a slightly different level.
    demo_prices = {
        sym: px * random.uniform(0.980, 1.020)
        for sym, px in _BASE_PRICES.items()
    }

    # Per-run randomised initial Polymarket lag: each market starts with a
    # different amount of stale pricing so different edges are visible.
    # current_up_mid is the live stateful mid — updated each cycle via geometric
    # convergence rather than re-computed from initial value each time.
    _run_markets = []
    for tmpl in _DEMO_MARKETS_TEMPLATE:
        lag_bias = random.uniform(-0.06, 0.10)   # negative = already priced in; positive = bigger lag
        init_mid = round(max(0.05, min(0.95, tmpl["up_mid_0"] + lag_bias)), 3)
        _run_markets.append({**tmpl, "up_mid_0": init_mid, "current_up_mid": init_mid})

    cycle_start = time.monotonic()

    # Simulated clock (seconds elapsed in the simulation).
    # Each cycle advances it by SIM_STEP seconds.
    sim_now: float = 0.0
    # In demo mode each cycle represents 1 simulated second — matching real-time
    # resolution and eliminating the blind spot between cycles that caused last-
    # second snipe reversals to be undetectable.  Live mode uses --interval.
    SIM_STEP: float = 1.0 if args.demo else args.interval

    # In demo mode, suppress per-cycle noise and only print events + periodic
    # snapshots every SNAPSHOT_INTERVAL simulated seconds.  Pass --verbose to
    # restore full per-cycle output (useful for debugging a single run).
    verbose: bool = not args.demo or getattr(args, "verbose", False)
    SNAPSHOT_INTERVAL: float = 60.0
    last_snapshot: float = -(SNAPSHOT_INTERVAL + 1.0)   # force first snapshot immediately

    print(f"\n{'═'*70}")
    print(f"  {BOLD}POLYMARKET PAPER-TRADING SIMULATOR{RESET}")
    mode_label = f"{YELLOW}DEMO (offline){RESET}" if args.demo else f"{GREEN}LIVE{RESET}"
    print(f"  Mode: {mode_label}  |  Starting wallet: ${args.wallet:,.2f}  |  seed={seed}")
    print(f"  Strategy: Momentum + ARB + Late-window SNIPER")
    print(f"  Sniper window: final {SNIPE_WINDOW:.0f}s  |  min confidence: {SNIPE_MIN_PROB:.0%}")
    if args.demo:
        print(f"  Sim resolution: 1s/cycle  |  output: events + 60s snapshots (--verbose for full)")
    print(f"{'═'*70}\n")

    while cycle < args.cycles:
        cycle += 1
        sim_now += SIM_STEP
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        elapsed = time.monotonic() - cycle_start

        is_snapshot = (sim_now - last_snapshot) >= SNAPSHOT_INTERVAL
        show_cycle  = verbose or is_snapshot

        if show_cycle:
            print(f"{'─'*70}")
            sim_ts = f"  sim={sim_now:.0f}s" if args.demo else ""
            print(f"  {BOLD}CYCLE {cycle}{RESET}  {now_str}{sim_ts}  (elapsed: {elapsed:.0f}s)")
            print(f"{'─'*70}")

        # ── Fetch / update prices ─────────────────────────────────────────────
        if args.demo:
            # Drift prices using randomised Gaussian moves each cycle.
            # Volatility (std) scales with sqrt(SIM_STEP) — Brownian motion.
            # Mean drift scales linearly so total expected drift is the same
            # regardless of step size.  At SIM_STEP=1 this gives per-second
            # moves; at SIM_STEP=90 (old default) it matched the original model.
            step_scale = math.sqrt(SIM_STEP / 90.0)
            for sym in TARGET_SYMBOLS:
                vol_per_step = _DRIFT_PCT[sym] * 2.5 * step_scale
                drift = random.gauss(vol_per_step * 0.3 * step_scale, vol_per_step)
                demo_prices[sym] = demo_prices[sym] * (1 + drift)
            prices = demo_prices.copy()
        else:
            try:
                prices = live_prices()
            except Exception as e:
                print(f"  {RED}Price fetch failed: {e}{RESET}")
                time.sleep(args.interval)
                continue

        if show_cycle:
            print(f"\n  {BOLD}LIVE PRICES{RESET}")
            for sym, px in sorted(prices.items()):
                print(f"    {CYAN}{sym:3s}{RESET}  ${px:>12,.4f}")

        # ── Fetch / build market list ─────────────────────────────────────────
        if args.demo:
            # Advance each market's t_rem by SIM_STEP (simulated seconds per cycle)
            secs_passed = SIM_STEP
            markets = []
            for tmpl in _run_markets:
                t_rem = max(0.0, tmpl["t_rem_0"] - (cycle - 1) * SIM_STEP)
                sym = tmpl["symbol"]
                p_live = prices.get(sym, tmpl["strike"])
                vol = VOLATILITY.get(sym, DEFAULT_VOL)
                fp = fair_prob_up(p_live, tmpl["strike"], max(t_rem, 5), vol)
                # Stateful geometric convergence: each second the market closes
                # 0.8–1.8% of the remaining gap toward fair value, plus small noise.
                # This is mathematically correct (unlike the old linear accumulation
                # formula) and works at any step size without diverging.
                prev_mid = tmpl["current_up_mid"]
                reprice_speed = random.uniform(0.008, 0.018) * SIM_STEP
                noise = random.gauss(0, 0.001 * math.sqrt(SIM_STEP))
                new_mid = prev_mid + (fp - prev_mid) * reprice_speed + noise
                new_mid = round(max(0.01, min(0.99, new_mid)), 3)
                tmpl["current_up_mid"] = new_mid   # persist for next cycle
                markets.append({**tmpl, "t_rem": t_rem, "up_mid": new_mid})
        else:
            try:
                raw_markets = live_markets()
            except Exception as e:
                print(f"  {RED}Market fetch failed: {e}{RESET}")
                time.sleep(args.interval)
                continue
            markets = []
            skipped_reasons: dict[str, int] = {}
            _UP_LABELS   = {"up", "yes", "higher", "above"}
            _DOWN_LABELS = {"down", "no", "lower", "below"}
            for m in raw_markets:
                q = m.get("question", m.get("title", ""))
                sym = parse_symbol(q)
                if sym not in prices:
                    skipped_reasons["no_symbol"] = skipped_reasons.get("no_symbol", 0) + 1
                    continue

                cid = m.get("conditionId", m.get("condition_id", q))

                # Strike: try dollar amount in text first; fall back to the
                # Coinbase opening price at startDate (used by "X Up or Down - 15 min"
                # markets where the strike IS the price when the window opened).
                strike = parse_strike(q, m.get("description", ""))
                if strike is None:
                    if cid in _strike_cache:
                        strike = _strike_cache[cid]
                    else:
                        strike = live_open_price(sym, m.get("startDate", ""))
                        if strike is None:
                            # Last resort: current live price (less accurate but
                            # lets the bot trade rather than dropping the market)
                            strike = prices.get(sym)
                        if strike is not None:
                            _strike_cache[cid] = strike
                if strike is None:
                    skipped_reasons["no_strike"] = skipped_reasons.get("no_strike", 0) + 1
                    continue

                t_rem = secs_until(m.get("endDate", m.get("end_date_iso", "")))

                # Token IDs: Gamma API returns either a tokens[] array OR
                # clobTokenIds + outcomes as JSON-encoded strings.
                tokens = m.get("tokens", [])
                up_tok = next((t["token_id"] for t in tokens
                               if t.get("outcome", "").lower() in _UP_LABELS), None)
                dn_tok = next((t["token_id"] for t in tokens
                               if t.get("outcome", "").lower() in _DOWN_LABELS), None)

                if not up_tok or not dn_tok:
                    # Fallback: parse clobTokenIds + outcomes (alternate Gamma format)
                    try:
                        clob_raw = m.get("clobTokenIds", "[]")
                        out_raw  = m.get("outcomes", "[]")
                        clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
                        outcomes_list = json.loads(out_raw) if isinstance(out_raw, str) else out_raw
                        for tid, outcome in zip(clob_ids, outcomes_list):
                            ol = outcome.lower()
                            if ol in _UP_LABELS:
                                up_tok = tid
                            elif ol in _DOWN_LABELS:
                                dn_tok = tid
                    except Exception:
                        pass

                if not up_tok or not dn_tok:
                    skipped_reasons["no_tokens"] = skipped_reasons.get("no_tokens", 0) + 1
                    all_outcomes = ([t.get("outcome","?") for t in tokens]
                                    or [m.get("outcomes","?")])
                    print(f"  {DIM}[debug] no Up/Down tokens {all_outcomes} in: {q[:60]}{RESET}")
                    continue

                # Midpoint: prefer CLOB live_midpoint; fall back to outcomePrices
                up_mid = live_midpoint(up_tok)
                if up_mid is None:
                    try:
                        px_raw = m.get("outcomePrices", "[]")
                        out_raw = m.get("outcomes", "[]")
                        px_list  = json.loads(px_raw) if isinstance(px_raw, str) else px_raw
                        out_list = json.loads(out_raw) if isinstance(out_raw, str) else out_raw
                        for outcome, px in zip(out_list, px_list):
                            if outcome.lower() in _UP_LABELS:
                                up_mid = float(px)
                    except Exception:
                        pass
                if up_mid is None:
                    up_mid = 0.5

                markets.append({
                    "cid": cid,
                    "question": q, "symbol": sym, "strike": strike,
                    "t_rem": t_rem, "up_mid": up_mid,
                    "up_token": up_tok, "down_token": dn_tok,
                })
            if skipped_reasons:
                print(f"  {YELLOW}[debug] dropped {sum(skipped_reasons.values())} markets: {skipped_reasons}{RESET}")

        # ── Settle expired positions ──────────────────────────────────────────
        # In demo mode use sim_now; in live mode use real time.
        _now = sim_now if args.demo else time.monotonic()
        expired = [p for p in positions if p.won is None and p.end_secs <= _now]
        if expired:
            print(f"\n  {BOLD}SETTLING EXPIRED POSITIONS{RESET}")
        for pos in expired:
            # Skip positions that were already closed by early exit
            if pos.won is not None:
                continue
            final_price = prices.get(pos.symbol)
            if final_price is None:
                continue
            if pos.bet == "Up":
                won = final_price > pos.strike
            else:
                won = final_price < pos.strike
            wallet.settle(pos, won, final_price)
            tag = f"{GREEN}WIN {RESET}" if won else f"{RED}LOSS{RESET}"
            print(
                f"  [{tag}]  {CYAN}{pos.symbol}{RESET}  {pos.mode}  "
                f"bet={pos.bet}  strike=${pos.strike:,.4f}  "
                f"final=${final_price:,.4f}  "
                f"staked=${pos.order_usdc:.2f}  "
                f"pnl={GREEN if pos.pnl>=0 else RED}{pos.pnl:+.2f}{RESET}"
            )

        # ── Early-exit scan (fair-value alignment + flat take-profit) ─────────
        # Build a quick lookup: cid → current market data for this cycle
        _mkt_lookup = {m["cid"]: m for m in markets}
        early_exits_this_cycle = []

        for pos in [p for p in positions if p.won is None]:
            m = _mkt_lookup.get(pos.cid)
            if m is None:
                continue

            t_rem_pos = m["t_rem"]
            if t_rem_pos < EXIT_MIN_T_REM:
                continue
            # Snipe positions near the end are near-locks — let them resolve
            if pos.is_snipe and t_rem_pos < SNIPE_WINDOW:
                continue

            current_up_mid = m["up_mid"]
            current_mid = current_up_mid if pos.bet == "Up" else (1.0 - current_up_mid)

            # Pre-compute current fair probability (needed for guard + triggers).
            live_px = prices.get(pos.symbol)
            current_fair = None
            if live_px is not None:
                _vol = VOLATILITY.get(pos.symbol, DEFAULT_VOL)
                _p_up = fair_prob_up(live_px, pos.strike, max(t_rem_pos, 1), _vol)
                current_fair = _p_up if pos.bet == "Up" else (1.0 - _p_up)

            # Guard: only sell at a loss when conviction is genuinely gone.
            # If net sell (mid − slippage) is below entry AND the model still
            # gives ≥ ARB_MIN_PROB, hold through the dip.
            # If fair_prob has dropped below ARB_MIN_PROB the trade thesis is
            # broken — allow a stop-loss exit even at a loss.
            effective_sell = current_mid - SLIPPAGE
            if effective_sell <= pos.entry_price:
                if current_fair is None or current_fair >= ARB_MIN_PROB:
                    continue  # still believe in trade — hold through the dip
                # conviction gone → fall through to stop-loss trigger

            should_exit = False
            exit_type   = ""

            # Trigger 1: fair-value alignment (profit capture)
            if EXIT_RESIDUAL_EDGE > 0.0 and current_fair is not None:
                residual = current_fair - current_mid
                if residual <= EXIT_RESIDUAL_EDGE:
                    should_exit = True
                    exit_type   = f"fair-val(fair={current_fair:.3f} mkt={current_mid:.3f})"

            # Trigger 1b: stop-loss — conviction dropped below entry threshold
            if not should_exit and current_fair is not None:
                if current_fair < ARB_MIN_PROB:
                    should_exit = True
                    exit_type   = f"stop-loss(fair={current_fair:.3f}<{ARB_MIN_PROB})"

            # Trigger 2: flat take-profit fallback
            if not should_exit and EXIT_TAKE_PROFIT > 0.0:
                if (current_mid - pos.entry_price) >= EXIT_TAKE_PROFIT:
                    should_exit = True
                    exit_type   = f"flat-tp(+{current_mid - pos.entry_price:.3f})"

            if not should_exit:
                continue

            sell_price = round(max(current_mid - SLIPPAGE, 0.01), 4)
            wallet.early_exit(pos, sell_price, exit_type)
            early_exits_this_cycle.append((pos, sell_price, exit_type))

        if early_exits_this_cycle:
            print(f"\n  {BOLD}EARLY EXITS ({len(early_exits_this_cycle)}){RESET}")
            for pos, sp, xt in early_exits_this_cycle:
                tag = f"{GREEN}PROFIT{RESET}" if pos.pnl >= 0 else f"{RED}LOSS{RESET}"
                print(
                    f"  [{tag}]  {CYAN}{pos.symbol}{RESET}  {pos.mode}  "
                    f"bet={pos.bet}  entry={pos.entry_price:.4f}  sell={sp:.4f}  "
                    f"shares={pos.shares:.2f}  pnl={GREEN if pos.pnl>=0 else RED}{pos.pnl:+.2f}{RESET}"
                    f"  [{xt}]"
                )

        # ── Scan markets ─────────────────────────────────────────────────────
        if show_cycle:
            print(f"\n  {BOLD}MARKET SCAN  ({len(markets)} active 15-min markets){RESET}")
        fired_this_cycle = 0

        for m in markets:
            sym      = m["symbol"]
            cid      = m["cid"]
            strike   = m["strike"]
            t_rem    = m["t_rem"]
            up_mid   = m["up_mid"]
            live_px  = prices.get(sym)

            if live_px is None or t_rem <= 0 or t_rem > 930:
                continue

            vol    = VOLATILITY.get(sym, DEFAULT_VOL)
            p_up   = fair_prob_up(live_px, strike, t_rem, vol)
            p_down = 1.0 - p_up
            edge_u = p_up   - up_mid
            edge_d = p_down - (1.0 - up_mid)

            is_snipe  = t_rem < SNIPE_WINDOW
            threshold = SNIPE_EDGE if is_snipe else ARB_EDGE

            if show_cycle:
                print(_fmt_market_scan(
                    sym, cid, strike, t_rem, live_px,
                    up_mid, p_up, edge_u, edge_d, threshold, is_snipe
                ))

            # Already have an open position on this market?
            already_open = any(
                p.cid == cid and p.won is None for p in positions
            )
            if already_open:
                if show_cycle:
                    print(f"      {DIM}↳ position already open — skip{RESET}")
                continue

            trade = evaluate(sym, cid, strike, t_rem, live_px, up_mid, wallet, cooldowns, sim_now)

            if trade is None:
                if show_cycle:
                    best = max(edge_u, edge_d)
                    best_fair = max(
                        p_up if edge_u >= edge_d else 0.0,
                        p_down if edge_d > edge_u else 0.0
                    )
                    if best < threshold:
                        reason = "no edge"
                    elif is_snipe and max(p_up, p_down) < SNIPE_MIN_PROB:
                        reason = f"confidence gate (fair={max(p_up,p_down):.2f}<{SNIPE_MIN_PROB})"
                    elif is_snipe and abs(live_px - strike) / strike < SNIPE_MIN_PRICE_DIST_PCT:
                        reason = f"too close to strike ({abs(live_px-strike)/strike:.1%}<{SNIPE_MIN_PRICE_DIST_PCT:.1%})"
                    elif not is_snipe and best_fair < ARB_MIN_PROB:
                        reason = f"low conviction (fair={best_fair:.2f}<{ARB_MIN_PROB})"
                    else:
                        reason = "no edge on near-certain side"
                    print(f"      {DIM}↳ no trade ({reason}){RESET}")
                continue

            # ── PAPER ORDER — always printed regardless of verbosity ──────────
            tag = f"{YELLOW}★ SNIPE{RESET}" if trade["is_snipe"] else f"{CYAN}◆ ARB{RESET}"
            print(
                f"\n  [sim={sim_now:.0f}s]  {tag}  {BOLD}BUY {trade['bet']}{RESET}"
                f"  {CYAN}{sym}{RESET}  strike=${strike:,.4f}"
                f"  @ {trade['limit_price']:.4f}  "
                f"shares={trade['shares']:.2f}  cost=${trade['order_usdc']:.2f}  "
                f"fair={trade['fair_prob']:.3f}  edge={trade['edge']:+.3f}  "
                f"t_rem={t_rem:.0f}s"
            )

            # Record position — use sim_now for demo, real time for live
            _now_for_expiry = sim_now if args.demo else time.monotonic()
            expire_at = _now_for_expiry + t_rem
            pos = Position(
                cid=cid,
                symbol=sym,
                bet=trade["bet"],
                fair_prob=trade["fair_prob"],
                entry_price=trade["limit_price"],
                order_usdc=trade["order_usdc"],
                shares=trade["shares"],
                strike=strike,
                end_secs=expire_at,
                is_snipe=trade["is_snipe"],
                mode=trade["mode"],
                live_price_at_entry=live_px,
            )
            positions.append(pos)
            wallet.open_order(cid, trade["order_usdc"])
            cooldowns[cid] = sim_now
            fired_this_cycle += 1

        # ── Open positions summary + wallet ──────────────────────────────────
        open_pos = [p for p in positions if p.won is None]
        had_activity = bool(expired or early_exits_this_cycle or fired_this_cycle)

        if show_cycle or had_activity:
            if open_pos:
                print(f"\n  {BOLD}OPEN POSITIONS ({len(open_pos)}){RESET}")
                for p in open_pos:
                    _now_disp = sim_now if args.demo else time.monotonic()
                    t_left = max(0.0, p.end_secs - _now_disp)
                    print(
                        f"    {CYAN}{p.symbol:3s}{RESET}  {p.mode:5s}  "
                        f"bet={p.bet:4s}  strike=${p.strike:,.4f}  "
                        f"entry={p.entry_price:.4f}  "
                        f"cost=${p.order_usdc:.2f}  "
                        f"t_left={t_left:.0f}s"
                    )
            print(f"\n  {_fmt_wallet(wallet)}\n")
            if is_snapshot:
                last_snapshot = sim_now

        # ── Auto-stop in demo mode when all markets have expired and no
        #    positions remain open (avoids running 999 empty cycles) ──────────
        if args.demo and all(m["t_rem"] <= 0 for m in markets) and not open_pos:
            break

        if not args.demo and cycle < args.cycles:
            time.sleep(args.interval)

    # ── Final summary ─────────────────────────────────────────────────────────
    # Force-settle remaining open positions using last known prices
    print(f"\n{'═'*70}")
    print(f"  {BOLD}FINAL SUMMARY{RESET}")
    print(f"{'═'*70}")
    still_open = [p for p in positions if p.won is None]
    if still_open:
        print(f"\n  {YELLOW}{len(still_open)} positions still open (unsettled — window not yet closed){RESET}")
        for p in still_open:
            _now_final = sim_now if args.demo else time.monotonic()
            t_left = max(0.0, p.end_secs - _now_final)
            print(f"    {p.symbol} {p.bet} @ {p.entry_price:.4f}  t_left={t_left:.0f}s")

    settled = [p for p in positions if p.won is not None]
    if settled:
        total_staked = sum(p.order_usdc for p in settled)
        total_pnl    = sum(p.pnl for p in settled)
        wins   = [p for p in settled if p.won]
        losses = [p for p in settled if not p.won]
        snipes = [p for p in settled if p.is_snipe]
        arbs   = [p for p in settled if not p.is_snipe]
        early  = [p for p in settled if p.exit_type not in ("resolution", "")]
        print(f"\n  Settled trades:     {len(settled)}")
        print(f"  Win / Loss:         {GREEN}{len(wins)}{RESET} / {RED}{len(losses)}{RESET}"
              + (f"  ({len(wins)/len(settled)*100:.0f}% win rate)" if settled else ""))
        print(f"  Early exits:        {len(early)}  "
              + f"(fair-val: {sum(1 for p in early if 'fair-val' in p.exit_type)}  "
              + f"flat-tp: {sum(1 for p in early if 'flat-tp' in p.exit_type)})")
        print(f"  SNIPE trades:       {len(snipes)}  wins={sum(1 for p in snipes if p.won)}")
        print(f"  ARB trades:         {len(arbs)}  wins={sum(1 for p in arbs if p.won)}")
        print(f"  Total staked:       ${total_staked:.2f}")
        print(f"  Total PnL:          {GREEN if total_pnl>=0 else RED}{total_pnl:+.2f}{RESET}")
        if total_staked:
            print(f"  ROI on staked:      {total_pnl/total_staked*100:+.1f}%")

    total_value = wallet.balance + wallet.deployed
    settled_pnl = total_value - wallet.starting
    pnl_col = GREEN if settled_pnl >= 0 else RED
    print(f"\n  Starting balance:   ${wallet.starting:,.2f}")
    print(f"  Liquid cash:        ${wallet.balance:,.2f}")
    print(f"  In open positions:  ${wallet.deployed:,.2f}")
    print(f"  Total portfolio:    ${total_value:,.2f}  ({pnl_col}{settled_pnl:+,.2f}{RESET})")
    print(f"{'═'*70}\n")

    return total_value   # caller can chain ending balance into next run


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polymarket paper-trading simulator"
    )
    parser.add_argument("--demo",     action="store_true", help="Offline demo mode")
    parser.add_argument("--cycles",   type=int,   default=999, help="Number of scan cycles")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between cycles")
    parser.add_argument("--wallet",   type=float, default=1000.0, help="Starting USDC")
    parser.add_argument("--runs",     type=int,   default=1,
                        help="Number of consecutive runs; each run starts with the previous run's ending balance")
    parser.add_argument("--seed",     type=int,   default=None,
                        help="Random seed for reproducibility (default: random each run)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print full per-cycle market scan in demo mode (default: events-only)")
    args = parser.parse_args()

    try:
        balance = args.wallet
        for run_num in range(1, args.runs + 1):
            if args.runs > 1:
                print(f"\n{'▓'*70}")
                print(f"  RUN {run_num} of {args.runs}  —  starting balance: ${balance:,.2f}")
                print(f"{'▓'*70}")
            run_args = argparse.Namespace(**vars(args))
            run_args.wallet = balance
            # Each run gets a fresh random seed (unless a fixed seed was given)
            run_args.seed = args.seed  # None = pick new seed each run inside run()
            ending = run(run_args)
            balance = ending if ending is not None else balance

        if args.runs > 1:
            print(f"\n{'▓'*70}")
            pnl = balance - args.wallet
            pnl_col = GREEN if pnl >= 0 else RED
            print(f"  {BOLD}ALL {args.runs} RUNS COMPLETE{RESET}")
            print(f"  Starting wallet:  ${args.wallet:,.2f}")
            print(f"  Final balance:    ${balance:,.2f}  ({pnl_col}{pnl:+,.2f}  {pnl/args.wallet*100:+.1f}%{RESET})")
            print(f"{'▓'*70}\n")
    except KeyboardInterrupt:
        print("\n\n  [Ctrl+C] — Simulator stopped.")


if __name__ == "__main__":
    main()
