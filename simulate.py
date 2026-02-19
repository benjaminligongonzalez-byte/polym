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
BINANCE_API  = "https://api.binance.com"

MARKET_KEYWORDS = [
    "up or down - 15",
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
SNIPE_WINDOW   = 300.0     # seconds — activate in final 5 min
SNIPE_MIN_PROB = 0.75      # minimum fair probability in sniper mode
SNIPE_EDGE     = 0.03      # lower edge bar (near-certain outcome)
SNIPE_KELLY    = 0.50      # half-Kelly on near-locks
ARB_KELLY      = 0.25      # quarter-Kelly for standard arb

MAX_ORDER_FRAC   = 0.05    # 5% of wallet per order
MAX_EXPOSURE_FRAC= 0.20    # 20% of wallet total
MAX_ORDER_HARD   = 50.0    # hard cap per single order
MIN_ORDER        = 2.0
SLIPPAGE         = 0.03
COOLDOWN         = 30.0

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

def live_prices() -> dict[str, float]:
    import requests
    syms = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
    smap = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "XRPUSDT": "XRP", "SOLUSDT": "SOL"}
    resp = requests.get(
        f"{BINANCE_API}/api/v3/ticker/price",
        params={"symbols": json.dumps(syms)},
        timeout=5,
    )
    resp.raise_for_status()
    return {smap[r["symbol"]]: float(r["price"]) for r in resp.json() if r["symbol"] in smap}

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
    return [m for m in mkts if is_15min(m.get("question", m.get("title", "")))]

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
    end_secs: float     # time.monotonic() when market expires
    is_snipe: bool
    mode: str
    live_price_at_entry: float
    won: Optional[bool] = None
    pnl: float = 0.0

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
        if won:
            pnl = pos.shares - pos.order_usdc   # net profit (shares × $1 − cost)
            self.balance += pos.shares           # collect $1 per winning share
            pos.pnl = pnl
            self.wins += 1
        else:
            pos.pnl = -pos.order_usdc            # total loss (already deducted at open)
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

    if edge_up >= edge_down:
        bet, token_side = "Up",   "up"
        fair, mkt_p, edge = p_up,   up_mid,   edge_up
    else:
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

    # In demo mode, prices drift each cycle
    demo_prices = dict(_BASE_PRICES)
    cycle_start = time.monotonic()

    # Simulated clock (seconds elapsed in the simulation).
    # Each cycle advances it by sim_step seconds.
    # This lets positions expire properly within the demo without
    # waiting for real wall-clock time.
    sim_now: float = 0.0
    # In demo mode each "cycle" represents 90 simulated seconds of trading
    SIM_STEP: float = 90.0 if args.demo else args.interval

    print(f"\n{'═'*70}")
    print(f"  {BOLD}POLYMARKET PAPER-TRADING SIMULATOR{RESET}")
    mode_label = f"{YELLOW}DEMO (offline){RESET}" if args.demo else f"{GREEN}LIVE{RESET}"
    print(f"  Mode: {mode_label}  |  Starting wallet: ${args.wallet:,.2f}")
    print(f"  Strategy: Momentum + ARB + Late-window SNIPER")
    print(f"  Sniper window: final {SNIPE_WINDOW:.0f}s  |  min confidence: {SNIPE_MIN_PROB:.0%}")
    print(f"{'═'*70}\n")

    while cycle < args.cycles:
        cycle += 1
        sim_now += SIM_STEP
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        elapsed = time.monotonic() - cycle_start

        print(f"{'─'*70}")
        print(f"  {BOLD}CYCLE {cycle}{RESET}  {now_str}  (elapsed: {elapsed:.0f}s)")
        print(f"{'─'*70}")

        # ── Fetch / update prices ─────────────────────────────────────────────
        if args.demo:
            # Drift prices slightly to simulate market movement
            for sym in TARGET_SYMBOLS:
                drift_dir = 1 if (cycle % 3 != 0) else -1  # mostly up, sometimes down
                drift = _DRIFT_PCT[sym] * drift_dir * (1 + (cycle % 2) * 0.5)
                demo_prices[sym] = demo_prices[sym] * (1 + drift)
            prices = demo_prices.copy()
        else:
            try:
                prices = live_prices()
            except Exception as e:
                print(f"  {RED}Price fetch failed: {e}{RESET}")
                time.sleep(args.interval)
                continue

        print(f"\n  {BOLD}LIVE PRICES{RESET}")
        for sym, px in sorted(prices.items()):
            print(f"    {CYAN}{sym:3s}{RESET}  ${px:>12,.4f}")

        # ── Fetch / build market list ─────────────────────────────────────────
        if args.demo:
            # Advance each market's t_rem by one cycle interval
            secs_passed = args.interval
            markets = []
            for tmpl in _DEMO_MARKETS_TEMPLATE:
                t_rem = max(0.0, tmpl["t_rem_0"] - (cycle - 1) * secs_passed)
                # Polymarket mid drifts toward fair value slowly
                sym = tmpl["symbol"]
                p_live = prices.get(sym, tmpl["strike"])
                vol = VOLATILITY.get(sym, DEFAULT_VOL)
                fp = fair_prob_up(p_live, tmpl["strike"], max(t_rem, 5), vol)
                # Market maker closes ~40% of the gap per cycle
                current_up_mid = tmpl["up_mid_0"] + (fp - tmpl["up_mid_0"]) * 0.40 * max(cycle-1,0)
                current_up_mid = round(max(0.01, min(0.99, current_up_mid)), 3)
                markets.append({**tmpl, "t_rem": t_rem, "up_mid": current_up_mid})
        else:
            try:
                raw_markets = live_markets()
            except Exception as e:
                print(f"  {RED}Market fetch failed: {e}{RESET}")
                time.sleep(args.interval)
                continue
            markets = []
            for m in raw_markets:
                q = m.get("question", m.get("title", ""))
                sym = parse_symbol(q)
                if sym not in prices:
                    continue
                strike = parse_strike(q, m.get("description", ""))
                if strike is None:
                    continue
                t_rem = secs_until(m.get("endDate", m.get("end_date_iso", "")))
                tokens = m.get("tokens", [])
                up_tok = next((t["token_id"] for t in tokens
                               if t.get("outcome", "").lower() in ("up","yes")), None)
                dn_tok = next((t["token_id"] for t in tokens
                               if t.get("outcome", "").lower() in ("down","no")), None)
                if not up_tok or not dn_tok:
                    continue
                up_mid = live_midpoint(up_tok) or 0.5
                markets.append({
                    "cid": m.get("conditionId", m.get("condition_id", q)),
                    "question": q, "symbol": sym, "strike": strike,
                    "t_rem": t_rem, "up_mid": up_mid,
                    "up_token": up_tok, "down_token": dn_tok,
                })

        # ── Settle expired positions ──────────────────────────────────────────
        # In demo mode use sim_now; in live mode use real time.
        _now = sim_now if args.demo else time.monotonic()
        expired = [p for p in positions if p.won is None and p.end_secs <= _now]
        if expired:
            print(f"\n  {BOLD}SETTLING EXPIRED POSITIONS{RESET}")
        for pos in expired:
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

        # ── Scan markets ─────────────────────────────────────────────────────
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

            print(_fmt_market_scan(
                sym, cid, strike, t_rem, live_px,
                up_mid, p_up, edge_u, edge_d, threshold, is_snipe
            ))

            # Already have an open position on this market?
            already_open = any(
                p.cid == cid and p.won is None for p in positions
            )
            if already_open:
                print(f"      {DIM}↳ position already open — skip{RESET}")
                continue

            trade = evaluate(sym, cid, strike, t_rem, live_px, up_mid, wallet, cooldowns, sim_now)

            if trade is None:
                reason = "no edge" if max(edge_u, edge_d) < threshold else "confidence gate"
                print(f"      {DIM}↳ no trade ({reason}){RESET}")
                continue

            # ── PAPER ORDER ──────────────────────────────────────────────────
            tag = f"{YELLOW}★ SNIPE{RESET}" if trade["is_snipe"] else f"{CYAN}◆ ARB{RESET}"
            print(
                f"      {tag}  {BOLD}BUY {trade['bet']}{RESET}"
                f"  @ {trade['limit_price']:.4f}  "
                f"shares={trade['shares']:.2f}  cost=${trade['order_usdc']:.2f}  "
                f"fair={trade['fair_prob']:.3f}  edge={trade['edge']:+.3f}  "
                f"kelly={trade['kelly_f']:.3f}×{trade['kelly_mult']}"
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

        # ── Open positions summary ────────────────────────────────────────────
        open_pos = [p for p in positions if p.won is None]
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

        # ── Wallet summary ────────────────────────────────────────────────────
        print(f"\n  {_fmt_wallet(wallet)}\n")

        if cycle < args.cycles:
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
        wins = [p for p in settled if p.won]
        losses = [p for p in settled if not p.won]
        snipes = [p for p in settled if p.is_snipe]
        arbs   = [p for p in settled if not p.is_snipe]
        print(f"\n  Settled trades:     {len(settled)}")
        print(f"  Win / Loss:         {GREEN}{len(wins)}{RESET} / {RED}{len(losses)}{RESET}"
              f"  ({len(wins)/len(settled)*100:.0f}% win rate)" if settled else "")
        print(f"  SNIPE trades:       {len(snipes)}  wins={sum(1 for p in snipes if p.won)}")
        print(f"  ARB trades:         {len(arbs)}  wins={sum(1 for p in arbs if p.won)}")
        print(f"  Total staked:       ${total_staked:.2f}")
        print(f"  Total PnL:          {GREEN if total_pnl>=0 else RED}{total_pnl:+.2f}{RESET}")
        print(f"  ROI on staked:      {total_pnl/total_staked*100:+.1f}%" if total_staked else "")

    total_value = wallet.balance + wallet.deployed
    settled_pnl = total_value - wallet.starting
    pnl_col = GREEN if settled_pnl >= 0 else RED
    print(f"\n  Starting balance:   ${wallet.starting:,.2f}")
    print(f"  Liquid cash:        ${wallet.balance:,.2f}")
    print(f"  In open positions:  ${wallet.deployed:,.2f}")
    print(f"  Total portfolio:    ${total_value:,.2f}  ({pnl_col}{settled_pnl:+,.2f}{RESET})")
    print(f"{'═'*70}\n")


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
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("\n\n  [Ctrl+C] — Simulator stopped.")


if __name__ == "__main__":
    main()
