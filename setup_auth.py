#!/usr/bin/env python3
"""
Polymarket credential setup helper.

Run this once to derive your CLOB API key from your Ethereum private key
and write the result to .env.

Usage:
    python setup_auth.py              # reads PK from .env or prompts
    python setup_auth.py --pk 0x...   # supply PK directly (careful with shell history)
    python setup_auth.py --verify     # verify existing credentials without re-deriving

What this script does:
  1. Takes your Ethereum wallet private key (your Polymarket account key)
  2. Signs an EIP-712 message against Polymarket's CLOB (L1 auth) to derive
     a scoped API key + secret + passphrase (L2 credentials)
  3. Verifies the credentials by fetching your USDC balance
  4. Appends / updates the values in your .env file

How to get your private key
---------------------------
Polymarket uses Polygon wallets (same as any EVM chain):

  MetaMask  → Account menu → Account Details → Export private key
  Privy / embedded wallet → see polymarket.com Account → Settings
  Magic / social login → export from your provider's dashboard

Keep your private key secret. This script never sends it anywhere except
the signed EIP-712 challenge to api.polymarket.com — the key itself stays
on your machine.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys

from dotenv import dotenv_values, load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


def _upsert_env(key: str, value: str) -> None:
    """Write or update a key=value line in .env."""
    env_path = ENV_FILE

    if os.path.exists(env_path):
        content = open(env_path).read()
    else:
        content = ""

    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    line = f"{key}={value}"
    if pattern.search(content):
        content = pattern.sub(line, content)
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += line + "\n"

    with open(env_path, "w") as fh:
        fh.write(content)


def _read_pk(args_pk: str | None) -> str:
    """Get the private key from CLI arg, .env, or interactive prompt."""
    # Priority: --pk flag → .env PK → interactive
    pk = args_pk or os.environ.get("PK", "").strip()
    if not pk:
        print("\nEnter your Ethereum private key (input hidden).")
        print("Tip: in MetaMask → Account Details → Export private key")
        pk = getpass.getpass("Private key (hex, with or without 0x): ").strip()

    # Normalize — py-clob-client expects no 0x prefix
    if pk.startswith("0x") or pk.startswith("0X"):
        pk = pk[2:]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", pk):
        print("ERROR: private key must be 64 hex characters.")
        sys.exit(1)
    return pk


def _build_client(pk: str, api_key: str = "", secret: str = "", passphrase: str = ""):
    """Build a py-clob-client ClobClient, optionally with existing creds."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
    except ImportError:
        print("ERROR: py-clob-client not installed. Run: pip install py-clob-client")
        sys.exit(1)

    creds = None
    if api_key:
        from py_clob_client.clob_types import ApiCreds
        creds = ApiCreds(
            api_key=api_key,
            api_secret=secret,
            api_passphrase=passphrase,
        )

    return ClobClient(
        host=CLOB_HOST,
        chain_id=POLYGON,
        key=pk,
        creds=creds,
        signature_type=1,
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def derive(pk: str) -> None:
    """Derive API credentials from the private key and save to .env."""
    print("\n[1/3] Connecting to Polymarket CLOB …")
    client = _build_client(pk)

    print("[2/3] Deriving API key (signing EIP-712 challenge) …")
    try:
        creds = client.create_or_derive_api_creds()
    except Exception as exc:
        print(f"ERROR deriving credentials: {exc}")
        sys.exit(1)

    api_key = creds.api_key
    secret   = creds.api_secret
    phrase   = creds.api_passphrase

    print("[3/3] Verifying credentials …")
    client.set_api_creds(creds)
    balance = _get_balance(client)

    print(f"\n✓ Auth successful!")
    print(f"  Wallet address : {_get_address(pk)}")
    print(f"  USDC balance   : ${balance:.4f}")
    print(f"  API key        : {api_key[:8]}…")

    # Write to .env
    _upsert_env("PK", pk)
    _upsert_env("CLOB_API_KEY", api_key)
    _upsert_env("CLOB_SECRET", secret)
    _upsert_env("CLOB_PASS_PHRASE", phrase)

    print(f"\n✓ Written to {ENV_FILE}")
    print(f"\nYou can now run:")
    print(f"  python bot.py --paper    # paper trading (no real bets)")
    print(f"  python bot.py            # live trading (real bets)")
    print(f"\nTo switch paper/live mode edit PAPER_TRADE in .env")


def verify() -> None:
    """Verify existing credentials from .env without re-deriving."""
    env = dotenv_values(ENV_FILE) if os.path.exists(ENV_FILE) else {}
    pk       = env.get("PK", os.environ.get("PK", ""))
    api_key  = env.get("CLOB_API_KEY", os.environ.get("CLOB_API_KEY", ""))
    secret   = env.get("CLOB_SECRET", os.environ.get("CLOB_SECRET", ""))
    phrase   = env.get("CLOB_PASS_PHRASE", os.environ.get("CLOB_PASS_PHRASE", ""))

    if not pk:
        print("ERROR: PK not found in .env. Run 'python setup_auth.py' first.")
        sys.exit(1)
    if pk.startswith("0x"):
        pk = pk[2:]

    client = _build_client(pk, api_key, secret, phrase)
    if not api_key:
        print("No API key in .env — deriving first …")
        try:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    balance = _get_balance(client)
    address = _get_address(pk)

    print(f"\n✓ Credentials valid")
    print(f"  Address   : {address}")
    print(f"  API key   : {api_key[:8] if api_key else '(derived)'}…")
    print(f"  Balance   : ${balance:.4f} USDC")
    print(f"  Paper mode: {env.get('PAPER_TRADE', 'false')}")


def _get_balance(client) -> float:
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        resp = client.get_balance_allowance(params)
        return float(resp.get("balance", 0)) / 1_000_000
    except Exception:
        return 0.0


def _get_address(pk_hex: str) -> str:
    try:
        from eth_account import Account
        return Account.from_key(pk_hex).address
    except Exception:
        return "(unknown)"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive or verify Polymarket CLOB credentials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pk",
        metavar="HEX",
        help="Private key (hex). Avoid this in shared environments — omit to be prompted.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing credentials in .env; do not re-derive.",
    )
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        pk = _read_pk(args.pk)
        derive(pk)


if __name__ == "__main__":
    main()
