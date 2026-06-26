"""Report realized + unrealized P&L from Binance futures. Read-only.

Pulls the authoritative figures straight from Binance's income ledger
(REALIZED_PNL / COMMISSION / FUNDING_FEE) rather than the bot's own
log estimates, then adds the current open positions' unrealized P&L.

Usage:
    .venv/bin/python -m scripts.realized_pnl                 # since first deploy
    .venv/bin/python scripts/realized_pnl.py --since 2026-06-01
    .venv/bin/python scripts/realized_pnl.py --since 2026-06-01 --until 2026-06-15
"""

from __future__ import annotations

import argparse
import os
import time

import ccxt
from dotenv import load_dotenv

# App went live on the server on this date; default window start.
DEFAULT_SINCE = "2026-05-31"


def _client() -> ccxt.binance:
    load_dotenv()
    key = os.environ.get("BINANCE_API_KEY")
    secret = os.environ.get("BINANCE_API_SECRET")
    if not key or not secret:
        raise SystemExit("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")
    return ccxt.binance(
        {"apiKey": key, "secret": secret, "options": {"defaultType": "future"}}
    )


def fetch_income(ex: ccxt.binance, since_ms: int, until_ms: int | None) -> dict[str, float]:
    """Sum income by type over the window, paging through the 1000-row limit."""
    agg: dict[str, float] = {}
    fills = 0
    cur = since_ms
    while True:
        params = {"startTime": cur, "limit": 1000}
        if until_ms is not None:
            params["endTime"] = until_ms
        rows = ex.fapiPrivateGetIncome(params)
        if not rows:
            break
        for r in rows:
            t = r["incomeType"]
            agg[t] = agg.get(t, 0.0) + float(r["income"])
            if t == "REALIZED_PNL":
                fills += 1
        nxt = int(rows[-1]["time"]) + 1
        if nxt <= cur or len(rows) < 1000:
            break
        cur = nxt
        time.sleep(0.2)
    agg["_fills"] = fills
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default=DEFAULT_SINCE, help="start date YYYY-MM-DD (UTC)")
    ap.add_argument("--until", default=None, help="end date YYYY-MM-DD (UTC), default now")
    args = ap.parse_args()

    ex = _client()
    since_ms = ex.parse8601(f"{args.since}T00:00:00Z")
    until_ms = ex.parse8601(f"{args.until}T00:00:00Z") if args.until else None

    agg = fetch_income(ex, since_ms, until_ms)
    pnl = agg.get("REALIZED_PNL", 0.0)
    comm = agg.get("COMMISSION", 0.0)
    fund = agg.get("FUNDING_FEE", 0.0)
    fills = int(agg.get("_fills", 0))
    net_realized = pnl + comm + fund

    window = f"since {args.since}" + (f" until {args.until}" if args.until else "")
    print(f"=== Realized P&L ({window}) ===")
    print(f"Realized PnL : {pnl:+.2f} USDT  ({fills} fills)")
    print(f"Commission   : {comm:+.2f} USDT")
    print(f"Funding      : {fund:+.2f} USDT")
    other = {
        k: v
        for k, v in agg.items()
        if k not in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE", "_fills")
    }
    if other:
        print("Other        :", {k: round(v, 2) for k, v in other.items()})
    print(f"NET realized : {net_realized:+.2f} USDT")

    # Open positions (unrealized) — only meaningful for an open-ended window.
    upnl_total = 0.0
    lines = []
    for p in ex.fetch_positions():
        amt = float(p["info"]["positionAmt"])
        if amt == 0:
            continue
        upnl = float(p["info"]["unRealizedProfit"])
        upnl_total += upnl
        side = "SHORT" if amt < 0 else "LONG"
        lines.append(
            f"  {p['symbol']:14} {side} qty={amt} entry={p['entryPrice']} "
            f"mark={p['markPrice']} uPnL={upnl:+.2f}"
        )
    if lines:
        print("\n=== Open positions (unrealized) ===")
        print("\n".join(lines))
        print(f"Total unrealized : {upnl_total:+.2f} USDT")

    bal = ex.fetch_balance()["info"]
    print("\n=== Account ===")
    print(f"Wallet balance   : {float(bal['totalWalletBalance']):.2f} USDT")
    print(f"Margin balance   : {float(bal['totalMarginBalance']):.2f} USDT")
    print(f"Available        : {float(bal['availableBalance']):.2f} USDT")
    print(f"\nTOTAL P&L (realized + unrealized): {net_realized + upnl_total:+.2f} USDT")


if __name__ == "__main__":
    main()
