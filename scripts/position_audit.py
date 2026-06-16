"""Cross-check local position state against Binance futures. Read-only.

Prints a plain-text audit consumed by scripts/healthcheck.sh:
  - tracked positions that no longer exist on the exchange (stale local state)
  - exchange positions not tracked locally (manual / untracked)
  - open positions WITHOUT a resting stop-loss order  -> CRITICAL
  - open positions without a resting take-profit       -> WARN
  - protective orders resting on symbols with no position (orphans)

Exits 0 with a short note if API keys are absent (read-only scanning setup).
Lines are prefixed OK / WARN / CRITICAL so the healthcheck can gate on them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Run from anywhere — state files live in the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402  (loads .env)
from src.data_fetcher import create_exchange  # noqa: E402
from src.position_tracker import PositionTracker  # noqa: E402
from src.trader import is_protective_order, _is_reduce_only  # noqa: E402

_SL_TYPES = {"stop_market", "stop"}


def _is_stop_loss(order: dict) -> bool:
    return (order.get("type") or "").lower() in _SL_TYPES and _is_reduce_only(order)


def main() -> int:
    config = load_config()  # loads .env — must run before the key check
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        print("SKIPPED: no Binance API keys in .env — exchange audit not possible")
        return 0

    exchange = create_exchange(config.exchange, api_key, api_secret, testnet=False, market_type="future")
    # The audit intentionally fetches ALL open orders (no symbol) to catch
    # orphans on untracked symbols; acknowledge ccxt's rate-limit warning.
    exchange.options["warnOnFetchOpenOrdersWithoutSymbol"] = False

    tracked = PositionTracker().all()

    try:
        raw_positions = exchange.fetch_positions()
    except Exception as e:
        print(f"CRITICAL: could not fetch positions from exchange: {type(e).__name__}: {e}")
        return 1
    open_positions: dict[str, dict] = {}
    for p in raw_positions:
        if abs(float(p.get("contracts") or 0)) > 0:
            open_positions[p["symbol"].split(":")[0]] = p

    try:
        open_orders = exchange.fetch_open_orders()
    except Exception as e:
        print(f"CRITICAL: could not fetch open orders from exchange: {type(e).__name__}: {e}")
        return 1

    # Binance parks conditional orders (STOP_MARKET / TAKE_PROFIT_MARKET) on the
    # algo-orders endpoint — fetch_open_orders() does NOT return them. Without this
    # the audit reports protected positions as having no stop-loss (false CRITICAL).
    try:
        for o in exchange.fapiPrivateGetOpenAlgoOrders({}):
            market = exchange.markets_by_id.get(o.get("symbol"))
            if isinstance(market, list):
                market = market[0] if market else None
            if not market:
                continue
            # Normalize to the ccxt unified shape is_protective_order expects
            open_orders.append({
                "symbol": market["symbol"],
                "type": (o.get("orderType") or o.get("type") or "").lower(),
                "reduceOnly": str(o.get("reduceOnly")).lower() == "true",
            })
    except Exception as e:
        print(f"CRITICAL: could not fetch algo orders from exchange: {type(e).__name__}: {e}")
        return 1

    orders_by_symbol: dict[str, list[dict]] = {}
    for o in open_orders:
        orders_by_symbol.setdefault(o["symbol"].split(":")[0], []).append(o)

    issues = 0

    # Tracked locally but gone on the exchange (closed without us noticing)
    for sym in tracked:
        if sym not in open_positions:
            print(f"WARN: {sym} tracked in position_state.json but no position on exchange (stale state)")
            issues += 1

    # Open on the exchange but not tracked
    for sym, p in open_positions.items():
        if sym not in tracked:
            print(f"WARN: {sym} position open on exchange ({p.get('side')}, {p.get('contracts')}) but NOT tracked locally")
            issues += 1

    # Protection check — the one that really matters
    for sym, p in open_positions.items():
        sym_orders = orders_by_symbol.get(sym, [])
        has_sl = any(_is_stop_loss(o) for o in sym_orders)
        has_tp = any(is_protective_order(o) and not _is_stop_loss(o) for o in sym_orders)
        if not has_sl:
            print(f"CRITICAL: {sym} position OPEN with NO resting stop-loss order")
            issues += 1
        if not has_tp:
            print(f"WARN: {sym} position open with no resting take-profit order")
            issues += 1
        if has_sl and has_tp:
            t = tracked.get(sym)
            sl = f"{t.sl:.4f}" if t and t.sl > 0 else "?"
            tp = f"{t.tp:.4f}" if t and t.tp > 0 else "?"
            print(f"OK: {sym} {p.get('side')} {p.get('contracts')} @ {p.get('entryPrice')} — SL {sl} / TP {tp} resting, uPnL {p.get('unrealizedPnl')}")

    # Orphaned protective orders (no position behind them)
    for sym, sym_orders in orders_by_symbol.items():
        if sym in open_positions:
            continue
        orphans = [o for o in sym_orders if is_protective_order(o)]
        if orphans:
            print(f"WARN: {sym} has {len(orphans)} protective order(s) resting but NO open position (orphans)")
            issues += 1

    if not open_positions and not tracked:
        print("OK: no open positions, no tracked state — nothing to audit")
    if issues == 0:
        print(f"OK: audit clean — {len(open_positions)} open position(s), all protected and in sync")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
