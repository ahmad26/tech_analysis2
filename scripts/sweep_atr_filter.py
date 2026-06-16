"""Sweep the per-timeframe min_atr_pct volatility floor.

Runs the backtest ONCE with no volatility gate, then post-filters the resulting
trades across a grid of ATR%/price thresholds per timeframe. Keep rule:

  keep trade if atr_pct >= threshold   (the live hard-mute gate in detect_patterns)

Post-filtering by `atr_pct >= threshold` is exactly equivalent to the live gate
(entry == close, so atr/entry == atr/close), so we avoid re-fetching per threshold.

Expectancy is reported NET of fees (fee_in_R = round_trip / risk_pct), in two columns:

  taker exp : exit always taker  (the old TAKE_PROFIT_MARKET behaviour)
  maker exp : take-profit fills are MAKER (the live reduceOnly LIMIT TP), stops/timeouts
              stay taker — this is the LIVE model, and the floor is chosen on it.

The maker saving is larger in R for tighter stops, so it lifts the low-ATR buckets the
floor decision turns on. (A Donchian-breakout "rescue" of sub-floor trades was tried
and removed — net-negative on both 4h and 15m; see memory.)

Usage:
    python -m scripts.sweep_atr_filter [candles] [tf ...] [--fee RT]

    candles   number of candles to fetch per symbol (default 1000)
    tf ...    optional timeframes to restrict the sweep to (default: all in config)
    --fee RT  round-trip taker fee fraction (default 0.0009 = 0.09%; --fee 0 = gross)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import replace
from statistics import median

from src.backtester import run_backtest, net_r, Fees, DEFAULT_FEE_RT, TradeResult
from src.config import load_config

# Threshold grid, as a fraction of price (0.005 == 0.5% ATR/price).
GRID = [0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, 0.040]


def _expectancy(trades: list[TradeResult], fees: Fees) -> tuple[int, float, float, float]:
    """Return (n_closed, win_pct, total_R, expectancy_R) for closed trades, net of fees."""
    closed = [t for t in trades if t.outcome in ("WIN", "LOSS")]
    if not closed:
        return 0, 0.0, 0.0, 0.0
    wins = [t for t in closed if t.outcome == "WIN"]
    losses = [t for t in closed if t.outcome == "LOSS"]
    total_r = sum(net_r(t, fees) for t in closed)
    win_rate = len(wins) / len(closed)
    avg_win = sum(net_r(t, fees) for t in wins) / len(wins) if wins else 0.0
    avg_los = sum(net_r(t, fees) for t in losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_los
    return len(closed), win_rate * 100, total_r, expectancy


def main() -> None:
    argv = sys.argv[1:]
    # Optional "--fee X" anywhere in argv (X = round-trip taker fraction). Default
    # DEFAULT_FEE_RT; pass "--fee 0" for the old gross/fee-free view.
    fee_rt = DEFAULT_FEE_RT
    if "--fee" in argv:
        i = argv.index("--fee")
        fee_rt = float(argv[i + 1])
        del argv[i:i + 2]

    candles = int(argv[0]) if argv else 1000
    tf_filter = argv[1:] if len(argv) > 1 else None

    cfg = load_config()
    # Force NO gate so every candle's signal is captured; we filter in memory.
    cfg = replace(cfg, min_atr_pct={})
    if tf_filter:
        cfg = replace(cfg, timeframes=tf_filter)

    # Live fee model (maker TP) and the all-taker baseline for comparison.
    fees_maker = Fees.free() if fee_rt <= 0 else Fees(taker_side=fee_rt / 2, maker_tp=True)
    fees_taker = Fees.free() if fee_rt <= 0 else Fees.all_taker(fee_rt)
    fee_note = (f"taker {fee_rt*100:.3f}% RT; TP leg maker {fees_maker.maker_side*100:.3f}%"
                if fee_rt > 0 else "NONE (gross)")
    print(f"Running backtest (ungated) on {len(cfg.symbols)} symbols, "
          f"{len(cfg.timeframes)} timeframes, {candles} candles each...")
    print(f"Expectancy NET of fees: {fee_note}. Floor chosen on the maker (live) column.\n")
    results, date_ranges = run_backtest(cfg, candles=candles)

    by_tf: dict[str, list[TradeResult]] = defaultdict(list)
    for t in results:
        by_tf[t.timeframe].append(t)

    best: dict[str, float] = {}

    for tf in cfg.timeframes:
        trades = by_tf.get(tf, [])
        if not trades:
            print(f"\n── {tf}: no trades ──")
            continue

        atrs = sorted(t.atr_pct for t in trades)
        rng = date_ranges.get(tf, ("?", "?"))
        print(f"\n{'═' * 78}")
        print(f"  Timeframe {tf}   ({rng[0]} → {rng[1]})   "
              f"ATR%/price: min={atrs[0]*100:.2f}%  median={median(atrs)*100:.2f}%  max={atrs[-1]*100:.2f}%")
        print(f"{'═' * 78}")
        print(f"  {'min_atr_pct':>12}  {'trades':>7}  {'win%':>6}  {'taker exp':>11}  {'maker exp':>11}")
        print(f"  {'-'*12}  {'-'*7}  {'-'*6}  {'-'*11}  {'-'*11}")

        base_n = base_exp = None
        best_score = None
        best_thr = 0.0
        for thr in GRID:
            kept = [t for t in trades if t.atr_pct >= thr]
            n, win_pct, _, exp_taker = _expectancy(kept, fees_taker)
            _, _, _, exp_maker = _expectancy(kept, fees_maker)
            if thr == 0.0:
                base_n, base_exp = n, exp_maker
            # Choose the floor on the LIVE (maker) expectancy, keeping >=40% of baseline trades.
            if base_n and n >= 0.40 * base_n:
                if best_score is None or exp_maker > best_score:
                    best_score = exp_maker
                    best_thr = thr
            print(f"  {thr*100:>10.2f}%  {n:>7}  {win_pct:>5.0f}%  {exp_taker:>+10.3f}R  {exp_maker:>+10.3f}R")

        best[tf] = best_thr
        print(f"  → best (maker, >=40% of baseline trades kept): min_atr_pct={best_thr*100:.2f}% "
              f"({best_thr})   baseline maker expectancy={base_exp:+.3f}R")

    print(f"\n{'═' * 78}")
    print("  Suggested config.yaml min_atr_pct block:")
    print(f"{'═' * 78}")
    print("min_atr_pct:")
    for tf in cfg.timeframes:
        print(f"  {tf+':':<5} {best.get(tf, 0.0)}")


if __name__ == "__main__":
    main()
