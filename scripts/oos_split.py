"""Out-of-sample / regime-stability checker.

Runs the real backtest once, then splits each timeframe's trades by signal date
into a first half and a second half (a cheap proxy for two market regimes) and
reports expectancy per half — overall and per pattern. A finding is "robust" only
if it holds in BOTH halves; if it only shows up in one, it was fit to that window.

Use it for two questions:
  - Are the DISABLED patterns (hammer / inverted_hammer / morning_star) genuinely
    bad, or was that a single-window/regime artifact? Pass --include-disabled.
  - Does the chandelier trail edge survive on held-out data? Run with and without
    --chandelier and compare the SECOND-half columns.

Run:
  .venv/bin/python -m scripts.oos_split --timeframes 4h 1d --include-disabled --chandelier 1.0
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from src.backtester import DEFAULT_FEE_RT, Fees, net_r, run_backtest
from src.config import load_config

DISABLED = ["hammer", "inverted_hammer", "morning_star"]


def _expect(trades, fees):
    if not trades:
        return 0.0, 0
    rs = [net_r(t, fees) for t in trades]
    return sum(rs) / len(rs), len(rs)


def run(timeframes, include_disabled, chandelier, no_bearish, candles, psych_round=False):
    config = load_config()
    fees = Fees(taker_side=DEFAULT_FEE_RT / 2, maker_tp=True)

    if psych_round:
        from src import trading_rules
        trading_rules.PSYCH_ROUND = True
        trading_rules.PSYCH_ROUND_TFS = None   # sweep all requested TFs, not just the 1d live gate

    patterns = list(config.patterns)
    if no_bearish and "bearish_engulfing" in patterns:
        patterns.remove("bearish_engulfing")
    if include_disabled:
        patterns += [p for p in DISABLED if p not in patterns]

    print(f"patterns: {patterns}")
    print(f"chandelier={chandelier}  psych_round={psych_round}  fees=on  (expectancy = net R/trade)\n")

    results, date_ranges = run_backtest(
        config, timeframes=timeframes, candles=candles, patterns=patterns,
        exit_model="rolling", htf=True, gate=True, chandelier=chandelier,
    )

    for tf in timeframes:
        tf_trades = sorted([t for t in results if t.timeframe == tf],
                           key=lambda t: t.signal_date)
        if not tf_trades:
            print(f"== {tf}: no trades ==\n")
            continue
        mid = len(tf_trades) // 2
        first, second = tf_trades[:mid], tf_trades[mid:]
        d_lo = tf_trades[0].signal_date.date()
        d_mid = first[-1].signal_date.date()
        d_hi = tf_trades[-1].signal_date.date()

        print(f"== {tf}  ({d_lo} … {d_mid} | {d_mid} … {d_hi}) ==")
        o1, n1 = _expect(first, fees)
        o2, n2 = _expect(second, fees)
        print(f"  {'OVERALL':<18} 1st: {o1:+.3f}R (n={n1:>4})   2nd: {o2:+.3f}R (n={n2:>4})")

        by_pat: dict[str, list] = defaultdict(list)
        for t in tf_trades:
            by_pat[t.pattern_name].append(t)
        for pat in sorted(by_pat):
            p_first = [t for t in first if t.pattern_name == pat]
            p_second = [t for t in second if t.pattern_name == pat]
            e1, c1 = _expect(p_first, fees)
            e2, c2 = _expect(p_second, fees)
            flag = ""
            # flag patterns that flip sign between halves (regime-dependent)
            if c1 >= 5 and c2 >= 5 and (e1 > 0) != (e2 > 0):
                flag = "  <- FLIPS sign"
            print(f"  {pat:<18} 1st: {e1:+.3f}R (n={c1:>4})   2nd: {e2:+.3f}R (n={c2:>4}){flag}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframes", nargs="+", default=["4h", "1d"])
    ap.add_argument("--include-disabled", action="store_true")
    ap.add_argument("--chandelier", type=float, default=0.0)
    ap.add_argument("--no-bearish", action="store_true")
    ap.add_argument("--candles", type=int, default=None)
    ap.add_argument("--psych-round", action="store_true")
    a = ap.parse_args()
    run(a.timeframes, a.include_disabled, a.chandelier, a.no_bearish, a.candles, a.psych_round)
