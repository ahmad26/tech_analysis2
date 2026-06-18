"""Head-to-head of exit-mechanism variants on the REAL ladder engine (priority B).

Holds the ENTRY side fixed — identical gated signals, wick anchors, R/R>=1.5 gate,
point-in-time HTF levels — and swaps ONLY the exit handling inside
`_check_outcome_rolling`, so any difference is purely the exit mechanism:

  baseline   : current live ladder (arms break-even on the first TP rung touched)
  chandelier : + continuous ~Nx ATR trail off the running peak
               (fixes give-back frozen between sparse structural levels)
  be-delay   : hold the protective SL until MFE >= Mx R, then ladder as usual
               (fixes the 61% scratch-at-0R; see finding_exit_capture_problem)
  both       : chandelier + delayed arm together

The two arms target DIFFERENT failure modes (give-back vs. early-scratch), so
"both" is the one to watch. Signals are detected once and all four variants run
on the same set, so the only moving part is the exit.

Each variant reports net expectancy (R/trade, fees on) per timeframe PLUS a
first-half / second-half OOS split. A variant is only trustworthy if it beats
baseline in BOTH halves (regime robustness — see finding_oos_regime_check); a
win in one half only was fit to that window.

Run:
  .venv/bin/python -m scripts.exit_headtohead --timeframes 4h 1d \
      --chandelier 1.0 --breakeven-r 1.0 [--candles 1000] [--no-bearish]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from src.backtester import (
    DEFAULT_FEE_RT,
    MAX_BARS_FORWARD,
    TF_SECONDS,
    Fees,
    TradeResult,
    _check_outcome_rolling,
    net_r,
)
from src.config import load_config
from src.data_fetcher import create_exchange, fetch_ohlcv
from src.pattern_detector import HTF_MAP, detect_patterns, extract_htf_levels
from src.trading_rules import MIN_RR

DEFAULT_CANDLES = 1000


def _expect(trades: list[TradeResult], fees: Fees) -> tuple[float, int]:
    if not trades:
        return 0.0, 0
    rs = [net_r(t, fees) for t in trades]
    return sum(rs) / len(rs), len(rs)


def run(timeframes, candles, chandelier, breakeven_r, no_bearish):
    config = load_config()
    fees = Fees(taker_side=DEFAULT_FEE_RT / 2, maker_tp=True)
    exchange = create_exchange(config.exchange)

    patterns = list(config.patterns)
    if no_bearish and "bearish_engulfing" in patterns:
        patterns.remove("bearish_engulfing")

    # (chandelier, breakeven_r) per variant — entry side is identical for all.
    variants: dict[str, tuple[float, float]] = {
        "baseline":   (0.0,        0.0),
        "chandelier": (chandelier, 0.0),
        "be-delay":   (0.0,        breakeven_r),
        "both":       (chandelier, breakeven_r),
    }

    print(f"patterns   : {patterns}")
    print(f"variants   : chandelier={chandelier}xATR  breakeven_r={breakeven_r}R  (fees on)")
    print(f"expectancy : net R/trade; OOS split = first vs second half by signal date\n")

    for tf in timeframes:
        max_bars = MAX_BARS_FORWARD.get(tf, 20)
        # variant -> list[TradeResult]; identical order/length across variants.
        bucket: dict[str, list[TradeResult]] = {v: [] for v in variants}

        for symbol in config.symbols:
            print(f"  scanning {symbol} {tf}...", flush=True)
            try:
                df = fetch_ohlcv(exchange, symbol, tf, limit=candles)
            except Exception:
                continue

            htf_frames = {}
            for htf_tf in HTF_MAP.get(tf, []):
                need = int(candles * TF_SECONDS[tf] / TF_SECONDS[htf_tf]) + 260
                try:
                    htf_frames[htf_tf] = fetch_ohlcv(exchange, symbol, htf_tf, limit=need)
                except Exception:
                    pass

            n = len(df)
            for i in range(208, n - 1):
                window = df.iloc[i - 208: i + 2]
                scan_ts = df.index[i + 1]
                htf_levels = []
                for htf_tf, hdf in htf_frames.items():
                    cut = hdf.index.searchsorted(scan_ts, side="right")
                    htf_levels.extend(extract_htf_levels(hdf.iloc[:cut], htf_tf))

                try:
                    found = detect_patterns(window, symbol, tf, patterns,
                                            htf_levels, config.min_atr_pct.get(tf))
                except Exception:
                    continue

                for p in found:
                    ts = p.trading_signal
                    if ts is None or ts.risk_reward < MIN_RR:  # mirror live gate
                        continue
                    risk_pct = (abs(ts.entry - ts.stop_loss) / ts.entry) if ts.entry else 0.0
                    for vname, (chand, be_r) in variants.items():
                        outcome, pnl_r, exit_kind = _check_outcome_rolling(
                            df, i, ts.action, ts.entry, ts.stop_loss,
                            ts.all_tp_candidates, max_bars, atr=p.atr,
                            chandelier=chand, breakeven_r=be_r,
                        )
                        bucket[vname].append(TradeResult(
                            symbol=symbol, timeframe=tf, pattern_name=p.pattern_name,
                            action=ts.action, setup=ts.setup, outcome=outcome,
                            pnl_r=pnl_r, signal_date=p.candle_timestamp,
                            risk_pct=risk_pct, exit_kind=exit_kind,
                        ))

        base = sorted(bucket["baseline"], key=lambda t: t.signal_date)
        if not base:
            print(f"== {tf}: no trades ==\n")
            continue
        mid = len(base) // 2
        split_date = base[mid].signal_date.date()
        be_overall, _ = _expect(base, fees)

        print(f"== {tf}  (n={len(base)}, OOS split @ {split_date}) ==")
        print(f"  {'variant':<12} {'overall':>9} {'1st half':>9} {'2nd half':>9}   robust?")
        for vname in variants:
            tr = sorted(bucket[vname], key=lambda t: t.signal_date)
            first, second = tr[:mid], tr[mid:]
            o, _ = _expect(tr, fees)
            e1, _ = _expect(first, fees)
            e2, _ = _expect(second, fees)
            if vname == "baseline":
                tag = "(reference)"
            else:
                # robust = beats baseline overall AND in both halves
                b1, _ = _expect(base[:mid], fees)
                b2, _ = _expect(base[mid:], fees)
                both_halves = e1 > b1 and e2 > b2
                tag = "YES" if (o > be_overall and both_halves) else \
                      ("one-half" if o > be_overall else "no")
            print(f"  {vname:<12} {o:>+8.3f}R {e1:>+8.3f}R {e2:>+8.3f}R   {tag}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframes", nargs="+", default=["4h", "1d"])
    ap.add_argument("--candles", type=int, default=DEFAULT_CANDLES)
    ap.add_argument("--chandelier", type=float, default=1.0,
                    help="continuous ATR-multiple trail (0 disables that arm)")
    ap.add_argument("--breakeven-r", type=float, default=1.0,
                    help="MFE in R required before the break-even arm fires (0 disables)")
    ap.add_argument("--no-bearish", action="store_true",
                    help="drop bearish_engulfing (matches finding_exit_capture config)")
    a = ap.parse_args()
    run(a.timeframes, a.candles, a.chandelier, a.breakeven_r, a.no_bearish)
