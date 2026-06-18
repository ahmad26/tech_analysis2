"""TP-buffer diagnostic — should the resting maker LIMIT sit a little INSIDE the final
ladder level so a near-miss still fills?

Two questions, one scan (signals detected once on the real live engine = rolling ladder,
wick anchors, R/R>=1.5 gate, point-in-time HTF):

1. NEAR-MISS FREQUENCY (intuition): for each gated trade, how close did the favorable
   excursion get to the FINAL ladder level (where the live LIMIT rests)? Report the share
   that reached it exactly (clean fill) and the share that fell within b*ATR short of it —
   that short-fall population is exactly what a b*ATR buffer would newly capture.

2. NET-R PER BUFFER (the decision): re-run the real `_check_outcome_rolling` with
   tp_buffer_atr in {0, 0.05, 0.10, 0.25} on the SAME signals and report net expectancy
   (R/trade, fees on) per timeframe with a first/second-half OOS split. A buffer earns its
   place only if it lifts expectancy in BOTH halves (regime-robust) — the gain from
   catching near-misses must beat the reward shaved off clean fills.

Note: the backtester counts a one-tick wick to the exact level as a fill, so buffer=0 is
already optimistic about exact-level fills; a positive buffer also narrows that
backtest-vs-live gap (live needs price to trade through the resting order).

Run:
  .venv/bin/python -m scripts.tp_buffer_check --timeframes 4h 1d --candles 5000
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
BUFFERS = [0.0, 0.05, 0.10, 0.25]   # ATR multiples to test


def _expect(trades: list[TradeResult], fees: Fees) -> tuple[float, int]:
    if not trades:
        return 0.0, 0
    rs = [net_r(t, fees) for t in trades]
    return sum(rs) / len(rs), len(rs)


def _peak_favorable(df, signal_idx, action, max_bars) -> float | None:
    """Best favorable price (max high for long, min low for short) over the window."""
    end = min(signal_idx + 1 + max_bars, len(df))
    if end <= signal_idx + 1:
        return None
    if action == "BUY":
        return float(df["high"].iloc[signal_idx + 1:end].max())
    return float(df["low"].iloc[signal_idx + 1:end].min())


def run(timeframes, candles):
    config = load_config()
    fees = Fees(taker_side=DEFAULT_FEE_RT / 2, maker_tp=True)
    exchange = create_exchange(config.exchange)

    print(f"buffers (xATR): {BUFFERS}   engine: live ladder (rolling+gate+htf, fees on)\n")

    for tf in timeframes:
        max_bars = MAX_BARS_FORWARD.get(tf, 20)
        bucket: dict[float, list[TradeResult]] = {b: [] for b in BUFFERS}
        # near-miss tally
        n = 0
        reached_final = 0
        within = {b: 0 for b in BUFFERS if b > 0}   # 0 < shortfall <= b*ATR

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

            ndf = len(df)
            for i in range(208, ndf - 1):
                window = df.iloc[i - 208: i + 2]
                scan_ts = df.index[i + 1]
                htf_levels = []
                for htf_tf, hdf in htf_frames.items():
                    cut = hdf.index.searchsorted(scan_ts, side="right")
                    htf_levels.extend(extract_htf_levels(hdf.iloc[:cut], htf_tf))

                try:
                    found = detect_patterns(window, symbol, tf, config.patterns_for(tf),
                                            htf_levels, config.min_atr_pct.get(tf))
                except Exception:
                    continue

                for p in found:
                    ts = p.trading_signal
                    if ts is None or ts.risk_reward < MIN_RR:
                        continue
                    risk_pct = (abs(ts.entry - ts.stop_loss) / ts.entry) if ts.entry else 0.0
                    for b in BUFFERS:
                        outcome, pnl_r, exit_kind = _check_outcome_rolling(
                            df, i, ts.action, ts.entry, ts.stop_loss,
                            ts.all_tp_candidates, max_bars, atr=p.atr, tp_buffer_atr=b,
                        )
                        bucket[b].append(TradeResult(
                            symbol=symbol, timeframe=tf, pattern_name=p.pattern_name,
                            action=ts.action, setup=ts.setup, outcome=outcome,
                            pnl_r=pnl_r, signal_date=p.candle_timestamp,
                            risk_pct=risk_pct, exit_kind=exit_kind,
                        ))

                    # near-miss vs the FINAL ladder level
                    final = ts.all_tp_candidates[-1][0]
                    peak = _peak_favorable(df, i, ts.action, max_bars)
                    if peak is None or not p.atr:
                        continue
                    n += 1
                    if ts.action == "BUY":
                        gap = final - peak       # >0 = fell short
                    else:
                        gap = peak - final
                    if gap <= 0:
                        reached_final += 1
                    else:
                        short_atr = gap / p.atr
                        for b in within:
                            if short_atr <= b:
                                within[b] += 1

        if not n:
            print(f"== {tf}: no trades ==\n")
            continue

        # NEAR-MISS report
        print(f"== {tf}  (n={n}) — near-miss vs final ladder level ==")
        print(f"  reached final exactly : {reached_final} ({reached_final/n*100:.1f}%)")
        for b in sorted(within):
            print(f"  fell <= {b:>4.2f}xATR short : {within[b]} ({within[b]/n*100:.1f}%)  "
                  f"<- a {b}xATR buffer would newly catch these")

        # NET-R per buffer, OOS split (split index from the buffer=0 ordering)
        base = sorted(bucket[0.0], key=lambda t: t.signal_date)
        mid = len(base) // 2
        split_date = base[mid].signal_date.date()
        be0, _ = _expect(base, fees)
        b0_1, _ = _expect(base[:mid], fees)
        b0_2, _ = _expect(base[mid:], fees)
        print(f"  -- net expectancy by buffer (OOS split @ {split_date}) --")
        print(f"  {'buffer':<10} {'overall':>9} {'1st half':>9} {'2nd half':>9}   robust?")
        for b in BUFFERS:
            tr = sorted(bucket[b], key=lambda t: t.signal_date)
            o, _ = _expect(tr, fees)
            e1, _ = _expect(tr[:mid], fees)
            e2, _ = _expect(tr[mid:], fees)
            if b == 0.0:
                tag = "(reference)"
            else:
                tag = "YES" if (o > be0 and e1 > b0_1 and e2 > b0_2) else \
                      ("one-half" if o > be0 else "no")
            print(f"  {b:<10.2f} {o:>+8.3f}R {e1:>+8.3f}R {e2:>+8.3f}R   {tag}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframes", nargs="+", default=["4h", "1d"])
    ap.add_argument("--candles", type=int, default=DEFAULT_CANDLES)
    a = ap.parse_args()
    run(a.timeframes, a.candles)
