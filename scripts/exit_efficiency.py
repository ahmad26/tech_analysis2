"""Exit-efficiency diagnostic.

Question this answers: when the app opens a good position, how much of the
*available* favorable move does it actually capture before the exit closes it?

For every gated signal it computes, over the same forward window the backtester
uses:
  - captured_R : the actual R the live ladder exit booked (real _check_outcome_rolling)
  - mfe_R      : the max favorable excursion (best unrealised R the trade ever showed)
  - mae_R      : the max adverse excursion (worst drawdown in R) before exit

and reports, per timeframe:
  - capture ratio   = mean(captured_R) / mean(mfe_R)   (1.0 = perfect harvesting)
  - scratch rate    = % of trades that exited at <= 0.1R despite mfe_R >= 1R
                      (i.e. "it was a winner and we gave it all back")
  - mean give-back  = mean(mfe_R - captured_R) over trades that reached mfe_R >= 1R

Run:  .venv/bin/python -m scripts.exit_efficiency --timeframes 4h 1d [--body-anchors]
"""
from __future__ import annotations

import argparse

import src.pattern_detector as pdmod
from src.backtester import (
    MAX_BARS_FORWARD,
    TF_SECONDS,
    _check_outcome_rolling,
)
from src.config import load_config
from src.data_fetcher import create_exchange, fetch_ohlcv
from src.pattern_detector import HTF_MAP, detect_patterns, extract_htf_levels
from src.trading_rules import MIN_RR

DEFAULT_CANDLES = 1000


def _mfe_mae(df, signal_idx, action, entry, sl, max_bars):
    """Max favorable / adverse excursion in R over the forward window."""
    initial_risk = abs(entry - sl)
    if initial_risk == 0:
        return None, None
    end = min(signal_idx + 1 + max_bars, len(df))
    best = 0.0
    worst = 0.0
    is_long = action == "BUY"
    for bar_i in range(signal_idx + 1, end):
        hi = float(df["high"].iloc[bar_i])
        lo = float(df["low"].iloc[bar_i])
        fav = (hi - entry) if is_long else (entry - lo)
        adv = (entry - lo) if is_long else (hi - entry)
        best = max(best, fav / initial_risk)
        worst = max(worst, adv / initial_risk)
    return best, worst


def run(timeframes, candles, body_anchors, dense_grid=0.0, chandelier=0.0):
    if body_anchors:
        pdmod.USE_BODY_ANCHORS = True
        print("[body-anchors] Fib windows anchored to candle bodies\n")
    if dense_grid or chandelier:
        print(f"[trail] dense_grid={dense_grid} chandelier={chandelier}\n")

    config = load_config()
    exchange = create_exchange(config.exchange)
    patterns_active = [p for p in config.patterns if p != "bearish_engulfing"]

    for tf in timeframes:
        max_bars = MAX_BARS_FORWARD.get(tf, 20)
        rows = []  # (captured_R, mfe_R, mae_R)

        for symbol in config.symbols:
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
                    found = detect_patterns(window, symbol, tf, patterns_active,
                                            htf_levels, config.min_atr_pct.get(tf))
                except Exception:
                    continue

                for p in found:
                    ts = p.trading_signal
                    if ts is None or ts.risk_reward < MIN_RR:  # mirror live gate
                        continue
                    _, captured_r, _ = _check_outcome_rolling(
                        df, i, ts.action, ts.entry, ts.stop_loss,
                        ts.all_tp_candidates, max_bars, atr=p.atr,
                        dense_grid=dense_grid, chandelier=chandelier,
                    )
                    mfe, mae = _mfe_mae(df, i, ts.action, ts.entry, ts.stop_loss, max_bars)
                    if mfe is None:
                        continue
                    rows.append((captured_r, mfe, mae))

        if not rows:
            print(f"{tf}: no trades")
            continue

        n_tr = len(rows)
        mean_cap = sum(r[0] for r in rows) / n_tr
        mean_mfe = sum(r[1] for r in rows) / n_tr
        capture = mean_cap / mean_mfe if mean_mfe else 0.0

        winners = [r for r in rows if r[1] >= 1.0]  # ever showed >= 1R profit
        scratches = [r for r in winners if r[0] <= 0.1]
        giveback = [r[1] - r[0] for r in winners]
        tp_reached = [r for r in rows if r[0] >= 1.5]  # actually booked >= target

        print(f"=== {tf} ({'body' if body_anchors else 'wick'} anchors)  n={n_tr} ===")
        print(f"  mean captured R     : {mean_cap:+.2f}R")
        print(f"  mean MFE (available): {mean_mfe:+.2f}R")
        print(f"  CAPTURE RATIO       : {capture*100:.0f}%   (of the favorable move we kept)")
        print(f"  trades that hit >=1R MFE : {len(winners)}/{n_tr} ({len(winners)/n_tr*100:.0f}%)")
        print(f"    of those, scratched <=0.1R : {len(scratches)} ({len(scratches)/max(1,len(winners))*100:.0f}%)  <-- gave a winner back")
        print(f"    mean give-back (MFE - captured): {sum(giveback)/max(1,len(winners)):.2f}R")
        print(f"  booked >= 1.5R target    : {len(tp_reached)}/{n_tr} ({len(tp_reached)/n_tr*100:.0f}%)")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframes", nargs="+", default=["4h", "1d"])
    ap.add_argument("--candles", type=int, default=DEFAULT_CANDLES)
    ap.add_argument("--body-anchors", action="store_true")
    ap.add_argument("--dense-grid", type=float, default=0.0)
    ap.add_argument("--chandelier", type=float, default=0.0)
    a = ap.parse_args()
    run(a.timeframes, a.candles, a.body_anchors, a.dense_grid, a.chandelier)
