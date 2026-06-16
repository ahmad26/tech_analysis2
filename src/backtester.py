from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.data_fetcher import create_exchange, fetch_ohlcv
from src.models import AppConfig
from src.pattern_detector import HTF_MAP, detect_patterns, extract_htf_levels

TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

logger = logging.getLogger(__name__)

DEFAULT_BACKTEST_CANDLES = 1000

# Per-side fees as a fraction of notional (~incl. BNB discount). Market entry is
# always taker; the stop-loss (STOP_MARKET) is taker; the maker take-profit (a resting
# reduceOnly LIMIT, see memory/feature_maker_tp.md) is maker. Trailing the stop is free
# (no fill until triggered), so each trade pays exactly one entry leg + one exit leg.
TAKER_FEE_SIDE = 0.00045
MAKER_FEE_SIDE = 0.00018
DEFAULT_FEE_RT = 2 * TAKER_FEE_SIDE   # 0.0009 — all-taker round trip (legacy default)

MAX_BARS_FORWARD: dict[str, int] = {
    "15m": 96,   # 1 day
    "1h":  72,   # 3 days
    "4h":  42,   # 1 week
    "1d":  30,   # 1 month
}


@dataclass(frozen=True)
class Fees:
    """Per-side fee model. The exit leg is maker only when the trade left via a
    take-profit fill (exit_kind == "tp") AND maker_tp is on — stop/timeout exits and
    every entry are taker."""
    taker_side: float = TAKER_FEE_SIDE
    maker_side: float = MAKER_FEE_SIDE
    maker_tp: bool = True

    @classmethod
    def free(cls) -> "Fees":
        return cls(0.0, 0.0, False)

    @classmethod
    def all_taker(cls, round_trip: float = DEFAULT_FEE_RT) -> "Fees":
        return cls(taker_side=round_trip / 2, maker_side=round_trip / 2, maker_tp=False)

    def round_trip(self, exit_kind: str) -> float:
        exit_fee = self.maker_side if (self.maker_tp and exit_kind == "tp") else self.taker_side
        return self.taker_side + exit_fee


def net_r(t: "TradeResult", fees: Fees) -> float:
    """P&L in R after deducting fees.

    fee_in_R = round_trip_fee / risk_pct, where risk_pct = initial SL distance / entry
    (the R denominator). Tighter stops (smaller risk_pct) pay proportionally more. Fees
    are charged on every opened trade — wins, losses and breakeven scratches alike.
    """
    if t.risk_pct <= 0:
        return t.pnl_r
    rt = fees.round_trip(t.exit_kind)
    if rt <= 0:
        return t.pnl_r
    return t.pnl_r - rt / t.risk_pct


@dataclass
class TradeResult:
    symbol: str
    timeframe: str
    pattern_name: str
    action: str        # "BUY" / "SELL"
    setup: str         # "REVERSAL" / "CONTINUATION"
    outcome: str       # "WIN" / "LOSS" / "OPEN"
    pnl_r: float       # actual R achieved (variable with rolling TP)
    signal_date: datetime = datetime.min
    atr_pct: float = 0.0  # ATR(14)/entry at signal — used for vol-filter threshold sweeps
    risk_pct: float = 0.0  # initial |entry - SL| / entry — R denominator, drives fee_in_R
    exit_kind: str = "stop"  # "tp" = left at a take-profit (maker), "stop"/"open" = taker


def _densify_ladder(
    tp_candidates: list[tuple[float, str]],
    entry: float,
    action: str,
    atr: float | None,
    step_mult: float,
) -> list[tuple[float, str]]:
    """Insert intermediate trailing rungs spaced `step_mult`*ATR apart between entry
    and the final (structural) TP, keeping the structural levels in place. Used by
    the dense-grid-no-lag experiment: the SL still trails tight to the just-touched
    rung (no lag), but with finer rungs it ratchets up during a run that stops short
    of the next structural level. Does NOT change the final exit level (the resting
    maker LIMIT) or the first structural TP the R/R gate is computed on — only the
    *exit-simulation* ladder is densified.
    """
    if step_mult <= 0 or atr is None or atr <= 0 or not tp_candidates:
        return tp_candidates
    is_long = action == "BUY"
    final = tp_candidates[-1][0]
    step = step_mult * atr
    grid: list[tuple[float, str]] = []
    k = 1
    while k <= 1000:
        price = entry + k * step if is_long else entry - k * step
        if (is_long and price >= final) or (not is_long and price <= final):
            break
        grid.append((price, f"grid {step_mult}xATR"))
        k += 1
    merged = sorted(list(tp_candidates) + grid, key=lambda x: abs(x[0] - entry))
    # Drop grid rungs that sit within half a step of a structural level (the
    # structural level already covers that price; keep the meaningful label).
    out: list[tuple[float, str]] = []
    structural = [c[0] for c in tp_candidates]
    for price, label in merged:
        if label.startswith("grid") and any(abs(price - s) < step / 2 for s in structural):
            continue
        out.append((price, label))
    return out


def _check_outcome_rolling(
    df: pd.DataFrame,
    signal_idx: int,
    action: str,
    entry: float,
    sl: float,
    tp_candidates: list[tuple[float, str]],
    max_bars: int,
    atr: float | None = None,
    dense_grid: float = 0.0,
) -> tuple[str, float, str]:
    """Cascade through TP levels. When each level is hit, SL advances to lock in profit.

    SL moves to entry after 1st TP, then to the previous TP price on each subsequent hit.
    Returns (outcome, pnl_r, exit_kind) where exit_kind is "tp" when the position left
    at a take-profit level (filled by the maker LIMIT live), "stop" when it left at the
    (trailing) stop, or "open" when it never closed within the window.

    When `dense_grid` > 0, intermediate rungs `dense_grid`*ATR apart are inserted
    between entry and the final TP (dense-grid-no-lag experiment) so the tight trail
    ratchets during runs that fall short of the next structural level.
    """
    if dense_grid > 0:
        tp_candidates = _densify_ladder(tp_candidates, entry, action, atr, dense_grid)
    if not tp_candidates:
        return "OPEN", 0.0, "open"

    initial_risk = abs(entry - sl)
    if initial_risk == 0:
        return "OPEN", 0.0, "open"

    current_sl = sl
    sl_moved = False
    tp_idx = 0
    end = min(signal_idx + 1 + max_bars, len(df))

    for bar_i in range(signal_idx + 1, end):
        hi = float(df["high"].iloc[bar_i])
        lo = float(df["low"].iloc[bar_i])
        current_tp = tp_candidates[tp_idx][0]

        # Check SL first (conservative — avoids over-optimistic results on wide candles)
        sl_hit = (lo <= current_sl) if action == "BUY" else (hi >= current_sl)
        tp_hit = (hi >= current_tp) if action == "BUY" else (lo <= current_tp)

        if sl_hit:
            pnl_r = (current_sl - entry) / initial_risk if action == "BUY" else (entry - current_sl) / initial_risk
            return ("WIN" if pnl_r > 0 else "LOSS"), pnl_r, "stop"

        if tp_hit:
            prev_tp = current_tp
            tp_idx += 1

            # No next level — exit at this TP (maker LIMIT fill live)
            if tp_idx >= len(tp_candidates):
                pnl_r = (prev_tp - entry) / initial_risk if action == "BUY" else (entry - prev_tp) / initial_risk
                return "WIN", pnl_r, "tp"

            # Advance SL: entry on first hit, previous TP price on subsequent hits
            current_sl = entry if not sl_moved else prev_tp
            sl_moved = True

    # Timed out
    if not sl_moved:
        return "OPEN", 0.0, "open"

    # SL was moved — exits at the locked-in (trailing) stop = taker
    pnl_r = (current_sl - entry) / initial_risk if action == "BUY" else (entry - current_sl) / initial_risk
    return ("WIN" if pnl_r > 0 else "WIN"), max(0.0, pnl_r), "stop"  # breakeven counts as WIN (no loss)


def _check_outcome_live(
    df: pd.DataFrame,
    signal_idx: int,
    action: str,
    entry: float,
    sl: float,
    tp: float,
    max_bars: int,
) -> tuple[str, float, str]:
    """Simulate the LIVE exit engine: one full-size TP at TP1 (the maker LIMIT the
    trader actually places) and a stop trailed per bar exactly like
    position_manager._update — ATR(14) chandelier + staged R-locks (breakeven at
    +1R, lock +1R at +2R), ratchet-only.

    Granularity caveat: live trails every 5 min on the forming candle; here the
    stop only moves at bar closes, so intra-bar chandelier tightening is not seen.
    Timed-out trades are marked to market at the window's last close (live has no
    time-stop), outcome "OPEN".
    """
    from src.position_manager import (
        BREAKEVEN_BUFFER_R, BREAKEVEN_TRIGGER_R, LOCK_1R_TRIGGER_R, TRAIL_ATR_MULT, _atr,
    )

    initial_risk = abs(entry - sl)
    if initial_risk == 0:
        return "OPEN", 0.0, "open"

    is_long = action == "BUY"
    current_sl = sl
    end = min(signal_idx + 1 + max_bars, len(df))

    for bar_i in range(signal_idx + 1, end):
        hi = float(df["high"].iloc[bar_i])
        lo = float(df["low"].iloc[bar_i])

        # Check SL first (conservative — same convention as the rolling model)
        sl_hit = (lo <= current_sl) if is_long else (hi >= current_sl)
        tp_hit = (hi >= tp) if is_long else (lo <= tp)

        if sl_hit:
            pnl_r = (current_sl - entry) / initial_risk if is_long else (entry - current_sl) / initial_risk
            return ("WIN" if pnl_r > 0 else "LOSS"), pnl_r, "stop"
        if tp_hit:
            pnl_r = (tp - entry) / initial_risk if is_long else (entry - tp) / initial_risk
            return "WIN", pnl_r, "tp"

        # Bar close: trail the stop like position_manager._update
        close = float(df["close"].iloc[bar_i])
        atr = _atr(df.iloc[: bar_i + 1])
        candidates: list[float] = []
        if atr is not None:
            candidates.append(close - TRAIL_ATR_MULT * atr if is_long else close + TRAIL_ATR_MULT * atr)
        profit = (close - entry) if is_long else (entry - close)
        r_mult = profit / initial_risk
        if r_mult >= LOCK_1R_TRIGGER_R:
            candidates.append(entry + initial_risk if is_long else entry - initial_risk)
        elif r_mult >= BREAKEVEN_TRIGGER_R:
            buf = BREAKEVEN_BUFFER_R * initial_risk
            candidates.append(entry + buf if is_long else entry - buf)
        if candidates:
            cand = max(candidates) if is_long else min(candidates)
            if (is_long and cand > current_sl) or (not is_long and cand < current_sl):
                current_sl = cand

    # Timed out — mark to market at the last close inside the window
    last_close = float(df["close"].iloc[end - 1])
    pnl_r = (last_close - entry) / initial_risk if is_long else (entry - last_close) / initial_risk
    return "OPEN", pnl_r, "open"


def run_backtest(
    config: AppConfig,
    timeframes: list[str] | None = None,
    candles: int | None = None,
    patterns: list[str] | None = None,
    exit_model: str = "rolling",
    htf: bool = False,
    gate: bool = False,
    dense_grid: float = 0.0,
) -> tuple[list[TradeResult], dict[str, tuple[str, str]]]:
    exchange = create_exchange(config.exchange)
    results: list[TradeResult] = []
    date_ranges: dict[str, tuple[str, str]] = {}
    n_candles = candles if candles is not None else DEFAULT_BACKTEST_CANDLES
    tfs = timeframes if timeframes else config.timeframes
    active_patterns = patterns if patterns else config.patterns

    for tf in tfs:
        max_bars = MAX_BARS_FORWARD.get(tf, 20)
        tf_start: str | None = None
        tf_end: str | None = None

        for symbol in config.symbols:
            print(f"  scanning {symbol} {tf}...", flush=True)

            try:
                df = fetch_ohlcv(exchange, symbol, tf, limit=n_candles)
            except Exception:
                logger.exception("Failed to fetch %s %s", symbol, tf)
                continue

            if tf_start is None:
                tf_start = str(df.index[0].date())
                tf_end   = str(df.index[-1].date())

            # Pre-fetch higher-TF history so each window can see the HTF levels
            # exactly as they stood at scan time (last forming HTF candle excluded
            # by extract_htf_levels' target=len-2, same as live).
            htf_frames: dict[str, pd.DataFrame] = {}
            if htf:
                for htf_tf in HTF_MAP.get(tf, []):
                    need = int(n_candles * TF_SECONDS[tf] / TF_SECONDS[htf_tf]) + 260
                    try:
                        htf_frames[htf_tf] = fetch_ohlcv(exchange, symbol, htf_tf, limit=need)
                    except Exception:
                        logger.exception("Failed to fetch HTF %s for %s", htf_tf, symbol)

            n = len(df)
            for i in range(208, n - 1):
                window = df.iloc[i - 208: i + 2]

                htf_levels = None
                if htf_frames:
                    # Live scans right after the signal candle closes, i.e. at the
                    # open of candle i+1 — slice HTF history to that moment.
                    scan_ts = df.index[i + 1]
                    htf_levels = []
                    for htf_tf, hdf in htf_frames.items():
                        cut = hdf.index.searchsorted(scan_ts, side="right")
                        htf_levels.extend(extract_htf_levels(hdf.iloc[:cut], htf_tf))

                try:
                    patterns = detect_patterns(window, symbol, tf, active_patterns, htf_levels, config.min_atr_pct.get(tf))
                except Exception:
                    continue

                for p in patterns:
                    if p.trading_signal is None:
                        continue
                    ts = p.trading_signal
                    if gate or exit_model == "live":
                        # Mirror main.py's gate: live drops signals whose R/R
                        # (computed on TP1 after obstacle demotion) is < MIN_RR.
                        from src.trading_rules import MIN_RR
                        if ts.risk_reward < MIN_RR:
                            continue
                    if exit_model == "live":
                        outcome, pnl_r, exit_kind = _check_outcome_live(
                            df, i,
                            ts.action, ts.entry, ts.stop_loss,
                            ts.take_profit, max_bars,
                        )
                    else:
                        outcome, pnl_r, exit_kind = _check_outcome_rolling(
                            df, i,
                            ts.action, ts.entry, ts.stop_loss,
                            ts.all_tp_candidates, max_bars,
                            atr=p.atr, dense_grid=dense_grid,
                        )
                    results.append(TradeResult(
                        symbol=symbol,
                        timeframe=tf,
                        pattern_name=p.pattern_name,
                        action=ts.action,
                        setup=ts.setup,
                        outcome=outcome,
                        pnl_r=pnl_r,
                        signal_date=p.candle_timestamp,
                        atr_pct=(p.atr / ts.entry) if (p.atr is not None and ts.entry) else 0.0,
                        risk_pct=(abs(ts.entry - ts.stop_loss) / ts.entry) if ts.entry else 0.0,
                        exit_kind=exit_kind,
                    ))

        if tf_start and tf_end:
            date_ranges[tf] = (tf_start, tf_end)

    return results, date_ranges


# ── Reporting ──────────────────────────────────────────────────────────────────

def _stats(trades: list[TradeResult], risk: float = 1.0, fees: Fees | None = None) -> dict:
    fees = fees or Fees.free()
    wins   = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]
    opens  = [t for t in trades if t.outcome == "OPEN"]
    closed = wins + losses
    win_pct    = len(wins) / len(closed) * 100 if closed else 0.0
    pnl_r      = sum(net_r(t, fees) for t in trades)
    avg_win_r  = sum(net_r(t, fees) for t in wins)   / len(wins)   if wins   else 0.0
    avg_los_r  = sum(net_r(t, fees) for t in losses) / len(losses) if losses else 0.0
    win_rate   = len(wins)   / len(closed) if closed else 0.0
    loss_rate  = len(losses) / len(closed) if closed else 0.0
    expectancy = win_rate * avg_win_r + loss_rate * avg_los_r
    pnl_usd    = pnl_r * risk
    risked_usd = len(trades) * risk
    roi_pct    = pnl_usd / risked_usd * 100 if risked_usd else 0.0
    return dict(
        n=len(trades), wins=len(wins), losses=len(losses), opens=len(opens),
        win_pct=win_pct, avg_win_r=avg_win_r, pnl_r=pnl_r, expectancy=expectancy,
        pnl_usd=pnl_usd, risked_usd=risked_usd, roi_pct=roi_pct,
    )


def _row(label: str, s: dict) -> str:
    sign     = "+" if s["pnl_r"]     >= 0 else ""
    exp_sign = "+" if s["expectancy"] >= 0 else ""
    usd_sign = "+" if s["pnl_usd"]   >= 0 else ""
    return (
        f"  {label:<22} {s['n']:>7} {s['wins']:>5} {s['losses']:>5} {s['opens']:>5}"
        f"  {s['win_pct']:>5.0f}%  {s['avg_win_r']:>7.2f}R"
        f"  {sign}{s['pnl_r']:>7.1f}R"
        f"  {usd_sign}${s['pnl_usd']:>8.2f}"
        f"  {s['roi_pct']:>+6.1f}%"
        f"  {exp_sign}{s['expectancy']:>5.2f}R"
    )


def _print_section(title: str, trades: list[TradeResult], risk: float = 1.0, fees: Fees | None = None) -> None:
    if not trades:
        return

    by_pattern: dict[str, list[TradeResult]] = defaultdict(list)
    for t in trades:
        by_pattern[t.pattern_name].append(t)

    hdr = (
        f"  {'Pattern':<22} {'Signals':>7} {'Win':>5} {'Loss':>5} {'Open':>5}"
        f"  {'Win%':>5}  {'AvgWin':>8}  {'P&L (R)':>8}  {'P&L ($)':>9}  {'ROI':>6}  {'Expect':>6}"
    )
    div = "  " + "-" * (len(hdr) - 2)

    print(f"\n{'═' * len(hdr)}")
    print(f"  {title}")
    print(f"{'═' * len(hdr)}")
    print(hdr)
    print(div)

    for pname in sorted(by_pattern):
        print(_row(pname, _stats(by_pattern[pname], risk, fees)))

    print(div)
    print(_row("TOTAL", _stats(trades, risk, fees)))
    print(f"{'═' * len(hdr)}")


def print_backtest_results(
    results: list[TradeResult],
    risk: float = 1.0,
    date_ranges: dict[str, tuple[str, str]] | None = None,
    fees: Fees | None = None,
) -> None:
    if not results:
        print("\nNo trades generated.")
        return

    fees = fees or Fees.free()
    total_risked = len(results) * risk
    print(f"\n  Total trades : {len(results)}")
    print(f"  Risk/trade   : ${risk:.2f}")
    print(f"  Total risked : ${total_risked:,.2f}")
    print("  Rolling TP   : SL advances to entry on 1st TP hit, then to each prior level.")
    if fees.taker_side > 0 or fees.maker_side > 0:
        tp_leg = f"{fees.maker_side*100:.3f}% maker" if fees.maker_tp else f"{fees.taker_side*100:.3f}% taker"
        print(f"  Fees         : entry {fees.taker_side*100:.3f}% taker + exit "
              f"({fees.taker_side*100:.3f}% taker on stops, {tp_leg} on TP fills); "
              f"charged per trade as fee/risk%.")
    else:
        print("  Fees         : NONE (gross, fee-free).")
    if date_ranges:
        for tf, (start, end) in sorted(date_ranges.items()):
            print(f"  {tf:<5}          : {start} → {end}")

    _print_section("ALL SYMBOLS / ALL TIMEFRAMES", results, risk, fees)

    by_tf: dict[str, list[TradeResult]] = defaultdict(list)
    for t in results:
        by_tf[t.timeframe].append(t)
    for tf in sorted(by_tf):
        _print_section(f"Timeframe: {tf}", by_tf[tf], risk, fees)

    _print_section("REVERSAL setups",     [t for t in results if t.setup == "REVERSAL"],     risk, fees)
    _print_section("CONTINUATION setups", [t for t in results if t.setup == "CONTINUATION"], risk, fees)


# ── % Wallet simulation ────────────────────────────────────────────────────────

def simulate_pct_wallet(
    results: list[TradeResult],
    starting_wallet: float,
    risk_pct: float,
    fees: Fees | None = None,
) -> None:
    if not results:
        print("\nNo trades to simulate.")
        return

    fees = fees or Fees.free()

    trades = sorted(results, key=lambda t: t.signal_date)

    wallet = starting_wallet
    peak   = wallet
    max_dd = 0.0
    wins = losses = 0

    # Monthly equity snapshots  {year-month: wallet}
    snapshots: dict[str, float] = {}
    current_month = trades[0].signal_date.strftime("%Y-%m")
    snapshots[current_month] = wallet

    for t in trades:
        risk    = wallet * risk_pct / 100
        wallet  = max(0.0, wallet + net_r(t, fees) * risk)
        peak    = max(peak, wallet)
        dd      = (peak - wallet) / peak * 100 if peak > 0 else 0.0
        max_dd  = max(max_dd, dd)
        if t.outcome == "WIN":
            wins += 1
        elif t.outcome == "LOSS":
            losses += 1
        month = t.signal_date.strftime("%Y-%m")
        snapshots[month] = wallet

    total_return = (wallet - starting_wallet) / starting_wallet * 100
    width = 62

    print(f"\n{'═' * width}")
    print(f"  % WALLET SIMULATION")
    print(f"{'═' * width}")
    print(f"  Starting balance : ${starting_wallet:>12,.2f}")
    print(f"  Final balance    : ${wallet:>12,.2f}  ({total_return:+.1f}%)")
    print(f"  Risk per trade   : {risk_pct}% of wallet")
    print(f"  Max drawdown     : {max_dd:.1f}%")
    print(f"  Trades           : {len(trades)}  ({wins} wins / {losses} losses)")
    print(f"  Period           : {trades[0].signal_date.date()} → {trades[-1].signal_date.date()}")
    print(f"\n  {'Month':<10}  {'Balance':>12}  {'Change':>8}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*8}")

    prev = starting_wallet
    for month in sorted(snapshots):
        bal    = snapshots[month]
        change = (bal - prev) / prev * 100 if prev > 0 else 0.0
        sign   = "+" if change >= 0 else ""
        print(f"  {month:<10}  ${bal:>12,.2f}  {sign}{change:>6.1f}%")
        prev = bal

    print(f"{'═' * width}")
