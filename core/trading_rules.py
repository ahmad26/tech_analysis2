from __future__ import annotations

import math

from core.models import DetectedPattern, TradingSignal

# Minimum acceptable reward-to-risk ratio.
# TP is selected as the nearest Fib level that satisfies this threshold.
# If no level qualifies, the furthest available level is used as fallback.
MIN_RR = 1.5

# Minimum SL distance as a fraction of entry price.
# Signals where risk is smaller than this are skipped (degenerate near-zero SL).
MIN_RISK_PCT = 0.001  # 0.1%

# SL buffer: push SL this many ATRs beyond the support/resistance level so that
# a wick touching the level does not immediately trigger the stop.
SL_BUFFER_ATR = 0.25

# Round-number (psychological-level) awareness on the TAKE-PROFIT. EXPERIMENT,
# default off; flip via --psych-round / module globals (like MAKER_TP). Order flow
# clusters at round numbers, which act as S/R: a TP that sits just BEYOND a round
# number (e.g. a Fib-extreme target at 6.996, just under 7) often never tags it —
# price stalls at the number and reverses. So pull such a TP to just BEFORE the
# number, where fills actually happen. Grid auto-scales with price magnitude:
# 10^floor(log10 price) and its half-step (LINK ~7 -> 1.0 & 0.5; BTC ~68k -> 10000
# & 5000). Distances are ATR-relative.
#
# NOTE: the mirror idea on the STOP (push it PAST the round number) was implemented
# and backtested but REMOVED — it was net-negative on both 1d and 4h (the larger
# loss when the level breaks outweighs the wick stop-outs it dodges; the stop
# already has SL_BUFFER_ATR). Only the TP-pull survived. See the 2026-06-26 finding.
# Per-TF: a 5000-candle (8.3yr 1d / 2.3yr 4h) OOS split on 2026-06-26 showed the
# TP-pull is robustly positive on 1d in BOTH halves (+0.004/+0.012R, ~+19R aggregate)
# but a tiny net drag on 4h (-0.003/-0.001R both halves). So it ships ON, gated to 1d.
# PSYCH_ROUND_TFS = None means "all timeframes" (used by the --psych-round experiment
# flag to sweep arbitrary TFs). See memory/finding_psych_round_tp.md.
PSYCH_ROUND = True
PSYCH_ROUND_TFS: set[str] | None = {"1d"}   # TFs the pull applies to (None = all)
PSYCH_BAND_ATR = 0.5      # within this many ATR beyond a round number => adjust
PSYCH_BUFFER_ATR = 0.1    # rest the adjusted TP this many ATR on the near side

_FIB_FRACS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
_FIB_LABELS = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]


def _all_fib_levels(high: float, low: float, source: str) -> list[tuple[float, str]]:
    r = high - low
    return [(low + f * r, f"{source} {lbl}") for f, lbl in zip(_FIB_FRACS, _FIB_LABELS)]


def _round_grid(price: float) -> tuple[float, float]:
    """Psychological-level grid for a price's magnitude: the full decade step and
    its half-step (e.g. price 7 -> (1.0, 0.5); 68000 -> (10000, 5000))."""
    step = 10.0 ** math.floor(math.log10(price))
    return step, step / 2.0


def _psych_adjust_tp(price: float, entry: float, action: str, atr: float | None,
                     tf: str | None = None) -> float:
    """Pull a TP that sits just BEYOND a round number to just BEFORE it (the near
    side, where fills cluster). Full step is checked before the half-step so the
    stronger barrier wins when both apply. Never crosses entry.

    Gated per-TF: only applied when PSYCH_ROUND_TFS is None (all) or contains `tf`."""
    if not PSYCH_ROUND or atr is None or atr <= 0 or price <= 0:
        return price
    if PSYCH_ROUND_TFS is not None and tf not in PSYCH_ROUND_TFS:
        return price
    band, buf = atr * PSYCH_BAND_ATR, atr * PSYCH_BUFFER_ATR
    for g in _round_grid(price):
        if action == "SELL":                       # TP below entry; round level above it
            r = math.ceil(price / g) * g
            if price < r <= entry and (r - price) <= band:
                return min(r + buf, entry)
        else:                                       # BUY; TP above entry; round level below it
            r = math.floor(price / g) * g
            if entry <= r < price and (price - r) <= band:
                return max(r - buf, entry)
    return price


def _tp_candidates(
    entry: float,
    action: str,
    high50: float, low50: float,
    high200: float, low200: float,
    extra: list[tuple[float, str]],
    atr: float | None = None,
    tf: str | None = None,
) -> list[tuple[float, str]]:
    """All Fib levels on the TP side of entry, sorted closest-first. MAs passed as extra."""
    raw = _all_fib_levels(high50, low50, "Fib50") + _all_fib_levels(high200, low200, "Fib200") + extra
    raw = [(_psych_adjust_tp(p, entry, action, atr, tf), s) for p, s in raw]

    if action == "BUY":
        candidates = [(p, s) for p, s in raw if p > entry]
        candidates.sort(key=lambda x: x[0])
    else:
        candidates = [(p, s) for p, s in raw if p < entry]
        candidates.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by price (keep first label when two windows share a level)
    seen: set[int] = set()
    result: list[tuple[float, str]] = []
    for price, src in candidates:
        key = round(price, 6)
        key_int = int(key * 1_000_000)
        if key_int not in seen:
            seen.add(key_int)
            result.append((price, src))
    return result


def compute_trading_signal(
    p: DetectedPattern,
    high50: float,
    low50: float,
    high200: float,
    low200: float,
    atr: float | None = None,
    htf_levels: list[tuple[float, str]] | None = None,
) -> TradingSignal | None:
    if p.fib50 is None or p.fib200 is None:
        return None

    (_, p_lo50), (_, p_hi50) = p.fib50
    (_, p_lo200), (_, p_hi200) = p.fib200

    # Tightest bracket: highest floor and lowest ceiling across both windows
    if p_lo50 >= p_lo200:
        tight_lo, tight_lo_src = p_lo50, f"Fib50 {p.fib50[0][0]}"
    else:
        tight_lo, tight_lo_src = p_lo200, f"Fib200 {p.fib200[0][0]}"

    if p_hi50 <= p_hi200:
        tight_hi, tight_hi_src = p_hi50, f"Fib50 {p.fib50[1][0]}"
    else:
        tight_hi, tight_hi_src = p_hi200, f"Fib200 {p.fib200[1][0]}"

    close = p.close_price
    action = "BUY" if p.signal == "bullish" else "SELL"

    if p.ma_position == "above_both":
        setup = "CONTINUATION" if action == "BUY" else "REVERSAL"
    elif p.ma_position == "below_both":
        setup = "CONTINUATION" if action == "SELL" else "REVERSAL"
    else:
        setup = "CONTINUATION"  # "between" or unknown — no clear trend to reverse

    # SL: pick the closest level on the wrong side of entry.
    # Own-TF MAs + Fib levels + higher-TF levels all compete equally.
    # Higher-TF levels are labelled (e.g. "4h MA50") so their origin is visible.
    all_sl_raw: list[tuple[float, str]] = (
        _all_fib_levels(high50, low50, "Fib50")
        + _all_fib_levels(high200, low200, "Fib200")
    )
    for val, name in [(p.ma7, "MA7"), (p.ma25, "MA25"), (p.ma50, "MA50"), (p.ma99, "MA99"), (p.ma200, "MA200")]:
        if val is not None:
            all_sl_raw.append((val, name))
    if htf_levels:
        all_sl_raw.extend(htf_levels)

    if action == "BUY":
        sl_candidates = sorted(
            [(price, src) for price, src in all_sl_raw if price < close],
            key=lambda x: x[0], reverse=True,
        )
    else:
        sl_candidates = sorted(
            [(price, src) for price, src in all_sl_raw if price > close],
            key=lambda x: x[0],
        )

    sl, sl_src = sl_candidates[0] if sl_candidates else (
        (tight_lo, tight_lo_src) if action == "BUY" else (tight_hi, tight_hi_src)
    )

    risk = abs(close - sl)
    if risk < close * MIN_RISK_PCT:
        return None

    # If SL is tighter than 1× ATR, widen to the nearest Fib/MA level
    # that is at least 1× ATR away (or use bare ATR if no level qualifies).
    if atr is not None and risk < atr:
        if action == "BUY":
            wider = [(price, src) for price, src in all_sl_raw if price < close and (close - price) >= atr]
            wider.sort(key=lambda x: x[0], reverse=True)
        else:
            wider = [(price, src) for price, src in all_sl_raw if price > close and (price - close) >= atr]
            wider.sort(key=lambda x: x[0])
        if wider:
            sl, sl_src = wider[0]
        else:
            sl = (close - atr) if action == "BUY" else (close + atr)
            sl_src = "ATR(14)"
        risk = abs(close - sl)

    # Push SL a quarter-ATR beyond the support/resistance level so a wick
    # touching the level does not immediately trigger the stop.
    if atr is not None:
        buffer = atr * SL_BUFFER_ATR
        sl = (sl - buffer) if action == "BUY" else (sl + buffer)
        risk = abs(close - sl)

    # TP candidates: own-TF MAs + HTF levels on the correct side of entry
    mas_tp: list[tuple[float, str]] = []
    for val, name in [(p.ma7, "MA7"), (p.ma25, "MA25"), (p.ma50, "MA50"), (p.ma99, "MA99"), (p.ma200, "MA200")]:
        if val is None:
            continue
        if action == "BUY" and val > close:
            mas_tp.append((val, name))
        elif action == "SELL" and val < close:
            mas_tp.append((val, name))
    if htf_levels:
        for val, name in htf_levels:
            if action == "BUY" and val > close:
                mas_tp.append((val, name))
            elif action == "SELL" and val < close:
                mas_tp.append((val, name))

    candidates = _tp_candidates(close, action, high50, low50, high200, low200, mas_tp, atr, p.timeframe)
    if not candidates:
        return None

    # Pick the closest level that meets MIN_RR; discard signal if none qualifies
    tp, tp_src = None, None
    for price, src in candidates:
        if abs(price - close) >= risk * MIN_RR:
            tp, tp_src = price, src
            break

    if tp is None:
        return None

    # If a structural level sits between entry and the Fib TP, the closest one
    # becomes TP1 and the Fib TP is demoted to TP2. Only strong, structural
    # levels count as obstacles: own-TF MA50/MA200 and higher-TF levels.
    # Fast MAs (MA7/MA25/MA99) hug price and would otherwise hijack TP1 right
    # next to entry, collapsing R/R and dropping otherwise-valid signals.
    # Higher-TF levels are weighted first since they are stronger barriers.
    tp2: float | None = None
    tp2_src: str | None = None
    own_ma_levels = [(v, n) for v, n in [(p.ma50, "MA50"), (p.ma200, "MA200")] if v is not None]
    all_ma_levels = (htf_levels or []) + own_ma_levels  # HTF first — higher priority
    if action == "BUY":
        between = sorted(
            [(v, n) for v, n in all_ma_levels if close < v < tp],
            key=lambda x: x[0],
        )
    else:
        between = sorted(
            [(v, n) for v, n in all_ma_levels if tp < v < close],
            key=lambda x: x[0], reverse=True,
        )
    if between:
        tp2, tp2_src = tp, tp_src
        tp, tp_src = between[0]

    reward = abs(tp - close)
    rr = reward / risk

    return TradingSignal(
        action=action,
        setup=setup,
        entry=close,
        stop_loss=sl,
        sl_source=sl_src,
        take_profit=tp,
        tp_source=tp_src,
        risk_reward=rr,
        all_tp_candidates=candidates,
        take_profit_2=tp2,
        tp_source_2=tp2_src,
    )
