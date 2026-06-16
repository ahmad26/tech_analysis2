from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models import DetectedPattern, FibBound
from src.trading_rules import compute_trading_signal

logger = logging.getLogger(__name__)

_DOJI_THRESHOLD = 0.10        # α: body/range <= 10% qualifies as Doji
_DOJI_SHADOW_BETA = 0.10      # β: max upper/lower shadow for Dragonfly/Gravestone
_LONG_LEGGED_K = 1.5          # range must exceed k × avg_range to be Long-Legged
_LONG_LEGGED_PERIODS = 20     # lookback for avg range in Long-Legged classification
_SMALL_BODY_THRESHOLD = 0.15
_SHADOW_RATIO = 2.0

DISPLAY_NAMES = {
    "hammer": "Hammer",
    "inverted_hammer": "Inverted Hammer",
    "bullish_engulfing": "Bullish Engulfing",
    "bearish_engulfing": "Bearish Engulfing",
    "doji": "Doji",
    "morning_star": "Morning Star",
    "evening_star": "Evening Star",
    "shooting_star": "Shooting Star",
    "dragonfly_doji": "Dragonfly Doji",
    "gravestone_doji": "Gravestone Doji",
    "long_legged_doji": "Long-Legged Doji",
}


def _body(o: float, c: float) -> float:
    return abs(c - o)


def _range(h: float, l: float) -> float:
    return h - l


def _is_doji(o: float, h: float, l: float, c: float) -> bool:
    r = _range(h, l)
    if r == 0:
        return True
    return _body(o, c) / r <= _DOJI_THRESHOLD


def _is_hammer(o: float, h: float, l: float, c: float) -> bool:
    """Small body near top, long lower shadow, tiny upper shadow."""
    body = _body(o, c)
    r = _range(h, l)
    if r == 0 or body == 0:
        return False
    body_top = max(o, c)
    body_bottom = min(o, c)
    upper_shadow = h - body_top
    lower_shadow = body_bottom - l
    return (
        lower_shadow >= _SHADOW_RATIO * body
        and upper_shadow <= body * 0.5
        and body / r <= 0.4
    )


def _is_inverted_hammer(o: float, h: float, l: float, c: float) -> bool:
    """Small body near bottom, long upper shadow, tiny lower shadow."""
    body = _body(o, c)
    r = _range(h, l)
    if r == 0 or body == 0:
        return False
    body_top = max(o, c)
    body_bottom = min(o, c)
    upper_shadow = h - body_top
    lower_shadow = body_bottom - l
    return (
        upper_shadow >= _SHADOW_RATIO * body
        and lower_shadow <= body * 0.5
        and body / r <= 0.4
    )


def _is_shooting_star(o: float, h: float, l: float, c: float) -> bool:
    """Same shape as inverted hammer but bearish context (checked separately)."""
    return _is_inverted_hammer(o, h, l, c)


def _check_single_candle(
    pattern: str,
    o: float, h: float, l: float, c: float,
) -> str | None:
    if pattern == "hammer":
        if _is_hammer(o, h, l, c):
            return "bullish"
    return None


def _check_doji_contextual(df: pd.DataFrame, target: int) -> list[tuple[str, str]]:
    """Context-aware doji detection. Returns list of (pattern_name, signal) pairs."""
    n = len(df)
    o = float(df["open"].iloc[target])
    h = float(df["high"].iloc[target])
    l = float(df["low"].iloc[target])
    c = float(df["close"].iloc[target])

    r = _range(h, l)
    if r == 0 or not _is_doji(o, h, l, c):
        return []

    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    # Classify doji shape using the formal ratio-based definition:
    # Dragonfly:  body at top  → upper_shadow/range <= β
    # Gravestone: body at bottom → lower_shadow/range <= β
    # Long-Legged: range is wide relative to recent average (range > k × avg_range)
    # Standard:   everything else
    if upper_shadow / r <= _DOJI_SHADOW_BETA:
        doji_type = "dragonfly"
    elif lower_shadow / r <= _DOJI_SHADOW_BETA:
        doji_type = "gravestone"
    else:
        start = max(0, target - _LONG_LEGGED_PERIODS)
        avg_range = (df["high"].iloc[start:target] - df["low"].iloc[start:target]).mean()
        if avg_range > 0 and r > _LONG_LEGGED_K * avg_range:
            doji_type = "long_legged"
        else:
            doji_type = "standard"

    if target > 0:
        o0 = float(df["open"].iloc[target - 1])
        c0 = float(df["close"].iloc[target - 1])
        h0 = float(df["high"].iloc[target - 1])
        l0 = float(df["low"].iloc[target - 1])
        prev_bullish = c0 > o0
        prev_bearish = c0 < o0
    else:
        prev_bullish = prev_bearish = False
        h0 = l0 = 0.0

    if doji_type == "dragonfly":
        # Shadow breach below prior low failed to hold — continuation bearish
        if prev_bearish and l < l0:
            return [("Dragonfly Doji", "bearish")]
        return []

    if doji_type == "gravestone":
        # Shadow breach above prior high failed to hold — continuation bullish
        if prev_bullish and h > h0:
            return [("Gravestone Doji", "bullish")]
        return []

    # Standard / long-legged: direction from confirmation candle
    if target + 1 >= n:
        return []
    c1 = float(df["close"].iloc[target + 1])
    o1 = float(df["open"].iloc[target + 1])
    name = "Long-Legged Doji" if doji_type == "long_legged" else "Doji"
    if c1 > o1:
        return [(name, "bullish")]
    if c1 < o1:
        return [(name, "bearish")]
    return []  # confirmation is also a doji — no clear direction


def _check_two_candle(
    pattern: str,
    o0: float, h0: float, l0: float, c0: float,
    o1: float, h1: float, l1: float, c1: float,
) -> str | None:
    if pattern == "bullish_engulfing":
        prev_bearish = c0 < o0
        curr_bullish = c1 > o1
        engulfs = c1 >= o0 and o1 <= c0
        if prev_bearish and curr_bullish and engulfs:
            return "bullish"
    elif pattern == "bearish_engulfing":
        prev_bullish = c0 > o0
        curr_bearish = c1 < o1
        engulfs = c1 <= o0 and o1 >= c0
        if prev_bullish and curr_bearish and engulfs:
            return "bearish"
    elif pattern == "inverted_hammer":
        # Bullish reversal: inverted-hammer shape after a bearish candle (downtrend context)
        if _is_inverted_hammer(o1, h1, l1, c1) and c0 < o0:
            return "bullish"
    elif pattern == "shooting_star":
        # Bearish reversal: same shape after a bullish candle (uptrend context)
        if _is_shooting_star(o1, h1, l1, c1) and c0 > o0:
            return "bearish"
    return None


def _check_three_candle(
    pattern: str,
    o0: float, h0: float, l0: float, c0: float,
    o1: float, h1: float, l1: float, c1: float,
    o2: float, h2: float, l2: float, c2: float,
) -> str | None:
    r1 = _range(h1, l1)
    body1_ratio = _body(o1, c1) / r1 if r1 > 0 else 0
    small_middle = body1_ratio <= _SMALL_BODY_THRESHOLD

    if pattern == "morning_star":
        first_bearish = c0 < o0 and _body(o0, c0) > 0
        third_bullish = c2 > o2 and _body(o2, c2) > 0
        closes_above_mid = c2 > (o0 + c0) / 2
        if first_bearish and small_middle and third_bullish and closes_above_mid:
            return "bullish"
    elif pattern == "evening_star":
        first_bullish = c0 > o0 and _body(o0, c0) > 0
        third_bearish = c2 < o2 and _body(o2, c2) > 0
        closes_below_mid = c2 < (o0 + c0) / 2
        if first_bullish and small_middle and third_bearish and closes_below_mid:
            return "bearish"
    return None


_FIB_LEVELS: list[tuple[float, str]] = [
    (0.0,   "0%"),
    (0.236, "23.6%"),
    (0.382, "38.2%"),
    (0.500, "50%"),
    (0.618, "61.8%"),
    (0.786, "78.6%"),
    (1.0,   "100%"),
]


def _fib_bounds(high: float, low: float, price: float) -> FibBound:
    """Return the nearest Fib level below and above price within [low, high]."""
    r = high - low
    levels = [(label, low + frac * r) for frac, label in _FIB_LEVELS]
    lower = levels[0]
    upper = levels[-1]
    for label, lp in levels:
        if lp <= price:
            lower = (label, lp)
    for label, lp in reversed(levels):
        if lp >= price:
            upper = (label, lp)
    return lower, upper


def _compute_atr(df: pd.DataFrame, target: int, period: int = 14) -> tuple[float | None, float | None, float | None]:
    """Compute ATR(period) and the recent price range high/low.

    Returns (atr, range_high, range_low). Returns (None, None, None) if not enough data.
    """
    if target < period:
        return None, None, None

    highs = df["high"].iloc[target - period + 1: target + 1].values
    lows = df["low"].iloc[target - period + 1: target + 1].values
    prev_closes = df["close"].iloc[target - period: target].values

    tr = np.maximum(highs - lows,
         np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))

    return float(tr.mean()), float(highs.max()), float(lows.min())


def _compute_context(
    df: pd.DataFrame, target: int
) -> tuple:
    """Compute MAs, MA position, Fibonacci brackets, window ranges, and ATR(14).

    Returns (ma7, ma25, ma50, ma99, ma200, ma_position,
             fib50, fib200, high50, low50, high200, low200,
             atr, atr_range_high, atr_range_low).
    ma_position ("above_both" | "below_both" | "between" | None) is based on MA50/MA200.
    """
    close = float(df["close"].iloc[target])
    roll = df["close"].rolling

    def _ma(period: int) -> float | None:
        v = roll(period).mean().iloc[target]
        return None if np.isnan(v) else float(v)

    ma7   = _ma(7)
    ma25  = _ma(25)
    ma50  = _ma(50)
    ma99  = _ma(99)
    ma200 = _ma(200)

    if ma50 is not None and ma200 is not None:
        if close > ma50 and close > ma200:
            ma_position: str | None = "above_both"
        elif close < ma50 and close < ma200:
            ma_position = "below_both"
        else:
            ma_position = "between"
    elif ma50 is not None:
        ma_position = "above_both" if close > ma50 else "below_both"
    else:
        ma_position = None

    w50 = df.iloc[max(0, target - 49):target + 1]
    h50, l50 = float(w50["high"].max()), float(w50["low"].min())
    fib50 = _fib_bounds(h50, l50, close)

    w200 = df.iloc[max(0, target - 199):target + 1]
    h200, l200 = float(w200["high"].max()), float(w200["low"].min())
    fib200 = _fib_bounds(h200, l200, close)

    atr, atr_range_high, atr_range_low = _compute_atr(df, target)

    return ma7, ma25, ma50, ma99, ma200, ma_position, fib50, fib200, h50, l50, h200, l200, atr, atr_range_high, atr_range_low


_SINGLE_CANDLE_PATTERNS = {"hammer"}  # doji handled separately via _check_doji_contextual
_TWO_CANDLE_PATTERNS = {"bullish_engulfing", "bearish_engulfing", "inverted_hammer", "shooting_star"}
_THREE_CANDLE_PATTERNS = {"morning_star", "evening_star"}

# Which timeframes to pull key levels from when scanning a lower timeframe.
# Higher-TF levels are stronger support/resistance and treated as primary obstacles.
HTF_MAP: dict[str, list[str]] = {
    "15m": ["1h", "4h", "1d"],
    "1h":  ["4h", "1d"],
    "4h":  ["1d"],
    "1d":  [],
}

# Candles to fetch when pre-loading higher-TF data (needs enough for MA200).
HTF_CANDLES = 250


def extract_htf_levels(df: pd.DataFrame, timeframe: str) -> list[tuple[float, str]]:
    """Extract MA50, MA200, and key Fib levels from a higher-timeframe dataframe.

    Returns a list of (price, label) pairs to be used as additional SL/TP/obstacle
    candidates when analysing a lower timeframe signal.
    """
    if len(df) < 2:
        return []
    target = len(df) - 2
    levels: list[tuple[float, str]] = []

    # Only the structural MAs (50, 200) are emitted as cross-TF levels. Fast MAs
    # (7/25/99) hug price and would hijack TP1 right next to entry, collapsing
    # R/R — matches the documented contract (MA50, MA200, key Fib levels).
    for period in (50, 200):
        raw = df["close"].rolling(period).mean().iloc[target]
        if not np.isnan(raw):
            levels.append((float(raw), f"{timeframe} MA{period}"))

    w50 = df.iloc[max(0, target - 49):target + 1]
    h50, l50 = float(w50["high"].max()), float(w50["low"].min())
    r = h50 - l50
    if r > 0:
        for frac, lbl in _FIB_LEVELS:
            levels.append((l50 + frac * r, f"{timeframe} Fib {lbl}"))

    return levels


def detect_patterns(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    patterns: list[str],
    htf_levels: list[tuple[float, str]] | None = None,
    min_atr_pct: float | None = None,
) -> list[DetectedPattern]:
    if len(df) < 4:
        logger.warning("Not enough candles for %s %s", symbol, timeframe)
        return []

    n = len(df)
    target = n - 2  # last completed candle (second-to-last row)

    o = df["open"].iloc[target]
    h = df["high"].iloc[target]
    l = df["low"].iloc[target]
    c = df["close"].iloc[target]
    candle_ts = df.index[target]

    ma7, ma25, ma50, ma99, ma200, ma_position, fib50, fib200, h50, l50, h200, l200, atr, atr_range_high, atr_range_low = _compute_context(df, target)

    # Low-volatility regime filter: reversal/continuation candle patterns whipsaw
    # in flat, quiet markets. Skip the candle when normalized volatility (ATR/price)
    # is below the per-timeframe floor. (A Donchian-breakout "rescue" was tried as an
    # alternative to muting but backtested net-negative on both 4h and 15m — the
    # rescued sub-floor trades have the tightest stops and so the worst fee/reward;
    # reverted to the hard mute. See memory/feature_atr_vol_filter.md.)
    if min_atr_pct is not None and atr is not None and c > 0 and (atr / c) < min_atr_pct:
        return []

    detected: list[DetectedPattern] = []

    for pattern in patterns:
        signal = None

        if pattern == "doji":
            # Use target-1 as the doji candle so both the doji and its confirmation
            # (target) are fully closed. target+1 would be the live forming candle.
            doji_ts = df.index[target - 1].to_pydatetime()
            for doji_name, doji_signal in _check_doji_contextual(df, target - 1):
                dp = DetectedPattern(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_name=doji_name,
                    signal=doji_signal,
                    close_price=c,       # entry at confirmation candle's close
                    candle_timestamp=doji_ts,
                    ma7=ma7,
                    ma25=ma25,
                    ma50=ma50,
                    ma99=ma99,
                    ma200=ma200,
                    ma_position=ma_position,
                    fib50=fib50,
                    fib200=fib200,
                    atr=atr,
                    atr_range_high=atr_range_high,
                    atr_range_low=atr_range_low,
                )
                dp.trading_signal = compute_trading_signal(dp, h50, l50, h200, l200, atr, htf_levels)
                detected.append(dp)
            continue

        if pattern in _SINGLE_CANDLE_PATTERNS:
            signal = _check_single_candle(pattern, o, h, l, c)

        elif pattern in _TWO_CANDLE_PATTERNS and n >= 3:
            o0 = df["open"].iloc[target - 1]
            h0 = df["high"].iloc[target - 1]
            l0 = df["low"].iloc[target - 1]
            c0 = df["close"].iloc[target - 1]
            signal = _check_two_candle(pattern, o0, h0, l0, c0, o, h, l, c)

        elif pattern in _THREE_CANDLE_PATTERNS and n >= 4:
            o0 = df["open"].iloc[target - 2]
            h0 = df["high"].iloc[target - 2]
            l0 = df["low"].iloc[target - 2]
            c0 = df["close"].iloc[target - 2]
            o1 = df["open"].iloc[target - 1]
            h1 = df["high"].iloc[target - 1]
            l1 = df["low"].iloc[target - 1]
            c1 = df["close"].iloc[target - 1]
            signal = _check_three_candle(pattern, o0, h0, l0, c0, o1, h1, l1, c1, o, h, l, c)

        if signal:
            dp = DetectedPattern(
                symbol=symbol,
                timeframe=timeframe,
                pattern_name=DISPLAY_NAMES.get(pattern, pattern),
                signal=signal,
                close_price=c,
                candle_timestamp=candle_ts.to_pydatetime(),
                ma7=ma7,
                ma25=ma25,
                ma50=ma50,
                ma99=ma99,
                ma200=ma200,
                ma_position=ma_position,
                fib50=fib50,
                fib200=fib200,
                atr=atr,
                atr_range_high=atr_range_high,
                atr_range_low=atr_range_low,
            )
            dp.trading_signal = compute_trading_signal(dp, h50, l50, h200, l200, atr, htf_levels)
            detected.append(dp)

    if detected:
        logger.info(
            "Found %d pattern(s) on %s %s: %s",
            len(detected),
            symbol,
            timeframe,
            ", ".join(p.pattern_name for p in detected),
        )
    return detected
