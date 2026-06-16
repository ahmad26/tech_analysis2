import pandas as pd
import numpy as np
import pytest

from src.pattern_detector import detect_patterns, DISPLAY_NAMES


def test_detect_patterns_returns_list(sample_ohlcv_df):
    results = detect_patterns(
        sample_ohlcv_df, "BTC/USDT", "1h",
        ["hammer", "doji", "shooting_star"],
    )
    assert isinstance(results, list)
    for r in results:
        assert r.symbol == "BTC/USDT"
        assert r.timeframe == "1h"
        assert r.signal in ("bullish", "bearish")


def test_detect_patterns_too_few_candles():
    dates = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [101, 102], "volume": [1, 2]},
        index=dates,
    )
    results = detect_patterns(df, "BTC/USDT", "1h", ["doji"])
    assert results == []


def test_detect_patterns_unknown_pattern(sample_ohlcv_df):
    results = detect_patterns(
        sample_ohlcv_df, "BTC/USDT", "1h", ["nonexistent_pattern"]
    )
    assert results == []


def test_detect_patterns_second_to_last_candle(sample_ohlcv_df):
    """Verify detection uses the second-to-last candle timestamp."""
    results = detect_patterns(
        sample_ohlcv_df, "BTC/USDT", "1h",
        list(DISPLAY_NAMES.keys()),
    )
    expected_ts = sample_ohlcv_df.index[-2]
    for r in results:
        assert r.candle_timestamp == expected_ts.to_pydatetime()


def test_detect_engulfing_direction():
    """Test that bullish/bearish engulfing only fires for the correct direction."""
    dates = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
    np.random.seed(99)
    n = 20
    closes = 100 + np.cumsum(np.random.randn(n) * 2)
    opens = closes + np.random.randn(n)
    highs = np.maximum(opens, closes) + 1
    lows = np.minimum(opens, closes) - 1

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [100] * n},
        index=dates,
    )
    df.index.name = "timestamp"

    bullish = detect_patterns(df, "BTC/USDT", "1h", ["bullish_engulfing"])
    for p in bullish:
        assert p.signal == "bullish"

    bearish = detect_patterns(df, "BTC/USDT", "1h", ["bearish_engulfing"])
    for p in bearish:
        assert p.signal == "bearish"


def test_hammer_detection(hammer_df):
    results = detect_patterns(hammer_df, "BTC/USDT", "1h", ["hammer"])
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].pattern_name == "Hammer"
    assert results[0].signal == "bullish"


def test_doji_detection():
    """A candle with open==close should be detected as doji."""
    dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open":  [100, 102, 104, 100.0, 106],
            "high":  [102, 104, 106, 105.0, 108],
            "low":   [99,  101, 103, 95.0,  105],
            "close": [102, 104, 106, 100.0, 107],  # idx 3: open==close, doji
            "volume": [100] * 5,
        },
        index=dates,
    )
    results = detect_patterns(df, "BTC/USDT", "1h", ["doji"])
    assert len(results) == 1
    assert results[0].pattern_name == "Doji"


def test_shooting_star_detection():
    """Shooting star: small body at bottom, long upper shadow, no lower shadow."""
    dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    # idx=-2 is idx 3. body=0.3 (106->105.7), upper_shadow=6.3, lower_shadow=0.1
    df = pd.DataFrame(
        {
            "open":  [100, 102, 104, 106.0,  108],
            "high":  [102, 104, 106, 112.0,  110],
            "low":   [99,  101, 103, 105.6,  107],
            "close": [102, 104, 106, 105.7,  109],
            "volume": [100] * 5,
        },
        index=dates,
    )
    results = detect_patterns(df, "BTC/USDT", "1h", ["shooting_star"])
    assert len(results) == 1
    assert results[0].signal == "bearish"


def test_bullish_engulfing_detection():
    """Bearish candle followed by larger bullish candle."""
    dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            #                            prev(idx2)  target(idx3)
            "open":  [100, 102, 105.0,   103.0,  108],
            "high":  [103, 104, 106.0,   107.0,  110],
            "low":   [99,  101, 102.5,   102.0,  107],
            "close": [102, 103, 103.0,   106.0,  109],  # prev: bearish (105->103), target: bullish (103->106) engulfs
            "volume": [100] * 5,
        },
        index=dates,
    )
    results = detect_patterns(df, "BTC/USDT", "1h", ["bullish_engulfing"])
    assert len(results) == 1
    assert results[0].signal == "bullish"


def test_morning_star_detection():
    """Three-candle morning star: bearish, small body, bullish closing above midpoint.

    For 6 candles, idx=-2 is idx 4 (target candle).
    Three-candle check uses idx-4=idx 2 (c0), idx-3=idx 3 (c1), idx-2=idx 4 (c2).
    """
    dates = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            #              idx0   idx1    idx2(c0)  idx3(c1)  idx4(target)  idx5
            "open":  [110, 110,   108.0,  100.0,    105.0,    110],
            "high":  [112, 112,   109.0,  100.5,    108.0,    112],
            "low":   [108, 108,   100.0,  99.5,     104.0,    109],
            "close": [109, 109,   100.0,  100.1,    107.0,    111],
            # c0(idx2): bearish 108->100, c1(idx3): tiny body 100->100.1
            # target(idx4): bullish 105->107, closes above mid of c0 = (108+100)/2 = 104
            "volume": [100] * 6,
        },
        index=dates,
    )
    results = detect_patterns(df, "BTC/USDT", "1h", ["morning_star"])
    assert len(results) == 1
    assert results[0].signal == "bullish"
    assert results[0].pattern_name == "Morning Star"
