import pandas as pd
import numpy as np
import pytest

from src.models import AppConfig


@pytest.fixture
def sample_config(tmp_path):
    return AppConfig(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1h", "4h"],
        patterns=["hammer", "doji", "bullish_engulfing", "bearish_engulfing"],
        exchange="binance",
        telegram_bot_token="test-token",
        telegram_chat_id="test-chat-id",
        state_file=str(tmp_path / "test_state.json"),
    )


@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame with 50 candles."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base = 42000.0
    closes = base + np.cumsum(np.random.randn(n) * 100)
    opens = closes + np.random.randn(n) * 50
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 30)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 30)
    volumes = np.abs(np.random.randn(n) * 1000) + 500

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def hammer_df():
    """DataFrame where the second-to-last candle forms a hammer pattern.

    Hammer: small real body at the top of the range, long lower shadow (>=2x body),
    little or no upper shadow.
    """
    dates = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")

    # Create a downtrend followed by a hammer
    opens = [100, 98, 96, 94, 92, 90, 88, 86, 85.0, 87]
    closes = [98, 96, 94, 92, 90, 88, 86, 84, 85.5, 88]
    highs = [101, 99, 97, 95, 93, 91, 89, 87, 85.7, 89]
    lows = [97, 95, 93, 91, 89, 87, 85, 83, 82.0, 86]  # idx 8: long lower shadow

    # Overwrite candle at index 8 (second-to-last) to be a clear hammer:
    # open=85, close=85.5, high=85.7, low=82 -> body=0.5, lower_shadow=3.0, upper_shadow=0.2
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000] * 10,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df
