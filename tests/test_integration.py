"""Integration tests that verify the scan pipeline end-to-end with mocks."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from src.alert_tracker import AlertTracker
from src.alerter import TelegramAlerter
from src.data_fetcher import fetch_ohlcv
from src.models import AppConfig, DetectedPattern
from src.pattern_detector import detect_patterns, DISPLAY_NAMES


def _make_ohlcv_data(n=50):
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    closes = 42000.0 + np.cumsum(np.random.randn(n) * 100)
    opens = closes + np.random.randn(n) * 50
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 30)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 30)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1000] * n},
        index=dates,
    )


def test_full_scan_pipeline(tmp_path):
    """Simulate a full scan: fetch -> detect -> track -> alert."""
    df = _make_ohlcv_data()
    patterns = list(DISPLAY_NAMES.keys())

    # Detect patterns
    detected = detect_patterns(df, "BTC/USDT", "1h", patterns)

    # Alert tracking
    tracker = AlertTracker(str(tmp_path / "state.json"))
    new_alerts = []
    for p in detected:
        if not tracker.is_duplicate(p.alert_key):
            new_alerts.append(p)
            tracker.record(p.alert_key)

    # Re-run detection — should produce no new alerts
    detected2 = detect_patterns(df, "BTC/USDT", "1h", patterns)
    repeat_alerts = []
    for p in detected2:
        if not tracker.is_duplicate(p.alert_key):
            repeat_alerts.append(p)

    assert repeat_alerts == [], "No new alerts should be generated for the same data"


def test_different_candle_generates_new_alert(tmp_path):
    """A new candle with the same pattern should generate a new alert."""
    tracker = AlertTracker(str(tmp_path / "state.json"))

    p1 = DetectedPattern(
        symbol="BTC/USDT",
        timeframe="1h",
        pattern_name="Doji",
        signal="bullish",
        close_price=42000.0,
        candle_timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    p2 = DetectedPattern(
        symbol="BTC/USDT",
        timeframe="1h",
        pattern_name="Doji",
        signal="bullish",
        close_price=42100.0,
        candle_timestamp=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
    )

    assert p1.alert_key != p2.alert_key

    tracker.record(p1.alert_key)
    assert tracker.is_duplicate(p1.alert_key)
    assert not tracker.is_duplicate(p2.alert_key)


def test_alert_message_format():
    p = DetectedPattern(
        symbol="ETH/USDT",
        timeframe="4h",
        pattern_name="Morning Star",
        signal="bullish",
        close_price=2500.50,
        candle_timestamp=datetime(2024, 3, 15, 8, 0, tzinfo=timezone.utc),
    )
    msg = p.format_message()
    assert "Morning Star" in msg
    assert "bullish" in msg
    assert "ETH/USDT" in msg
    assert "4h" in msg
    assert "2500.5" in msg


@pytest.mark.asyncio
async def test_alerter_integration():
    """Test that the alerter correctly calls the Telegram API."""
    with patch("src.alerter.telegram.Bot") as MockBot:
        mock_bot = MockBot.return_value
        mock_bot.send_message = AsyncMock()

        alerter = TelegramAlerter("tok", "chat")
        p = DetectedPattern(
            symbol="SOL/USDT",
            timeframe="15m",
            pattern_name="Shooting Star",
            signal="bearish",
            close_price=150.0,
            candle_timestamp=datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc),
        )
        result = await alerter.send_alert(p)
        assert result is True
        mock_bot.send_message.assert_awaited_once()
