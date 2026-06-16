from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.alerter import TelegramAlerter
from src.models import DetectedPattern


@pytest.fixture
def sample_pattern():
    return DetectedPattern(
        symbol="BTC/USDT",
        timeframe="1h",
        pattern_name="Hammer",
        signal="bullish",
        close_price=42000.0,
        candle_timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_send_alert_success(sample_pattern):
    with patch("src.alerter.telegram.Bot") as MockBot:
        mock_bot = MockBot.return_value
        mock_bot.send_message = AsyncMock(return_value=True)

        alerter = TelegramAlerter("test-token", "test-chat")
        result = await alerter.send_alert(sample_pattern)

        assert result is True
        mock_bot.send_message.assert_awaited_once()
        call_kwargs = mock_bot.send_message.call_args
        assert call_kwargs.kwargs["chat_id"] == "test-chat"
        assert "Hammer" in call_kwargs.kwargs["text"]
        assert "BTC/USDT" in call_kwargs.kwargs["text"]


@pytest.mark.asyncio
async def test_send_alert_failure(sample_pattern):
    with patch("src.alerter.telegram.Bot") as MockBot:
        import telegram.error

        mock_bot = MockBot.return_value
        mock_bot.send_message = AsyncMock(
            side_effect=telegram.error.TelegramError("Network error")
        )

        alerter = TelegramAlerter("test-token", "test-chat")
        result = await alerter.send_alert(sample_pattern)

        assert result is False
