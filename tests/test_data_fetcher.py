from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_fetcher import create_exchange, fetch_ohlcv


def test_create_exchange():
    with patch("src.data_fetcher.ccxt") as mock_ccxt:
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        exchange = create_exchange("binance")
        mock_ccxt.binance.assert_called_once_with({"enableRateLimit": True})
        assert exchange is mock_exchange


def test_fetch_ohlcv():
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.return_value = [
        [1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 100.0],
        [1704070800000, 42200.0, 42700.0, 42100.0, 42600.0, 150.0],
        [1704074400000, 42600.0, 42900.0, 42400.0, 42800.0, 120.0],
    ]

    df = fetch_ohlcv(mock_exchange, "BTC/USDT", "1h", limit=3)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert df.index.name == "timestamp"
    assert df["close"].iloc[0] == 42200.0
    assert df["close"].iloc[-1] == 42800.0


def test_fetch_ohlcv_empty():
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.return_value = []

    df = fetch_ohlcv(mock_exchange, "BTC/USDT", "1h", limit=50)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
