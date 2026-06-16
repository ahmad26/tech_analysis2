import os

import pytest
import yaml

from src.config import load_config
from src.models import AppConfig


def test_load_config(tmp_path, monkeypatch):
    config_data = {
        "symbols": ["BTC/USDT"],
        "timeframes": ["1h"],
        "patterns": ["doji"],
        "exchange": "binance",
        "candles_to_fetch": 30,
        "state_file": "test_state.json",
        "state_ttl_hours": 24,
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat456")

    cfg = load_config(str(config_file))

    assert isinstance(cfg, AppConfig)
    assert cfg.symbols == ["BTC/USDT"]
    assert cfg.timeframes == ["1h"]
    assert cfg.patterns == ["doji"]
    assert cfg.exchange == "binance"
    assert cfg.telegram_bot_token == "tok123"
    assert cfg.telegram_chat_id == "chat456"
    assert cfg.candles_to_fetch == 30
    assert cfg.state_ttl_hours == 24


def test_load_config_defaults(tmp_path, monkeypatch):
    config_data = {
        "symbols": ["ETH/USDT"],
        "timeframes": ["4h"],
        "patterns": ["hammer"],
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    # Prevent load_dotenv from loading the project .env file
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **kw: None)

    cfg = load_config(str(config_file))

    assert cfg.exchange == "binance"
    assert cfg.telegram_bot_token == ""
    assert cfg.telegram_chat_id == ""
    assert cfg.candles_to_fetch == 50
    assert cfg.state_ttl_hours == 48
