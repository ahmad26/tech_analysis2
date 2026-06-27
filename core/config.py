from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from core.models import AppConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> AppConfig:
    load_dotenv(_PROJECT_ROOT / ".env")

    if config_path is None:
        config_path = str(_PROJECT_ROOT / "config.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    return AppConfig(
        symbols=cfg["symbols"],
        timeframes=cfg["timeframes"],
        patterns=cfg["patterns"],
        exchange=cfg.get("exchange", "binance"),
        telegram_bot_token=bot_token,
        telegram_chat_id=chat_id,
        state_file=cfg.get("state_file", "alert_state.json"),
        candles_to_fetch=cfg.get("candles_to_fetch", 50),
        state_ttl_hours=cfg.get("state_ttl_hours", 48),
        min_atr_pct=cfg.get("min_atr_pct") or {},
        patterns_by_timeframe=cfg.get("patterns_by_timeframe") or {},
    )
