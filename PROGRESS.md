# Crypto Candlestick Pattern Alert App - Progress

## Status: Core app complete and working

## What's done

### Project setup
- `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`
- Virtual environment at `.venv/` with all deps installed

### Source code (all in `src/`)
- **models.py** - `AppConfig` and `DetectedPattern` dataclasses
- **config.py** - Loads `config.yaml` + `.env` secrets
- **data_fetcher.py** - Fetches OHLCV candles from Binance via ccxt (no API key needed)
- **pattern_detector.py** - Pure Python/numpy pattern detection (no TA-Lib dependency)
- **alert_tracker.py** - Duplicate prevention via JSON state file, auto-cleanup after 48h
- **alerter.py** - Telegram bot push notifications
- **main.py** - APScheduler orchestration + `--dry-run` mode

### Patterns implemented (pure Python, no TA-Lib needed)
1. Hammer
2. Inverted Hammer
3. Bullish Engulfing
4. Bearish Engulfing
5. Doji
6. Morning Star
7. Evening Star
8. Shooting Star

### Tests
- 27 tests, all passing (`pytest tests/`)
- Covers: config loading, data fetching, pattern detection, alerting, alert tracking, integration

### Config
- `config.yaml` - 10 coins (BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, DOT, LINK as USDT pairs), 4 timeframes (15m, 1h, 4h, 1d)

## Key change from original plan
- **Dropped `pandas-ta-classic`** - it still requires the TA-Lib C library for most candlestick patterns. Replaced with pure Python/numpy implementations in `pattern_detector.py`. All 8 patterns work without any C dependencies.

## How to run

```bash
cd tech_analysis
source .venv/bin/activate

# Quick check (no Telegram needed):
python -m src.main --dry-run

# Full mode with Telegram alerts:
cp .env.example .env   # then fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
python -m src.main

# Tests:
pytest tests/ -v
```

## Verified working
- Dry-run against live Binance data detected 18 patterns across all timeframes and coins (May 2, 2026 run)

## Still to do / possible next steps
- Set up Telegram bot and fill in `.env` credentials to enable live alerts
- Initial git commit
- Consider adding more patterns or configurable thresholds
- Consider adding logging to file
