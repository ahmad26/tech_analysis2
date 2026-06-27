"""Binance app entrypoint.

Runs the shared scanner/trader against Binance USDT-M futures. State lives in
binance/state/ so it never collides with the OKX app.

    python -m binance.main --trade --trade-risk-pct 2.0 --leverage 5
    python -m binance.main --manage-positions
    python -m binance.main --dry-run
"""
from __future__ import annotations

from pathlib import Path

from core.app import main
from core.venue import VenueContext
from binance.adapter import BinanceAdapter

CTX = VenueContext(
    adapter=BinanceAdapter(),
    state_dir=Path(__file__).resolve().parent / "state",
    api_key_env="BINANCE_API_KEY",
    api_secret_env="BINANCE_API_SECRET",
    label="Binance",
    demo=False,
)

if __name__ == "__main__":
    main(CTX)
