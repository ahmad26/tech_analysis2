"""OKX app entrypoint.

Runs the shared scanner/trader against OKX USDT perpetual swaps. State lives in
okx/state/ so it never collides with the Binance app. Signals are scanned from Binance
public spot data (in core.app), identical to the Binance venue — only execution differs.

    python -m okx.main --trade --trade-risk-pct 2.0 --leverage 5
    python -m okx.main --manage-positions
    python -m okx.main --dry-run

Requires OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSWORD in .env. The OKX adapter is
not yet sandbox-validated — run on demo (set demo=True below) before live trading.
"""
from __future__ import annotations

from pathlib import Path

from core.app import main
from core.venue import VenueContext
from okx.adapter import OKXAdapter

CTX = VenueContext(
    # EEA account (registered on my.okx.com) — API calls MUST go to eea.okx.com,
    # else every authenticated request returns 50119. Drop hostname for a global account.
    adapter=OKXAdapter(hostname="eea.okx.com"),
    state_dir=Path(__file__).resolve().parent / "state",
    api_key_env="OKX_API_KEY",
    api_secret_env="OKX_API_SECRET",
    api_password_env="OKX_API_PASSWORD",
    label="OKX",
    demo=False,   # LIVE — EEA demo geo-blocks swaps, so OKX is validated/run on live.
)

if __name__ == "__main__":
    main(CTX)
