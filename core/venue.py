"""Per-venue runtime context.

Holds everything that differs between the Binance and OKX apps: which adapter to use,
where this app's state files live (so the two apps never share position/risk/alert
state), which environment variables carry its credentials, and whether to use the
venue's demo/sandbox. The shared app runner (core.app) takes a VenueContext and is
otherwise venue-agnostic."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from core.exchange_adapter import ExchangeAdapter

logger = logging.getLogger(__name__)


@dataclass
class VenueContext:
    adapter: ExchangeAdapter
    state_dir: Path
    api_key_env: str
    api_secret_env: str
    label: str
    demo: bool = False
    api_password_env: str | None = None  # API passphrase env var (OKX); None if unused

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # State files — all under this venue's own directory, so the Binance and OKX
    # apps can run concurrently without ever touching each other's tracking.
    @property
    def position_state(self) -> str:
        return str(self.state_dir / "position_state.json")

    @property
    def risk_state(self) -> str:
        return str(self.state_dir / "risk_state.json")

    @property
    def alert_state(self) -> str:
        return str(self.state_dir / "alert_state.json")

    def credentials(self) -> tuple[str | None, str | None, str | None]:
        key = os.environ.get(self.api_key_env)
        secret = os.environ.get(self.api_secret_env)
        password = os.environ.get(self.api_password_env) if self.api_password_env else None
        return key, secret, password

    def missing_credentials(self) -> list[str]:
        """Names of required credential env vars that are unset (passphrase included
        only when this venue declares one)."""
        key, secret, password = self.credentials()
        missing = []
        if not key:
            missing.append(self.api_key_env)
        if not secret:
            missing.append(self.api_secret_env)
        if self.api_password_env and not password:
            missing.append(self.api_password_env)
        return missing

    def build_trading_exchange(self):
        """Authenticated futures client for this venue (raises if creds are missing —
        callers should check missing_credentials() first for a friendly message)."""
        key, secret, password = self.credentials()
        return self.adapter.build_exchange(key, secret, demo=self.demo, password=password)
