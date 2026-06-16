import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default TTL per timeframe (hours). Alert key format: symbol|timeframe|pattern|ts
_DEFAULT_TTL: dict[str, int] = {
    "15m": 1,
    "1h":  2,
    "4h":  6,
    "1d":  26,
}
_FALLBACK_TTL_HOURS = 48


class AlertTracker:
    def __init__(self, state_file: str, ttl_hours: int = 48, ttl_map: dict[str, int] | None = None):
        self._path = Path(state_file)
        self._ttl_map: dict[str, int] = ttl_map if ttl_map is not None else _DEFAULT_TTL
        self._fallback_ttl = ttl_hours * 3600
        self._seen: dict[str, float] = {}
        self._load()

    def _ttl_for_key(self, key: str) -> float:
        parts = key.split("|")
        tf = parts[1] if len(parts) >= 2 else ""
        hours = self._ttl_map.get(tf)
        return hours * 3600 if hours is not None else self._fallback_ttl

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._seen = json.load(f)
                logger.info("Loaded %d alert keys from %s", len(self._seen), self._path)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load state file, starting fresh")
                self._seen = {}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._seen, f)

    def is_duplicate(self, alert_key: str) -> bool:
        if alert_key not in self._seen:
            return False
        return time.time() - self._seen[alert_key] < self._ttl_for_key(alert_key)

    def record(self, alert_key: str) -> None:
        self._seen[alert_key] = time.time()
        self._save()

    def cleanup(self) -> None:
        now = time.time()
        before = len(self._seen)
        self._seen = {k: v for k, v in self._seen.items() if now - v < self._ttl_for_key(k)}
        removed = before - len(self._seen)
        if removed:
            logger.info("Cleaned up %d stale alert keys", removed)
            self._save()
