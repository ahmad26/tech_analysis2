from __future__ import annotations

import fcntl
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrackedPosition:
    symbol: str
    side: str          # "long" or "short"
    contracts: float
    entry_price: float
    opened_at_ms: int  # epoch ms, used to query income history
    sl: float = 0.0
    initial_sl: float = 0.0  # entry SL, frozen — sl trails but this stays for R calc
    tp: float = 0.0
    signal_timeframe: str = "1h"
    # Ladder exit engine: all TP candidate levels (closest-first). When non-empty,
    # the position is managed by level-ladder trailing (SL → entry on first touch,
    # then SL → last touched level); `tp` is the FINAL ladder level where the
    # resting maker LIMIT sits. Empty = legacy chandelier/staged-R management.
    tp_ladder: list[float] = field(default_factory=list)


class PositionTracker:
    def __init__(self, state_file: str = "position_state.json"):
        self.path = Path(state_file)
        self._state: dict[str, TrackedPosition] = self._load()

    def _load(self) -> dict[str, TrackedPosition]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text())
            positions = {}
            for k, v in raw.items():
                v.setdefault("sl", 0.0)
                v.setdefault("initial_sl", v.get("sl", 0.0))
                v.setdefault("tp", 0.0)
                v.setdefault("signal_timeframe", "1h")
                v.setdefault("tp_ladder", [])
                positions[k] = TrackedPosition(**v)
            return positions
        except Exception:
            logger.warning("Could not load position state from %s", self.path)
            return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps({k: asdict(v) for k, v in self._state.items()}, indent=2))

    def sync(self, current_positions: list[dict]) -> tuple[list[TrackedPosition], list[TrackedPosition]]:
        """Compare exchange positions against stored state.
        Returns (newly_opened, newly_closed) lists.

        Uses an exclusive file lock so concurrent scanner instances that unblock
        simultaneously (e.g. after a network outage) each only report closes once.
        """
        lock_path = self.path.with_suffix(".lock")
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                # Re-read under the lock: the first instance writes "closed" state,
                # so subsequent instances see an already-updated file and skip re-reporting.
                on_disk = self._load()

                current: dict[str, TrackedPosition] = {}
                for p in current_positions:
                    contracts = abs(float(p.get("contracts") or 0))
                    if contracts <= 0:
                        continue
                    symbol = p["symbol"].split(":")[0]
                    side = "long" if (p.get("side") or "").lower() == "long" else "short"
                    entry_price = float(p.get("entryPrice") or p.get("info", {}).get("entryPrice") or 0)
                    if symbol in on_disk:
                        prev = on_disk[symbol]
                        current[symbol] = TrackedPosition(
                            symbol=symbol, side=prev.side, contracts=contracts,
                            entry_price=prev.entry_price, opened_at_ms=prev.opened_at_ms,
                            sl=prev.sl, initial_sl=prev.initial_sl, tp=prev.tp,
                            signal_timeframe=prev.signal_timeframe,
                            tp_ladder=prev.tp_ladder,
                        )
                    else:
                        current[symbol] = TrackedPosition(
                            symbol=symbol, side=side, contracts=contracts,
                            entry_price=entry_price, opened_at_ms=int(time.time() * 1000),
                        )

                newly_opened = [p for sym, p in current.items() if sym not in on_disk]
                newly_closed = [p for sym, p in on_disk.items() if sym not in current]

                self._state = current
                self._save()
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

        return newly_opened, newly_closed

    def record(
        self,
        symbol: str,
        side: str,
        contracts: float,
        entry_price: float,
        sl: float = 0.0,
        tp: float = 0.0,
        signal_timeframe: str = "1h",
        tp_ladder: list[float] | None = None,
    ) -> None:
        self._state[symbol] = TrackedPosition(
            symbol=symbol, side=side, contracts=contracts,
            entry_price=entry_price, opened_at_ms=int(time.time() * 1000),
            sl=sl, initial_sl=sl, tp=tp, signal_timeframe=signal_timeframe,
            tp_ladder=list(tp_ladder) if tp_ladder else [],
        )
        self._save()

    def get(self, symbol: str) -> TrackedPosition | None:
        return self._state.get(symbol)

    def all(self) -> dict[str, TrackedPosition]:
        return dict(self._state)

    def update_sl(self, symbol: str, sl: float) -> None:
        if symbol in self._state:
            self._state[symbol].sl = sl
            self._save()

    def update_tp(self, symbol: str, tp: float, tp_ladder: list[float] | None = None) -> None:
        if symbol in self._state:
            self._state[symbol].tp = tp
            if tp_ladder is not None:
                self._state[symbol].tp_ladder = list(tp_ladder)
            self._save()

    def remove(self, symbol: str) -> None:
        self._state.pop(symbol, None)
        self._save()

    def log_closed_trade(
        self,
        pos: "TrackedPosition",
        *,
        realized_pnl: float,
        commission: float = 0.0,
        funding: float = 0.0,
        closed_at_ms: int | None = None,
    ) -> None:
        """Append one JSON line per closed position to closed_trades.jsonl.

        The money ledger (Binance income) records P&L but not per-trade risk, so R
        can't be reconstructed from it alone. Here we have the frozen entry SL, so we
        persist `one_r_usdt` = |entry - initial_sl| x contracts alongside the realized
        P&L, making net/gross R recoverable later (see scripts/realized_pnl.py --by-trade).
        Best-effort: never raises into the close-handling path."""
        try:
            closed_at_ms = closed_at_ms if closed_at_ms is not None else int(time.time() * 1000)
            one_r = (
                abs(pos.entry_price - pos.initial_sl) * pos.contracts
                if pos.entry_price and pos.initial_sl
                else 0.0
            )
            net = realized_pnl + commission + funding
            rec = {
                "closed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(closed_at_ms / 1000)),
                "closed_at_ms": closed_at_ms,
                "opened_at_ms": pos.opened_at_ms,
                "symbol": pos.symbol,
                "side": pos.side,
                "timeframe": pos.signal_timeframe,
                "entry": pos.entry_price,
                "initial_sl": pos.initial_sl,
                "sl_at_close": pos.sl,
                "tp": pos.tp,
                "contracts": pos.contracts,
                "one_r_usdt": round(one_r, 6),
                "realized_pnl": round(realized_pnl, 6),
                "commission": round(commission, 6),
                "funding": round(funding, 6),
                "net_pnl": round(net, 6),
                "r_gross": round(realized_pnl / one_r, 4) if one_r else None,
                "r_net": round(net / one_r, 4) if one_r else None,
            }
            log_path = self.path.with_name("closed_trades.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            logger.warning("Could not append closed trade for %s to closed_trades.jsonl",
                           getattr(pos, "symbol", "?"), exc_info=True)
