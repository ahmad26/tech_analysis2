from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

_CET = ZoneInfo("Europe/Warsaw")

# (label, price_below), (label, price_above)
FibBound = tuple[tuple[str, float], tuple[str, float]]


@dataclass
class AppConfig:
    symbols: list[str]
    timeframes: list[str]
    patterns: list[str]
    exchange: str
    telegram_bot_token: str
    telegram_chat_id: str
    state_file: str = "alert_state.json"
    candles_to_fetch: int = 50
    state_ttl_hours: int = 48
    # Per-timeframe volatility floor. A signal is skipped when the candle's
    # ATR(14)/close is below the threshold for its timeframe (low-vol chop filter).
    # Empty dict = no filtering. Keys are timeframe strings (e.g. "4h").
    min_atr_pct: dict[str, float] = field(default_factory=dict)
    # Per-timeframe pattern overrides. A timeframe present here uses EXACTLY its list
    # (the global `patterns` is ignored for that TF); any TF not listed falls back to
    # `patterns`. Lets a pattern be active on one TF and off on another — pattern edge
    # is timeframe-specific (e.g. hammer is strong on 1d but toxic on 4h, see
    # memory/finding_oos_regime_check). Empty dict = global list everywhere.
    patterns_by_timeframe: dict[str, list[str]] = field(default_factory=dict)

    def patterns_for(self, timeframe: str) -> list[str]:
        """Effective pattern list for a timeframe — the per-TF override if one exists,
        otherwise the global `patterns` list."""
        return self.patterns_by_timeframe.get(timeframe, self.patterns)


@dataclass
class TradingSignal:
    action: str      # "BUY" or "SELL"
    setup: str       # "REVERSAL" or "CONTINUATION"
    entry: float
    stop_loss: float
    sl_source: str   # e.g. "Fib200 0%", "MA50"
    take_profit: float       # primary TP (MA if one sits in the way, else nearest Fib meeting MIN_RR)
    tp_source: str
    risk_reward: float
    all_tp_candidates: list[tuple[float, str]]
    take_profit_2: float | None = None   # secondary TP: original Fib target when MA became TP1
    tp_source_2: str | None = None


@dataclass
class DetectedPattern:
    symbol: str
    timeframe: str
    pattern_name: str
    signal: str  # "bullish" or "bearish"
    close_price: float
    candle_timestamp: datetime
    ma7: float | None = None
    ma25: float | None = None
    ma50: float | None = None
    ma99: float | None = None
    ma200: float | None = None
    ma_position: str | None = None  # "above_both" | "below_both" | "between" (based on MA50/MA200)
    fib50: FibBound | None = None   # Fib bracket within 50-period high/low
    fib200: FibBound | None = None  # Fib bracket within 200-period high/low
    atr: float | None = None        # ATR(14) at signal candle
    atr_range_high: float | None = None  # 14-candle high
    atr_range_low: float | None = None   # 14-candle low
    trading_signal: TradingSignal | None = None

    @property
    def alert_key(self) -> str:
        ts = int(self.candle_timestamp.timestamp())
        return f"{self.symbol}|{self.timeframe}|{self.pattern_name}|{ts}"

    def format_message(self) -> str:
        arrow = "⬆️" if self.signal == "bullish" else "⬇️"

        if self.timeframe == "1d":
            candle_str = self.candle_timestamp.strftime("%Y-%m-%d")
        else:
            utc_str = self.candle_timestamp.strftime("%Y-%m-%d %H:%M UTC")
            cet_str = self.candle_timestamp.astimezone(_CET).strftime("%H:%M CET")
            candle_str = f"{utc_str} | {cet_str}"

        lines = [
            f"{arrow} {self.pattern_name} ({self.signal})",
            f"Coin: {self.symbol}",
            f"Timeframe: {self.timeframe}",
            f"Close: {self.close_price:.4f}",
            f"Candle: {candle_str}",
        ]

        binance_ma_parts = []
        for label, val in [("MA7", self.ma7), ("MA25", self.ma25), ("MA99", self.ma99)]:
            if val is not None:
                binance_ma_parts.append(f"{label}: {val:.4f}")
        if binance_ma_parts:
            lines.append(" | ".join(binance_ma_parts))

        if self.ma50 is not None or self.ma200 is not None:
            ma_parts = []
            if self.ma50 is not None:
                ma_parts.append(f"MA50: {self.ma50:.4f}")
            if self.ma200 is not None:
                ma_parts.append(f"MA200: {self.ma200:.4f}")
            if self.ma50 is not None and self.ma200 is not None:
                spread_pct = (self.ma50 - self.ma200) / self.ma200 * 100
                trend = "uptrend" if self.ma50 > self.ma200 else "downtrend"
                ma_parts.append(f"{trend} ({spread_pct:+.2f}%)")
            lines.append(" | ".join(ma_parts))

        if self.ma_position is not None:
            label = {"above_both": "above both MAs", "below_both": "below both MAs", "between": "between MA50 and MA200"}.get(self.ma_position, self.ma_position)
            lines.append(f"Price is {label}")

        if self.fib50 is not None:
            (lbl_lo, p_lo), (lbl_hi, p_hi) = self.fib50
            lines.append(f"Fib50:  {lbl_lo} ({p_lo:.4f}) — {lbl_hi} ({p_hi:.4f})")

        if self.fib200 is not None:
            (lbl_lo, p_lo), (lbl_hi, p_hi) = self.fib200
            lines.append(f"Fib200: {lbl_lo} ({p_lo:.4f}) — {lbl_hi} ({p_hi:.4f})")

        if self.atr is not None:
            atr_pct = self.atr / self.close_price * 100
            range_str = ""
            if self.atr_range_high is not None and self.atr_range_low is not None:
                range_str = f" | Range14: {self.atr_range_low:.4f} – {self.atr_range_high:.4f}"
            lines.append(f"ATR(14): {self.atr:.4f} ({atr_pct:.2f}%){range_str}")

        if self.trading_signal is not None:
            ts = self.trading_signal
            risk = abs(ts.entry - ts.stop_loss)
            lines.append(f"{'─' * 24}")
            lines.append(f"{ts.action}  [{ts.setup}]  Entry: {ts.entry:.4f}")
            lines.append(f"SL:  {ts.stop_loss:.4f}  [{ts.sl_source} +buffer]")
            if ts.take_profit_2 is not None:
                lines.append(f"TP1: {ts.take_profit:.4f}  [{ts.tp_source}]")
                lines.append(f"TP2: {ts.take_profit_2:.4f}  [{ts.tp_source_2}]")
            else:
                lines.append(f"TP:  {ts.take_profit:.4f}  [{ts.tp_source}]")
            if risk > 0:
                lines.append(f"R/R: 1:{ts.risk_reward:.1f}")

        return "\n".join(lines)
