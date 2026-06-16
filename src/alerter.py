import logging

import requests

from src.models import DetectedPattern

logger = logging.getLogger(__name__)

_TIMEOUT = 10


class TelegramAlerter:
    def __init__(self, bot_token: str, chat_id: str):
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id

    def send_alert(self, pattern: DetectedPattern) -> bool:
        try:
            resp = requests.post(
                self._url,
                json={"chat_id": self._chat_id, "text": pattern.format_message()},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info("Sent alert: %s %s %s", pattern.symbol, pattern.timeframe, pattern.pattern_name)
            return True
        except Exception as e:
            logger.error("Failed to send Telegram alert: %s", e)
            return False

    def send_text(self, text: str) -> bool:
        try:
            resp = requests.post(
                self._url,
                json={"chat_id": self._chat_id, "text": text},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    def send_position_opened(
        self, symbol: str, side: str, contracts: float, entry: float, sl: float, tp: float, rr: float
    ) -> bool:
        arrow = "⬆️ LONG" if side == "long" else "⬇️ SHORT"
        text = (
            f"{arrow} {symbol} opened\n"
            f"Size: {contracts:.4f} @ {entry:.4f}\n"
            f"SL: {sl:.4f} | TP: {tp:.4f}\n"
            f"R/R: 1:{rr:.1f}"
        )
        logger.info("Position opened notification: %s %s qty=%.4f", side.upper(), symbol, contracts)
        return self.send_text(text)

    def send_position_closed(
        self, symbol: str, side: str, contracts: float, entry: float, pnl: float
    ) -> bool:
        arrow = "⬆️ LONG" if side == "long" else "⬇️ SHORT"
        sign = "+" if pnl >= 0 else ""
        pnl_pct = (pnl / (entry * contracts)) * 100 if entry > 0 and contracts > 0 else 0.0
        emoji = "✅" if pnl >= 0 else "❌"
        text = (
            f"{emoji} {arrow} {symbol} closed\n"
            f"Size: {contracts:.4f} @ {entry:.4f}\n"
            f"P&L: {sign}{pnl:.2f} USDT ({sign}{pnl_pct:.2f}%)"
        )
        logger.info("Position closed notification: %s %s P&L=%.2f", side.upper(), symbol, pnl)
        return self.send_text(text)
