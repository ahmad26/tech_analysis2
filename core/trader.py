from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import ccxt

from core.exchange_adapter import ExchangeAdapter
from core.models import DetectedPattern

logger = logging.getLogger(__name__)

_RECALC_INTERVAL_S = 12 * 3600  # 43 200 s — at most twice a day
# Backstop: recalc regardless of open positions once the cached risk amount is this
# old — otherwise holding 2+ positions for weeks would freeze sizing on a stale
# balance (and keep risking pre-drawdown amounts through a drawdown).
_RECALC_MAX_AGE_S = 7 * 24 * 3600


class Trader:
    """Venue-agnostic order execution. All exchange-specific operations (balance,
    stop/TP placement, protective-order cancellation) go through `adapter`; the sizing
    and level logic here is identical across venues."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        adapter: ExchangeAdapter,
        alerter=None,
        position_tracker=None,
        risk_pct: float = 1.0,
        leverage: int = 1,
        risk_state_file: str = "risk_state.json",
    ):
        self.exchange = exchange
        self.adapter = adapter
        self.alerter = alerter
        self.position_tracker = position_tracker
        self.risk_pct = risk_pct
        self.leverage = leverage
        self._risk_state_path = Path(risk_state_file)

    # ------------------------------------------------------------------
    # Risk-amount cache
    # ------------------------------------------------------------------

    def _load_risk_state(self) -> dict:
        try:
            if self._risk_state_path.exists():
                return json.loads(self._risk_state_path.read_text())
        except Exception:
            logger.warning("Could not read %s, starting fresh", self._risk_state_path)
        return {}

    def _save_risk_state(self, state: dict) -> None:
        try:
            self._risk_state_path.write_text(json.dumps(state, indent=2))
        except Exception:
            logger.warning("Could not write %s", self._risk_state_path)

    def _get_risk_amount(self) -> float:
        """Return the cached risk amount in USDT, recalculating when conditions are met.

        Recalculates only when open positions ≤ 1 AND ≥ 12 h have elapsed since the
        last calculation, so all concurrent trades share the same baseline risk amount.
        Falls back to live balance if no cached value exists yet.
        """
        state = self._load_risk_state()
        now = time.time()
        elapsed = now - state.get("last_calc_ts", 0.0)
        open_count = len(self.position_tracker.all()) if self.position_tracker else 0

        if (open_count <= 1 and elapsed >= _RECALC_INTERVAL_S) or elapsed >= _RECALC_MAX_AGE_S:
            balance = self._get_usdt_balance()
            risk_amount = balance * self.risk_pct / 100
            self._save_risk_state({"risk_amount": risk_amount, "last_calc_ts": now})
            logger.info(
                "Risk amount recalculated: %.2f USDT (balance=%.2f USDT, %d position(s) open)",
                risk_amount, balance, open_count,
            )
            return risk_amount

        cached = state.get("risk_amount")
        if cached is not None:
            return float(cached)

        # No cached value and conditions not met — fall back to live balance.
        balance = self._get_usdt_balance()
        return balance * self.risk_pct / 100

    def execute_signal(self, p: DetectedPattern) -> bool:
        """Place market entry + SL + TP for a futures signal. Returns True if executed."""
        if p.trading_signal is None:
            return False
        ts = p.trading_signal

        risk_per_unit = abs(ts.entry - ts.stop_loss)
        if risk_per_unit <= 0:
            return False

        if not self.adapter.supports_symbol(self.exchange, p.symbol):
            logger.info("%s not tradable on %s, skipping", p.symbol, self.adapter.label)
            return False

        # Venue symbol for direct ccxt calls (identity on Binance, swap form on OKX).
        # Tracker/alerts/logs keep the base symbol p.symbol; adapter methods translate
        # internally, so they still receive p.symbol.
        msym = self.adapter.market_symbol(p.symbol)

        risk_amount = self._get_risk_amount()

        balance = self._get_usdt_balance()
        if balance <= 0:
            logger.warning("No free USDT balance, skipping %s", p.symbol)
            return False

        # Size so loss at SL = risk_amount, capped so margin <= 20% of balance
        risk_based_qty = risk_amount / risk_per_unit
        max_margin = balance * 0.20
        margin_based_qty = max_margin * self.leverage / ts.entry
        raw_qty = min(risk_based_qty, margin_based_qty)
        if risk_based_qty > margin_based_qty:
            logger.info("Position capped by margin limit for %s (risk qty=%.4f, capped to=%.4f)", p.symbol, risk_based_qty, raw_qty)
        # raw_qty is in base-coin units; the adapter converts to the venue's order
        # amount (base coin on Binance, contracts on OKX) and applies precision.
        try:
            quantity = self.adapter.order_amount(self.exchange, p.symbol, raw_qty)
        except ccxt.InvalidOrder:
            # ccxt raises (rather than returning 0) when qty is below the
            # exchange minimum amount precision
            logger.warning("Quantity below exchange minimum for %s (%.8f), skipping", p.symbol, raw_qty)
            return False
        if quantity <= 0:
            logger.warning("Quantity too small for %s (%.8f)", p.symbol, raw_qty)
            return False

        # Min-notional guard: USDT-M venues reject orders whose notional (qty*entry) is
        # below the market minimum. Risk-based sizing on a wide stop (e.g. a 1d signal)
        # can clear the min-amount check above yet still fall under this floor. Sizing up
        # would breach the risk cap, so skip cleanly rather than let the exchange reject
        # the market entry every scan.
        try:
            min_notional = self.exchange.market(msym)["limits"]["cost"]["min"] or self.adapter.min_notional_floor
        except Exception:
            min_notional = self.adapter.min_notional_floor
        # notional must be in USD: on OKX `quantity` is contracts, so convert back to
        # coin units first (identity on Binance).
        notional = self.adapter.amount_to_coins(self.exchange, p.symbol, quantity) * ts.entry
        if notional < min_notional:
            logger.warning(
                "Notional %.2f below exchange minimum %.2f for %s (qty=%.8f) — skipping",
                notional, min_notional, p.symbol, quantity,
            )
            return False

        is_long = ts.action == "BUY"
        entry_side = "buy" if is_long else "sell"
        exit_side  = "sell" if is_long else "buy"

        # Set leverage before placing orders
        try:
            self.adapter.set_leverage(self.exchange, self.leverage, p.symbol)
        except Exception:
            logger.warning("Could not set leverage for %s, continuing with default", p.symbol)

        # Market entry (adapter attaches any venue params, e.g. OKX tdMode)
        try:
            order = self.adapter.market_order(self.exchange, p.symbol, entry_side, quantity)
            fill_price = float(order.get("average") or order.get("price") or ts.entry)
            logger.info("%s %s qty=%.6f @ %.4f", ts.action, p.symbol, quantity, fill_price)
        except Exception:
            logger.exception("Market entry failed for %s", p.symbol)
            return False

        # Recalculate SL from actual fill price to avoid immediate trigger
        sl_distance = abs(ts.entry - ts.stop_loss)
        sl_price = fill_price - sl_distance if is_long else fill_price + sl_distance
        sl = float(self.exchange.price_to_precision(msym, sl_price))

        # Ladder exit engine: the TP candidates are absolute structural levels
        # (Fib/MA/HTF). The resting maker LIMIT goes at the FINAL level; the
        # position manager ratchets the stop to each intermediate level as price
        # touches it. Levels the fill already slipped past are dropped so the first
        # touch can't be instant.
        ladder = [lv for lv, _src in ts.all_tp_candidates
                  if (lv > fill_price) == is_long and lv != fill_price]
        if ladder:
            tp_price = ladder[-1]
        else:
            # Extreme slippage ate every level — fall back to TP1 at its
            # signal distance, managed as a legacy (non-ladder) position.
            tp_distance = abs(ts.take_profit - ts.entry)
            tp_price = fill_price + tp_distance if is_long else fill_price - tp_distance
        tp = float(self.exchange.price_to_precision(msym, tp_price))

        # Cancel stale protective orders for this symbol before placing new ones
        self._cancel_protective(p.symbol)

        # Stop-loss (STOP_MARKET, reduceOnly)
        try:
            self.adapter.place_stop(self.exchange, p.symbol, exit_side, quantity, sl)
            logger.info("SL placed for %s: %.4f", p.symbol, sl)
        except Exception:
            logger.exception("SL order failed for %s — flattening unprotected position", p.symbol)
            self._flatten_unprotected(p.symbol, exit_side, quantity, fill_price, is_long)
            return False

        # Take-profit — maker reduceOnly LIMIT
        try:
            self.adapter.place_take_profit(self.exchange, p.symbol, exit_side, quantity, tp)
            logger.info("TP placed for %s: %.4f", p.symbol, tp)
        except Exception:
            # The SL is resting, so the position stays protected. Keep the trade:
            # the tracked TP lets the position manager exit in software, and the
            # next trailing ratchet re-places the order exchange-side.
            logger.exception("TP order failed for %s — continuing with SL only", p.symbol)
            if self.alerter is not None:
                self.alerter.send_text(
                    f"⚠️ {p.symbol}: TP order failed — position protected by SL; "
                    f"software exit at {tp:.4f} until the TP order is re-placed"
                )

        if self.position_tracker is not None:
            side = "long" if is_long else "short"
            self.position_tracker.record(
                p.symbol, side, quantity, fill_price,
                sl=sl, tp=tp, signal_timeframe=p.timeframe,
                tp_ladder=ladder,
            )

        if self.alerter is not None:
            self.alerter.send_position_opened(
                symbol=p.symbol,
                side="long" if is_long else "short",
                contracts=quantity,
                entry=fill_price,
                sl=sl,
                tp=tp,
                rr=ts.risk_reward,
                timeframe=p.timeframe,
            )

        return True

    def _flatten_unprotected(self, symbol: str, exit_side: str, quantity: float, fill_price: float, is_long: bool) -> None:
        """Close a just-opened position whose stop-loss could not be placed —
        an unprotected position is worse than a missed trade. If even the close
        fails, record the position so the manager and audit can see it."""
        try:
            self.adapter.market_order(self.exchange, symbol, exit_side, quantity, reduce_only=True)
            logger.info("Emergency-closed %s after SL placement failure", symbol)
            if self.alerter is not None:
                self.alerter.send_text(
                    f"⚠️ {symbol}: SL order failed right after entry — position closed immediately, trade abandoned"
                )
        except Exception:
            logger.exception("CRITICAL: could not flatten %s — position is OPEN with NO RESTING STOP", symbol)
            if self.position_tracker is not None:
                self.position_tracker.record(symbol, "long" if is_long else "short", quantity, fill_price)
            if self.alerter is not None:
                self.alerter.send_text(
                    f"🚨 CRITICAL: {symbol} position is open WITHOUT a stop-loss and could not be closed — manual action required"
                )

    def cancel_conditional_orders(self, symbol: str) -> None:
        """Cancel our SL/TP exit orders for a symbol, leaving unrelated orders untouched."""
        self._cancel_protective(symbol)

    def _cancel_protective(self, symbol: str) -> None:
        """Cancel every resting SL/TP order for `symbol` (regular + algo), via the adapter."""
        try:
            for order in self.adapter.fetch_protective_orders(self.exchange, symbol):
                try:
                    self.adapter.cancel_protective_order(self.exchange, symbol, order)
                    logger.info("Cancelled protective order %s for %s", order.ids or order.kind, symbol)
                except Exception:
                    logger.warning("Could not cancel protective order %s for %s", order.ids, symbol)
        except Exception:
            logger.warning("Could not fetch/cancel protective orders for %s", symbol)

    def _get_usdt_balance(self) -> float:
        return self.adapter.available_balance(self.exchange)
