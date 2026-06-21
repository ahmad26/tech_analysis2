from __future__ import annotations

import logging

import ccxt
import numpy as np
import pandas as pd

from src.data_fetcher import fetch_ohlcv
from src.position_tracker import PositionTracker, TrackedPosition
from src.trader import is_protective_order, place_take_profit

logger = logging.getLogger(__name__)

TF_RANK: dict[str, int] = {"15m": 1, "1h": 2, "4h": 3, "1d": 4}
TRAIL_ATR_MULT = 1.5

# Staged R-multiple stop. R = original entry→SL distance (frozen at entry).
# At +1R profit the stop jumps to breakeven (+ a small buffer to cover fees and
# slippage so a return to entry doesn't scratch the trade). At +2R it locks +1R.
# Beyond that the ATR chandelier takes over; the tightest candidate always wins
# and the stop only ever ratchets toward profit.
BREAKEVEN_TRIGGER_R = 1.0
LOCK_1R_TRIGGER_R = 2.0
BREAKEVEN_BUFFER_R = 0.1


def _order_ids(order) -> set[str]:
    """All identifiers an exchange order may be referenced by. A STOP_MARKET
    placed via create_order comes back as a Binance *algo* order (algoId), while
    the maker LIMIT take-profit is a regular order (id/orderId); collecting every
    namespace lets _cancel_conditional reliably exclude a just-placed order."""
    if not isinstance(order, dict):
        return set()
    info = order.get("info") or {}
    raw = {order.get("id"), info.get("algoId"), info.get("orderId"), info.get("clientAlgoId")}
    return {str(i) for i in raw if i}


_SL_TYPES = {"stop_market", "stop"}


def _is_stop_order(order: dict) -> bool:
    """True for a stop-loss style order (vs a take-profit). Reads the unified
    `type`, the ccxt-parsed `info.orderType`, and the raw Binance algo
    `orderType`, so it classifies regular and algo orders alike."""
    if not isinstance(order, dict):
        return False
    info = order.get("info") or {}
    otype = (order.get("type") or order.get("orderType")
             or info.get("orderType") or info.get("type") or "").lower()
    return otype in _SL_TYPES


def _atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if len(df) < period + 1:
        return None
    highs = df["high"].iloc[-period:].values
    lows = df["low"].iloc[-period:].values
    prev_closes = df["close"].iloc[-(period + 1):-1].values
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)),
    )
    return float(tr.mean())


class PositionManager:
    def __init__(self, exchange, position_tracker: PositionTracker, alerter=None):
        self.exchange = exchange
        self.tracker = position_tracker
        self.alerter = alerter

    def _fetch_candles(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Candles for trailing/exit decisions, fetched from self.exchange — the
        futures venue the positions actually trade on. The scanner detects signals
        on spot data, but ladder touches and SL/TP-hit checks must be evaluated on
        the prices that fill the resting orders; spot/futures basis (~0.05-0.1%)
        otherwise causes near-boundary rungs to be missed (or hit) on the wrong
        feed. On fetch failure return None so the caller skips the trail this
        cycle rather than acting on stale data."""
        try:
            return fetch_ohlcv(self.exchange, symbol, timeframe, limit=50)
        except Exception:
            logger.warning("Could not fetch %s %s (futures) for position management", symbol, timeframe)
            return None

    def run(self) -> None:
        """Trail stops and check SL/TP for all open positions. Called once per scan cycle."""
        for symbol, pos in list(self.tracker.all().items()):
            df = self._fetch_candles(symbol, pos.signal_timeframe)
            if df is None or df.empty:
                logger.warning("No candle data for %s %s — skipping position update", symbol, pos.signal_timeframe)
                continue
            self._update(pos, df)

    def handle_signal(self, pattern, data_cache: dict) -> bool:
        """
        Called when a new tradeable signal fires for a symbol that already has an open position.
        Returns True  → signal handled internally, caller should NOT open a new position.
        Returns False → caller should proceed to open a new position (e.g. after a reversal close).
        """
        pos = self.tracker.get(pattern.symbol)
        if pos is None:
            return False

        if pattern.trading_signal is None:
            return True

        new_side = "long" if pattern.trading_signal.action == "BUY" else "short"
        new_rank = TF_RANK.get(pattern.timeframe, 0)
        old_rank = TF_RANK.get(pos.signal_timeframe, 0)

        if new_side == pos.side:
            # Same rank rule as reversals: a lower-TF signal must not rewrite the
            # TP of a position opened on a higher timeframe (15m structure is
            # noise relative to a 1d target).
            if new_rank < old_rank:
                logger.debug(
                    "Ignored same-side %s signal for %s — rank %d < existing %s rank %d",
                    pattern.timeframe, pattern.symbol, new_rank, pos.signal_timeframe, old_rank,
                )
                return True
            # Refresh the ladder from the new signal's candidate levels; the
            # resting TP moves to the new final level. Legacy positions are
            # migrated to ladder management here. SL can only ratchet, so a
            # refreshed ladder can never loosen the stop.
            new_ladder = [lv for lv, _src in pattern.trading_signal.all_tp_candidates]
            new_tp = new_ladder[-1] if new_ladder else pattern.trading_signal.take_profit
            df = data_cache.get(pattern.symbol, {}).get(pos.signal_timeframe)
            self._replace_orders(pos, pos.sl, new_tp, df)
            self.tracker.update_tp(pattern.symbol, new_tp, tp_ladder=new_ladder)
            logger.info(
                "Updated TP for %s → %.4f (%d ladder levels)  (signal: %s %s)",
                pattern.symbol, new_tp, len(new_ladder), pattern.timeframe, pattern.pattern_name,
            )
            return True

        # Opposite direction
        if new_rank >= old_rank:
            logger.info(
                "Reversing %s: closing %s, new signal is %s %s (rank %d >= %d)",
                pattern.symbol, pos.side, pattern.timeframe, new_side, new_rank, old_rank,
            )
            df = data_cache.get(pattern.symbol, {}).get(pos.signal_timeframe)
            price = float(df["close"].iloc[-1]) if df is not None and not df.empty else None
            closed = self._close(pos, price, reason="REVERSED")
            # Only open the opposite position once the old one is confirmed gone —
            # otherwise the new entry would net against the still-open position.
            return not closed
        else:
            logger.debug(
                "Ignored opposite %s signal for %s — rank %d < existing %s rank %d",
                pattern.timeframe, pattern.symbol, new_rank, pos.signal_timeframe, old_rank,
            )
            return True

    # ------------------------------------------------------------------ #

    def _update(self, pos: TrackedPosition, df: pd.DataFrame) -> None:
        close = float(df["close"].iloc[-1])

        if pos.sl > 0:
            sl_hit = (pos.side == "long" and close <= pos.sl) or \
                     (pos.side == "short" and close >= pos.sl)
            if sl_hit:
                logger.info("SL hit for %s at %.4f (sl=%.4f)", pos.symbol, close, pos.sl)
                self._close(pos, close, reason="SL")
                return

        if pos.tp > 0:
            tp_hit = (pos.side == "long" and close >= pos.tp) or \
                     (pos.side == "short" and close <= pos.tp)
            if tp_hit:
                logger.info("TP hit for %s at %.4f (tp=%.4f)", pos.symbol, close, pos.tp)
                self._close(pos, close, reason="TP")
                return

        # Ladder positions: level-ladder trailing only (the validated exit
        # engine — no chandelier, no staged R locks; they were the old engine).
        if pos.tp_ladder:
            self._trail_ladder(pos, df)
            return

        # Legacy (pre-ladder) positions: trail SL via ATR chandelier combined
        # with staged R-multiple profit locks. Compute every candidate, take the
        # tightest, and only move the stop if it ratchets toward profit (never loosen).
        atr = _atr(df)
        if atr is None or pos.sl <= 0:
            return

        candidates: list[float] = []

        # ATR chandelier, anchored on the latest close.
        if pos.side == "long":
            candidates.append(close - TRAIL_ATR_MULT * atr)
        else:
            candidates.append(close + TRAIL_ATR_MULT * atr)

        # Staged R-multiple stop. R is the entry→SL distance frozen at entry
        # (initial_sl); pos.sl itself trails, so it can't be used here. Legacy
        # positions without an initial_sl (0.0) fall back to the ATR trail only.
        R = abs(pos.entry_price - pos.initial_sl) if pos.initial_sl > 0 else 0.0
        if R > 0:
            profit = (close - pos.entry_price) if pos.side == "long" else (pos.entry_price - close)
            r_mult = profit / R
            step = None
            if r_mult >= LOCK_1R_TRIGGER_R:
                step = pos.entry_price + R if pos.side == "long" else pos.entry_price - R
            elif r_mult >= BREAKEVEN_TRIGGER_R:
                buf = BREAKEVEN_BUFFER_R * R
                step = pos.entry_price + buf if pos.side == "long" else pos.entry_price - buf
            if step is not None:
                candidates.append(step)

        candidate_sl = max(candidates) if pos.side == "long" else min(candidates)

        ratchet = (pos.side == "long" and candidate_sl > pos.sl) or \
                  (pos.side == "short" and candidate_sl < pos.sl)
        if not ratchet:
            return

        status = self._replace_orders(pos, candidate_sl, pos.tp, df)
        if status == "immediate":
            logger.info("Trailing SL for %s would trigger immediately at %.4f — closing", pos.symbol, candidate_sl)
            self._close(pos, candidate_sl, reason="SL")
            return
        if status == "failed":
            logger.error("Trailing SL update for %s failed — keeping previous SL %.4f", pos.symbol, pos.sl)
            return
        logger.info("Trailed SL for %s %s: %.4f → %.4f", pos.symbol, pos.side.upper(), pos.sl, candidate_sl)
        self.tracker.update_sl(pos.symbol, candidate_sl)

    def _trail_ladder(self, pos: TrackedPosition, df: pd.DataFrame) -> None:
        """Level-ladder trailing — mirrors the backtester's rolling exit model:
        first touched ladder level moves the stop to entry, each later touch
        moves it to the last touched level. Touches are recomputed from price
        history since the position opened, so missed runs or restarts cannot
        lose ladder progress. The resting maker LIMIT at the final level
        (pos.tp) is left in place."""
        if pos.sl <= 0:
            return
        opened = pd.Timestamp(pos.opened_at_ms, unit="ms", tz="UTC")
        # Include the candle the position was opened inside
        start = max(0, int(df.index.searchsorted(opened, side="right")) - 1)
        since = df.iloc[start:]
        if since.empty:
            return

        if pos.side == "long":
            extreme = float(since["high"].max())
            touched = [lv for lv in pos.tp_ladder if lv <= extreme]
        else:
            extreme = float(since["low"].min())
            touched = [lv for lv in pos.tp_ladder if lv >= extreme]
        if not touched:
            return

        new_sl = pos.entry_price if len(touched) == 1 else touched[-1]
        ratchet = (pos.side == "long" and new_sl > pos.sl) or \
                  (pos.side == "short" and new_sl < pos.sl)
        if not ratchet:
            return

        status = self._replace_orders(pos, new_sl, pos.tp, df)
        if status == "immediate":
            # Price has already returned through the ratcheted level; the stop is
            # met. The exchange refused to rest it, so close now (what the
            # software check would do at the next poll anyway).
            logger.info("Ladder SL for %s would trigger immediately at %.4f — closing", pos.symbol, new_sl)
            self._close(pos, new_sl, reason="SL")
            return
        if status == "failed":
            # New stop did not place; the previous stop is still resting. Do NOT
            # advance the tracked SL past what is actually on the exchange.
            logger.error("Ladder SL update for %s failed — keeping previous SL %.4f", pos.symbol, pos.sl)
            return
        logger.info(
            "Ladder SL for %s %s: %.4f → %.4f (%d/%d levels touched)",
            pos.symbol, pos.side.upper(), pos.sl, new_sl, len(touched), len(pos.tp_ladder),
        )
        self.tracker.update_sl(pos.symbol, new_sl)

    def _close(self, pos: TrackedPosition, price: float | None, reason: str) -> bool:
        """Close a position at market. Returns True when the position is confirmed
        gone (tracker entry removed), False when the close failed and the position
        stays tracked so the next management run retries it."""
        try:
            self._cancel_conditional(pos.symbol)
        except Exception:
            logger.warning("Could not cancel orders for %s before closing", pos.symbol)

        close_side = "sell" if pos.side == "long" else "buy"
        try:
            order = self.exchange.create_market_order(pos.symbol, close_side, pos.contracts, params={"reduceOnly": True})
            fill = float(order.get("average") or order.get("price") or price or 0)
            logger.info("Closed %s %s qty=%.6f @ %.4f (%s)", pos.side.upper(), pos.symbol, pos.contracts, fill, reason)
            if self.alerter:
                pnl = (fill - pos.entry_price) * pos.contracts * (1 if pos.side == "long" else -1)
                self.alerter.send_position_closed(pos.symbol, pos.side, pos.contracts, pos.entry_price, pnl)
        except Exception as exc:
            # A reduceOnly close can fail because the position is already flat
            # (e.g. the exchange-side SL filled first) — the common case, since
            # the resting stop fires intra-candle while we only poll candle
            # closes. Only drop the tracker entry when the exchange confirms
            # the position is gone — otherwise keep tracking so the next run
            # re-detects the SL/TP hit and retries.
            if self._is_flat(pos.symbol):
                logger.info(
                    "Close order for %s rejected (%s) — already flat on exchange, "
                    "exchange-side %s filled first; removing from tracker",
                    pos.symbol, exc, reason,
                )
                if self.alerter:
                    # The exchange-side protective order filled before our close;
                    # estimate the fill from the order that fired.
                    fill = {"SL": pos.sl, "TP": pos.tp}.get(reason) or price or pos.entry_price
                    pnl = (fill - pos.entry_price) * pos.contracts * (1 if pos.side == "long" else -1)
                    self.alerter.send_position_closed(pos.symbol, pos.side, pos.contracts, pos.entry_price, pnl)
                self.tracker.remove(pos.symbol)
                return True
            logger.exception("Failed to close %s", pos.symbol)
            logger.error("%s may still be open after failed close — keeping tracked, will retry next run", pos.symbol)
            # _cancel_conditional already ran, so the position has no resting
            # protection — re-place SL/TP while it waits for the retry.
            self._replace_orders(pos, pos.sl, pos.tp, None)
            if self.alerter:
                try:
                    self.alerter.send_text(
                        f"⚠️ Failed to close {pos.symbol} {pos.side} ({reason}) — will retry next run"
                    )
                except Exception:
                    logger.warning("Could not send failed-close alert for %s", pos.symbol)
            return False

        self.tracker.remove(pos.symbol)
        return True

    def _is_flat(self, symbol: str) -> bool:
        """True only when the exchange confirms there is no open position for
        `symbol`. Unverifiable (network error) counts as NOT flat — keep tracking."""
        try:
            positions = self.exchange.fetch_positions([symbol])
            return all(
                abs(float(p.get("contracts") or 0)) <= 0
                for p in positions
                if p["symbol"].split(":")[0] == symbol
            )
        except Exception:
            logger.warning("Could not verify position state for %s", symbol)
            return False

    def _replace_orders(self, pos: TrackedPosition, new_sl: float, new_tp: float, df) -> str:
        """Place the new SL/TP, THEN cancel the superseded orders.

        Placing BEFORE cancelling (step 3) means a rejected new stop leaves the
        existing protection resting — the position is never left naked by a
        failed replace. Returns the SL-leg status so the caller can react:
          'placed'    — the new stop is resting on the exchange;
          'immediate' — the exchange rejected it as already-triggerable (-2021):
                         the stop level is already met, the caller should CLOSE
                         now. The old orders are left intact;
          'failed'    — placement errored for another reason; old orders intact;
          'skipped'   — no SL requested (new_sl <= 0).
        """
        exit_side = "sell" if pos.side == "long" else "buy"
        new_ids: set[str] = set()
        sl_status = "skipped"

        if new_sl > 0:
            try:
                sl_price = float(self.exchange.price_to_precision(pos.symbol, new_sl))
                order = self.exchange.create_order(
                    symbol=pos.symbol,
                    type="STOP_MARKET",
                    side=exit_side,
                    amount=pos.contracts,
                    params={"stopPrice": sl_price, "reduceOnly": True},
                )
                new_ids |= _order_ids(order)
                sl_status = "placed"
            except ccxt.OrderImmediatelyFillable:
                # -2021: market is already at/through the stop level. Resting it
                # is pointless — the position should be closed now. The existing
                # orders have NOT been cancelled, so protection stays intact.
                logger.warning(
                    "Updated SL for %s at %.4f would trigger immediately (-2021) — "
                    "leaving existing orders, caller should close", pos.symbol, new_sl,
                )
                return "immediate"
            except Exception:
                logger.exception(
                    "Failed to place updated SL for %s at %.4f — leaving existing orders intact",
                    pos.symbol, new_sl,
                )
                return "failed"

        if new_tp > 0:
            tp_price = float(self.exchange.price_to_precision(pos.symbol, new_tp))
            resting_tp = self._resting_tp_ids(pos.symbol)
            cur_tp = (float(self.exchange.price_to_precision(pos.symbol, pos.tp))
                      if pos.tp > 0 else 0.0)
            if resting_tp and tp_price == cur_tp:
                # Target unchanged and already resting — leave it. The maker TP is a
                # reduceOnly LIMIT, whose quantity Binance counts against the
                # position's reducible size the moment it rests (unlike STOP_MARKET,
                # which only counts once triggered). Placing a SECOND full-size
                # reduceOnly LIMIT at the same level is therefore rejected (-2022,
                # "ReduceOnly Order is rejected") — the recurring failure seen on
                # every ladder ratchet, which re-passes the unchanged pos.tp. Just
                # keep the resting order and exclude it from the cancel sweep below.
                new_ids |= resting_tp
            else:
                # Target moved (or the TP went missing, e.g. after _close cancelled
                # it): cancel the resting LIMIT TP BEFORE placing the new one — two
                # full-size reduceOnly LIMITs can't coexist (-2022), so the SL's
                # place-before-cancel trick is impossible for the LIMIT TP. The SL
                # placed above keeps the position protected during the brief swap; a
                # failed re-place leaves only the TP missing, recovered next run
                # (the tracker's tp then matches no resting order, so this branch
                # runs again instead of the skip above).
                self._cancel_resting_tp(pos.symbol)
                try:
                    tp_order = place_take_profit(self.exchange, pos.symbol, exit_side, pos.contracts, tp_price)
                    new_ids |= _order_ids(tp_order)
                except Exception:
                    logger.exception(
                        "Failed to place updated TP for %s at %.4f — position keeps its SL, "
                        "TP retried next run", pos.symbol, new_tp,
                    )

        # New orders are resting — now cancel the superseded ones (never the new).
        try:
            self._cancel_conditional(pos.symbol, exclude_ids=new_ids)
        except Exception:
            logger.warning("Could not cancel superseded orders for %s after replace", pos.symbol)

        return sl_status

    def _resting_tp_ids(self, symbol: str) -> set[str]:
        """Ids of take-profit orders currently resting for `symbol` — protective
        and non-stop — across both the regular and algo endpoints. Used to keep
        the existing TP when a re-placement fails so the cancel sweep does not
        orphan it."""
        ids: set[str] = set()
        try:
            for order in self.exchange.fetch_open_orders(symbol):
                if is_protective_order(order) and not _is_stop_order(order):
                    ids |= _order_ids(order)
        except Exception:
            logger.warning("Could not fetch regular orders for %s while preserving TP", symbol)
        try:
            raw_symbol = self.exchange.market_id(symbol)
            for o in self.exchange.fapiPrivateGetOpenAlgoOrders({}):
                if o.get("symbol") != raw_symbol or _is_stop_order(o):
                    continue
                if str(o.get("reduceOnly")).lower() == "true":
                    ids.add(str(o.get("algoId")))
        except Exception:
            logger.warning("Could not fetch algo orders for %s while preserving TP", symbol)
        return ids

    def _cancel_resting_tp(self, symbol: str) -> None:
        """Cancel only the resting take-profit order(s) for `symbol` (the maker
        reduceOnly LIMIT, or an algo TAKE_PROFIT_MARKET if MAKER_TP is off),
        leaving the stop-loss untouched. Required before re-placing the maker
        LIMIT TP at a NEW level: a resting reduceOnly LIMIT counts against the
        position's reducible size, so a second full-size reduceOnly LIMIT is
        rejected (-2022). The SL keeps protecting the position throughout."""
        try:
            for order in self.exchange.fetch_open_orders(symbol):
                if not is_protective_order(order) or _is_stop_order(order):
                    continue
                try:
                    self.exchange.cancel_order(order["id"], symbol)
                except Exception:
                    logger.warning("Could not cancel TP order %s for %s", order.get("id"), symbol)
        except Exception:
            logger.warning("Could not fetch regular orders for %s while cancelling TP", symbol)

        try:
            raw_symbol = self.exchange.market_id(symbol)
            for o in self.exchange.fapiPrivateGetOpenAlgoOrders({}):
                if o.get("symbol") != raw_symbol or _is_stop_order(o):
                    continue
                if str(o.get("reduceOnly")).lower() != "true":
                    continue
                try:
                    self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": o["algoId"]})
                except Exception:
                    logger.warning("Could not cancel TP algo order %s for %s", o.get("algoId"), symbol)
        except Exception:
            logger.warning("Could not fetch algo orders for %s while cancelling TP", symbol)

    def _cancel_conditional(self, symbol: str, exclude_ids: set[str] | None = None) -> None:
        exclude_ids = exclude_ids or set()
        # Regular STOP_MARKET / TAKE_PROFIT_MARKET orders + the maker LIMIT take-profit
        try:
            for order in self.exchange.fetch_open_orders(symbol):
                if not is_protective_order(order):
                    continue
                if _order_ids(order) & exclude_ids:
                    continue
                try:
                    self.exchange.cancel_order(order["id"], symbol)
                except Exception:
                    logger.warning("Could not cancel order %s for %s", order["id"], symbol)
        except Exception:
            logger.warning("Could not fetch regular orders for %s", symbol)

        # Algo (conditional) orders visible in Binance UI
        try:
            raw_symbol = self.exchange.market_id(symbol)
            for o in self.exchange.fapiPrivateGetOpenAlgoOrders({}):
                if o.get("symbol") != raw_symbol:
                    continue
                if {str(o.get("algoId"))} & exclude_ids:
                    continue
                try:
                    self.exchange.fapiPrivateDeleteAlgoOrder({"algoId": o["algoId"]})
                except Exception:
                    logger.warning("Could not cancel algo order %s for %s", o.get("algoId"), symbol)
        except Exception:
            logger.warning("Could not fetch algo orders for %s", symbol)
