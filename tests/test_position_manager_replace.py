"""Tests for the SL replace/trailing failure paths (2026-06-15 fix).

The bug: when a trailing ratchet cancelled the resting stop, then failed to
place the new one, the position was left naked on the exchange yet the tracker
recorded the (non-existent) tighter SL. These tests pin the three fixes:
  1. _replace_orders reports the SL-leg outcome instead of swallowing it;
  2. callers react — close on -2021 ("would immediately trigger"), keep the
     previous SL on other failures, advance only on success;
  3. place-before-cancel — a failed replace never cancels the old protection,
     and a successful replace cancels the OLD orders but not the just-placed ones.
"""
from __future__ import annotations

import pandas as pd
import pytest

import src.trader as trader
from src.position_manager import PositionManager
from src.position_tracker import TrackedPosition

import ccxt


class FakeExchange:
    def __init__(self, sl_behavior: str = "ok", tp_behavior: str = "ok"):
        self.sl_behavior = sl_behavior  # "ok" | "immediate" | "error"
        self.tp_behavior = tp_behavior  # "ok" | "error"
        self.cancelled: list[str] = []        # regular cancel_order ids
        self.algo_cancelled: list[str] = []   # cancelled algoIds
        self.market_orders: list[tuple] = []
        self.open_orders: list[dict] = []
        self.algo_orders: list[dict] = []
        self.new_sl_order = {"id": "NEW_SL", "info": {"algoId": "A_NEW"}}

    def price_to_precision(self, symbol, price):
        return round(float(price), 4)

    def market_id(self, symbol):
        return symbol.replace("/", "")

    def create_order(self, **kwargs):
        if kwargs.get("type") == "STOP_MARKET":
            if self.sl_behavior == "immediate":
                raise ccxt.OrderImmediatelyFillable('{"code":-2021,"msg":"Order would immediately trigger."}')
            if self.sl_behavior == "error":
                raise ccxt.ExchangeError("boom")
            return dict(self.new_sl_order)
        # take-profit LIMIT
        if self.tp_behavior == "error":
            raise ccxt.ExchangeError("tp boom")
        return {"id": "NEW_TP", "info": {"orderId": "NEW_TP"}}

    def create_market_order(self, symbol, side, amount, params=None):
        self.market_orders.append((symbol, side, amount))
        return {"average": 0.0, "price": 0.0}

    def fetch_open_orders(self, symbol):
        return list(self.open_orders)

    def cancel_order(self, oid, symbol):
        self.cancelled.append(oid)

    def fapiPrivateGetOpenAlgoOrders(self, params=None):
        return list(self.algo_orders)

    def fapiPrivateDeleteAlgoOrder(self, params):
        self.algo_cancelled.append(params["algoId"])


class FakeTracker:
    def __init__(self, pos: TrackedPosition):
        self._pos = {pos.symbol: pos}

    def all(self):
        return dict(self._pos)

    def get(self, s):
        return self._pos.get(s)

    def update_sl(self, s, sl):
        if s in self._pos:
            self._pos[s].sl = sl

    def update_tp(self, s, tp, tp_ladder=None):
        if s in self._pos:
            self._pos[s].tp = tp
            if tp_ladder is not None:
                self._pos[s].tp_ladder = list(tp_ladder)

    def remove(self, s):
        self._pos.pop(s, None)


def _pos(**over) -> TrackedPosition:
    base = dict(
        symbol="DOT/USDT", side="long", contracts=47.7, entry_price=1.02,
        opened_at_ms=int(pd.Timestamp("2026-06-15 11:00", tz="UTC").timestamp() * 1000),
        sl=0.994, initial_sl=0.994, tp=1.05, signal_timeframe="4h",
        tp_ladder=[1.025, 1.029, 1.05],
    )
    base.update(over)
    return TrackedPosition(**base)


def _df(high: float, low: float = 0.9, close: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2026-06-15 12:00", periods=2, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"high": [high, high], "low": [low, low], "close": [close, close]}, index=idx
    )


@pytest.fixture(autouse=True)
def _maker_tp(monkeypatch):
    # TP re-placement goes through place_take_profit; force the maker LIMIT path
    # so the fake exchange's create_order TP branch is exercised deterministically.
    monkeypatch.setattr(trader, "MAKER_TP", True)


# ---- step 1 + 3: _replace_orders outcome + place-before-cancel ----------- #

def test_replace_immediate_leaves_old_orders_intact():
    ex = FakeExchange(sl_behavior="immediate")
    ex.open_orders = [{"id": "OLD_TP", "type": "LIMIT", "reduceOnly": True}]
    ex.algo_orders = [{"symbol": "DOTUSDT", "algoId": "OLD_SL"}]
    mgr = PositionManager(ex, FakeTracker(_pos()))

    status = mgr._replace_orders(_pos(), 1.029, 1.05, None)

    assert status == "immediate"
    # Nothing cancelled — the existing stop is still protecting the position.
    assert ex.cancelled == []
    assert ex.algo_cancelled == []


def test_replace_failed_leaves_old_orders_intact():
    ex = FakeExchange(sl_behavior="error")
    ex.open_orders = [{"id": "OLD_TP", "type": "LIMIT", "reduceOnly": True}]
    ex.algo_orders = [{"symbol": "DOTUSDT", "algoId": "OLD_SL"}]
    mgr = PositionManager(ex, FakeTracker(_pos()))

    status = mgr._replace_orders(_pos(), 1.029, 1.05, None)

    assert status == "failed"
    assert ex.cancelled == []
    assert ex.algo_cancelled == []


def test_replace_unchanged_tp_keeps_resting_tp_cancels_old_sl():
    ex = FakeExchange(sl_behavior="ok")
    # new_tp (1.05) == pos.tp (1.05): the resting maker LIMIT TP is already correct.
    # The maker TP is a reduceOnly LIMIT whose qty Binance counts against the
    # position's reducible size the moment it rests, so re-placing a second
    # full-size reduceOnly LIMIT is rejected (-2022). The fix leaves the resting TP
    # untouched and only cancels the superseded SL (the just-placed SL algoId A_NEW
    # is preserved).
    ex.open_orders = [{"id": "RESTING_TP", "type": "LIMIT", "reduceOnly": True}]
    ex.algo_orders = [
        {"symbol": "DOTUSDT", "algoId": "OLD_SL"},
        {"symbol": "DOTUSDT", "algoId": "A_NEW"},
    ]
    mgr = PositionManager(ex, FakeTracker(_pos()))

    status = mgr._replace_orders(_pos(), 1.029, 1.05, None)

    assert status == "placed"
    assert ex.cancelled == []                  # resting TP kept (target unchanged)
    assert ex.algo_cancelled == ["OLD_SL"]     # superseded SL cancelled, new SL kept


def test_replace_changed_tp_cancels_old_tp_then_places_new():
    ex = FakeExchange(sl_behavior="ok")
    ex.open_orders = [{"id": "OLD_TP", "type": "LIMIT", "reduceOnly": True}]
    ex.algo_orders = [{"symbol": "DOTUSDT", "algoId": "OLD_SL"}]
    mgr = PositionManager(ex, FakeTracker(_pos()))

    # new_tp (1.06) != pos.tp (1.05): two full-size reduceOnly LIMITs can't coexist,
    # so the old TP must be cancelled BEFORE the new one is placed (place-before-
    # cancel is impossible for a LIMIT TP). The SL stays protected throughout.
    status = mgr._replace_orders(_pos(), 1.029, 1.06, None)

    assert status == "placed"
    assert "OLD_TP" in ex.cancelled            # superseded TP cancelled to free reducible qty
    assert ex.algo_cancelled == ["OLD_SL"]     # superseded SL cancelled too


# ---- TP leg: a CHANGED target swaps the LIMIT TP cancel-then-place -------- #
# The maker LIMIT TP can't use place-before-cancel (a second full-size reduceOnly
# LIMIT is rejected -2022), so a changed target cancels the old TP first. If the
# re-place then fails the position keeps its (freshly placed) SL and is briefly
# TP-less — recovered next run, when the tracker's tp matches no resting order.

def test_replace_changed_tp_failure_keeps_sl():
    ex = FakeExchange(sl_behavior="ok", tp_behavior="error")
    ex.open_orders = [{"id": "OLD_TP", "type": "LIMIT", "reduceOnly": True}]
    ex.algo_orders = [{"symbol": "DOTUSDT", "algoId": "OLD_SL"}]
    mgr = PositionManager(ex, FakeTracker(_pos()))

    status = mgr._replace_orders(_pos(), 1.029, 1.06, None)

    # The SL leg still placed, so the ratchet is allowed to advance.
    assert status == "placed"
    assert ex.algo_cancelled == ["OLD_SL"]
    assert "OLD_TP" in ex.cancelled            # old LIMIT TP cancelled before the failed re-place


def test_replace_changed_tp_failure_cancels_algo_tp():
    # Same, but the resting TP is a TAKE_PROFIT_MARKET algo order rather than the
    # maker LIMIT: a changed target cancels it first across the algo endpoint.
    ex = FakeExchange(sl_behavior="ok", tp_behavior="error")
    ex.algo_orders = [
        {"symbol": "DOTUSDT", "algoId": "OLD_SL", "orderType": "STOP_MARKET", "reduceOnly": "true"},
        {"symbol": "DOTUSDT", "algoId": "OLD_TP", "orderType": "TAKE_PROFIT_MARKET", "reduceOnly": "true"},
    ]
    mgr = PositionManager(ex, FakeTracker(_pos()))

    status = mgr._replace_orders(_pos(), 1.029, 1.06, None)

    assert status == "placed"
    # Both the superseded stop and the old algo take-profit are cancelled.
    assert "OLD_SL" in ex.algo_cancelled
    assert "OLD_TP" in ex.algo_cancelled


# ---- step 2: _trail_ladder reacts to the outcome ------------------------- #

def test_trail_ladder_immediate_closes_position():
    ex = FakeExchange(sl_behavior="immediate")
    tracker = FakeTracker(_pos())
    mgr = PositionManager(ex, tracker)

    mgr._trail_ladder(tracker.get("DOT/USDT"), _df(high=1.030))

    # Stop already met → position closed at market, removed from tracker.
    assert len(ex.market_orders) == 1
    assert "DOT/USDT" not in tracker.all()


def test_trail_ladder_failed_keeps_previous_sl():
    ex = FakeExchange(sl_behavior="error")
    tracker = FakeTracker(_pos())
    mgr = PositionManager(ex, tracker)

    mgr._trail_ladder(tracker.get("DOT/USDT"), _df(high=1.030))

    # New stop never placed → tracker must not advance past the resting SL.
    assert tracker.get("DOT/USDT").sl == pytest.approx(0.994)
    assert ex.market_orders == []


def test_trail_ladder_success_advances_sl():
    ex = FakeExchange(sl_behavior="ok")
    tracker = FakeTracker(_pos())
    mgr = PositionManager(ex, tracker)

    mgr._trail_ladder(tracker.get("DOT/USDT"), _df(high=1.030))

    # 2 levels touched (1.025, 1.029) → SL ratchets to the last touched level.
    assert tracker.get("DOT/USDT").sl == pytest.approx(1.029)
