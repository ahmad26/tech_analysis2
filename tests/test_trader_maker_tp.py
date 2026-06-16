"""Unit tests for the maker take-profit: order classification + placement.

The live order lifecycle can't be exercised without an exchange, but the two pieces
that gate correctness ARE pure functions: is_protective_order (used to cancel the
right orders before replace/close) and place_take_profit (maker LIMIT vs taker
TP_MARKET). A misclassification here orphans or duplicates a TP on a live position.
"""
from __future__ import annotations

import src.trader as trader
from src.trader import is_protective_order, place_take_profit


def test_is_protective_order_conditional_types():
    assert is_protective_order({"type": "STOP_MARKET"})
    assert is_protective_order({"type": "take_profit_market"})
    assert is_protective_order({"type": "STOP"})


def test_is_protective_order_maker_limit_tp():
    # Maker TP: reduceOnly LIMIT, via ccxt-unified bool and via raw info string.
    assert is_protective_order({"type": "limit", "reduceOnly": True})
    assert is_protective_order({"type": "LIMIT", "info": {"reduceOnly": "true"}})


def test_is_protective_order_leaves_benign_orders():
    # A normal (non-reduceOnly) limit order must NOT be cancelled.
    assert not is_protective_order({"type": "limit", "reduceOnly": False})
    assert not is_protective_order({"type": "limit", "info": {"reduceOnly": "false"}})
    assert not is_protective_order({"type": "market"})


class _FakeExchange:
    def __init__(self):
        self.calls = []

    def create_order(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": "1"}


def test_place_take_profit_maker(monkeypatch):
    monkeypatch.setattr(trader, "MAKER_TP", True)
    ex = _FakeExchange()
    place_take_profit(ex, "BTC/USDT", "sell", 0.5, 70000.0)
    (call,) = ex.calls
    assert call["type"] == "LIMIT"
    assert call["price"] == 70000.0
    assert call["params"]["reduceOnly"] is True
    assert "stopPrice" not in call["params"]  # a maker limit rests, it is not triggered


def test_place_take_profit_taker(monkeypatch):
    monkeypatch.setattr(trader, "MAKER_TP", False)
    ex = _FakeExchange()
    place_take_profit(ex, "BTC/USDT", "sell", 0.5, 70000.0)
    (call,) = ex.calls
    assert call["type"] == "TAKE_PROFIT_MARKET"
    assert call["params"]["stopPrice"] == 70000.0
    assert call["params"]["reduceOnly"] is True
