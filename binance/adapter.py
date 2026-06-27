"""Binance USDT-M futures adapter.

Encapsulates every Binance-specific (non-ccxt-unified) operation: the income ledger
(`fapiPrivateGetIncome`), the algo-order endpoints (`fapiPrivateGetOpenAlgoOrders` /
`fapiPrivateDeleteAlgoOrder`) on which STOP_MARKET orders live, and the
`info.availableBalance` balance field. This is a behaviour-preserving extraction of the
code that previously lived inline in trader.py / position_manager.py / main.py.
"""
from __future__ import annotations

import logging

from core.data_fetcher import create_exchange
from core.exchange_adapter import ExchangeAdapter, ProtectiveOrder

logger = logging.getLogger(__name__)

# Order types that are SL/TP protective exits (Binance unified + algo `orderType`).
_CONDITIONAL_TYPES = {"stop_market", "take_profit_market", "stop", "take_profit"}
_SL_TYPES = {"stop_market", "stop"}


def _is_reduce_only(order: dict) -> bool:
    ro = order.get("reduceOnly")
    if ro is None:
        ro = str(order.get("info", {}).get("reduceOnly", "")).lower() == "true"
    return bool(ro)


def _order_type(order: dict) -> str:
    info = order.get("info") or {}
    return (order.get("type") or order.get("orderType")
            or info.get("orderType") or info.get("type") or "").lower()


def _regular_is_protective(order: dict) -> bool:
    """A resting regular order that is one of our exits — a conditional type, or the
    maker take-profit (a reduceOnly LIMIT)."""
    type_l = (order.get("type") or "").lower()
    if type_l in _CONDITIONAL_TYPES:
        return True
    return type_l == "limit" and _is_reduce_only(order)


def _order_ids(order: dict) -> set[str]:
    info = order.get("info") or {}
    raw = {order.get("id"), info.get("algoId"), info.get("orderId"), info.get("clientAlgoId")}
    return {str(i) for i in raw if i}


class BinanceAdapter(ExchangeAdapter):
    id = "binance"
    label = "Binance"

    def build_exchange(self, api_key: str, api_secret: str, *, demo: bool = False,
                       password: str | None = None):
        # Binance needs no passphrase; `password` is accepted for a uniform signature.
        return create_exchange(self.id, api_key, api_secret, testnet=demo, market_type="future")

    # -- account / ledger --------------------------------------------------- #
    def available_balance(self, exchange) -> float:
        try:
            bal = exchange.fetch_balance({"type": "future"})
            available = float(bal.get("info", {}).get("availableBalance", 0))
            if available > 0:
                return available
            for asset in ("USDT", "USDC"):
                free = float((bal.get(asset) or {}).get("free") or 0)
                if free > 0:
                    return free
            return 0.0
        except Exception:
            logger.exception("Failed to fetch balance")
            return 0.0

    def realized_pnl(self, exchange, symbol: str, since_ms: int) -> float:
        try:
            market_id = exchange.market_id(symbol)
            start_time = max(0, since_ms - 1000)
            rows = exchange.fapiPrivateGetIncome({
                "incomeType": "REALIZED_PNL", "symbol": market_id,
                "startTime": start_time, "limit": 1000,
            })
            return sum(float(r["income"]) for r in rows)
        except Exception:
            logger.warning("Could not fetch realized PnL for %s", symbol, exc_info=True)
            return 0.0

    def trade_costs(self, exchange, symbol: str, since_ms: int) -> tuple[float, float]:
        try:
            market_id = exchange.market_id(symbol)
            start_time = max(0, since_ms - 1000)
            comm = fund = 0.0
            for income_type, is_comm in (("COMMISSION", True), ("FUNDING_FEE", False)):
                rows = exchange.fapiPrivateGetIncome({
                    "incomeType": income_type, "symbol": market_id,
                    "startTime": start_time, "limit": 1000,
                })
                total = sum(float(r["income"]) for r in rows)
                if is_comm:
                    comm = total
                else:
                    fund = total
            return comm, fund
        except Exception:
            logger.warning("Could not fetch trade costs for %s", symbol, exc_info=True)
            return 0.0, 0.0

    # -- protective orders -------------------------------------------------- #
    def fetch_protective_orders(self, exchange, symbol: str) -> list[ProtectiveOrder]:
        orders: list[ProtectiveOrder] = []
        # Regular order book — maker LIMIT take-profit + any conditional shown here.
        try:
            for o in exchange.fetch_open_orders(symbol):
                if not _regular_is_protective(o):
                    continue
                orders.append(ProtectiveOrder(
                    ids=_order_ids(o), is_stop=_order_type(o) in _SL_TYPES,
                    reduce_only=_is_reduce_only(o), kind="regular", raw=o,
                ))
        except Exception:
            logger.warning("Could not fetch regular orders for %s", symbol)
        # Algo order book — STOP_MARKET / TAKE_PROFIT_MARKET live here on Binance.
        try:
            raw_symbol = exchange.market_id(symbol)
            for o in exchange.fapiPrivateGetOpenAlgoOrders({}):
                if o.get("symbol") != raw_symbol:
                    continue
                otype = (o.get("orderType") or o.get("type") or "").lower()
                orders.append(ProtectiveOrder(
                    ids={str(o.get("algoId"))} if o.get("algoId") else set(),
                    is_stop=otype in _SL_TYPES,
                    reduce_only=str(o.get("reduceOnly")).lower() == "true",
                    kind="algo", raw=o,
                ))
        except Exception:
            logger.warning("Could not fetch algo orders for %s", symbol)
        return orders

    def cancel_protective_order(self, exchange, symbol: str, order: ProtectiveOrder) -> None:
        if order.kind == "algo":
            exchange.fapiPrivateDeleteAlgoOrder({"algoId": order.raw["algoId"]})
        else:
            exchange.cancel_order(order.raw["id"], symbol)
