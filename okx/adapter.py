"""OKX USDT perpetual-swap adapter.

The trading-critical path (market entry/close, set_leverage, fetch_positions, the
maker LIMIT take-profit, the STOP via `stopLossPrice`, and the EEA XPERP symbol mapping)
has been exercised on the OKX live account — EEA demo geo-blocks swaps, so there is no
sandbox to fall back to. The only piece still wanting confirmation against a real closed
position is the realized-P&L / fee / funding readout from /account/positions-history
(logging-only, not safety-critical); it is tagged `# VALIDATE`.

Key OKX differences from Binance, handled here:
  * a third credential, the API passphrase (`password`);
  * a sandbox toggle via `set_sandbox_mode`, not Binance's demo-trading flag;
  * the perpetual's unified symbol is 'BTC/USDT:USDT' (see market_symbol);
  * every order needs a trade mode (`tdMode`, cross/isolated);
  * stop orders are *algo* (trigger) orders on a separate order book.
"""
from __future__ import annotations

import logging

import ccxt

from core.exchange_adapter import ExchangeAdapter, ProtectiveOrder

logger = logging.getLogger(__name__)

# OKX bill sub-types we treat as realized trading P&L vs costs (for the ledger pull).
_SL_ORDTYPES = {"conditional", "trigger", "move_order_stop"}  # algo stop families


class OKXAdapter(ExchangeAdapter):
    id = "okx"
    label = "OKX"
    # OKX min order cost is small (~$1) and enforced via min contracts; don't apply
    # the Binance $20 floor, which would wrongly skip valid small OKX orders.
    min_notional_floor = 1.0

    def __init__(self, td_mode: str = "isolated", hostname: str | None = None):
        # tdMode is required on every OKX order. EEA XPERP positions opened via the OKX
        # UI come back as 'isolated' (USDC-margined), so default to that.
        self.td_mode = td_mode
        # API host. OKX routes regional entities to their own domain: accounts
        # registered on my.okx.com (EEA) MUST call eea.okx.com or every authenticated
        # request returns 50119 "API key doesn't exist". None = ccxt default
        # (www.okx.com, the global entity). ccxt builds the REST URL from
        # `exchange.hostname` (urls.api.rest = "https://{hostname}").
        self.hostname = hostname
        self._exchange = None        # set in build_exchange, used to discover instruments
        self._sym_map: dict[str, str] = {}   # coin -> ccxt XPERP symbol

    # -- symbol dialect ----------------------------------------------------- #
    # OKX EEA does NOT offer the global USDT swaps to retail (they 51155). It offers
    # USDC-margined "XPERP" perpetuals with per-coin, periodically-rolling instrument
    # ids (e.g. ETH-USD_UM_XPERP-310404 / ccxt 'ETH/USD:USD-310404'). We map the bot's
    # canonical 'COIN/USDT' to the live XPERP symbol dynamically from the loaded market
    # list, so a roll to a new suffix is picked up automatically each run.
    @staticmethod
    def _coin(symbol: str) -> str:
        return symbol.split("/")[0]

    def _ensure_map(self) -> None:
        if self._sym_map or self._exchange is None:
            return
        if not getattr(self._exchange, "markets", None):
            try:
                self._exchange.load_markets()
            except Exception:
                return
        for sym, m in self._exchange.markets.items():
            instId = (m.get("info") or {}).get("instId", "")
            if "XPERP" in instId and m.get("active") and m.get("base"):
                self._sym_map.setdefault(m["base"], sym)

    def market_symbol(self, symbol: str) -> str:
        """Canonical 'COIN/USDT' -> the live OKX EEA XPERP ccxt symbol. Falls back to the
        input if no XPERP exists for the coin (e.g. DOT) — the caller's order then fails
        cleanly rather than hitting the wrong instrument."""
        self._ensure_map()
        return self._sym_map.get(self._coin(symbol), symbol)

    def base_symbol(self, venue_symbol: str) -> str:
        """Inverse: an OKX position/order symbol ('ETH/USD:USD-310404') -> canonical
        'ETH/USDT', so the tracker keys OKX positions the same as Binance-scanned signals.
        Coin-only, so it works without the market map loaded."""
        return f"{self._coin(venue_symbol)}/USDT"

    def supports_symbol(self, exchange, symbol: str) -> bool:
        """True only if the coin has a live XPERP instrument (e.g. DOT does not on EEA).
        Without this the order falls back to the raw 'DOT/USDT' and fails 51155 every scan."""
        self._ensure_map()
        return self._coin(symbol) in self._sym_map

    def is_immediate_trigger_error(self, exc: Exception) -> bool:
        # OKX rejects a stop already through the market with 51280 ("SL trigger price must
        # be less/greater than the last price") as a plain InvalidOrder — the twin of
        # Binance's -2021. Match it so the ladder trail closes instead of erroring.
        if isinstance(exc, ccxt.OrderImmediatelyFillable):
            return True
        return "51280" in str(exc)

    def market_order(self, exchange, symbol: str, side: str, amount, *, reduce_only: bool = False):
        params = {"tdMode": self.td_mode}
        if reduce_only:
            params["reduceOnly"] = True
        return exchange.create_market_order(self.market_symbol(symbol), side, amount, params=params)

    def set_leverage(self, exchange, leverage: int, symbol: str) -> None:
        exchange.set_leverage(leverage, self.market_symbol(symbol), params={"mgnMode": self.td_mode})

    # -- order sizing ------------------------------------------------------- #
    def order_amount(self, exchange, symbol: str, coin_qty: float) -> float:
        msym = self.market_symbol(symbol)
        ct = float(exchange.market(msym).get("contractSize") or 1)
        return float(exchange.amount_to_precision(msym, coin_qty / ct))

    def amount_to_coins(self, exchange, symbol: str, amount: float) -> float:
        msym = self.market_symbol(symbol)
        ct = float(exchange.market(msym).get("contractSize") or 1)
        return amount * ct

    # -- construction ------------------------------------------------------- #
    def build_exchange(self, api_key: str, api_secret: str, *, demo: bool = False,
                       password: str | None = None):
        exchange = ccxt.okx({
            "apiKey": api_key,
            "secret": api_secret,
            "password": password,           # OKX API passphrase (required)
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {"defaultType": "swap", "fetchCurrencies": False},
        })
        if demo:
            exchange.set_sandbox_mode(True)
        if self.hostname:
            # Set AFTER sandbox so it can't be clobbered. EEA demo = eea.okx.com host
            # + the x-simulated-trading header that set_sandbox_mode adds.
            exchange.hostname = self.hostname
        self._exchange = exchange
        self._sym_map = {}   # rebuilt from this exchange's markets on first use
        return exchange

    # -- account / ledger --------------------------------------------------- #
    def available_balance(self, exchange) -> float:
        try:
            bal = exchange.fetch_balance()
            free = float((bal.get("USDT") or {}).get("free") or 0)
            if free > 0:
                return free
            # Fallback: USDC-margined accounts.
            return float((bal.get("USDC") or {}).get("free") or 0)
        except Exception:
            logger.exception("Failed to fetch OKX balance")
            return 0.0

    def _closed_positions(self, exchange, symbol: str, since_ms: int) -> list[dict]:
        """Raw OKX positions-history rows for this instrument since the position opened.

        OKX reports per-closed-position realized P&L already split the same way Binance's
        income ledger is: `pnl` = gross realized P&L (excludes fees/funding), `fee` and
        `fundingFee` = signed costs (negative). Reading these keeps the OKX and Binance
        adapters numerically consistent. The earlier bills-ledger approach mismapped the
        codes — OKX bill type 8 is *funding*, not realized P&L (per ccxt's own map).
        # VALIDATE: confirm these fields are populated on a real closed XPERP position;
        # `pnl`/`fee`/`fundingFee` come straight from /account/positions-history.
        """
        msym = self.market_symbol(symbol)
        inst_id = exchange.market_id(msym)
        rows = exchange.fetch_positions_history([msym], since=max(0, since_ms - 1000))
        out: list[dict] = []
        for p in rows:
            info = p.get("info") or {}
            if info.get("instId") and info.get("instId") != inst_id:
                continue
            out.append(info)
        return out

    def realized_pnl(self, exchange, symbol: str, since_ms: int) -> float:
        # Gross realized P&L (excludes fees/funding), matching Binance's REALIZED_PNL.
        # Best-effort — P&L logging is not safety-critical, so failures return 0.0.
        try:
            return sum(float(p.get("pnl") or 0) for p in self._closed_positions(exchange, symbol, since_ms))
        except Exception:
            logger.warning("Could not fetch OKX realized PnL for %s", symbol, exc_info=True)
            return 0.0

    def trade_costs(self, exchange, symbol: str, since_ms: int) -> tuple[float, float]:
        # (commission, funding), both signed (negative = cost) — matches Binance's
        # COMMISSION / FUNDING_FEE split. Best-effort (0, 0) on failure.
        try:
            comm = fund = 0.0
            for p in self._closed_positions(exchange, symbol, since_ms):
                comm += float(p.get("fee") or 0)
                fund += float(p.get("fundingFee") or 0)
            return comm, fund
        except Exception:
            logger.warning("Could not fetch OKX trade costs for %s", symbol, exc_info=True)
            return 0.0, 0.0

    # -- protective orders -------------------------------------------------- #
    def fetch_protective_orders(self, exchange, symbol: str) -> list[ProtectiveOrder]:
        msym = self.market_symbol(symbol)
        orders: list[ProtectiveOrder] = []
        # Regular order book — the maker LIMIT take-profit (reduceOnly limit).
        try:
            for o in exchange.fetch_open_orders(msym):
                if not self._regular_is_protective(o):
                    continue
                orders.append(ProtectiveOrder(
                    ids=self._ids(o), is_stop=False,
                    reduce_only=self._reduce_only(o), kind="regular", raw=o,
                ))
        except Exception:
            logger.warning("Could not fetch OKX regular orders for %s", symbol)
        # Algo (trigger/conditional) order book — stop-losses and TP-market live here.
        try:
            algos = exchange.fetch_open_orders(msym, params={"ordType": "conditional", "stop": True})
            for o in algos:
                info = o.get("info") or {}
                ordtype = (info.get("ordType") or o.get("type") or "").lower()
                has_sl = bool(info.get("slTriggerPx") or info.get("slOrdPx"))
                orders.append(ProtectiveOrder(
                    ids=self._ids(o),
                    is_stop=has_sl or ordtype in _SL_ORDTYPES,
                    reduce_only=self._reduce_only(o), kind="algo", raw=o,
                ))
        except Exception:
            logger.warning("Could not fetch OKX algo orders for %s (validate ordType filter)", symbol)
        return orders

    def cancel_protective_order(self, exchange, symbol: str, order: ProtectiveOrder) -> None:
        msym = self.market_symbol(symbol)
        oid = order.raw.get("id") or (order.raw.get("info") or {}).get("algoId")
        params = {"trigger": True} if order.kind == "algo" else {}
        exchange.cancel_order(oid, msym, params=params)

    # -- order placement ---------------------------------------------------- #
    def place_stop(self, exchange, symbol: str, exit_side: str, quantity, stop_price):
        # OKX stop-market = algo order with a stop trigger. ccxt maps `stopLossPrice`
        # to OKX's slTriggerPx; slOrdPx -1 makes it fill at market on trigger.
        return exchange.create_order(
            symbol=self.market_symbol(symbol),
            type="market",
            side=exit_side,
            amount=quantity,
            params={"reduceOnly": True, "tdMode": self.td_mode, "stopLossPrice": stop_price},
        )

    def place_take_profit(self, exchange, symbol: str, exit_side: str, quantity, tp_price):
        # Maker reduceOnly LIMIT (rests as a maker; no negative slippage).
        return exchange.create_order(
            symbol=self.market_symbol(symbol),
            type="limit",
            side=exit_side,
            amount=quantity,
            price=tp_price,
            params={"reduceOnly": True, "tdMode": self.td_mode},
        )

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _reduce_only(order: dict) -> bool:
        ro = order.get("reduceOnly")
        if ro is None:
            ro = str((order.get("info") or {}).get("reduceOnly", "")).lower() == "true"
        return bool(ro)

    @classmethod
    def _regular_is_protective(cls, order: dict) -> bool:
        type_l = (order.get("type") or "").lower()
        return type_l == "limit" and cls._reduce_only(order)

    @staticmethod
    def _ids(order: dict) -> set[str]:
        info = order.get("info") or {}
        raw = {order.get("id"), info.get("algoId"), info.get("ordId"), info.get("clOrdId")}
        return {str(i) for i in raw if i}
