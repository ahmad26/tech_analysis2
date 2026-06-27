"""Exchange-adapter seam.

The strategy, sizing, and exit-decision logic (pattern_detector, trading_rules,
backtester, and the Trader/PositionManager *decision* code) are venue-agnostic — they
work on abstract prices and ccxt's *unified* API. Only a handful of operations are not
unified across exchanges; those live behind this adapter so a new venue (OKX) is added
by writing one subclass, never by touching the decision code.

The non-unified seam:
  * building the authenticated futures client (demo/sandbox + extra credentials);
  * reading the free USDT balance (exchange-specific balance field);
  * the realized-P&L / commission / funding ledger (Binance income vs OKX bills);
  * enumerating and cancelling resting protective orders — Binance returns STOP_MARKET
    as *algo* orders on a separate endpoint, the maker LIMIT take-profit as a *regular*
    order; other venues split this differently. fetch_protective_orders normalises both
    into ProtectiveOrder so the manager never sees the split.

Everything else (market entry/close, set_leverage, *_to_precision, market limits,
fetch_positions, placing the STOP_MARKET / maker-LIMIT orders) is ccxt-unified and stays
in the generic Trader/PositionManager.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import ccxt


@dataclass
class ProtectiveOrder:
    """A resting SL/TP order, normalised across a venue's regular and algo order books.

    `kind` records which order book it came from so cancel_protective_order can route the
    cancellation to the right endpoint. `ids` collects every identifier the order may be
    referenced by (so a just-placed order can be excluded from a cancel sweep)."""
    ids: set[str]
    is_stop: bool          # True = stop-loss, False = take-profit
    reduce_only: bool
    kind: str              # venue-specific routing tag, e.g. "regular" | "algo"
    raw: dict = field(default_factory=dict)


class ExchangeAdapter:
    """Base adapter. Subclasses implement the venue-specific operations below.

    `id` is the ccxt exchange id; `label` is for logs/alerts/state-file naming."""

    id: str = "base"
    label: str = "base"
    # USD floor used when a market reports no minimum cost (Binance USDT-M rejects
    # sub-$20 notionals with -4164; OKX min cost is ~$1). See min-notional guard.
    min_notional_floor: float = 20.0

    # -- order sizing (coin units <-> exchange order amount) ----------------- #
    def order_amount(self, exchange, symbol: str, coin_qty: float) -> float:
        """Convert a quantity in BASE-COIN units (how the strategy sizes) into the
        amount this venue's order API expects, precision-applied. Binance USDT-M takes
        the base-coin amount directly (identity); OKX takes the number of CONTRACTS
        (coin_qty / contractSize), so it overrides this. May raise ccxt.InvalidOrder
        when the result is below the exchange minimum."""
        return float(exchange.amount_to_precision(symbol, coin_qty))

    def amount_to_coins(self, exchange, symbol: str, amount: float) -> float:
        """Inverse of order_amount: the base-coin quantity an order `amount` represents,
        used for notional/PnL math. Identity for Binance; amount*contractSize for OKX."""
        return amount

    # -- symbol dialect ----------------------------------------------------- #
    def market_symbol(self, symbol: str) -> str:
        """Translate a base unified symbol (e.g. 'BTC/USDT', as used for state keys and
        public-data scanning) into the form this venue's futures market expects for
        order/position/candle calls. Identity for Binance USDT-M (ccxt resolves
        'BTC/USDT' to the perpetual under defaultType=future); OKX overrides this to its
        EEA XPERP symbol."""
        return symbol

    def base_symbol(self, venue_symbol: str) -> str:
        """Inverse of market_symbol: a symbol returned by the exchange (positions/orders)
        -> the canonical 'COIN/USDT' used for state keys and signal matching. Binance's
        'ETH/USDT:USDT' -> 'ETH/USDT'; OKX overrides (its venue symbols are 'ETH/USD:…')."""
        return venue_symbol.split(":")[0]

    def supports_symbol(self, exchange, symbol: str) -> bool:
        """Whether this venue can actually trade `symbol`. Default True (Binance lists all
        scanned pairs). OKX overrides: its EEA XPERP set omits some coins (e.g. DOT), and
        attempting them returns 51155 — so the trader skips them up front instead."""
        return True

    def is_immediate_trigger_error(self, exc: Exception) -> bool:
        """True if `exc` is the venue's 'stop would trigger immediately' rejection — the
        market is already at/through the stop level, so the trail should CLOSE rather than
        rest a doomed order. Binance surfaces -2021 as ccxt.OrderImmediatelyFillable; OKX
        returns 51280 as a plain InvalidOrder, so it overrides to also match that."""
        return isinstance(exc, ccxt.OrderImmediatelyFillable)

    def market_order(self, exchange, symbol: str, side: str, amount, *, reduce_only: bool = False):
        """Market entry/close. Centralised so a venue can attach its required params
        (e.g. OKX's tdMode). `symbol` is the canonical base symbol; translated here."""
        params = {"reduceOnly": True} if reduce_only else {}
        return exchange.create_market_order(self.market_symbol(symbol), side, amount, params=params)

    def set_leverage(self, exchange, leverage: int, symbol: str) -> None:
        """Set leverage for `symbol`. OKX requires a margin-mode param, so it overrides."""
        exchange.set_leverage(leverage, self.market_symbol(symbol))

    # -- construction ------------------------------------------------------- #
    def build_exchange(self, api_key: str, api_secret: str, *, demo: bool = False,
                       password: str | None = None):
        """Build the authenticated futures client. `password` is the API passphrase
        some venues require (e.g. OKX); venues that don't need it ignore it."""
        raise NotImplementedError

    # -- account / ledger --------------------------------------------------- #
    def available_balance(self, exchange) -> float:
        raise NotImplementedError

    def realized_pnl(self, exchange, symbol: str, since_ms: int) -> float:
        raise NotImplementedError

    def trade_costs(self, exchange, symbol: str, since_ms: int) -> tuple[float, float]:
        """(commission, funding) over the position's window; both typically negative."""
        raise NotImplementedError

    # -- protective orders (SL/TP) ------------------------------------------ #
    def fetch_protective_orders(self, exchange, symbol: str) -> list[ProtectiveOrder]:
        raise NotImplementedError

    def cancel_protective_order(self, exchange, symbol: str, order: ProtectiveOrder) -> None:
        raise NotImplementedError

    # -- order placement (ccxt-unified defaults; override only if a venue differs) -- #
    def place_stop(self, exchange, symbol: str, exit_side: str, quantity, stop_price):
        """STOP_MARKET reduceOnly. May raise ccxt.OrderImmediatelyFillable when the
        market is already through the level — callers treat that as 'close now'."""
        return exchange.create_order(
            symbol=symbol,
            type="STOP_MARKET",
            side=exit_side,
            amount=quantity,
            params={"stopPrice": stop_price, "reduceOnly": True},
        )

    def place_take_profit(self, exchange, symbol: str, exit_side: str, quantity, tp_price):
        """Maker reduceOnly LIMIT take-profit (rests above price for a long / below for a
        short, so it fills as a maker with no negative slippage)."""
        return exchange.create_order(
            symbol=symbol,
            type="LIMIT",
            side=exit_side,
            amount=quantity,
            price=tp_price,
            params={"reduceOnly": True, "timeInForce": "GTC"},
        )
