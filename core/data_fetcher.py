from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

_FETCH_RETRIES = 3
_RETRY_BACKOFF = (2, 5)  # seconds: 2s after 1st fail, 5s after 2nd


def create_exchange(
    exchange_id: str,
    api_key: str | None = None,
    api_secret: str | None = None,
    testnet: bool = False,
    market_type: str = "spot",
) -> ccxt.Exchange:
    options: dict = {
        "defaultType": market_type,
        "fetchCurrencies": False,
    }
    params: dict = {"enableRateLimit": True, "timeout": 30000, "options": options}
    if api_key and api_secret:
        params["apiKey"] = api_key
        params["secret"] = api_secret
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class(params)
    if testnet and market_type == "future":
        exchange.enable_demo_trading(True)
    return exchange


_PAGE_SIZE = 1000  # Binance hard cap per request


def _fetch_ohlcv_once(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int,
) -> list:
    """Single attempt — no retry. Called by fetch_ohlcv which handles retries."""
    if limit <= _PAGE_SIZE:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # Paginate backwards from now until we have enough candles. Each iteration
    # recomputes `since` from the current earliest candle (one page further back)
    # and keeps only candles older than what we already hold, so overlapping bars
    # are dropped and the window genuinely advances toward the start of history.
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    raw: list = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=_PAGE_SIZE)
    if not raw:
        return raw

    while len(raw) < limit:
        earliest = raw[0][0]
        since = earliest - tf_ms * _PAGE_SIZE
        page = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=_PAGE_SIZE)
        if not page:
            break
        page = [c for c in page if c[0] < earliest]
        if not page:
            break  # no candles older than what we have — reached start of history
        raw = page + raw

    return raw[-limit:]


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 50,
) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(_FETCH_RETRIES):
        try:
            if attempt == 0:
                logger.debug("Fetching %s %s (%d candles)", symbol, timeframe, limit)
            else:
                logger.warning("Retrying %s %s (attempt %d/%d)", symbol, timeframe, attempt + 1, _FETCH_RETRIES)
            raw = _fetch_ohlcv_once(exchange, symbol, timeframe, limit)
            break
        except (ccxt.RequestTimeout, ccxt.NetworkError) as exc:
            last_exc = exc
            if attempt < _FETCH_RETRIES - 1:
                delay = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                logger.warning("Timeout/network error fetching %s %s — retrying in %ds", symbol, timeframe, delay)
                time.sleep(delay)
    else:
        raise last_exc  # type: ignore[misc]

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df
