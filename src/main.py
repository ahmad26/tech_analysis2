from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from src.alerter import TelegramAlerter
from src.alert_tracker import AlertTracker
from src.config import load_config
from src.data_fetcher import create_exchange, fetch_ohlcv
from src.models import AppConfig
from src.pattern_detector import detect_patterns, extract_htf_levels, HTF_MAP, HTF_CANDLES
from src.trading_rules import MIN_RR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)



def _fetch_realized_pnl(exchange, position) -> float:
    try:
        market_id = exchange.market_id(position.symbol)
        # 1-second buffer before open time to handle fill timestamps arriving slightly early
        start_time = max(0, position.opened_at_ms - 1000)
        result = exchange.fapiPrivateGetIncome({
            "incomeType": "REALIZED_PNL",
            "symbol": market_id,
            "startTime": start_time,
            "limit": 1000,
        })
        return sum(float(item["income"]) for item in result)
    except Exception:
        logger.warning("Could not fetch realized PnL for %s", position.symbol, exc_info=True)
        return 0.0


TF_PRIORITY: dict[str, int] = {"15m": 1, "1h": 2, "4h": 3, "1d": 4}


def _collect_signals(
    config: AppConfig,
    timeframes: list,
    data_cache: dict,
    tracker: AlertTracker,
) -> tuple[list, dict[str, dict[str, set[str]]]]:
    """Phase 1: scan all timeframes and return every valid, non-duplicate signal.

    Also returns all_directions: {symbol: {tf: set("bullish"|"bearish")}} for
    every detected pattern regardless of R/R or dedup status — used by phase 2
    to veto lower-TF signals that conflict with a higher-TF detected direction.
    """
    signals = []
    all_directions: dict[str, dict[str, set[str]]] = {}
    for tf in timeframes:
        logger.info("Starting scan for timeframe %s", tf)
        for symbol in config.symbols:
            symbol_data = data_cache.get(symbol, {})
            if tf not in symbol_data:
                logger.error("Failed to fetch data for %s %s", symbol, tf)
                continue
            htf_levels = _build_htf_levels(tf, symbol_data)
            patterns = detect_patterns(symbol_data[tf], symbol, tf, config.patterns_for(tf), htf_levels, config.min_atr_pct.get(tf))
            for p in patterns:
                all_directions.setdefault(symbol, {}).setdefault(tf, set()).add(p.signal)
                if p.trading_signal is None or p.trading_signal.risk_reward < MIN_RR:
                    rr = f"{p.trading_signal.risk_reward:.2f}" if p.trading_signal else "n/a"
                    logger.info("Dropped low R/R: %s (R/R=%s)", p.alert_key, rr)
                    continue
                if tracker.is_duplicate(p.alert_key):
                    logger.debug("Skipping duplicate: %s", p.alert_key)
                    continue
                signals.append(p)
        logger.info("Scan complete for timeframe %s", tf)
    return signals, all_directions


def _resolve_signals(signals: list, all_directions: dict[str, dict[str, set[str]]]) -> list:
    """Phase 2: per symbol keep only the highest-TF signal.

    Prevents a lower-TF signal from triggering a trade that contradicts a
    higher-TF signal on the same symbol (higher timeframe always wins).

    Additionally vetoes any candidate signal if a higher TF has detected
    pattern(s) exclusively in the opposite direction — even if that higher-TF
    signal didn't qualify for trading due to poor R/R or deduplication.
    """
    by_symbol: dict = {}
    for p in signals:
        by_symbol.setdefault(p.symbol, []).append(p)

    resolved = []
    for sym_signals in by_symbol.values():
        sym_signals.sort(key=lambda p: TF_PRIORITY.get(p.timeframe, 0), reverse=True)
        winner = sym_signals[0]
        suppressed = sym_signals[1:]

        # Veto: if any higher TF detected patterns exclusively in the opposite
        # direction (not mixed), block this signal regardless of its R/R.
        action = winner.trading_signal.action if winner.trading_signal else None
        if action:
            opposite = "bearish" if action == "BUY" else "bullish"
            same = "bullish" if action == "BUY" else "bearish"
            symbol_dirs = all_directions.get(winner.symbol, {})
            veto_tf = None
            for tf, directions in symbol_dirs.items():
                if TF_PRIORITY.get(tf, 0) <= TF_PRIORITY.get(winner.timeframe, 0):
                    continue
                if opposite in directions and same not in directions:
                    veto_tf = tf
                    break
            if veto_tf:
                logger.info(
                    "%s: vetoing %s %s — %s has conflicting direction",
                    winner.symbol, winner.timeframe, action, veto_tf,
                )
                continue

        if suppressed:
            logger.info(
                "%s: using %s %s, suppressing %s",
                winner.symbol,
                winner.timeframe,
                winner.trading_signal.action if winner.trading_signal else "?",
                ", ".join(
                    f"{p.timeframe} {p.trading_signal.action if p.trading_signal else '?'}"
                    for p in suppressed
                ),
            )
        resolved.append(winner)
    return resolved


def _print_signal_matrix(
    all_detected: list[DetectedPattern],
    timeframes: list[str],
    symbols: list[str],
) -> None:
    from collections import defaultdict

    grid: dict[tuple[str, str], set[str]] = defaultdict(set)
    for p in all_detected:
        grid[(p.symbol, p.timeframe)].add(p.signal)

    active = [s for s in symbols if any((s, tf) in grid for tf in timeframes)]
    if not active:
        return

    def cell(symbol: str, tf: str) -> str:
        signals = grid.get((symbol, tf), set())
        if not signals:
            return "  -  "
        if "bullish" in signals and "bearish" in signals:
            return "MIXED"
        return " BULL" if "bullish" in signals else " BEAR"

    sym_w = max(len(s) for s in active)
    col_w = max(len(tf) for tf in timeframes)
    col_w = max(col_w, 5)  # at least wide enough for "MIXED"

    header = " " * (sym_w + 3) + "  ".join(tf.center(col_w) for tf in timeframes)
    divider = "-" * (sym_w + 3) + "--".join("-" * col_w for _ in timeframes)

    print(f"\n{'SIGNAL MATRIX':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(divider)
    for symbol in active:
        cells = "  ".join(cell(symbol, tf).center(col_w) for tf in timeframes)
        print(f"{symbol:>{sym_w}}   {cells}")
    print("=" * len(header))


def _prefetch_all_data(
    exchange, config: AppConfig, timeframes: list[str]
) -> tuple[dict, list[tuple[str, str, Exception]]]:
    """Fetch all needed OHLCV data (own + HTF) for every symbol, deduped across timeframes."""
    import ccxt as _ccxt
    needed_tfs: set[str] = set(timeframes)
    for tf in timeframes:
        needed_tfs.update(HTF_MAP.get(tf, []))
    candles = max(HTF_CANDLES, config.candles_to_fetch)
    cache: dict = {}
    errors: list[tuple[str, str, Exception]] = []

    # Load markets once upfront so all fetch_ohlcv calls can reuse the cached result.
    # Without this, ccxt calls exchangeInfo inside every fetch_ohlcv, so a single
    # transient failure causes 40 identical errors.
    _MARKET_RETRIES = 3
    _MARKET_BACKOFF = (5, 15)
    market_exc: Exception | None = None
    for attempt in range(_MARKET_RETRIES):
        try:
            exchange.load_markets()
            market_exc = None
            break
        except (_ccxt.RequestTimeout, _ccxt.NetworkError) as e:
            market_exc = e
            if attempt < _MARKET_RETRIES - 1:
                delay = _MARKET_BACKOFF[min(attempt, len(_MARKET_BACKOFF) - 1)]
                logger.warning("Could not load markets (attempt %d/%d) — retrying in %ds: %s", attempt + 1, _MARKET_RETRIES, delay, e)
                import time as _time; _time.sleep(delay)
    if market_exc is not None:
        logger.error("Market load failed after %d attempts: %s", _MARKET_RETRIES, market_exc)
        for symbol in config.symbols:
            for tf in needed_tfs:
                errors.append((symbol, tf, market_exc))
        return cache, errors

    for symbol in config.symbols:
        cache[symbol] = {}
        for tf in needed_tfs:
            try:
                cache[symbol][tf] = fetch_ohlcv(exchange, symbol, tf, limit=candles)
            except Exception as e:
                logger.warning("Could not fetch %s %s", symbol, tf)
                errors.append((symbol, tf, e))
    return cache, errors


def _send_fetch_error_alert(alerter: TelegramAlerter, errors: list[tuple[str, str, Exception]]) -> None:
    from collections import Counter
    total = len(errors)
    summaries = []
    for _, _, e in errors:
        first_line = str(e).split("\n")[0][:120]
        summaries.append(f"{type(e).__name__}: {first_line}")
    counts = Counter(summaries)
    most_common_msg, most_common_count = counts.most_common(1)[0]
    # If every error points to exchangeInfo, it's a market-load failure — say so clearly.
    is_market_load = "exchangeInfo" in most_common_msg and most_common_count == total
    if is_market_load:
        text = f"⚠️ Scanner: market load failed — scan skipped\n{most_common_msg}"
    elif most_common_count == total:
        text = f"⚠️ Scanner: {total} pair(s) failed to fetch\n{most_common_msg}"
    else:
        lines = [f"⚠️ Scanner: {total} pair(s) failed to fetch"]
        for msg, cnt in counts.most_common(3):
            lines.append(f"• {cnt}× {msg}")
        text = "\n".join(lines)
    alerter.send_text(text)


def _build_htf_levels(tf: str, tf_data: dict) -> list[tuple[float, str]]:
    """Combine key levels from all higher timeframes for a given signal timeframe."""
    levels: list[tuple[float, str]] = []
    for higher_tf in HTF_MAP.get(tf, []):
        if higher_tf in tf_data:
            levels.extend(extract_htf_levels(tf_data[higher_tf], higher_tf))
    return levels


def _manage_positions_only(config: AppConfig) -> None:
    """Fetch data only for open positions and trail stops. Lightweight — no pattern scan."""
    from src.trader import Trader
    from src.position_tracker import PositionTracker
    from src.position_manager import PositionManager

    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        sys.exit(1)

    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env for position management")
        sys.exit(1)

    alerter = TelegramAlerter(config.telegram_bot_token, config.telegram_chat_id)
    position_tracker = PositionTracker()
    positions = position_tracker.all()

    if not positions:
        logger.info("No open positions to manage.")
        return

    trading_exchange = create_exchange(config.exchange, api_key, api_secret, testnet=False, market_type="future")
    position_manager = PositionManager(trading_exchange, position_tracker, alerter=alerter)

    # Fetch data only for symbols/timeframes that are actually needed
    scan_exchange = create_exchange(config.exchange)
    data_cache: dict = {}
    for symbol, pos in positions.items():
        if pos.sl <= 0:
            continue
        tf = pos.signal_timeframe
        data_cache.setdefault(symbol, {})
        try:
            data_cache[symbol][tf] = fetch_ohlcv(scan_exchange, symbol, tf, limit=50)
        except Exception:
            logger.warning("Could not fetch %s %s for position management", symbol, tf)

    position_manager.run(data_cache)
    logger.info("Position management complete — %d position(s) checked.", len(positions))


def _status(config: AppConfig) -> None:
    """Print current signal recommendations and open position status."""
    from src.position_tracker import PositionTracker

    exchange = create_exchange(config.exchange)
    data_cache, _ = _prefetch_all_data(exchange, config, config.timeframes)
    positions = PositionTracker().all()

    all_detected: list = []
    for symbol in config.symbols:
        symbol_data = data_cache.get(symbol, {})
        for tf in config.timeframes:
            if tf not in symbol_data:
                continue
            htf_levels = _build_htf_levels(tf, symbol_data)
            detected = detect_patterns(symbol_data[tf], symbol, tf, config.patterns_for(tf), htf_levels, config.min_atr_pct.get(tf))
            all_detected.extend(detected)

    # Open positions without a matching current signal
    signalled_symbols = {p.symbol for p in all_detected if p.trading_signal and p.trading_signal.risk_reward >= MIN_RR}

    print(f"\n{'='*56}")
    print(f"  STATUS REPORT — {len(all_detected)} signal(s), {len(positions)} open position(s)")
    print(f"{'='*56}")

    # --- Signals with optional position info ---
    printed_signals = set()
    for p in all_detected:
        if p.trading_signal is None or p.trading_signal.risk_reward < MIN_RR:
            continue
        ts = p.trading_signal
        key = p.symbol
        pos = positions.get(key)

        direction = "⬆ BUY " if ts.action == "BUY" else "⬇ SELL"
        print(f"\n{direction}  {p.symbol}  [{p.timeframe}]  {p.pattern_name}")
        print(f"  Entry: {ts.entry:.4f}   Signal SL: {ts.stop_loss:.4f}   Signal TP: {ts.take_profit:.4f}   R/R: 1:{ts.risk_reward:.1f}")

        if pos is not None:
            side_arrow = "⬆ LONG" if pos.side == "long" else "⬇ SHORT"
            sl_str = f"{pos.sl:.4f}" if pos.sl > 0 else "not set"
            tp_str = f"{pos.tp:.4f}" if pos.tp > 0 else "not set"
            pnl_pct = (ts.entry - pos.entry_price) / pos.entry_price * 100 * (1 if pos.side == "long" else -1)
            print(f"  ✅ POSITION OPEN  {side_arrow}  qty={pos.contracts}  entry={pos.entry_price:.4f}  P&L={pnl_pct:+.2f}%")
            print(f"     Current SL: {sl_str}   Current TP: {tp_str}   TF: {pos.signal_timeframe}")
        else:
            print(f"  — no open position")

        printed_signals.add(key)

    # --- Open positions with no current signal ---
    orphan_positions = {sym: pos for sym, pos in positions.items() if sym not in printed_signals}
    if orphan_positions:
        print(f"\n{'─'*56}")
        print("  Open positions with no active signal:")
        for sym, pos in orphan_positions.items():
            side_arrow = "⬆ LONG" if pos.side == "long" else "⬇ SHORT"
            sl_str = f"{pos.sl:.4f}" if pos.sl > 0 else "not set"
            tp_str = f"{pos.tp:.4f}" if pos.tp > 0 else "not set"
            print(f"\n  {side_arrow}  {sym}  qty={pos.contracts}  entry={pos.entry_price:.4f}")
            print(f"     Current SL: {sl_str}   Current TP: {tp_str}   TF: {pos.signal_timeframe}")

    print(f"\n{'='*56}\n")


def _dry_run(config: AppConfig) -> None:
    exchange = create_exchange(config.exchange)
    data_cache, _ = _prefetch_all_data(exchange, config, config.timeframes)
    all_detected: list[DetectedPattern] = []

    for symbol in config.symbols:
        symbol_data = data_cache.get(symbol, {})
        for tf in config.timeframes:
            if tf not in symbol_data:
                continue
            htf_levels = _build_htf_levels(tf, symbol_data)
            detected = detect_patterns(symbol_data[tf], symbol, tf, config.patterns_for(tf), htf_levels, config.min_atr_pct.get(tf))
            all_detected.extend(detected)

    if not all_detected:
        print("\nNo patterns detected.")
        return

    _print_signal_matrix(all_detected, config.timeframes, config.symbols)

    print(f"\n{'='*50}")
    print(f" {len(all_detected)} pattern(s) detected")
    print(f"{'='*50}\n")
    for p in all_detected:
        print(p.format_message())
        print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto candlestick pattern alerts")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single scan, print results to console, and exit (no Telegram needed)",
    )
    parser.add_argument(
        "--trade",
        action="store_true",
        help="Enable live order execution (requires BINANCE_API_KEY and BINANCE_API_SECRET in .env)",
    )
    parser.add_argument(
        "--trade-risk-pct",
        type=float,
        default=1.0,
        metavar="PCT",
        help="Percentage of free USDT balance to risk per trade when --trade is active (default: 1.0)",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=1,
        metavar="X",
        help="Futures leverage multiplier (default: 1)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current signal recommendations and open position status, then exit",
    )
    parser.add_argument(
        "--manage-positions",
        action="store_true",
        help="Check and trail open positions only — no pattern scan (run every 5 min via cron)",
    )
    parser.add_argument(
        "--cancel-orders",
        action="store_true",
        help="Cancel all open futures orders (all symbols) and exit — use to clear stale orders",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest on 1000 candles of historical data per symbol/timeframe",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=1.0,
        metavar="USD",
        help="Dollar amount risked per trade for P&L calculation (default: 1.0)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        metavar="TF",
        help="Limit backtest to specific timeframes, e.g. --timeframes 15m 1h",
    )
    parser.add_argument(
        "--candles",
        type=int,
        default=None,
        metavar="N",
        help="Number of candles to fetch per symbol for backtest (default: 1000)",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        metavar="PATTERN",
        help="Limit backtest to specific patterns, e.g. --patterns bullish_engulfing bearish_engulfing",
    )
    parser.add_argument(
        "--wallet",
        type=float,
        default=1000.0,
        metavar="USD",
        help="Starting wallet size for %%-of-wallet simulation (default: 1000.0)",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=None,
        metavar="PCT",
        help="Risk this %% of current wallet per trade (enables compound sizing simulation)",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=None,
        metavar="RT",
        help="Round-trip taker fee as a fraction of notional for backtest P&L "
             "(default: 0.0009 = 0.09%%; pass 0 for gross/fee-free)",
    )
    parser.add_argument(
        "--no-maker-tp",
        action="store_true",
        help="Model the take-profit as taker too (default: TP fills are maker, matching "
             "the live reduceOnly LIMIT TP)",
    )
    parser.add_argument(
        "--exit-model",
        choices=["rolling", "live"],
        default="rolling",
        help="Backtest exit simulation: 'rolling' = legacy cascading multi-TP model; "
             "'live' = the actual trader mechanics (full exit at TP1, ATR chandelier + "
             "staged R-locks, R/R>=1.5 gate)",
    )
    parser.add_argument(
        "--htf",
        action="store_true",
        help="Include higher-timeframe levels (MA50/MA200 + Fib50) as SL/TP candidates "
             "and TP obstacles in the backtest, point-in-time, matching the live scanner",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Apply the live R/R >= 1.5 signal gate to the rolling exit model too "
             "(the live exit model always gates)",
    )
    parser.add_argument(
        "--body-anchors",
        action="store_true",
        help="Experimental: anchor the Fibonacci windows to candle body boundaries "
             "(max(open,close)/min(open,close)) instead of wick extremes (high/low)",
    )
    parser.add_argument(
        "--dense-grid",
        type=float,
        default=0.0,
        metavar="ATR_MULT",
        help="EXPERIMENT (rolling model): insert intermediate trailing rungs this many "
             "ATR apart between entry and the final TP (e.g. 0.5). Tight trail, no lag — "
             "tests whether finer rungs lock more profit on runs that fall short of the "
             "next structural level. 0 = off (default).",
    )
    parser.add_argument(
        "--chandelier",
        type=float,
        default=0.0,
        metavar="ATR_MULT",
        help="EXPERIMENT (rolling model): once break-even is reached, layer a CONTINUOUS "
             "ATR chandelier (running_peak - MULT*ATR) on the structural ladder, ratcheted "
             "every bar. Decouples the trail from sparse structural levels so the stop "
             "reacts between Fib/MA rungs (e.g. 3.0). 0 = off (default).",
    )
    args = parser.parse_args()

    config = load_config()

    if args.status:
        _status(config)
        return

    if args.manage_positions:
        _manage_positions_only(config)
        return

    if args.cancel_orders:
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            logger.error("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")
            sys.exit(1)
        from src.position_tracker import PositionTracker
        exchange = create_exchange(config.exchange, api_key, api_secret, testnet=False, market_type="future")

        try:
            positions = exchange.fetch_positions()
        except Exception:
            logger.exception("Failed to fetch positions")
            sys.exit(1)
        try:
            algo_orders = exchange.fapiPrivateGetOpenAlgoOrders({})
        except Exception:
            logger.exception("Failed to fetch algo orders")
            sys.exit(1)

        protected_symbols = {p["symbol"].split(":")[0] for p in positions if abs(p.get("contracts") or 0) > 0}
        tracked = PositionTracker().all()

        # Normalise raw Binance symbol "SOLUSDT" → ccxt unified "SOL/USDT"
        exchange.load_markets()
        def _normalise(raw: str) -> str:
            for sym, mkt in exchange.markets.items():
                if mkt.get("id") == raw:
                    return sym.split(":")[0]
            return raw

        # Group algo orders by normalised symbol
        from collections import defaultdict
        algo_by_symbol: dict = defaultdict(list)
        for o in algo_orders:
            algo_by_symbol[_normalise(o["symbol"])].append(o)

        logger.info(
            "Found %d algo order(s) across %d symbol(s) — %d position(s) open",
            len(algo_orders), len(algo_by_symbol), len(protected_symbols),
        )

        cancelled = 0
        for symbol, orders in algo_by_symbol.items():
            if symbol not in protected_symbols:
                # No open position — cancel all
                for o in orders:
                    try:
                        exchange.fapiPrivateDeleteAlgoOrder({"algoId": o["algoId"]})
                        logger.info("Cancelled orphaned algo order for %s (algoId=%s)", symbol, o["algoId"])
                        cancelled += 1
                    except Exception as e:
                        logger.warning("Could not cancel algoId=%s for %s: %s", o["algoId"], symbol, e)
                continue

            pos = tracked.get(symbol)
            if pos is None or pos.sl <= 0:
                logger.info("Keeping %d algo order(s) for unmanaged position %s", len(orders), symbol)
                continue

            # Managed position — cancel all and re-place exactly 1 SL + 1 TP
            logger.info("Normalising %d algo order(s) for %s → 1 SL + 1 TP", len(orders), symbol)
            for o in orders:
                try:
                    exchange.fapiPrivateDeleteAlgoOrder({"algoId": o["algoId"]})
                    cancelled += 1
                except Exception as e:
                    logger.warning("Could not cancel algoId=%s for %s: %s", o["algoId"], symbol, e)

            exit_side = "sell" if pos.side == "long" else "buy"
            try:
                sl_price = float(exchange.price_to_precision(symbol, pos.sl))
                exchange.fapiPrivatePostAlgoOrder({
                    "symbol": exchange.market_id(symbol),
                    "side": exit_side.upper(),
                    "positionSide": "LONG" if pos.side == "long" else "SHORT",
                    "strategyType": "VP",
                    "quantity": pos.contracts,
                    "stopPrice": sl_price,
                    "reduceOnly": "true",
                })
                logger.info("Placed algo SL for %s at %.4f", symbol, sl_price)
            except Exception:
                # Fall back to regular STOP_MARKET order
                try:
                    exchange.create_order(symbol=symbol, type="STOP_MARKET", side=exit_side,
                                          amount=pos.contracts, params={"stopPrice": pos.sl, "reduceOnly": True})
                    logger.info("Placed STOP_MARKET SL for %s at %.4f", symbol, pos.sl)
                except Exception:
                    logger.exception("Failed to place SL for %s", symbol)
            try:
                tp_price = float(exchange.price_to_precision(symbol, pos.tp))
                exchange.fapiPrivatePostAlgoOrder({
                    "symbol": exchange.market_id(symbol),
                    "side": exit_side.upper(),
                    "positionSide": "LONG" if pos.side == "long" else "SHORT",
                    "strategyType": "VP",
                    "quantity": pos.contracts,
                    "stopPrice": tp_price,
                    "reduceOnly": "true",
                })
                logger.info("Placed algo TP for %s at %.4f", symbol, tp_price)
            except Exception:
                try:
                    exchange.create_order(symbol=symbol, type="TAKE_PROFIT_MARKET", side=exit_side,
                                          amount=pos.contracts, params={"stopPrice": pos.tp, "reduceOnly": True})
                    logger.info("Placed TAKE_PROFIT_MARKET TP for %s at %.4f", symbol, pos.tp)
                except Exception:
                    logger.exception("Failed to place TP for %s", symbol)

        logger.info("Done — cancelled %d algo order(s)", cancelled)
        return

    if args.dry_run:
        _dry_run(config)
        return

    if args.backtest:
        from src.backtester import run_backtest, print_backtest_results, simulate_pct_wallet, Fees, DEFAULT_FEE_RT
        fee_rt = DEFAULT_FEE_RT if args.fee is None else args.fee
        if fee_rt <= 0:
            fees = Fees.free()
        else:
            # Round-trip taker = fee_rt; the take-profit leg is maker unless --no-maker-tp.
            fees = Fees(taker_side=fee_rt / 2, maker_tp=not args.no_maker_tp)
        print("Fetching historical data and running backtest...")
        if args.body_anchors:
            import src.pattern_detector as _pd
            _pd.USE_BODY_ANCHORS = True
            print("  [body-anchors] Fib windows anchored to candle bodies, not wicks")
        results, date_ranges = run_backtest(config, timeframes=args.timeframes, candles=args.candles, patterns=args.patterns, exit_model=args.exit_model, htf=args.htf, gate=args.gate, dense_grid=args.dense_grid, chandelier=args.chandelier)
        print_backtest_results(results, risk=args.risk, date_ranges=date_ranges, fees=fees)
        if args.risk_pct is not None:
            simulate_pct_wallet(results, args.wallet, args.risk_pct, fees=fees)
        return

    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        sys.exit(1)

    alerter = TelegramAlerter(config.telegram_bot_token, config.telegram_chat_id)
    tracker = AlertTracker(config.state_file, config.state_ttl_hours)

    trader = None
    position_manager = None
    if args.trade:
        from src.trader import Trader
        from src.position_tracker import PositionTracker
        from src.position_manager import PositionManager
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            logger.error("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env for --trade")
            sys.exit(1)
        trading_exchange = create_exchange(
            config.exchange, api_key, api_secret, testnet=False, market_type="future"
        )
        position_tracker = PositionTracker()
        trader = Trader(trading_exchange, alerter=alerter, position_tracker=position_tracker, risk_pct=args.trade_risk_pct, leverage=args.leverage)
        position_manager = PositionManager(trading_exchange, position_tracker, alerter=alerter)
        logger.info(
            "Trading enabled on Binance futures MAINNET, risk=%.1f%% per trade, leverage=%dx",
            args.trade_risk_pct, args.leverage,
        )

        # Detect positions closed since last run, cancel orphaned SL/TP, and notify
        try:
            current_positions = trading_exchange.fetch_positions()
            _, closed = position_tracker.sync(current_positions)
            for pos in closed:
                trader.cancel_conditional_orders(pos.symbol)
                pnl = _fetch_realized_pnl(trading_exchange, pos)
                alerter.send_position_closed(pos.symbol, pos.side, pos.contracts, pos.entry_price, pnl)
        except Exception:
            logger.exception("Failed to check position changes")

    tfs = args.timeframes if args.timeframes else config.timeframes
    scan_start = time.monotonic()
    scan_exchange = create_exchange(config.exchange)
    data_cache, fetch_errors = _prefetch_all_data(scan_exchange, config, tfs)
    if fetch_errors:
        _send_fetch_error_alert(alerter, fetch_errors)
    if position_manager is not None:
        position_manager.run(data_cache)

    # Phase 1: collect all valid signals across all timeframes
    # Phase 2: resolve conflicts — per symbol, only the highest-TF signal is kept
    signals = _resolve_signals(*_collect_signals(config, tfs, data_cache, tracker))

    executed = 0
    for p in signals:
        if trader is not None:
            try:
                if position_manager is not None:
                    handled = position_manager.handle_signal(p, data_cache)
                    if handled:
                        tracker.record(p.alert_key)
                        executed += 1
                        continue
                if trader.execute_signal(p):
                    tracker.record(p.alert_key)
                    executed += 1
            except Exception:
                logger.exception("Error executing signal for %s", p.alert_key)
            continue
        try:
            success = alerter.send_alert(p)
        except Exception:
            logger.exception("Error sending alert for %s", p.alert_key)
            continue
        if success:
            tracker.record(p.alert_key)
            executed += 1

    tracker.cleanup()
    # One-line machine-readable summary — parsed by scripts/healthcheck.sh
    logger.info(
        "Scan summary: duration=%.1fs timeframes=%s fetch_errors=%d signals=%d executed=%d mode=%s",
        time.monotonic() - scan_start,
        ",".join(tfs),
        len(fetch_errors),
        len(signals),
        executed,
        "trade" if trader is not None else "alert",
    )
    logger.info("Scan complete.")


if __name__ == "__main__":
    main()
