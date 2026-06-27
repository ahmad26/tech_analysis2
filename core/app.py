"""Shared application runner for all trading venues.

This is the former monolithic `src/main.py`, made venue-agnostic. Everything that is
specific to an exchange (which adapter, where state files live, which credential env
vars, demo vs mainnet) arrives via a `VenueContext`; the binance/ and okx/ apps each
provide one and call `main(ctx)`. Signal scanning always uses Binance public spot OHLCV
(`config.exchange`), so both venues trade byte-identical signals — only execution and
position-management candles use the venue's own futures feed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from core.alerter import TelegramAlerter
from core.alert_tracker import AlertTracker
from core.config import load_config
from core.data_fetcher import create_exchange, fetch_ohlcv
from core.models import AppConfig, DetectedPattern
from core.pattern_detector import detect_patterns, extract_htf_levels, HTF_MAP, HTF_CANDLES
from core.trading_rules import MIN_RR
from core.venue import VenueContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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


def _manage_positions_only(config: AppConfig, ctx: VenueContext) -> None:
    """Fetch data only for open positions and trail stops. Lightweight — no pattern scan."""
    from core.position_tracker import PositionTracker
    from core.position_manager import PositionManager

    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        sys.exit(1)

    missing = ctx.missing_credentials()
    if missing:
        logger.error("%s must be set in .env for position management", ", ".join(missing))
        sys.exit(1)

    alerter = TelegramAlerter(config.telegram_bot_token, config.telegram_chat_id, venue_label=ctx.label)
    position_tracker = PositionTracker(ctx.position_state, symbol_normalizer=ctx.adapter.base_symbol)
    positions = position_tracker.all()

    if not positions:
        logger.info("No open positions to manage.")
        return

    trading_exchange = ctx.build_trading_exchange()
    # Markets must be loaded before any trail: _replace_orders calls
    # price_to_precision/amount_to_precision, which need market metadata.
    trading_exchange.load_markets()
    position_manager = PositionManager(trading_exchange, ctx.adapter, position_tracker, alerter=alerter)

    position_manager.run()
    logger.info("Position management complete — %d position(s) checked.", len(positions))


def _status(config: AppConfig, ctx: VenueContext) -> None:
    """Print current signal recommendations and open position status."""
    from core.position_tracker import PositionTracker

    exchange = create_exchange(config.exchange)
    data_cache, _ = _prefetch_all_data(exchange, config, config.timeframes)
    positions = PositionTracker(ctx.position_state, symbol_normalizer=ctx.adapter.base_symbol).all()

    all_detected: list = []
    for symbol in config.symbols:
        symbol_data = data_cache.get(symbol, {})
        for tf in config.timeframes:
            if tf not in symbol_data:
                continue
            htf_levels = _build_htf_levels(tf, symbol_data)
            detected = detect_patterns(symbol_data[tf], symbol, tf, config.patterns_for(tf), htf_levels, config.min_atr_pct.get(tf))
            all_detected.extend(detected)

    print(f"\n{'='*56}")
    print(f"  STATUS REPORT [{ctx.label}] — {len(all_detected)} signal(s), {len(positions)} open position(s)")
    print(f"{'='*56}")

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


def _cancel_orders(config: AppConfig, ctx: VenueContext) -> None:
    """Normalise resting protective orders: managed positions get exactly 1 SL + 1 TP;
    orphaned protective orders on flat symbols are cancelled. Venue-agnostic via the
    adapter (replaces the old Binance-algo-specific implementation)."""
    from core.position_tracker import PositionTracker
    from core.position_manager import PositionManager

    missing = ctx.missing_credentials()
    if missing:
        logger.error("%s must be set in .env", ", ".join(missing))
        sys.exit(1)

    exchange = ctx.build_trading_exchange()
    exchange.load_markets()
    tracker = PositionTracker(ctx.position_state, symbol_normalizer=ctx.adapter.base_symbol)
    manager = PositionManager(exchange, ctx.adapter, tracker)

    try:
        positions = exchange.fetch_positions()
    except Exception:
        logger.exception("Failed to fetch positions")
        sys.exit(1)

    open_symbols = {p["symbol"].split(":")[0] for p in positions if abs(float(p.get("contracts") or 0)) > 0}
    tracked = tracker.all()

    # Managed positions: re-place exactly 1 SL + 1 TP (the replace cancels the rest).
    for symbol, pos in tracked.items():
        if symbol in open_symbols and pos.sl > 0:
            logger.info("Normalising orders for %s → 1 SL + 1 TP", symbol)
            manager._replace_orders(pos, pos.sl, pos.tp, None)

    # Orphaned protective orders on symbols with no open position: cancel all.
    cancelled = 0
    for symbol in set(config.symbols) | set(tracked) | open_symbols:
        if symbol in open_symbols:
            continue
        for order in ctx.adapter.fetch_protective_orders(exchange, symbol):
            try:
                ctx.adapter.cancel_protective_order(exchange, symbol, order)
                cancelled += 1
                logger.info("Cancelled orphaned protective order for %s", symbol)
            except Exception as e:
                logger.warning("Could not cancel order for %s: %s", symbol, e)

    logger.info("Done — cancelled %d orphaned order(s)", cancelled)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crypto candlestick pattern alerts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single scan, print results to console, and exit (no Telegram needed)")
    parser.add_argument("--trade", action="store_true",
                        help="Enable live order execution (requires this venue's API credentials in .env)")
    parser.add_argument("--trade-risk-pct", type=float, default=1.0, metavar="PCT",
                        help="Percentage of free USDT balance to risk per trade when --trade is active (default: 1.0)")
    parser.add_argument("--leverage", type=int, default=1, metavar="X",
                        help="Futures leverage multiplier (default: 1)")
    parser.add_argument("--status", action="store_true",
                        help="Print current signal recommendations and open position status, then exit")
    parser.add_argument("--manage-positions", action="store_true",
                        help="Check and trail open positions only — no pattern scan (run every 5 min via cron)")
    parser.add_argument("--cancel-orders", action="store_true",
                        help="Normalise/cancel resting protective orders and exit — use to clear stale orders")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest on 1000 candles of historical data per symbol/timeframe")
    parser.add_argument("--risk", type=float, default=1.0, metavar="USD",
                        help="Dollar amount risked per trade for P&L calculation (default: 1.0)")
    parser.add_argument("--timeframes", nargs="+", metavar="TF",
                        help="Limit backtest to specific timeframes, e.g. --timeframes 15m 1h")
    parser.add_argument("--candles", type=int, default=None, metavar="N",
                        help="Number of candles to fetch per symbol for backtest (default: 1000)")
    parser.add_argument("--patterns", nargs="+", metavar="PATTERN",
                        help="Limit backtest to specific patterns")
    parser.add_argument("--wallet", type=float, default=1000.0, metavar="USD",
                        help="Starting wallet size for %%-of-wallet simulation (default: 1000.0)")
    parser.add_argument("--risk-pct", type=float, default=None, metavar="PCT",
                        help="Risk this %% of current wallet per trade (enables compound sizing simulation)")
    parser.add_argument("--fee", type=float, default=None, metavar="RT",
                        help="Round-trip taker fee as a fraction of notional for backtest P&L "
                             "(default: 0.0009 = 0.09%%; pass 0 for gross/fee-free)")
    parser.add_argument("--no-maker-tp", action="store_true",
                        help="Model the take-profit as taker too (default: TP fills are maker)")
    parser.add_argument("--exit-model", choices=["rolling", "live"], default="rolling",
                        help="Backtest exit simulation: 'rolling' = legacy cascading multi-TP model; "
                             "'live' = the actual trader mechanics")
    parser.add_argument("--htf", action="store_true",
                        help="Include higher-timeframe levels as SL/TP candidates and TP obstacles in the backtest")
    parser.add_argument("--gate", action="store_true",
                        help="Apply the live R/R >= 1.5 signal gate to the rolling exit model too")
    parser.add_argument("--body-anchors", action="store_true",
                        help="Experimental: anchor the Fibonacci windows to candle body boundaries")
    parser.add_argument("--dense-grid", type=float, default=0.0, metavar="ATR_MULT",
                        help="EXPERIMENT (rolling model): insert intermediate trailing rungs this many ATR apart")
    parser.add_argument("--chandelier", type=float, default=0.0, metavar="ATR_MULT",
                        help="EXPERIMENT (rolling model): continuous ATR chandelier once break-even is reached")
    parser.add_argument("--breakeven-floor", type=float, default=0.0, metavar="R_MULT",
                        help="(rolling model): arm a break-even stop once profit reaches this many R")
    parser.add_argument("--psych-round", action="store_true",
                        help="Round-number TP awareness forced ON for ALL scanned timeframes (sweep mode)")
    parser.add_argument("--psych-band", type=float, default=0.5, metavar="ATR_MULT",
                        help="How far (in ATR) beyond a round number a TP must be to get pulled in. Default 0.5.")
    parser.add_argument("--psych-buffer", type=float, default=0.1, metavar="ATR_MULT",
                        help="How far (in ATR) on the near side of the round number to rest the adjusted TP. Default 0.1.")
    return parser


def main(ctx: VenueContext) -> None:
    args = _build_arg_parser().parse_args()

    # Round-number TP adjustment is a signal-compute setting read via module globals.
    if args.psych_round:
        from core import trading_rules
        trading_rules.PSYCH_ROUND = True
        trading_rules.PSYCH_ROUND_TFS = None   # force on for all scanned TFs (sweep mode)
        trading_rules.PSYCH_BAND_ATR = args.psych_band
        trading_rules.PSYCH_BUFFER_ATR = args.psych_buffer

    config = load_config()

    if args.status:
        _status(config, ctx)
        return

    if args.manage_positions:
        _manage_positions_only(config, ctx)
        return

    if args.cancel_orders:
        _cancel_orders(config, ctx)
        return

    if args.dry_run:
        _dry_run(config)
        return

    if args.backtest:
        from core.backtester import run_backtest, print_backtest_results, simulate_pct_wallet, Fees, DEFAULT_FEE_RT
        fee_rt = DEFAULT_FEE_RT if args.fee is None else args.fee
        if fee_rt <= 0:
            fees = Fees.free()
        else:
            fees = Fees(taker_side=fee_rt / 2, maker_tp=not args.no_maker_tp)
        print("Fetching historical data and running backtest...")
        if args.body_anchors:
            import core.pattern_detector as _pd
            _pd.USE_BODY_ANCHORS = True
            print("  [body-anchors] Fib windows anchored to candle bodies, not wicks")
        results, date_ranges = run_backtest(config, timeframes=args.timeframes, candles=args.candles, patterns=args.patterns, exit_model=args.exit_model, htf=args.htf, gate=args.gate, dense_grid=args.dense_grid, chandelier=args.chandelier, breakeven_floor_r=args.breakeven_floor)
        print_backtest_results(results, risk=args.risk, date_ranges=date_ranges, fees=fees)
        if args.risk_pct is not None:
            simulate_pct_wallet(results, args.wallet, args.risk_pct, fees=fees)
        return

    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        sys.exit(1)

    alerter = TelegramAlerter(config.telegram_bot_token, config.telegram_chat_id, venue_label=ctx.label)
    tracker = AlertTracker(ctx.alert_state, config.state_ttl_hours)

    trader = None
    position_manager = None
    if args.trade:
        from core.trader import Trader
        from core.position_tracker import PositionTracker
        from core.position_manager import PositionManager
        missing = ctx.missing_credentials()
        if missing:
            logger.error("%s must be set in .env for --trade", ", ".join(missing))
            sys.exit(1)
        trading_exchange = ctx.build_trading_exchange()
        position_tracker = PositionTracker(ctx.position_state, symbol_normalizer=ctx.adapter.base_symbol)
        trader = Trader(trading_exchange, ctx.adapter, alerter=alerter, position_tracker=position_tracker,
                        risk_pct=args.trade_risk_pct, leverage=args.leverage, risk_state_file=ctx.risk_state)
        position_manager = PositionManager(trading_exchange, ctx.adapter, position_tracker, alerter=alerter)
        logger.info(
            "Trading enabled on %s futures MAINNET, risk=%.1f%% per trade, leverage=%dx",
            ctx.label, args.trade_risk_pct, args.leverage,
        )

        # Detect positions closed since last run, cancel orphaned SL/TP, and notify
        try:
            current_positions = trading_exchange.fetch_positions()
            _, closed = position_tracker.sync(current_positions)
            for pos in closed:
                trader.cancel_conditional_orders(pos.symbol)
                pnl = ctx.adapter.realized_pnl(trading_exchange, pos.symbol, pos.opened_at_ms)
                commission, funding = ctx.adapter.trade_costs(trading_exchange, pos.symbol, pos.opened_at_ms)
                position_tracker.log_closed_trade(
                    pos, realized_pnl=pnl, commission=commission, funding=funding,
                )
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
        position_manager.run()

    # Phase 1: collect all valid signals; Phase 2: per symbol keep highest-TF signal
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
        "Scan summary: venue=%s duration=%.1fs timeframes=%s fetch_errors=%d signals=%d executed=%d mode=%s",
        ctx.label,
        time.monotonic() - scan_start,
        ",".join(tfs),
        len(fetch_errors),
        len(signals),
        executed,
        "trade" if trader is not None else "alert",
    )
    logger.info("Scan complete.")
