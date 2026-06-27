# Binance → OKX migration (parallel dual-venue)

Status: **structure built and verified on the `migration-okx` branch; OKX trading path
exercised on the live account; NOT yet cut over.**
The live Binance bot still runs on the old `src/` package via cron — nothing here has
touched it.

## Structure

```
core/        SHARED, venue-agnostic. Strategy + execution *decisions*.
             pattern_detector, trading_rules, models, config, data_fetcher,
             alerter, alert_tracker, backtester, position_tracker,
             trader (generic), position_manager (generic), app (shared runner),
             venue (VenueContext), exchange_adapter (the seam).
binance/     Thin app: adapter.py (BinanceAdapter) + main.py + state/
okx/         Thin app: adapter.py (OKXAdapter) + main.py + state/
src/         LEGACY — the currently-live Binance bot. Removed only at cutover.
```

Both apps share one rulebook (`core/`): edit a strategy rule once, both venues change.
Both scan **Binance public spot OHLCV** (`config.exchange`), so signals are identical;
only order execution and position-management candles use each venue's own futures feed.
State is fully partitioned (`binance/state/`, `okx/state/`) so the two apps can run
concurrently without ever corrupting each other's position/risk/alert tracking.

## Run

```bash
python -m binance.main --trade --trade-risk-pct 2.0 --leverage 5   # = current live behaviour
python -m binance.main --manage-positions
python -m okx.main --dry-run
python -m okx.main --trade ...                                     # after sandbox validation
```

## OKX validation status

1. OKX API key (key + secret + **passphrase**) is in `.env` (see `.env.example`).
2. EEA accounts: API calls go to `eea.okx.com` (else 50119), and OKX EEA retail gets
   **USDC-margined XPERP** perpetuals, not global USDT swaps (else 51155) — the adapter
   maps `COIN/USDT` → the live XPERP symbol dynamically. EEA **demo geo-blocks swaps**, so
   `okx/main.py` runs `demo=False`; validation was done on the live account.
3. Exercised live: entry → SL (`stopLossPrice` algo) / maker-LIMIT TP placement → ladder
   trail → close, plus `set_leverage`, `tdMode=isolated`, and net position mode.
4. **Still `# VALIDATE`** (logging-only, not safety-critical): realized-P&L / fee / funding
   readout. Now pulled from `/account/positions-history` (`pnl` / `fee` / `fundingFee`),
   replacing an earlier bills-ledger mapping that used the wrong bill-type codes (OKX type
   8 is *funding*, not P&L). Confirm the fields populate on a real closed XPERP position.

## Cutover checklist (deliberate, brief pause — do when ready)

The Binance bot keeps running on `src/` until you do this. To move it onto the new
`binance/` app:

1. Stop cron (comment the 3 `tech_analysis` lines).
2. Copy live state into the app dir:
   `cp position_state.json risk_state.json alert_state.json closed_trades.jsonl binance/state/`
3. Repoint cron at the new entrypoints (own locks/logs), e.g.:
   ```cron
   2,17,32,47 * * * * flock -n /tmp/ta_binance_scan.lock bash -c "cd /home/kub/tech_analysis && timeout 840 .venv/bin/python -m binance.main --trade --trade-risk-pct 2.0 --leverage 5 >> binance/scanner.log 2>&1"
   */5 * * * * flock -n /tmp/ta_binance_mgmt.lock bash -c "cd /home/kub/tech_analysis && timeout 240 .venv/bin/python -m binance.main --manage-positions >> binance/scanner.log 2>&1"
   # ...and the OKX equivalents once validated:
   # 2,17,32,47 * * * * flock -n /tmp/ta_okx_scan.lock ... python -m okx.main --trade ...
   # */5 * * * * flock -n /tmp/ta_okx_mgmt.lock ... python -m okx.main --manage-positions ...
   ```
4. Verify the first managed run advances the 5 open positions (check `binance/scanner.log`).
5. Still pending at cutover: update `scripts/` (position_audit, realized_pnl) + `healthcheck.sh`
   to the per-venue state paths and the adapter; then remove the legacy `src/`.

When Binance becomes unavailable: just comment out the Binance cron lines — OKX keeps running.
