# tech_analysis — Crypto Candlestick Pattern Scanner

## What this project does

Scans 10 Binance crypto pairs across 4 timeframes for candlestick reversal/continuation patterns. For each detected pattern it computes TP/SL levels using Fibonacci retracements, moving averages, ATR-based volatility, and higher-timeframe key levels. Only signals with R/R ≥ 1.5 are alerted or traded. Sends Telegram alerts and opens Binance futures positions automatically. Deduplication TTL varies by timeframe.

## How to run

```bash
# Quick scan — prints results to console, no Telegram needed
python -m src.main --dry-run

# Live mode — scans and sends Telegram alerts (requires .env)
python -m src.main

# Backtest on historical data (P&L is NET of fees by default: 0.09% round-trip taker)
python -m src.main --backtest --timeframes 4h --risk 10
python -m src.main --backtest --timeframes 4h --fee 0       # gross / fee-free view

# Backtest variants (added 2026-06-12):
#   --exit-model live  = simulate the ACTUAL trader mechanics (full exit at TP1,
#                        ATR chandelier + staged R-locks, R/R>=1.5 gate);
#                        default 'rolling' = legacy cascading multi-TP ladder model
#   --htf              = include point-in-time higher-TF levels (matches live scanner)
#   --gate             = apply the live R/R>=1.5 gate to the rolling model too
python -m src.main --backtest --timeframes 4h --exit-model live --htf

# Backtest with compound wallet simulation
python -m src.main --backtest --risk-pct 2 --wallet 1000

# Live trading on Binance futures mainnet (requires API keys in .env)
python -m src.main --trade --trade-risk-pct 2.0 --leverage 5
```

## Project structure

```
src/
  main.py             — entry point, scan loop, HTF data pre-fetch
  pattern_detector.py — candle pattern logic, ATR, context, HTF levels
  trading_rules.py    — SL/TP/RR computation
  models.py           — DetectedPattern, TradingSignal, format_message()
  config.py           — loads config.yaml + .env
  data_fetcher.py     — ccxt Binance OHLCV fetch
  alerter.py          — Telegram send
  alert_tracker.py    — 48h deduplication via alert_state.json
  backtester.py       — rolling historical backtest
  trader.py           — live order execution (Binance futures testnet)
  position_tracker.py — tracks open positions across runs
config.yaml           — symbols, timeframes, active patterns
.env                  — secrets (not in git)
.env.example          — template
scripts/server_setup.sh — one-time server setup (run manually on server)
scripts/healthcheck.sh  — 6-hourly LLM healthcheck (cron) → Telegram verdict
scripts/position_audit.py — read-only cross-check: position_state.json vs Binance
```

## Active patterns (config.yaml)

Patterns are configured **per timeframe**. The global `patterns:` list is the default
for any TF not overridden; `patterns_by_timeframe:` gives an exact list for a TF (it
replaces, not extends, the global list for that TF). `AppConfig.patterns_for(tf)` resolves
the effective list; an explicit backtest `--patterns` override still applies to all TFs.
Pattern edge is timeframe-specific, so the same pattern can be on for one TF and off for
another.

```yaml
patterns:                 # default for 15m, 1h (and any TF not overridden)
  - bullish_engulfing
  - bearish_engulfing
  - doji
  - evening_star
  - shooting_star
patterns_by_timeframe:
  4h: [ …global 5…, morning_star ]
  1d: [ …global 5…, hammer, morning_star ]
```

**Per-TF enablement (set 2026-06-18 from a long-window OOS sweep — 5000 candles,
1d≈8.3yr / 4h≈2.3yr; see `memory/finding_oos_regime_check.md`):**
- `hammer` — **enabled on 1d only** (+0.32/+0.15R both halves over 8yr); TOXIC on 4h
  (+0.06/−0.38R) so it stays off there. The old blanket disable judged it on a short 4h
  window and wrongly applied it everywhere.
- `morning_star` — **enabled on 1d and 4h** (positive both halves on both: 1d +0.18/+0.21,
  4h +0.04/+0.28). Was a false-negative ("too rare").
- `inverted_hammer` — **still disabled everywhere** (flips sign / marginal, no robust edge).

Earlier note (now superseded): these three were disabled "after backtesting on 4h data"
(hammer −0.20R, inverted_hammer −0.01R, morning_star +0.02R) — that read was a short-window
4h artifact applied globally. The long-window per-TF sweep corrected it.

## Symbols and timeframes

10 pairs: BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, DOT, LINK (all /USDT)
4 timeframes: 15m, 1h, 4h, 1d
`candles_to_fetch: 210` (needs 200 for MA200 + buffer)

## TP/SL logic (trading_rules.py)

### Stop Loss
All of these compete; closest level on the wrong side of entry wins:
- Own-TF Fibonacci retracements (50-period and 200-period windows, 7 levels each)
- Own-TF MA50 and MA200
- Higher-TF levels (labelled e.g. `"4h MA50"`, `"1d Fib 61.8%"`)

**ATR floor**: if risk < ATR(14), SL is widened to the nearest level ≥ 1× ATR away. If no level qualifies, bare ATR is used.

**SL buffer**: SL is pushed 0.25× ATR beyond the support/resistance level so a wick touching the level doesn't trigger the stop immediately.
- Constants: `MIN_RR = 1.5`, `MIN_RISK_PCT = 0.001`, `SL_BUFFER_ATR = 0.25`

### Take Profit
- TP candidates: all Fib levels + own-TF MAs + HTF levels on the correct side of entry
- Closest level meeting `MIN_RR = 1.5` is chosen as TP

**Order type**: the live TP is a **maker** reduceOnly `LIMIT` order (resting at the TP price, ~0.018%/side) rather than `TAKE_PROFIT_MARKET` (taker, ~0.045%). SL stays `STOP_MARKET` (must be guaranteed to fill). Controlled by `MAKER_TP` in `trader.py` (set False to revert); single source of truth `trader.place_take_profit()`, and `trader.is_protective_order()` identifies SL/TP (incl. the limit TP) for cancel/replace. NOT yet testnet-validated; the `--cancel-orders` algo-order tool was not updated. See `memory/feature_maker_tp.md`.

**TP1/TP2 obstacle logic**: if a *structural* level sits *between* entry and the Fib TP, the closest obstacle becomes TP1 and the Fib target is demoted to TP2. HTF levels are weighted first (stronger barriers). Only structural levels count as obstacles: own-TF **MA50/MA200** and HTF levels. Fast MAs (MA7/MA25/MA99) are deliberately **excluded** — they hug price, so one almost always lands just above/below entry, hijacking TP1 and collapsing R/R to ~0 (which silently dropped otherwise-valid higher-TF signals). **R/R is still gated on TP1** (signal filter), but since 2026-06-12 TP1 is no longer the live exit. Do not re-add fast MAs here. See `memory/fix_fast_ma_obstacle.md`.

**Ladder exit engine (live since 2026-06-12)**: the trader no longer exits full-size at TP1. All TP candidate levels (`all_tp_candidates`, closest-first) are stored as `tp_ladder` in `position_state.json`; the resting maker LIMIT sits at the **final** ladder level. `position_manager._trail_ladder()` ratchets the stop as price touches each level: first touch → SL to entry, each later touch → SL to the last touched level (no buffer; touches recomputed from price history since open, so restarts can't lose progress). This mirrors the backtester's rolling model, which beats the old TP1-exit engine by ~0.11R/trade on identical 4h signals (see `memory/finding_exit_engine_divergence.md`). Positions opened before the change (no `tp_ladder`) keep the legacy ATR-chandelier + staged-R management until a same-side signal migrates them.

**Break-even floor (live 2026-06-23)**: the ladder only ratchets on rung *touches*, so a **sparse ladder whose only rung is the final TP** (a 1d signal with no MA/HTF/Fib level between entry and the Fib target — e.g. LINK/BTC shorts) never moves its stop until the trade is essentially over, leaving a large unrealised gain fully exposed to the initial SL. `_trail_ladder()` now also arms a break-even stop at **entry + 0.1R** once profit (from the favourable extreme since open) reaches `BREAKEVEN_FLOOR_R` × initial risk (**+1R**, `position_manager.py`), *independent* of rung touches. The ladder and the floor coexist — the tighter stop wins, ratchet-only — so dense ladders (which usually arm break-even below +1R on their first rung) are unaffected; the floor only rescues sparse ones. Backtestable with `--breakeven-floor 1.0` (rolling model; default 0 = off, the legacy "arm on rung touch only"). Enabled live as a risk-protection rule; expectancy not yet swept on the long window (break-even arming can scratch winners — see `memory/finding_exit_capture_problem.md`). See `memory/feature_breakeven_floor.md`.

**Round-number TP-pull (live 2026-06-26, 1d only)**: Fib extremes cluster just past psychological round numbers, which act as S/R — a TP resting just *beyond* a round level (e.g. a LINK 1d short target at 6.996, under 7.0) often never tags it; price stalls at the number and reverses. `trading_rules._psych_adjust_tp()` pulls such a TP to just *before* the round level (the near side, where fills cluster), on an auto-scaling grid (`10^floor(log10 price)` + half-step: LINK→1.0/0.5, BTC→10k/5k). It feeds the R/R gate, chosen TP, ladder rungs and the resting maker LIMIT. **Gated per-TF via `PSYCH_ROUND_TFS` (default `{"1d"}`)**: a 5000-candle OOS split (1d≈8.3yr) showed it robustly net-positive on 1d in BOTH halves (+0.004/+0.012R, ~+19R aggregate) but a tiny drag on 4h, so it's on for 1d only — same per-TF shape as hammer/morning_star. The CLI `--psych-round` forces it on for all scanned TFs (sweep mode). The mirror SL-push was tested and **dropped** (net-negative both TFs). See `memory/finding_psych_round_tp.md`.

### Higher-timeframe (HTF) level awareness

```python
HTF_MAP = {
    "15m": ["1h", "4h", "1d"],
    "1h":  ["4h", "1d"],
    "4h":  ["1d"],
    "1d":  [],
}
```

`extract_htf_levels()` pulls **only MA50, MA200**, and all 7 Fib50 levels from each higher TF (fast MAs are intentionally not emitted — they previously leaked in as e.g. `"4h MA7"` and acted as noise obstacles). These are used as SL/TP candidates and TP obstacles. Pre-fetched once per symbol per scan (`HTF_CANDLES = 250`).

Until 2026-06-12 a propagation bug meant only doji signals actually received HTF levels; all patterns get them now. Backtested (4h, 16mo, live exit model): HTF improves expectancy and halves trade count.

## Signal format

Each alert shows:
- Pattern name, coin, timeframe, close price, candle timestamp (UTC + CET)
- MA50 / MA200 with trend direction and spread %
- Price position relative to MAs
- Fib50 and Fib200 brackets
- ATR(14) value and 14-candle range
- BUY/SELL action, CONTINUATION/REVERSAL setup
- SL with source label (e.g. `Fib200 0% +buffer`)
- TP1 / TP2 when an MA/HTF obstacle is present, or single TP otherwise
- R/R ratio

## Deployment (Ubuntu home server)

**The server (`/home/kub/tech_analysis`) is the sole source of truth.** The Mac is
retired and the old `deploy.sh` (Mac → server rsync) has been removed — make all
edits, venv rebuilds, and dependency installs directly on the server. This is NOT a
git repo, so there is no version-control safety net.

One-time system setup (already done on this server):
```bash
bash scripts/server_setup.sh          # apt deps + project dir (needs sudo once)
python3 -m venv .venv && .venv/bin/pip install -e .   # venv + deps
cp .env.example .env                  # then fill in TELEGRAM_* (and BINANCE_* for trading)
```

Cron jobs (live; installed in the user crontab — edit with `crontab -e`):
```cron
# Scan + trade, 4×/hour at candle-close +2 min; flock skips overlap, timeout hard-kills a hang
2,17,32,47 * * * * flock -n /tmp/tech_analysis_scan.lock bash -c "cd /home/kub/tech_analysis && timeout 840 .venv/bin/python -m src.main --trade --trade-risk-pct 2.0 --leverage 5 >> /home/kub/tech_analysis/scanner.log 2>&1"
# Position manager every 5 min
*/5 * * * * flock -n /tmp/tech_analysis_mgmt.lock bash -c "cd /home/kub/tech_analysis && timeout 240 .venv/bin/python -m src.main --manage-positions >> /home/kub/tech_analysis/scanner.log 2>&1"
# LLM healthcheck every 6h, +10 min so it sees a fresh scan
10 */6 * * * flock -n /tmp/tech_analysis_health.lock bash -c "timeout 300 bash /home/kub/tech_analysis/scripts/healthcheck.sh >> /home/kub/tech_analysis/healthcheck.log 2>&1"
```

Check logs: `tail -f ~/tech_analysis/scanner.log`

## Monitoring (LLM healthcheck)

`scripts/healthcheck.sh` runs every 6h via cron (`10 */6 * * *`):
1. Collects evidence: last-6h log window, scan cadence vs expected 24, error lines,
   `Scan summary:` duration trend (one-line summary logged by `main.py` per scan),
   and `scripts/position_audit.py` (tracked state vs Binance positions/orders —
   flags CRITICAL if a position has no resting stop-loss).
2. Deterministic red-flag gate (no scan in 30 min, CRITICAL audit, "unprotected").
3. `claude -p` (haiku, headless) produces a `STATUS: OK|WARN|PROBLEM` verdict;
   falls back to a deterministic verdict if the `claude` CLI is missing.
4. Sends the verdict to Telegram; full report in `healthcheck_last_report.txt`.

Test without sending: `bash scripts/healthcheck.sh --no-send`

## Pre-existing test failures (do not investigate)

These 4 tests fail before any changes — they existed in the original codebase and are not regressions:
- `test_alerter`
- `test_data_fetcher`
- `test_integration`
- `test_pattern_detector` (doji)

All other tests (22) pass. Only investigate failures beyond these 4.

## Environment variables (.env)

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
# Optional — only needed for live trading
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
```

No API key is needed for read-only scanning (Binance public OHLCV via ccxt).
