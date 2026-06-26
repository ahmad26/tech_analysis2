#!/usr/bin/env bash
# Break-even-floor expectancy sweep — run on the SERVER (needs Binance network).
#
# Compares the rolling exit model with vs without the +1R break-even floor
# (memory/feature_breakeven_floor.md), on the long window for 1d and 4h, with HTF
# and the live R/R>=1.5 gate so it matches what the live trader actually does.
#
# The floor is LIVE-enabled (position_manager.BREAKEVEN_FLOOR_R = 1.0); this checks
# whether arming break-even at +1R helps or scratches winners — watch 1d especially,
# where the sparse single-rung ladders it targets are concentrated. If 1d goes
# net-negative vs baseline, lift the threshold or set it default-off.
#
# Usage:  bash scripts/breakeven_floor_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TFS="1d 4h"
CANDLES=5000
COMMON="--backtest --timeframes ${TFS} --candles ${CANDLES} --htf --gate"
PY=".venv/bin/python"

echo "============================================================"
echo "BASELINE  (no break-even floor — legacy arm-on-rung-touch)"
echo "  ${PY} -m src.main ${COMMON}"
echo "============================================================"
$PY -m src.main $COMMON

echo
echo "============================================================"
echo "FLOOR     (--breakeven-floor 1.0 — matches live)"
echo "  ${PY} -m src.main ${COMMON} --breakeven-floor 1.0"
echo "============================================================"
$PY -m src.main $COMMON --breakeven-floor 1.0

echo
echo "Compare R/trade per timeframe between the two blocks above."
echo "Decision: if 1d R/trade drops with the floor, raise BREAKEVEN_FLOOR_R"
echo "or set it to 0 (default-off) in src/position_manager.py."
