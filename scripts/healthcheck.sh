#!/usr/bin/env bash
# LLM-powered healthcheck — run every 6h via cron (see CLAUDE.md 'Deployment').
#
# 1. Collects evidence: log window, scan cadence, error lines, scan-duration
#    trend, state files, and a position audit against Binance.
# 2. Cheap deterministic gate first — hard failures are flagged without an LLM.
# 3. Asks Claude (haiku, headless `claude -p`) for a verdict on the report.
# 4. Sends verdict to Telegram. Full report kept in healthcheck_last_report.txt.
#
# Requirements on the server: `claude` CLI (optional — falls back to a raw
# deterministic summary if missing), curl, and .env with Telegram credentials.
#
# Usage: scripts/healthcheck.sh [--no-send]   (--no-send: print verdict only)

set -uo pipefail  # no -e: partial evidence is still worth reporting

# cron runs with a minimal PATH — claude lives in ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

LOG=scanner.log
WINDOW_HOURS=6
EXPECTED_SCANS=$((WINDOW_HOURS * 4))   # scanner cron fires every 15 min
REPORT="$DIR/healthcheck_last_report.txt"

# Telegram creds (server logs/cron run in UTC — see memory/project_server_config)
set -a; [ -f .env ] && source .env; set +a

# ---------------------------------------------------------------- collect ----
CUTOFF="$(date -u -d "${WINDOW_HOURS} hours ago" '+%Y-%m-%d %H:%M')"
WINDOW="$(mktemp)"
trap 'rm -f "$WINDOW"' EXIT

# Everything from the first in-window timestamped line onward (keeps tracebacks)
awk -v c="$CUTOFF" 'on {print; next} /^20[0-9][0-9]-/ && substr($0,1,16) >= c {on=1; print}' "$LOG" > "$WINDOW"

SCANS=$(grep -c "Scan complete\." "$WINDOW" || true)
SUMMARIES=$(grep "Scan summary:" "$WINDOW" | tail -25)
ERRORS=$(grep -iE "\[ERROR\]|\[CRITICAL\]|exception|traceback|unprotected" "$WINDOW" | tail -40)
ERROR_COUNT=$(grep -icE "\[ERROR\]|\[CRITICAL\]" "$WINDOW" || true)
WARN_COUNT=$(grep -c "\[WARNING\]" "$WINDOW" || true)
LAST_SCAN_TS=$(grep "Scan complete\." "$LOG" | tail -1 | cut -c1-19)

# Minutes since last completed scan (UTC)
LAST_SCAN_AGE_MIN="unknown"
if [ -n "$LAST_SCAN_TS" ]; then
  LAST_EPOCH=$(date -u -d "$LAST_SCAN_TS" +%s 2>/dev/null || echo 0)
  [ "$LAST_EPOCH" -gt 0 ] && LAST_SCAN_AGE_MIN=$(( ($(date -u +%s) - LAST_EPOCH) / 60 ))
fi

# Position audit vs Binance (read-only)
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
AUDIT=$("$PY" scripts/position_audit.py 2>&1 || true)

ALERT_ENTRIES=$("$PY" -c "import json;print(len(json.load(open('alert_state.json'))))" 2>/dev/null || echo "?")
DISK=$(df -h "$DIR" | tail -1 | awk '{print $5 " used, " $4 " free"}')

# ----------------------------------------------------- deterministic gate ----
RED_FLAGS=""
[ "$LAST_SCAN_AGE_MIN" != "unknown" ] && [ "$LAST_SCAN_AGE_MIN" -gt 30 ] && \
  RED_FLAGS+="- No completed scan in ${LAST_SCAN_AGE_MIN} min (cron fires every 15 min)"$'\n'
[ "$SCANS" -lt $((EXPECTED_SCANS / 2)) ] && \
  RED_FLAGS+="- Only ${SCANS}/${EXPECTED_SCANS} expected scans completed in the last ${WINDOW_HOURS}h"$'\n'
echo "$AUDIT" | grep -q "^CRITICAL" && \
  RED_FLAGS+="- Position audit reports CRITICAL (see audit section)"$'\n'
echo "$ERRORS" | grep -qi "unprotected" && \
  RED_FLAGS+="- Log mentions an UNPROTECTED position"$'\n'

# ----------------------------------------------------------------- report ----
{
  echo "HEALTHCHECK REPORT — $(date -u '+%Y-%m-%d %H:%M UTC') — window: last ${WINDOW_HOURS}h"
  echo
  echo "== Cadence =="
  echo "Completed scans: ${SCANS} (expected ~${EXPECTED_SCANS})"
  echo "Last completed scan: ${LAST_SCAN_TS:-none found} (${LAST_SCAN_AGE_MIN} min ago)"
  echo
  echo "== Hard red flags (deterministic) =="
  echo "${RED_FLAGS:-none}"
  echo
  echo "== Scan summaries (duration trend, newest last) =="
  echo "${SUMMARIES:-none — scanner may predate the summary log line}"
  echo
  echo "== Errors / exceptions in window (${ERROR_COUNT} ERROR, ${WARN_COUNT} WARNING lines total) =="
  echo "${ERRORS:-none}"
  echo
  echo "== Position audit (local state vs Binance) =="
  echo "$AUDIT"
  echo
  echo "== State =="
  echo "alert_state.json entries: ${ALERT_ENTRIES}"
  echo "Disk: ${DISK}"
  echo "Log size: $(du -h "$LOG" | cut -f1)"
} > "$REPORT"

# -------------------------------------------------------------------- LLM ----
PROMPT='You are the health monitor for a crypto pattern scanner + live futures
trader that runs via cron every 15 min on an Ubuntu server (all times UTC).
Analyze the report below. Judge:
1. Correctness — running on schedule? errors? positions protected and in sync?
2. Efficiency — scan duration creeping up? repeated fetch retries/timeouts?
3. Anything anomalous worth a human look.
Transient single fetch failures with successful retries are normal noise.
Reply in plain text (no markdown), max 12 lines:
line 1 exactly "STATUS: OK", "STATUS: WARN" or "STATUS: PROBLEM",
then a terse summary of what you found and, if not OK, what to do about it.'

VERDICT=""
if command -v claude >/dev/null 2>&1; then
  # Cap report at 8 KB before feeding to the LLM — large inputs bloat RSS.
  # timeout 120 kills the claude process if it hangs before the outer timeout 300 fires.
  VERDICT=$(head -c 8192 "$REPORT" | timeout 120 claude -p "$PROMPT" --model haiku 2>/dev/null | head -20)
fi
if [ -z "$VERDICT" ]; then
  # Fallback: deterministic verdict without an LLM
  if [ -n "$RED_FLAGS" ]; then
    VERDICT="STATUS: PROBLEM (LLM unavailable — deterministic check)"$'\n'"$RED_FLAGS"
  elif [ "$ERROR_COUNT" -gt 20 ]; then
    VERDICT="STATUS: WARN (LLM unavailable) — ${ERROR_COUNT} error lines in last ${WINDOW_HOURS}h, scans on schedule (${SCANS}/${EXPECTED_SCANS})."
  else
    VERDICT="STATUS: OK (LLM unavailable) — ${SCANS}/${EXPECTED_SCANS} scans, ${ERROR_COUNT} error lines, audit: $(echo "$AUDIT" | tail -1)"
  fi
fi

echo "$VERDICT"

# --------------------------------------------------------------- telegram ----
if [ "${1:-}" = "--no-send" ]; then
  echo "(--no-send: skipping Telegram; full report in $REPORT)"
elif [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
  ICON="🩺"
  case "$VERDICT" in
    "STATUS: PROBLEM"*) ICON="🚨" ;;
    "STATUS: WARN"*)    ICON="⚠️" ;;
  esac
  MSG="${ICON} Healthcheck $(date -u '+%H:%M UTC')
${VERDICT}"
  curl -s -m 30 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" \
    --data-urlencode text="${MSG:0:3900}" > /dev/null \
    || echo "WARNING: failed to send Telegram message"
else
  echo "WARNING: Telegram credentials not set — verdict printed only"
fi
