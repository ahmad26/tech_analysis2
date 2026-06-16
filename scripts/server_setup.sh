#!/usr/bin/env bash
# Run this ONCE manually on the server after SSHing in.
# It installs system deps and creates the project directory.
# Usage: bash server_setup.sh

set -euo pipefail

REMOTE_DIR="${HOME}/tech_analysis"

echo "--- Installing system dependencies (requires sudo once) ---"
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv

echo "--- Creating project directory ---"
mkdir -p "${REMOTE_DIR}"

echo ""
echo "Done. Next, on the server:"
echo "  cd ${REMOTE_DIR} && python3 -m venv .venv && .venv/bin/pip install -e ."
echo "  cp .env.example .env   # then fill in TELEGRAM_* (and BINANCE_* for trading)"
echo "  See CLAUDE.md 'Deployment' for the cron entries to install with: crontab -e"
