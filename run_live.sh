#!/usr/bin/env bash
# One-command live run. Your API key is read from .env (already in place).
set -e
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 -m researchpilot.cli "${1:-What is causing 2026 AI agent reliability concerns in enterprise adoption?}" --json-out report.json
