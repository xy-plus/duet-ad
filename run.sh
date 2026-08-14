#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-3211}"
