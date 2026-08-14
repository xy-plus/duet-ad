#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
UVICORN="$ROOT/.venv/bin/uvicorn"
[ -x "$UVICORN" ] || UVICORN="$(command -v uvicorn)"
exec "$UVICORN" app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-3211}"
