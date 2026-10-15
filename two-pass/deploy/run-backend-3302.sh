#!/usr/bin/env bash
set -euo pipefail

# The source DeepSeek credential file is a shell-style file whose assignments
# use `export KEY=value`; systemd EnvironmentFile intentionally does not parse
# that syntax.  Load it here without printing or persisting any secret.
set -a
source /home/xy/.config/duet-ad1/service.env
source /home/xy/.config/claude/deepseek.env
source /home/xy/duet-ad1-h3-max-r2v/.env
set +a

exec /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3302
