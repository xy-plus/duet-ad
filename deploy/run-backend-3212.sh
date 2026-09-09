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

# Production data stays in the release tree; only the code runs from this
# pinned online worktree (pre-adaptation-prompt version, ms/grayblock-explainer).
export DATA_DIR=/home/xy/duet-ad1-h3-max-r2v-release/data
export PUBLIC_API_ENABLED=1
export PUBLIC_API_CLIENTS_FILE=/home/xy/.config/duet-ad1-h3-r2v/public-api-clients.json
# Asset URLs must resolve back to THIS instance: 3211 is a dedicated public
# port, so no path prefix is needed (unlike the shared 443 edge on :3301).
export PUBLIC_BASE_URL=https://8.166.140.227:3211

exec /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3212
