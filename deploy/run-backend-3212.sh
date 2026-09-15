#!/usr/bin/env bash
set -euo pipefail

# Production 3212 launcher for the experiment/object-gray-mask-multiref branch.
# Layered env sources, later wins: shared credentials first, main-workspace
# keys next, then this worktree's .env (FAL_KEY / FAL_PROXY / MULTIREF_MODE).
# The DeepSeek credential file uses `export KEY=value`, which systemd
# EnvironmentFile does not parse; source it here without printing or
# persisting any secret.
set -a
source /home/xy/.config/duet-ad1/service.env
source /home/xy/.config/claude/deepseek.env
source /home/xy/duet-ad1-h3-max-r2v/.env
source /home/xy/duet-ad1-h3-max-r2v-object-mask/.env
set +a

# Production data stays in the release tree; only the code runs from this
# worktree.
export DATA_DIR=/home/xy/duet-ad1-h3-max-r2v-release/data
export PUBLIC_API_ENABLED=1
export PUBLIC_API_CLIENTS_FILE=/home/xy/.config/duet-ad1-h3-r2v/public-api-clients.json
# Asset URLs must resolve back to THIS instance: 3211 is a dedicated public
# port, so no path prefix is needed (unlike the shared 443 edge on :3301).
export PUBLIC_BASE_URL=https://8.166.140.227:3211
export ARTIFACT_RETENTION=lean
export OPENCV_FOR_THREADS_NUM=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

exec /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3212
