#!/usr/bin/env bash
set -euo pipefail

# Two-pass H3 experiment backend (3262) — worktree .worktree/two-pass-h3,
# branch experiment/two-pass-h3 (base 7abb4d6, the version that produced
# c57d9d6c on 2026-09-23). Layered env sources, later wins; never print secrets.
set -a
source /home/xy/.config/duet-ad1/service.env
source /home/xy/.config/claude/deepseek.env
source /home/xy/duet-ad1-h3-max-r2v/.worktree/two-pass-h3/.env
set +a

# Isolated data + public asset base through the dedicated 3261 caddy.
export DATA_DIR=/home/xy/duet-ad1-h3-max-r2v/.worktree/two-pass-h3/data
export PUBLIC_BASE_URL=https://8.166.140.227:3261
export ARTIFACT_RETENTION=full
export OPENCV_FOR_THREADS_NUM=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

exec /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3262
