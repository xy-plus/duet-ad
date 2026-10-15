#!/usr/bin/env bash
set -euo pipefail

# Experimental instance for the experiment/object-gray-mask-multiref branch.
# Mirrors the production 3212 launcher: layered env sources, later wins.
# Order matters: shared credentials first, main-workspace keys next, then the
# worktree .env (FAL_KEY / MULTIREF_MODE) overrides.
set -a
source /home/xy/.config/duet-ad1/service.env
source /home/xy/.config/claude/deepseek.env
source /home/xy/duet-ad1-h3-max-r2v/.env
source /home/xy/duet-ad1-h3-max-r2v-object-mask/.env
set +a

# Experiment data stays inside the object-mask worktree; never touch the
# release/production data directories.
export DATA_DIR=/home/xy/duet-ad1-h3-max-r2v-object-mask/data
# 落盘保留级别（app/artifact_retention.py）：本实例仍是测试期，显式 full
# 保持全量落盘；生产上线时启动脚本应显式改为 lean（成功项目完成后幂等
# 清理节点媒体副本与抽帧，指纹缓存/回执/state/成片保留）。
export ARTIFACT_RETENTION=full
# Public v1 API enabled on this test port for external-API acceptance runs
# (same clients file as production; paid runs still require explicit user approval).
export PUBLIC_API_ENABLED=1
export PUBLIC_API_CLIENTS_FILE=/home/xy/.config/duet-ad1-h3-r2v/public-api-clients.json
# Asset URLs must resolve back to THIS instance: 3241 is a dedicated public
# port fronted by the experiment caddy (same cert as production 3211).
export PUBLIC_BASE_URL=https://8.166.140.227:3241
export OPENCV_FOR_THREADS_NUM=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

exec /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3242
