#!/usr/bin/env bash
set -euo pipefail

# Seedance 2.5 实验实例 3262 launcher（experiment/seedance25-r2v worktree）。
# Layered env sources, later wins: shared credentials first, then this
# worktree's .env（FAL_KEY / FAL_PROXY / OBJECT_MASK_* / INITIAL_ADMIN_*）。
# DeepSeek 凭证文件用 `export KEY=value`，systemd EnvironmentFile 不解析，
# 在这里 source（不打印、不落盘任何密钥）。
set -a
source /home/xy/.config/duet-ad1/service.env
source /home/xy/.config/claude/deepseek.env
source /home/xy/duet-ad1-h3-max-r2v/.worktree/seedance25-r2v/.env
set +a

# Seedance 推理 key 与 BytePlus 白名单 AK/SK 只从侦察取回文件逐键提取导出
# （不整体 source，避免无关 key 泄漏进进程环境；值不打印、不写日志）。
_seedance_env=/home/xy/duet-ad1-h3-max-r2v/data-verify/seedance25-recon-20260929/duet.env
for _key in SEEDANCE_INTL_API_KEY BYTEPLUS_ACCESS_KEY_ID BYTEPLUS_SECRET_ACCESS_KEY; do
  _value="$(grep -E "^${_key}=" "$_seedance_env" | head -n 1 | cut -d= -f2-)"
  if [ -z "$_value" ]; then
    echo "missing ${_key} in ${_seedance_env}" >&2
    exit 1
  fi
  export "${_key}=${_value}"
done
unset _key _value _seedance_env

# 本实例独立 DATA_DIR（全新，无兼容包袱）；代码与数据都在 worktree 内。
export DATA_DIR=/home/xy/duet-ad1-h3-max-r2v/.worktree/seedance25-r2v/data
# Asset intake URL 必须回指本实例：3261 是它的专用公网端口。
export PUBLIC_BASE_URL=https://8.166.140.227:3261
# BytePlus 国际站（推理 + 白名单控制面）出网走本地代理。
export SEEDANCE_PROXY=http://127.0.0.1:7897
# 实验期全量落盘；PUBLIC_API_ENABLED 不设（缺省 false，不重定价）。
export ARTIFACT_RETENTION=full
export OPENCV_FOR_THREADS_NUM=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

exec /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3262
