#!/usr/bin/env bash
set -euo pipefail

# SAM 统一人脸+物品遮盖改造的验证实例（空端口，独立于 3242 实验实例与生产）。
# env 层叠顺序与 3242 一致，后者覆盖前者。
set -a
source /home/xy/.config/duet-ad1/service.env
source /home/xy/.config/claude/deepseek.env
source /home/xy/duet-ad1-h3-max-r2v/.env
source /home/xy/duet-ad1-h3-max-r2v-object-mask/.env
set +a

# 独立数据目录，不碰生产/3242 的数据；指纹缓存因此独立，fal 调用为真实新调用。
export DATA_DIR=/home/xy/duet-ad1-h3-max-r2v-object-mask/data-mask-opacity-90
export ARTIFACT_RETENTION=full
# 仅内部 API 验证，不开对外 v1 API。
# 灰块不透明度 A/B：本实例 0.90（留 10% 原画面）。
export OBJECT_MASK_BLOCK_OPACITY=0.90
export PUBLIC_BASE_URL=https://8.166.140.227:3261
export OPENCV_FOR_THREADS_NUM=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_THREADS=2
export NUMEXPR_NUM_THREADS=2

exec /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3262
