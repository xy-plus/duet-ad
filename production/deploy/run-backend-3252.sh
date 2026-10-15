#!/usr/bin/env bash
set -euo pipefail

# Dev 3252 launcher (this workspace): isolated tts20-ark development instance.
# Same layered env sources as production 3212, then dev-3252.env LAST so its
# overrides win (ARK_AUDIO_* / VOCAL_SEPARATION_* / DATA_DIR=data-dev).
# dev-3252.env lives outside the repo (contains the ARK key); never commit it.
# No PUBLIC_API_* / PUBLIC_BASE_URL: dev has no caddy front, public API off.
set -a
source /home/xy/.config/duet-ad1/service.env
source /home/xy/.config/claude/deepseek.env
source /home/xy/duet-ad1-h3-max-r2v/.env
source /home/xy/.config/duet-ad1/dev-3252.env
set +a

# No PUBLIC_API_*: dev has no caddy front, public API off. PUBLIC_BASE_URL is
# inherited from .env (config.py requires a public HTTPS origin, so a local
# override is rejected); harmless while no projects are created on dev.
export ARTIFACT_RETENTION=lean
export OPENCV_FOR_THREADS_NUM=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

exec /home/xy/duet-ad1/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3252
