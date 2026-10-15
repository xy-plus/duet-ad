#!/bin/bash
# 注入外层工作区 .env 的环境变量（含 ARK_API_KEY）后执行 hypit CLI；绝不打印 .env 内容。
set -a
source /home/xy/duet-ad1-h3-max-r2v/.env
set +a
exec node /home/xy/duet-ad1-h3-max-r2v/hypit/bin/hypit.mjs "$@"
