# duet-ad1 · H3 与双前端生产部署

目标拓扑固定为同一个 `duet-ad1-caddy.service` 提供两个 HTTPS 入口：

- `:3211 → 127.0.0.1:3212 uvicorn`，保留原有全路径反代语义；
- `:3213 /api/* → 127.0.0.1:3212 uvicorn`，不剥离 `/api`；其余路径从 `/home/xy/duet-ad1/web-next/dist` 提供 SPA。

不新增前端 service、CORS 或第二个 uvicorn。Caddy 配置只启用 HTTP/1.1、HTTP/2；H3 是模型名，不是 HTTP/3。

本 runbook 的纯文档部分 TDD=N/A；部署前仍必须运行下面的自动化测试。上线采用 fix-forward，不恢复已删除的 Seedance 提交路径。

## 1. Preflight

在仓库根目录执行：

```bash
cd /home/xy/duet-ad1
test -x .venv/bin/python
command -v ffmpeg
command -v ffprobe
command -v codex
command -v bwrap
test -x /home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli
test -s /home/xy/.local/share/duet-asr/ggml-small.bin
printf '%s  %s\n' \
  1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b \
  /home/xy/.local/share/duet-asr/ggml-small.bin | sha256sum -c -
.venv/bin/python -m compileall -q app
bash -n run.sh .deploy/smoke-h3.sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path('.deploy/caddy/config.json').read_text())
server = config['apps']['http']['servers']['srv3211']
assert server['listen'] == [':3211']
assert server['protocols'] == ['h1', 'h2']
upstream = server['routes'][0]['handle'][0]['upstreams'][0]['dial']
assert upstream == '127.0.0.1:3212'

server = config['apps']['http']['servers']['srv3213']
assert server['listen'] == [':3213']
assert server['protocols'] == ['h1', 'h2']
api = server['routes'][0]
assert api['match'] == [{'path': ['/api/*']}]
assert api['handle'][0]['handler'] == 'reverse_proxy'
assert api['handle'][0]['upstreams'][0]['dial'] == '127.0.0.1:3212'
PY
/usr/local/libexec/duet/caddy validate --config "$PWD/.deploy/caddy/config.json"
```

Codex 使用 bwrap 创建自己的 user/mount/network namespace。Ubuntu AppArmor
下不能再在外层 user service 叠加 `PrivateTmp` / `ProtectSystem` /
`ProtectControlGroups` / `ProtectKernelTunables`，否则
服务会进入 `unprivileged_userns` profile，内层沙箱在解析开始前就失败。
仓库 unit 保留 `NoNewPrivileges` / `LockPersonality` / `RestrictSUIDSGID`；
`RestrictAddressFamilies` 显式加入 bwrap 配置 loopback 所需的 `AF_NETLINK`。
CodexRunner 仍固定 `workspace-write` 且禁网。

在发布前用与 unit 相同的保留属性执行一次无付费沙箱探针：

```bash
systemd-run --user --wait --pipe --collect \
  --property=NoNewPrivileges=true \
  --property=LockPersonality=true \
  --property=RestrictSUIDSGID=true \
  --property='RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK' \
  codex sandbox -P :workspace -C "$PWD" -- /bin/true
```

该命令必须退出 0；它只启动本地 sandbox，不调用模型或视频 API。

确认磁盘可写且现有恢复状态不会被清理：

```bash
test -d data
test -w data
find data -maxdepth 4 -path '*/.h3/attempts/*/attempt.json' -type f -print | head
```

不要删除或改写既有 `prepared_input.json`、`.h3/`、`generated.mp4`；它们是恢复和防重复扣费的依据。

## 2. 测试

```bash
/home/xy/duet-ad1/.venv/bin/python -m pytest tests -q
git diff --check
systemd-analyze --user verify .deploy/systemd/duet-ad1.service
/usr/local/libexec/duet/caddy validate --config "$PWD/.deploy/caddy/config.json"
```

全量测试必须通过。此阶段不运行 `.deploy/smoke-h3.sh`；3213 的生产 smoke 只允许第 6 节列出的 GET/HEAD，不创建会话、不提交生成。

## 3. 唯一 EnvironmentFile 与原 user unit

生产只保留现有 `systemctl --user` 的 `duet-ad1.service`。本次把同名 unit 原地替换，不创建 root system service，也不创建第二个 user service。所有服务环境——包括非秘密的监听、PATH 和开关——统一放在：

```text
%h/.config/duet-ad1/service.env
```

先用不会把值写进 shell history 的编辑器创建/迁移环境文件：

```bash
install -d -m 0700 ~/.config/duet-ad1
test -e ~/.config/duet-ad1/service.env || install -m 0600 /dev/null ~/.config/duet-ad1/service.env
chmod 0600 ~/.config/duet-ad1/service.env
${EDITOR:?set EDITOR first} ~/.config/duet-ad1/service.env
chmod 0600 ~/.config/duet-ad1/service.env
```

首条命令只在文件不存在时创建它；重复执行不得截断或重建已有文件。在编辑器中把原 unit、旧 drop-in 和旧环境文件里的所有服务变量迁入这一份文件；不要在终端打印、复制到工单或提交到仓库。文件按实际启用能力填写以下键，秘密值在现场录入：

```text
HOST=127.0.0.1
PORT=3212
PATH=/home/xy/.local/bin:/home/xy/duet-ad1/.venv/bin:/usr/local/bin:/usr/bin:/bin
DATA_DIR=/home/xy/duet-ad1/data

ACCESS_TOKEN=
ENABLE_PIPELINE=1
MAX_UPLOAD_MB=500
CODEX_TIMEOUT_S=1800
CODEX_CONCURRENCY=10
AUTO_RETRY_COUNT=2
AUTO_RETRY_INTERVAL_S=15
MAX_QUEUED=100
VOCAL_FILTER=on
YAMNET_MODEL_PATH=/home/xy/duet-ad1/models/yamnet.tflite
ASR_CLI=/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli
ASR_MODEL=/home/xy/.local/share/duet-asr/ggml-small.bin
ASR_TIMEOUT_S=600
ASR_THREADS=4

ENABLE_H3_SUBMIT=1
AUTODL_ART_TOKEN=
H3_REQUEST_TIMEOUT_S=30
H3_POLL_TIMEOUT_S=1500
H3_DOWNLOAD_TIMEOUT_S=180
H3_POLL_INTERVAL_S=3

ENABLE_MEDIAKIT_ERASE=0
VOLC_MEDIAKIT_API_KEY=
MEDIAKIT_CONCURRENCY=4
MEDIAKIT_TIMEOUT_S=180

TIKTOK_PROXY=
DOWNLOAD_TIMEOUT_S=120
```

只有确实使用显式 Codex 认证目录时，才另加 `CODEX_HOME=/absolute/path`；否则整行省略，让 Codex 使用默认认证目录，禁止用空值覆盖。未启用的可选能力保留空凭据并关闭对应开关。`HOST=127.0.0.1`、`PORT=3212` 是 Caddy 拓扑的一部分，不得放宽为公网监听。

确认 `service.env` 已完整迁移后，用仓库 unit 覆盖同名原 unit：

```bash
install -d -m 0700 ~/.config/systemd/user
install -m 0644 .deploy/systemd/duet-ad1.service \
  ~/.config/systemd/user/duet-ad1.service
systemd-analyze --user verify ~/.config/systemd/user/duet-ad1.service
```

不要将上述 namespace 限制作为 drop-in 加回去；如果需要改变 unit 硬化策略，
必须先在同样的 user-service 属性下完成一次 bwrap 与 Codex 断网写入测试。

新 unit 只有一个 `EnvironmentFile=`，没有任何 inline `Environment=`。在 `daemon-reload` 前删除仅指向旧 `h3.env` 的 drop-in 和旧环境文件，确保本次重启已经只有一个环境源：

```bash
rm -f ~/.config/systemd/user/duet-ad1.service.d/h3.conf
rm -f ~/.config/duet-ad1/h3.env
```

删除前必须确认所有变量已经迁移；上述两个精确旧路径删除后不可由本 runbook 恢复。不要运行会展开值的 `systemctl --user show ... Environment`，也不要 `cat`、`env`、`set` 或 shell trace 读取 `service.env`。

## 4. 原地发布

确保代码和依赖已经位于 `/home/xy/duet-ad1`，再执行：

```bash
systemctl --user daemon-reload
systemctl --user enable duet-ad1.service
systemctl --user restart duet-ad1.service
systemctl --user is-active duet-ad1.service
journalctl --user -u duet-ad1.service --since '-2 minutes' --no-pager
```

服务必须是单进程；不要给 uvicorn 增加 `--workers`。重启时 schema v2 generation 默认只执行 GET-only resume；唯一补发供应商 POST 的例外是完整确认的 `h3_provider_failed` 仍有自动额度，或该失败后已落盘的 ready 自动 attempt。

## 5. 3213 静态前端与同一 Caddy unit 发布

先确认构建产物和候选配置；`validate` 只解析候选文件，不监听端口、不重启服务：

```bash
cd /home/xy/duet-ad1
test -s web-next/dist/index.html
test -d web-next/dist/assets
/usr/local/libexec/duet/caddy validate --config "$PWD/.deploy/caddy/config.json"
```

部署前生成本次 3213 变更的精确 3211-only 回滚配置，并验证它可加载：

```bash
install -d -m 0700 /home/xy/.local/state/duet-ad1
ROLLBACK_CONFIG=/home/xy/.local/state/duet-ad1/caddy-config.before-3213.json
git show 2d6567b:.deploy/caddy/config.json > "${ROLLBACK_CONFIG}.tmp"
/usr/local/libexec/duet/caddy validate --config "${ROLLBACK_CONFIG}.tmp"
install -m 0600 "${ROLLBACK_CONFIG}.tmp" "$ROLLBACK_CONFIG"
rm -f "${ROLLBACK_CONFIG}.tmp"
```

3213 当前的旧预览不是 service。只允许按完整命令行和工作目录同时确认后，精确停止这一个 Vite 进程；零个进程是合法状态，多个匹配则立即停止发布并调查：

```bash
OLD_VITE_COMMAND='node ./node_modules/.bin/vite preview --host 0.0.0.0 --port 3213'
mapfile -t OLD_VITE_PIDS < <(pgrep -fx -- "$OLD_VITE_COMMAND" || true)
test "${#OLD_VITE_PIDS[@]}" -le 1
if test "${#OLD_VITE_PIDS[@]}" -eq 1; then
  OLD_VITE_PID="${OLD_VITE_PIDS[0]}"
  test "$(readlink -f "/proc/$OLD_VITE_PID/cwd")" = \
    /home/xy/duet-ad1/.worktree/antd-x-prototype/web-next
  kill -TERM -- "$OLD_VITE_PID"
  for _ in {1..50}; do
    kill -0 "$OLD_VITE_PID" 2>/dev/null || break
    sleep 0.2
  done
  ! kill -0 "$OLD_VITE_PID" 2>/dev/null
fi
```

不创建新 service，不启动第二个 uvicorn；原地重启同一个 Caddy unit：

```bash
systemctl --user restart duet-ad1-caddy.service
systemctl --user is-active duet-ad1-caddy.service
journalctl --user -u duet-ad1-caddy.service --since '-2 minutes' --no-pager
```

## 6. 健康检查与只读生产 smoke

先查本机 uvicorn：

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:3212/api/health
```

应返回 `{"ok":true}`。再查 Caddy 公网入口（替换成实际域名/IP；保留 3211）：

```bash
PUBLIC_BASE_URL='https://<public-host>:3211'
curl --fail --silent --show-error \
  "$PUBLIC_BASE_URL/api/health"
```

健康接口只证明进程和反代可达，不验证 H3 凭据、余额或付费链路。

3213 生产 smoke 只允许 GET/HEAD；禁止 POST/PUT/PATCH/DELETE，禁止创建会话、提交生成或运行 `.deploy/smoke-h3.sh`。先验证本机 HTTPS、SPA 回退和缓存头，再把 `PUBLIC_3213` 替换为实际公网域名/IP 重复同样的只读请求：

```bash
curl --fail --silent --show-error --insecure \
  https://127.0.0.1:3213/api/health
curl --fail --silent --show-error --insecure --head \
  https://127.0.0.1:3213/ | rg -i '^cache-control: no-store'
curl --fail --silent --show-error --insecure --head \
  https://127.0.0.1:3213/a-client-route | rg -i '^cache-control: no-store'

ASSET_PATH="$(find web-next/dist/assets -maxdepth 1 -type f \
  -printf '/assets/%f\n' | head -n 1)"
test -n "$ASSET_PATH"
curl --fail --silent --show-error --insecure --head \
  "https://127.0.0.1:3213$ASSET_PATH" | \
  rg -i '^cache-control: public, max-age=31536000, immutable'

PUBLIC_3213='https://<public-host>:3213'
curl --fail --silent --show-error "$PUBLIC_3213/api/health"
curl --fail --silent --show-error --head "$PUBLIC_3213/" | \
  rg -i '^cache-control: no-store'
```

### 3213 回滚

若 Caddy restart 或上述只读检查失败，恢复已验证的 3211-only 配置并重启同一 unit；回滚会关闭 3213，不要重新启动旧 Vite：

```bash
ROLLBACK_CONFIG=/home/xy/.local/state/duet-ad1/caddy-config.before-3213.json
test -s "$ROLLBACK_CONFIG"
/usr/local/libexec/duet/caddy validate --config "$ROLLBACK_CONFIG"
install -m 0644 "$ROLLBACK_CONFIG" \
  /home/xy/duet-ad1/.deploy/caddy/config.json
systemctl --user restart duet-ad1-caddy.service
systemctl --user is-active duet-ad1-caddy.service
curl --fail --silent --show-error --insecure \
  https://127.0.0.1:3211/api/health
```

回滚后将配置恢复变更作为独立 fix-forward commit 审核、测试；不要用 `git reset --hard` 或共享工作区 `git checkout` 掩盖现场状态。

## 7. 可选显式付费验收（不属于生产 smoke）

以下流程不属于 3213 生产部署和生产 smoke，只能在另行批准真实付费验收后人工执行。脚本要求显式 `RUN_PAID_SMOKE=1`，默认走本机 3212，通过正式 API 创建、轮询、提交和验证 H3 成片。每执行一次脚本都会生成新的创建 id 和 generation `client_request_id`，创建一个新会话，并按冻结计划产生真实付费 H3 task；不得把下面三次验收放进自动重试器。脚本会打印 cid、generation request id 和付费子任务数，不会打印 access token、供应商凭据或 task id。

先准备实测时长分别为 10、15、30 秒的样本，并在付费前确认：

```bash
SAMPLE_10=/absolute/path/to/10s.mp4
SAMPLE_15=/absolute/path/to/15s.mp4
SAMPLE_30=/absolute/path/to/30s.mp4
for sample in "$SAMPLE_10" "$SAMPLE_15" "$SAMPLE_30"; do
  ffprobe -v error -select_streams v:0 -show_entries stream=duration,duration_ts,time_base \
    -of default=noprint_wrappers=1:nokey=1 "$sample"
  # WebM/Matroska 若 stream duration/ticks 为空，以 v:0 packet PTS 起止验收；禁止用 format.duration 或 frame_count/fps。
  ffprobe -v error -select_streams v:0 -show_packets \
    -show_entries packet=pts_time,dts_time,duration_time -of csv=p=0 "$sample"
done

read -rsp 'ACCESS_TOKEN: ' ACCESS_TOKEN; printf '\n'
export ACCESS_TOKEN

# 第 1 次新建/付费：10s，原 Ref2VA 单任务
RUN_PAID_SMOKE=1 .deploy/smoke-h3.sh "$SAMPLE_10"

# 第 2 次新建/付费：15s，FL2VA 长链；冻结计划通常为 1 个子任务
RUN_PAID_SMOKE=1 .deploy/smoke-h3.sh "$SAMPLE_15"

# 第 3 次新建/付费：30s，FL2VA 长链；以脚本打印的 segment_count 为准
RUN_PAID_SMOKE=1 .deploy/smoke-h3.sh "$SAMPLE_30"

unset ACCESS_TOKEN
```

非 9:16 样本默认使用 `pad`，可在人工确认内容可裁时加 `FIT_MODE=crop`。脚本默认台词模式为 `auto`；无音轨同样合法。长视频只允许 `DIALOGUE_MODE=auto|none`，其中 auto 复用源音轨：长于画面时裁剪、短于画面时补静音，画面时长不变；none 输出静音。

成功标准：创建阶段到 `done`；10 秒输入通过 prepared receipt v1，15/30 秒输入通过 64 位 plan receipt 和正整数 `segment_count`；submit 返回 202；generation 到 `succeeded` 且 `has_video=true`。记录每次输出的 cid 和 `client_request_id`，不要复制 token、EnvironmentFile 或供应商响应。

把三次输出的 cid 人工填入以下变量，逐个验证最终文件。长链拼接目标是与源时长误差不超过 1/24 秒；同时确认视频编码/像素格式和音轨。若 `DIALOGUE_MODE=none`，音频 stream 查询应为空。

```bash
CID_10='replace-with-10s-cid'
CID_15='replace-with-15s-cid'
CID_30='replace-with-30s-cid'
for cid in "$CID_10" "$CID_15" "$CID_30"; do
  output="data/$cid/generated.mp4"
  test -s "$output"
  ffprobe -v error -select_streams v:0 -show_entries stream=duration,duration_ts,time_base \
    -of default=noprint_wrappers=1:nokey=1 "$output"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,pix_fmt,r_frame_rate \
    -of default=noprint_wrappers=1 "$output"
  ffprobe -v error -select_streams a:0 \
    -show_entries stream=codec_name,channels,duration,duration_ts,time_base \
    -of default=noprint_wrappers=1 "$output"
  # keep 模式核对 v:0 presentation start 与成片实际起音位置；不要把 AAC/Opus priming packet 当有效音频起点。
  source=$(find "data/$cid" -maxdepth 1 -type f -name 'source.*' -print -quit)
  for selector in v:0 a:0; do
    ffprobe -v error -select_streams "$selector" -read_intervals '%+#1' \
      -show_packets -show_entries packet=pts_time,dts_time -of csv=p=0 "$source"
  done
done
```

脚本遇到最终 `failed`、`submission_unknown`、`resume_required` 或超时都会退出，且脚本自身绝不重复调用提交 API。服务内部对完整确认且不计费的 `h3_provider_failed` 会按配置自动补交，默认单逻辑请求总计最多 3 次 POST；其他失败不会。保留脚本打印的 cid 和原 `client_request_id`，按下一节判断是原 attempt 继续、自动额度耗尽后的新 id、拼接失败的原 id 本地重拼，还是先去供应商核对。

## 8. 失败时 fix-forward

1. 停止重复点击和重复 smoke；保留 cid、`prepared_input.json`、`long_video_plan.json`、`.h3/` 和 `meta.json`。
2. 用 detail API 或 journal 确认是输入准备、凭据、H3 查询/下载还是公开反代问题；不要打印环境。
3. 修复当前 H3 代码/配置，重新运行相关测试和全量测试。
4. 再次 `daemon-reload`（unit 有改动时）并原地 `restart`，重复本地和公网 `/health`。
5. `resume_required` 只通过 UI 用原 request id、原台词和原画幅继续同一 attempt；完整确认的 `h3_provider_failed` 先等待服务按额度自动收敛，额度耗尽后才点“重试生成”并使用新 id。其他 H3 确定失败仍按人工新 id 处理，并以 detail API 的 `generation.retry_paid_segment_count` 为本次新增付费任务数。长链 `failed + stage=stitch` 点“重试拼接”，必须用原 id，只本地重拼且新增付费 H3 子任务为 0；半发布成片不会隐藏恢复入口。`submission_unknown` 不得继续或重试，先到 AutoDL 核对原 POST，服务端会固定返回 409 `submission_outcome_unknown`。

禁止以 Seedance 代码、旧 unit 或旧提交开关回退；它们不再属于生产契约。若 H3 仍不可用，保持服务可读、关闭 `ENABLE_H3_SUBMIT`，修复后再开启。
