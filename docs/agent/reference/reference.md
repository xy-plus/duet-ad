---
name: backend-api
type: reference
status: done
owner: agent
updated: 2026-08-17
links: [conversation-task, app/main.py, app/storage.py, app/downloader.py, app/pipeline.py, app/codex_runner.py, app/seedance.py, app/seedance_task.py]
---

# backend-api · 接口（How/Now）

## 公开接口（HTTP）

除标注外均需 `Authorization: Bearer <ACCESS_TOKEN>`，失败一律 401 `{"detail":"unauthorized"}`（口令比对用 `hmac.compare_digest`）。

### `GET /api/health`（无鉴权）

- 200 → `{"ok": true}`

### `POST /api/login`（无鉴权）

- 请求：`{"token": "<口令>"}`（JSON）
- 200 → `{"ok": true}`；口令错 → 401 `{"detail":"invalid token"}`

### `GET /api/conversations`

- 200 → 数组，按 `created_at` 倒序，每项恰 6 字段：`id, title, note, status, created_at, has_video`
- `has_video` 为磁盘实况探测（`generated.mp4` 是否存在），不读 meta

### `POST /api/conversations`

- 请求：multipart，`file` 与 `reference_url`（http(s) 直链 / TikTok 视频页）恰好一个 + `note`（可选，默认 `""`）+ `client_request_id`（可选幂等键，格式 `^[0-9A-Za-z-]{8,64}$`，空 = 不参与幂等）
- 201 → `{"id": "<32位hex>", "status": "queued"}`；`enable_pipeline` 开时登记后台流水线（经管道闸，见架构）
- 200 → `{"id", "status"}`：`client_request_id` 命中既有会话 meta（扫 meta.json 查重），不建目录、不重复入队
- 400 `provide exactly one of file or reference_url`：`file`/`reference_url` 都不给或都给
- 400 `invalid client_request_id`：幂等键非空但不合格式
- 429 `too many uploads`：同 IP 1 分钟超 10 次（内存滑动窗口，进程重启清零）
- 429 `too many queued tasks`：`queued` 状态会话数达 `MAX_QUEUED`（processing/done/failed 不计；查重+计数+建目录在同一把锁内，无竞态）
- 422：上传/下载校验链失败（`UploadError`/`DownloadError`），detail 为具体原因；失败即回滚删除整个会话目录

上传校验链（顺序执行，`storage.save_upload` → `storage.probe_video`）：

1. 扩展名 ∈ `{.mp4, .mov, .webm}`（小写化），否则 `unsupported extension: <ext>`
2. 流式落盘（1MB 块），累计超 `MAX_UPLOAD_MB*1024*1024` → `file exceeds <n> bytes`，已写部分删除
3. ffprobe 实探（30s 超时）：打不开 → `ffprobe failed: ...` / `unreadable video file`；时长解析失败 → `cannot parse video duration`；时长超 `MAX_DURATION_S` → `duration <d>s exceeds <n>s`

URL 分支（`downloader.fetch_reference`，线程池执行不堵事件循环）：TikTok 视频页先经 TikWM API 解析出 play 直链；下载带 SSRF 防护（见架构安全模型），任一步失败（含解析到私网、HTTP 非 2xx、超限/超时/空文件、连接/读取异常归一）抛 `DownloadError` → 422 + 回滚；落盘 `source.<ext>`（后缀取 URL path，白名单外默认 `.mp4`）后与文件分支汇合同一 `probe_video`

### `GET /api/conversations/{cid}`

- 200 → **冻结的 12 字段契约**（显式键，meta 内部字段不外泄）：

| 字段 | 来源 |
| --- | --- |
| `id, title, note, status, error, created_at, updated_at` | meta.json |
| `keyframes` | meta.json（字符串数组，缺省 `[]`） |
| `prompt` | meta.json（缺省 `null`） |
| `has_source` | `source.*` 磁盘探测 |
| `has_video` | `generated.mp4` 磁盘探测 |
| `submit_enabled` | `settings.enable_seedance_submit` |

- 404 `not found`：cid 非法（非 `^[0-9a-f]{32}$`）或目录/meta 不存在

### `GET /api/conversations/{cid}/files/{name:path}`

- 200 → `FileResponse`
- files 白名单（`storage.resolve_file`，此外一律 404）：`source.mp4`（映射唯一的 `source.*`，扩展名不定）、`preview.mp4`、`generated.mp4`、`contact_sheet.jpg`（映射 `work/contact_sheet.jpg`）、`keyframes/<fn>`（映射 `work/keyframes/<fn>`，`<fn>` 必须是不含路径的纯文件名）
- 防御：cid 正则校验 + `resolve()` 后必须 `is_relative_to` 会话目录且是文件；symlink 越界/穿越一律 404

### `POST /api/conversations/{cid}/submit`（预留，默认 501）

- 请求：JSON，仅接受 `{"confirm": true}`；不接受任何 prompt/参数覆盖（提交时由 `work/prompt.txt` + 关键帧现构建请求）
- 成功 200 → `{"status": "succeeded", "video": "generated.mp4"}`，并回写 meta：`has_video/submitted_at/task_id`（内部字段，不进响应）
- 门控矩阵（`seedance.submit`，**固定顺序**）：

| 序 | 条件 | 状态码 / detail |
| --- | --- | --- |
| 1 | `enable_seedance_submit` 关 | 501 `Seedance submission is disabled.` |
| 2 | 会话不存在 | 404 `not found` |
| 3 | `confirm` 不为 `true` | 409 `confirmation required` |
| 4 | `status != "done"` | 409 `artifacts not ready` |
| 5 | `has_video` 已真 | 409 `already submitted` |
| 6 | 每会话锁内复查：meta 消失或 `has_video` 已真 | 409 `already submitted` |
| 7 | `work/prompt.txt` 缺失或为空 | 409 `payload changed since review` |
| 8 | `work/keyframes/` 下无 PNG | 409 `payload changed since review` |
| 9 | dry-run 预检（120s 超时）构建 payload 失败或未落 payload-out | 409 `payload changed since review` |
| 10 | 服务进程无 `ARK_API_KEY` | 503 `ARK_API_KEY not configured` |
| 11 | 真实提交超时（1800s）/ 执行器不可用 / 非零退出 | 502 `seedance task timed out` / `seedance runner unavailable` / 脱敏后输出（≤300 字） |

- 提交 argv：提交时由当前 `work/prompt.txt` + `work/keyframes/*.png` 现构建 `app/seedance_task.py create --confirm-submit --wait --state-file work/task.json --download generated.mp4`，建模固定 `doubao-seedance-2-0-260128 / 9:16 / 15s / 720p / --generate-audio / --no-watermark`；argv 列表、无 shell、env 缺省继承服务进程（`ARK_API_KEY` 由此进入脚本）
- 幂等：成功后 `has_video=true` 落盘，重复/并发提交一律 409；锁常驻内存，进程重启即失效

## 公开接口（Python 模块）

- `app.config.get_settings() -> Settings` — 读环境变量建配置；缺 `ACCESS_TOKEN` 抛 RuntimeError
- `app.config.Settings` — frozen dataclass，11 字段（见架构配置表）；直建默认 `enable_pipeline=False`
- `app.auth.require_auth(request, cred)` — FastAPI 依赖，Bearer 校验
- `app.main.create_app(settings) -> FastAPI` — 应用工厂（测试注入用）；模块级 `app = create_app(get_settings())`
- `app.storage.new_conversation(data_dir, note, orig_name, client_request_id="") -> dict` — 建目录 + 初始 meta（status=queued）；幂等键非空才落 meta
- `app.voice.extract_audio(cdir) -> Path | None` — ffmpeg 抽音轨为 work/voice.mp3；无音轨 → None；失败 → PipelineError
- `app.voice.validate_voice_lines(raw, duration_s) -> list[dict]` — 台词 JSON 白名单校验（raw ≤ 32KB、条目 ≤ 200、每行 text ≤ 500 字、text/start_s/end_s 三字段、时间单调且落在时长内）；返回净化列表
- `app.storage.update_meta(data_dir, cid, **changes) -> dict | None` — 合并写字段并刷新 `updated_at`
- `app.storage.load_meta(data_dir, cid) -> dict | None` — cid 正则不过/文件缺 → None
- `app.storage.list_conversations(data_dir) -> list[dict]` — 扫描合法目录，按 `created_at` 倒序
- `app.storage.remove_conversation(data_dir, cid)` — 回滚删目录（cid 校验）
- `app.storage.mark_submitted(data_dir, cid, task_id) -> dict` — 提交标记回写（供幂等门控）
- `app.storage.save_upload(cdir, upload, max_bytes) -> Path` — 流式落盘 `source.<ext>`；`UploadError` 由 HTTP 层转 422
- `app.downloader.fetch_reference(url, cdir, settings) -> Path` — URL 分流（TikTok 经 TikWM / http(s) 直链）下载落盘 `source.<ext>`；`DownloadError` 由 HTTP 层转 422
- `app.storage.probe_video(path, max_duration_s) -> float` — ffprobe 探测，返回时长
- `app.storage.resolve_file(data_dir, cid, name) -> Path | None` — files 白名单解析
- `app.pipeline.run(settings, cid, runner)` — 后台任务入口；任何失败 → `failed`+`error`，不抛
- `app.pipeline.validate_work_dir(work) -> (list[str], str)` — agent 产物白名单校验，返回 (关键帧名, prompt)
- `app.codex_runner.CodexRunner(timeout_s, concurrency)` — `.build_argv(workdir, prompt)` / `.run(workdir, prompt)`；`CodexError` 包装超时/非零/找不到二进制
- `app.codex_runner.clean_stderr(text, limit=500)` — 剔环境变量行 + 截断（pipeline 的 `_run_cmd` 复用）
- `app.seedance.submit(settings, cid, payload, locks) -> dict` — 门控 + 执行；`SubmitError(status, detail)` 由 HTTP 层转响应

## 关键数据形

meta.json（`data/<cid>/meta.json`）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | str | uuid4 hex，32 位小写 |
| `title` | str | `note` 或净化文件名（去路径/控制字符，≤80 字，空则 `untitled`） |
| `note` | str | 用户备注（可无长度上限，见 OPEN_ISSUE nits） |
| `status` | str | `queued/processing/done/failed` |
| `error` | str \| null | 失败原因，≤500 字 |
| `created_at` / `updated_at` | str | ISO8601 UTC |
| `keyframes` | list[str] | 关键帧文件名（done 后写入） |
| `prompt` | str \| null | Seedance prompt（done 后写入） |
| `voice_lines` | list[dict] | 口播台词（`voice_mode≠none` 时 ASR 校验后写入；内部字段，不进 detail 响应） |
| `client_request_id` | str | 前端幂等键（仅提交时带才存在；内部字段，查重依据） |
| `has_video` | bool | 提交标记（仅提交后存在；内部字段） |
| `submitted_at` / `task_id` | str | 提交时间 / Ark 任务 id（内部字段，读不到 task.json 则为 null） |

产物白名单校验（`pipeline.validate_work_dir`，任一不过即 PipelineError → `failed`）：

- `work/keyframes/*.png`：数量 ∈ 1..9（新契约该目录只有选定帧 `01.png…N.png`）
- `work/prompt.txt`：存在、非空、≤ 32KB（`MAX_PROMPT_BYTES`）

## 依赖

- Python 包（`requirements.txt`）：fastapi、uvicorn[standard]、python-multipart、opencv-python-headless（skill 脚本用）、pytest、httpx（TestClient）
- 外部可执行：ffmpeg/ffprobe（探测+抽帧+测试造样例）、codex CLI（0.147.0 实证基线，仅流水线用）
- 技能脚本：`skills/video-maker/scripts/extract_keyframes.py`（`--fps`/`--times`/`--sample-count`/`--prefix`/`--columns`/`--out-dir`）、`skills/video-maker/scripts/crop_image.py`（裁字幕/水印）；提交脚本 `app/seedance_task.py`（`create --dry-run|--confirm-submit --wait`，模型默认 `doubao-seedance-2-0-260128`，Ark `https://ark.cn-beijing.volces.com/api/v3`）
- 流水线固定参数：抽帧 `--fps 4`（分页联系表落 `work/`）；提交建模 `9:16 / 15s / 720p / --generate-audio / --no-watermark`（提交时现构建，无评审 payload）
