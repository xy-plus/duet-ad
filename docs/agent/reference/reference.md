---
name: h3-runtime
type: reference
status: done
owner: agent
updated: 2026-08-20
tdd: N/A
links: [conversation-task, app/main.py, app/h3.py, app/prepared_input.py]
---

# H3 runtime · 接口（How/Now）

## HTTP 通则

- `/api/health`、`/api/login` 无需鉴权；其余 `/api/conversations*` 使用 `Authorization: Bearer <ACCESS_TOKEN>`。
- JSON 错误为 `{"detail":"<safe message>"}`。供应商响应正文、URL、token 和本机路径不作为公开错误。
- H3 提交是 202 + 轮询；`POST /submit` 不等待最终视频。

### `GET /api/health`

`200 {"ok":true}`。只证明 Web 进程可响应，不探测供应商凭据或余额。

### `POST /api/login`

请求 `{"token":"..."}`。匹配 `ACCESS_TOKEN` 返回 `200 {"ok":true}`，否则 401 `invalid token`。

### `GET /api/conversations`

按 `created_at` 倒序返回：

```json
[
  {
    "id": "32-char-hex",
    "title": "...",
    "note": "...",
    "status": "queued|processing|done|failed",
    "created_at": "ISO-8601 UTC",
    "has_video": false
  }
]
```

### `POST /api/conversations`

multipart 字段：

| 字段 | 契约 |
| --- | --- |
| `file` / `reference_url` | 必须且只能提供一个；上传扩展名 `.mp4/.mov/.webm`，URL 分支有 SSRF/大小/超时保护 |
| `note` | 可选；为空时标题取净化后的文件名/URL |
| `client_request_id` | 可选，`^[0-9A-Za-z-]{8,64}$`；命中既有创建请求返回 200，不重复入队 |
| `voice_mode` | `keep/rewrite/translate`，默认 `keep`；只控制 auto 输入准备 |
| `target_language` | `voice_mode=translate` 时必填；其他模式忽略 |

新建成功返回 `201 {"id":"...","status":"queued"}`；创建幂等命中返回 200 同形。有效视频时长为 `0 < duration_s <= min(MAX_DURATION_S,15)`，文件大小默认 ≤500MB。无音轨合法。常见错误：400 来源数量错误或创建 id 非法；401；422 下载/媒体/模式校验失败；429 IP 限流或排队已满。

### `GET /api/conversations/{cid}`

返回固定公开字段：

```json
{
  "id": "...",
  "title": "...",
  "note": "...",
  "status": "queued|processing|done|failed",
  "error": null,
  "created_at": "...",
  "updated_at": "...",
  "keyframes": ["01.png"],
  "prompt": "...",
  "segments": [],
  "voice_lines": [],
  "read_only": false,
  "duration_s": 9.2,
  "fit_required": false,
  "dialogue": {
    "mode": "auto|edit|custom|none",
    "lines": [],
    "auto_lines": []
  },
  "receipt_version": 1,
  "generation": {
    "status": "queued|running|resume_required|succeeded|failed|submission_unknown",
    "error": null,
    "attempt": 1,
    "client_request_id": "request-123456"
  },
  "fit_mode": "none",
  "has_source": true,
  "has_video": false,
  "submit_enabled": true,
  "postprocess": null,
  "postprocess_enabled": false
}
```

`generation`、`receipt_version` 和 `fit_mode` 在尚未创建时为 null。`dialogue.lines` 是当前 mode 的有效公开台词；`auto_lines` 永远保留自动有效台词供 edit 预填。`read_only` 由 `schema_version != 2` 派生，不相信旧 meta 自报。`has_source/has_video` 按磁盘实况计算。

### `GET /api/conversations/{cid}/files/{name}`

白名单：`source.mp4`（映射唯一 `source.*`）、`preview.mp4`、`generated.mp4`、`contact_sheet.jpg`、`keyframes/<basename>`、`postprocessed/<basename>`，以及旧分段路径 `segments/N/work/{keyframes|postprocessed}/<basename>`。路径穿越、非白名单或文件不存在均 404。

### `POST /api/conversations/{cid}/submit`

严格 JSON 形状：

```json
{
  "confirm": true,
  "client_request_id": "request-123456",
  "dialogue_mode": "auto",
  "fit_mode": "none"
}
```

允许键只有 `confirm/client_request_id/dialogue_mode/lines/fit_mode`：

- `confirm` 必须是 JSON boolean `true`。
- `client_request_id` 必须完整匹配 `^[0-9A-Za-z-]{8,64}$`。
- `dialogue_mode=auto`：禁止 `lines`，使用 `voice_line_provenance` 中 `kept=true` 的内部 ASR 行；启用 vocal filter 时每句必须为 `spoken` 或 `sung`。
- `dialogue_mode=edit|custom`：必须有非空 `lines`；每项只能含 `text/start_s/end_s`，文本非空、时间有序且落在实际视频时长内。`edit` provenance 固定 `asr+edited`，`custom` 固定 `manual`。
- `dialogue_mode=none`：禁止 `lines`，有效台词为空。
- `fit_required=false` 时只允许 `fit_mode=none`；为 true 时只允许 `crop` 或 `pad`。该值只在 pipeline `done` 时按实际选中关键帧计算，源视频 9:16 不能豁免非 9:16 关键帧。

接受后返回 `202 {"status":"queued","attempt":N}`。后台状态写入 `generation`，客户端轮询 detail。

门控和错误：

| HTTP | detail | 条件 |
| --- | --- | --- |
| 501 | `H3 submission is disabled.` | `ENABLE_H3_SUBMIT` 未开启；此门在会话查找前 |
| 404 | `not found` | cid 不存在/非法 |
| 409 | `read_only` | 非 schema v2 |
| 409 | `confirmation required` | 非严格 true |
| 422 | `invalid_submit_request` | 出现未知键 |
| 422 | `invalid_client_request_id` | id 不合规 |
| 422 | `invalid_dialogue` | mode、lines 形状或台词内容不合规 |
| 422 | `invalid_fit_mode` / `fit_mode_required` / `fit_mode_not_allowed` | 画幅选择不合规 |
| 409 | `artifacts not ready` | 输入准备 status 不是 done |
| 503 | `h3_credentials_missing` | MiniMax/AutoDL 任一凭据缺失 |
| 503 | `h3_configuration_invalid` | 冻结后无法构造合法 H3Request/timeout 配置 |
| 409 | `prepared_input_invalid` / `frame_fit_failed` | 冻结输入或画幅派生失败 |
| 409 | `generation in progress` / `already submitted` | active/succeeded 使用不同 id |
| 409 | `new client_request_id required` | 确定 failed 后复用旧 id |
| 409 | `resume_request_id_mismatch` | resume_required 没有使用原 client_request_id |
| 409 | `resume_parameters_changed` | resume_required 的 mode、归一化 lines 或 fit 与冻结值不一致 |
| 409 | `submission_outcome_unknown` | 既有 generation 为 submission_unknown；任意 id 均拒绝 |
| 409 | `generation_state_invalid` | 已持久化 generation status/attempt 不满足安全状态形状 |

相同 id 在 `queued/running/succeeded` 时返回现有 `{status,attempt}`，不重复 POST。确定 `failed` 只有新 id 才进入人工 retry。

`resume_required` 表示 provider task 已知或 Context IR 已完成：只接受原 `client_request_id`，且 dialogue mode、标准化 lines、`fit_mode` 必须与 meta 和 prepared receipt 完全一致。合法继续返回 `202 {"status":"queued","attempt":<原值>}`，不重写 receipt、不递增 attempt，后台调用幂等 `h3.start` 而非 `h3.retry`。已知 task 错误包括 `ir_query_failed/ir_timeout/h3_query_failed/h3_timeout/download_failed/download_dns_failed/download_peer_unverified/output_write_failed/output_probe_failed`；`ready_for_h3`、`ir_running`、`h3_running` 也进入此状态。

确定性输出安全拒绝 `download_url_rejected/download_redirect_rejected/download_too_large/download_invalid_video` 映射为 `failed`，只有用户明确使用新 id 才创建 retry attempt。它们不属于会因同参数继续而消失的传输故障。

`submission_unknown` 是唯一完全锁死状态：前端隐藏操作，服务端对任何 id 返回 409，必须先在 MiniMax/AutoDL 侧核对原 POST 是否已创建任务。

### `POST /api/conversations/{cid}/postprocess`

可选 Seedream 后处理：

```json
{
  "confirm": true,
  "options": {
    "remove_subtitle": true,
    "remove_brand": false
  }
}
```

至少一项为 true；未知 option 或非 bool 返回 422。禁用返回 501，旧会话返回 409 `read_only`，输入未 done/正在运行/重跑改变选项返回 409。接受后返回 `{"status":"running","frames":[]}`，detail 的 `postprocess` 轮询到 done/failed。后处理产物不进入 H3 receipt。

## Conversation meta schema v2

`data/<cid>/meta.json` 是内部状态，不等于 detail 响应。创建时的稳定字段：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `schema_version` | int | 固定 2；缺失或其他版本派生为只读 |
| `id` | str | uuid4 hex |
| `title`, `note` | str | 展示信息 |
| `status`, `error` | str, str/null | 输入准备状态与安全错误 |
| `created_at`, `updated_at` | str | ISO-8601 UTC |
| `keyframes`, `prompt` | list[str], str/null | 准备产物 |
| `voice_mode` | str | 创建阶段 auto 的 keep/rewrite/translate 方式 |
| `duration_s` | float/null | ffprobe 实际时长 |
| `fit_required` | bool/null | pipeline done 时按实际选中关键帧计算，任一非 9:16 即 true |
| `dialogue_mode` | str | 默认 auto；最终提交选择 |
| `generation` | object/null | coarse H3 attempt 状态 |

准备/提交后可增加：

| 字段 | 语义 |
| --- | --- |
| `voice_lines` | 当前有效裸台词 `{text,start_s,end_s}` |
| `voice_line_provenance` | 自动识别行 + `classification/provenance/kept` 决策 |
| `vocal_filter_enabled`, `voice_warnings`, `voice_lines_vocal_dropped` | 声学过滤留痕 |
| `prepared_dialogue` | 本次提交冻结的完整有效台词（含 classification/provenance） |
| `prepared_input_receipt` / `receipt_version` | `prepared_input.json` / 1 |
| `fit_mode` | `none/crop/pad` |
| `generation` | `{status,error,attempt,client_request_id}`；status 含 resume_required |
| `postprocess` | `{status,options,frames,error}`，与 H3 输入隔离 |

旧 `segments` 字段只用于读取历史产物；schema v2 的 1–15 秒准备路径要求单段。

## Prepared input receipt v1

固定顶层形状：

```json
{
  "schema": "duet.prepared-input",
  "version": 1,
  "bindings": {
    "source": {"path": "source.mp4", "sha256": "..."},
    "normalized_audio": null,
    "keyframes": [{"path": "work/keyframes/01.png", "sha256": "..."}],
    "visual_prompt": {"path": "work/visual_prompt.txt", "sha256": "..."},
    "final_prompt": {"path": "work/prompt.txt", "sha256": "..."}
  },
  "dialogue": {"mode": "none", "lines": [], "sha256": "..."},
  "vocal_filter": {"enabled": true},
  "video": {"duration_s": 9.2, "ratio": "9:16", "fit_mode": "none"},
  "engine_request": {
    "context_ir": {"model": "MiniMax-H3", "duration": 10, "ratio": "9:16"},
    "h3": {"workflow": "minimax_h3_lightx2v_v5", "duration": 10, "resolution": "768p竖"}
  }
}
```

严格要求顶层和子对象 key 集合完全匹配；绑定路径必须位于会话根内；关键帧 1–9 张且路径唯一；每个文件重读校验 SHA-256；台词 canonical JSON、provenance 和 hash 必须匹配；最终 prompt 必须等于当前视觉 prompt + 当前结构化发声块。任何偏差抛 `PreparedInputError`，HTTP 层归一为 409 `prepared_input_invalid`。

`normalized_audio=null` 是无音轨的合法表示。crop/pad 时 keyframe binding 指向 `work/h3_frames/<mode>/`，且不会出现 `postprocessed`。

## H3 attempt state v1

路径 `.h3/attempts/<六位递增号>/attempt.json`。最小形状：

```json
{
  "schema_version": 1,
  "cid": "...",
  "attempt_id": "000001",
  "client_request_id": "request-123456",
  "input": {},
  "input_receipt": "sha256",
  "status": "ir_submitting",
  "retryable": false,
  "ir": {"status": "submitting"},
  "h3": {"status": "not_started"}
}
```

内部 status：`ir_submitting/ir_running/ready_for_h3/h3_submitting/h3_running/succeeded/retryable_failure/failed/submission_unknown`。任务 id 出现后必须同时存在对应 receipt；optimized prompt 的全部严格小写 `<d>...</d>` 在去掉每段可选 `[Language]` 前缀后，必须与冻结 `voice_texts` 数量、顺序、文本全等；最终 output receipt 只能是 `{name:"generated.mp4",sha256,size}`。`result_url` 明确禁止落状态文件。

### start / inspect / resume / retry

- `h3.start(request)`：同 client id 幂等；新 attempt 允许 Context IR 和 H3 POST，已有 attempt 只按已知状态推进。
- `h3.inspect(request)`：纯读，不写、不联网；验证 session/attempt receipts。
- `h3.resume(request)`：获取会话 flock 后 GET-only 推进已有任务；`allow_submit=false`，不创建供应商任务。
- `h3.retry(request,new_id)`：底层显式创建新 attempt；runtime 只在公开 generation 已确定为 `failed` 且用户提交新 id 时调用。`resume_required` 调 `start` 续同 attempt，`submission_unknown` 不调用。

供应商顺序固定：上传 1–9 张冻结帧 → `POST /v2/h3_context_ir` → GET 查询并精确验证 optimized prompt 台词 → `POST /api/v1/comfyui/comfyui_workflow/minimax_h3_lightx2v_v5` → GET 结果 → 安全下载并原子写 `generated.mp4`。

IR 台词门禁：标签少、多、改写、乱序、只有裸文本而无标签或标签残缺均为 `ir_dialogue_mismatch`，H3 POST 不会发生；冻结台词为空时必须没有 `<d>`，并额外拒绝新增台词、角色/旁白发声和 OCR/字幕朗读语义。

输出下载门禁：只接受无 userinfo 的 HTTPS；hostname/IP 预解析必须全为公网地址，私网、loopback、local、reserved、multicast 均确定拒绝。owned httpx client 使用 `trust_env=false`；响应后在读取 status/body 前通过 `extensions.network_stream.get_extra_info("server_addr")` 验证实际 socket peer 也是公网地址。实际 peer 为私网仍是 `download_url_rejected`；DNS 临时失败为 `download_dns_failed`，缺失/异常/非 IP peer 为 `download_peer_unverified`，后二者同 id 恢复。

显式不跟随 3xx；Content-Length 和实际流都不得超过 `200 * 1024 * 1024` 字节。内容先以 0600 写同目录临时文件。ffprobe 缺失、OS 错误或超时为可恢复 `output_probe_failed`；ffprobe 正常执行但媒体无法解析、无 video stream 或时长非正有限值为确定失败 `download_invalid_video`。验证通过后才 `os.replace`、目录 fsync。状态只保存输出 name/SHA-256/size，不保存 URL。

## Python 公开接口

- `app.main.create_app(settings) -> FastAPI` — 应用工厂；启动时扫描可恢复 generation。
- `app.storage.new_conversation(...) -> dict` — 创建 schema v2 meta 和 `work/`。
- `app.pipeline.run(settings,cid,runner) -> None` — 输入准备；失败只写 meta，不向调用者抛。
- `app.prepared_input.prepare_dialogue(...) -> tuple[dict,...]` — 规范化 auto/edit/custom/none 来源。
- `app.prepared_input.write_prepared_input(...) -> PreparedInput` — 原子写 receipt 并复用 loader 验证。
- `app.prepared_input.load_prepared_input(...) -> PreparedInput` — 读取冻结 bytes，漂移即拒绝。
- `app.frame_fit.fit_frames(paths,output_dir,mode) -> tuple[Path,...]` — 显式 crop/pad 9:16 派生。
- `app.h3.start/inspect/resume/retry` — 可恢复状态机；runtime 对 resume_required 暴露同 id start，对确定 failed 暴露新 id retry，不对 submission_unknown 暴露操作。
- `app.postprocess.start/run_task` — 可选展示后处理；与 H3 请求隔离。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ACCESS_TOKEN` | 无，必填 | 共享 Bearer 口令 |
| `ENABLE_PIPELINE` | `1` | 生产是否启动输入准备；直接构造 Settings 的测试默认 false |
| `DATA_DIR` | `data` | 会话根目录 |
| `MAX_UPLOAD_MB` | `500` | 上传/URL 下载上限 |
| `MAX_DURATION_S` | `15` | 配置上限；HTTP 创建仍额外钳制到 15 秒 |
| `CODEX_TIMEOUT_S` | `1800` | 单次 codex 硬超时 |
| `CODEX_CONCURRENCY` | `10` | pipeline 并发闸 |
| `MAX_QUEUED` | `100` | queued 会话上限 |
| `VOCAL_FILTER` | `on` | off/false/0 才关闭 keep/drop；未知值保持开启 |
| `ENABLE_H3_SUBMIT` | false | H3 提交总开关 |
| `MINIMAX_API_KEY` | 空 | Context IR 凭据 |
| `AUTODL_ART_TOKEN` | 空 | AutoDL H3 凭据 |
| `H3_REQUEST_TIMEOUT_S` | `30` | 普通供应商请求超时 |
| `H3_UPLOAD_TIMEOUT_S` | `60` | 单关键帧上传超时 |
| `H3_IR_POLL_TIMEOUT_S` | `900` | Context IR 轮询总时限 |
| `H3_POLL_TIMEOUT_S` | `1500` | H3 轮询总时限 |
| `H3_DOWNLOAD_TIMEOUT_S` | `180` | 成片下载超时 |
| `H3_POLL_INTERVAL_S` | `3` | 两次查询间隔，可为 0 |
| `ENABLE_SEEDREAM_EDIT` | false | 可选关键帧后处理开关 |
| `SEEDREAM_MODEL` | `doubao-seedream-5-0-pro-260628` | 后处理模型 |
| `SEEDREAM_CONCURRENCY` | `10` | 后处理并发，最小钳制为 1 |
| `TIKTOK_PROXY` | 空 | TikTok/DoH 下载代理 |
| `DOWNLOAD_TIMEOUT_S` | `120` | URL 下载整体时限 |
| `HOST` / `PORT` | `0.0.0.0` / `3211` | `run.sh` 默认；生产 unit 必须覆盖为 `127.0.0.1/3212` |

全部服务环境只放 `%h/.config/duet-ad1/service.env` 这一份 0600 EnvironmentFile，不写 unit、仓库或命令行。部署步骤见 [.deploy/runbook.md](../../../.deploy/runbook.md)。

## 依赖

- Python：FastAPI、uvicorn、httpx、OpenCV、ai-edge-litert 等（以 `requirements.txt` 为准）。
- 可执行：`ffmpeg`、`ffprobe`、已认证的 `codex` CLI。
- 外部服务：MiniMax Context IR、AutoDL Art H3；可选 Seedream 图像编辑。
- 运行约束：Linux `flock/fsync` 语义、单 uvicorn 进程、Caddy 本机反代。
