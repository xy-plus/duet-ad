---
name: h3-runtime
type: reference
status: done
owner: agent
updated: 2026-08-25
tdd: N/A
links: [conversation-task, app/main.py, app/h3.py, app/prepared_input.py, app/long_generation.py]
---

# H3 runtime · 接口（How/Now）

## HTTP 通则

- `/api/health`、`/api/login` 无需鉴权；其余 `/api/conversations*` 使用 `Authorization: Bearer <ACCESS_TOKEN>`。
- JSON 错误 detail 为安全字符串或 `{code,message}`。结构化冲突只公开固定 code/中文提示；供应商响应正文、URL、token 和本机路径不作为公开错误。
- H3 提交是 202 + 轮询；`POST /submit` 不等待最终视频。

### `GET /api/health`

`200 {"ok":true}`。只证明 Web 进程可响应，不探测供应商凭据或余额。

### `POST /api/login`

请求只能是 `{"token":"..."}`。额外键或非字符串 token 返回 422 `invalid_login_request`；字符串不匹配 `ACCESS_TOKEN` 返回 401 `invalid token`；匹配返回 `200 {"ok":true}`。

### `GET /api/conversations`

按 `created_at` 倒序返回：

```json
[
  {
    "id": "32-char-hex",
    "title": "...",
    "note": "...",
    "status": "queued|processing|done|failed",
    "navigation_status": "analysis_queued|...|completed",
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

multipart 只允许表中字段及 `file`，未知或重复字段返回 422 `invalid_create_request`，且不会创建会话。已知旧页面提交 `voice_mode=none` 时返回纯文本中文刷新提示，不写文件、不入队；混入未知字段不会被归为旧页面。

新建成功返回 `201 {"id":"...","status":"queued"}`；创建幂等命中返回 200 同形。有效视频时长是 `v:0` 视觉时长：优先 `stream.duration`，其次 `duration_ts*time_base`，最后扫描视频包 PTS 范围并对缺失的末包 duration 使用相邻 PTS/帧率补尾；不得使用 OpenCV 帧数/FPS、音轨或 `format.duration` 补长。该值须正有限且不超过 300 秒；文件大小默认 ≤500MB。无音轨合法。`>15s` 只接受 `voice_mode=keep`，否则 422 `long_video_audio_mode_unsupported`。超时长返回结构化 `422`，`detail.code=video_duration_exceeds_h3_limit`，不保留刚创建的会话。其他常见错误：400 来源数量错误或创建 id 非法；401；422 下载/媒体/模式校验失败；429 IP 限流或排队已满。

### `GET /api/conversations/{cid}`

返回固定公开字段：

```json
{
  "id": "...",
  "title": "...",
  "note": "...",
  "status": "queued|processing|done|failed",
  "navigation_status": "analysis_queued|...|completed",
  "error": null,
  "created_at": "...",
  "updated_at": "...",
  "keyframes": ["01.png"],
  "prompt": "...",
  "source_prompt": "...",
  "source_prompt_sha256": "64 lowercase hex characters",
  "segments": [],
  "voice_lines": [],
  "read_only": false,
  "duration_s": 9.2,
  "fit_required": false,
  "fit_profiles": {
    "16:9": {"fit_required": true, "default_fit_mode": "crop"},
    "9:16": {"fit_required": false, "default_fit_mode": "none"}
  },
  "aspect_ratio": "9:16",
  "resolution": "768p",
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
  "postprocess_capabilities": {
    "remove_subtitle": true,
    "remove_brand": true,
    "optimize_image": true
  },
  "postprocess_enabled": true,
  "image_optimization_prompt": {
    "text": "...",
    "default_text": "...",
    "sha256": "64-hex"
  }
}
```

列表和详情的 `navigation_status` 来自同一个服务端投影，客户端不得再组合
`status/generation/postprocess/has_video` 猜测。完整枚举及优先级：

| 条件 | `navigation_status` |
| --- | --- |
| analysis `queued/processing/failed` | `analysis_queued/analysis_processing/analysis_failed` |
| analysis 未知 | `analysis_unknown` |
| analysis `done` 且无 generation 证据 | `analysis_complete`；即使存在孤立 `generated.mp4` 也不算完成 |
| generation `queued/running/failed` | `generation_queued/generation_running/generation_failed` |
| generation `submission_unknown/resume_required` | `generation_submission_unknown/generation_resume_required` |
| generation 未知 | `generation_unknown` |
| generation `succeeded` 但最终输出未通过服务端验收 | `output_missing` |
| generation `succeeded` 且最终输出有效 | `completed` |
| 有效最终输出上的 postprocess `running/failed` | `postprocessing/postprocess_failed`，优先于 `completed` |
| generation `succeeded`、最终输出有效，且 postprocess 未运行或已 `done` | `completed` |

analysis 的非终态/失败优先于所有 generation/postprocess 状态。投影只返回枚举字符串，
不包含供应商 task id、文件路径或其他内部恢复字段。

短视频使用上面的 `receipt_version=1`，不返回 plan 字段。长视频的同一响应还包含：

```json
{
  "duration_s": 30.0,
  "receipt_version": null,
  "plan_receipt": "64 lowercase hex characters",
  "segment_count": 2,
  "segments": [
    {"index": 1, "start_s": 0.0, "end_s": 15.0, "chain_id": "chain-001", "join_mode": "hard_cut"},
    {"index": 2, "start_s": 15.0, "end_s": 30.0, "chain_id": "chain-001", "join_mode": "continue"}
  ],
  "generation": {
    "status": "running",
    "stage": "h3",
    "fast_mode": false,
    "segments": [
      {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut", "status": "succeeded", "attempt": 1, "error": null},
      {"index": 2, "chain_id": "chain-001", "join_mode": "continue", "status": "running", "attempt": 1, "error": null}
    ]
  }
}
```

`generation`、短链 `receipt_version` 和 `fit_mode` 在尚未创建时为 null。长链 `generation.fast_mode` 是首次提交冻结的 boolean，历史 generation 缺失时公开为 `false`；恢复和重试参数比较都以冻结值为准。当前浏览器对尚未创建 generation 的长链固定初始化 `fastMode=true`，确认页不显示模式开关或说明，生成结果参数摘要也不展示该模式；一旦详情出现 generation，浏览器必须丢弃本地可编辑草稿，以详情中的画幅、清晰度、台词、fit 和快速模式重新同步并锁定参数控件，跨标签页提交不能继续显示另一个草稿值。`aspect_ratio/resolution` 是服务端推荐值，首次提交后即为冻结值；闭集分别为 `16:9|9:16` 和 `480p|768p`。`fit_profiles` 同时公开两个画幅的 `fit_required/default_fit_mode`，浏览器改变画幅时只能据此切换适配选项。历史 meta 缺少新字段时精确投影为 `9:16 + 768p`，不会改写旧 receipt。长链 `plan_receipt` 是 canonical `long_video_plan.json` 的 SHA-256，`segment_count` 来自冻结计划；`generation.segments` 只公开 `index/chain_id/join_mode/status/attempt/error`，不公开供应商 task id、内部 child request id 或文件路径。长链当前为 `failed` 时，`generation.retry_paid_segment_count` 以冻结 `meta.segments` 的完整索引集合为基数，由服务端结合持久化状态与分段 `generated.mp4` 文件实况计算：缺项计入，重复、未知或乱序状态整批不复用；快速模式成功段彼此独立，默认模式仍要求 `continue` 上游可复用；`stage=stitch` 固定为 0。该复用判定与 retry 初始化共用，前端不得从公开 segment status 再次推断费用。`source_prompt` 来自受 receipt 绑定的 `work/visual_prompt.txt`，配套 SHA-256 用于首次 H3 attempt 前的编辑 CAS；`prompt` 是机械追加结构化台词后的最终输入。`dialogue.lines` 是当前 mode 的有效公开台词；`auto_lines` 永远保留自动有效台词供短链 edit 预填。`read_only` 由 `schema_version != 2` 派生，不相信旧 meta 自报。`has_source` 按源文件实况计算；`has_video` 与 `navigation_status` 共用当前服务端最终输出验收结果。成功页继续播放成片，并从这些服务端冻结值生成只读参数摘要。

### `PATCH /api/conversations/{cid}/prompt`

请求严格为 `{confirm:true, expected_sha256:"...", prompt:"..."}`。该接口只适用于 `≤15s` 短链：在 schema v2、输入准备完成且生成 attempt 尚未创建时，以 SHA-256 CAS 更新 `work/visual_prompt.txt`，重新机械组合结构化台词并重写 prepared receipt。CAS 不一致返回结构化 409 `prompt_changed` 和中文刷新提示，不改写 prompt/meta/receipt；attempt 已创建后返回 409 `prompt_frozen`。长链的分段提示词已由 plan receipt 逐段绑定，不提供此顶层编辑接口。

### `GET /api/conversations/{cid}/files/{name}`

白名单：`source.mp4`（映射唯一 `source.*`）、`preview.mp4`、`generated.mp4`、`contact_sheet.jpg`、`keyframes/<basename>`、`postprocessed/<basename>`，以及长视频 `segments/N/work/{keyframes|postprocessed}/<basename>`。路径穿越、非白名单或文件不存在均 404。

### `POST /api/conversations/{cid}/submit`

短视频直接提交 frozen prepared input：

```json
{
  "confirm": true,
  "client_request_id": "request-123456",
  "dialogue_mode": "none",
  "fit_mode": "none",
  "aspect_ratio": "9:16",
  "resolution": "768p"
}
```

短视频允许键只有 `confirm/client_request_id/dialogue_mode/lines/fit_mode/aspect_ratio/resolution`：

- `confirm` 必须是 JSON boolean `true`。
- `client_request_id` 必须完整匹配 `^[0-9A-Za-z-]{8,64}$`。
- `dialogue_mode=auto`：禁止 `lines`，使用 `voice_line_provenance` 中 `kept=true` 的内部 ASR 行；启用 vocal filter 时每句必须为 `spoken` 或 `sung`。
- `dialogue_mode=edit|custom`：必须有非空 `lines`；每项只能含 `text/start_s/end_s`，文本非空、时间有序且落在实际视频时长内。`edit` provenance 固定 `asr+edited`，`custom` 固定 `manual`。
- `dialogue_mode=none`：禁止 `lines`，有效台词为空。
- `aspect_ratio` 只允许 `16:9|9:16`，`resolution` 只允许 `480p|768p`；两者都必须显式提交。
- 所选画幅的 `fit_required=false` 时只允许 `fit_mode=none`；为 true 时只允许 `crop` 或 `pad`。该值只在 pipeline `done` 时按实际 H3 输入帧计算，浏览器不能从源媒体自行推断。

长视频请求严格为：

```json
{
  "confirm": true,
  "client_request_id": "request-123456",
  "dialogue_mode": "auto",
  "fit_mode": "none",
  "aspect_ratio": "9:16",
  "resolution": "768p",
  "expected_plan_receipt": "64 lowercase hex characters",
  "fast_mode": true
}
```

长视频允许上述八个键；当前网页新建长视频固定发送 `fast_mode=true` 且不提供选择控件。后端契约不变：`fast_mode` 可缺失且安全等价 `false`，显式 `false` 仍受支持，出现时必须是 JSON boolean，非法类型返回 422 `invalid_fast_mode`。`dialogue_mode` 只能为 `auto`（复用源音轨；长于画面时裁剪、短于画面时补静音，画面时长不变）或 `none`（静音）；不接受 `lines`、`edit/custom`，也不接受创建阶段的 rewrite/translate。`expected_plan_receipt` 必须是 detail 当前返回的 64 位小写十六进制值。服务在任何付费 POST 前用它做 CAS，并重新校验 plan、meta、关键帧、提示词和文件哈希；SHA 校验、优化帧选择、画幅派生与 H3 请求消费同一份不可变 bytes，不会在校验后重新读取路径。所有段共用冻结的画幅、清晰度和 fit。历史未冻结 null 会话从安全 plan 纯派生，无法派生时付费前拒绝；已有冻结提交继续以原参数和 workflow 为准。

旧标签页可能仍按四键长视频契约提交，缺少 `expected_plan_receipt`。服务仅对这个精确旧请求返回结构化 `409 client_refresh_required`，提示刷新页面；不会自动采用服务端当前 receipt，也不会创建付费任务。`/`、`/index.html` 和 `/app.js` 均使用 `Cache-Control: no-store`，刷新后会取得当前提交契约。

接受后返回 `202 {"status":"queued","attempt":N}`。后台状态写入 `generation`，客户端轮询 detail。

门控和错误：

| HTTP | detail | 条件 |
| --- | --- | --- |
| 501 | `H3 submission is disabled.` | `ENABLE_H3_SUBMIT` 未开启；此门在会话查找前 |
| 404 | `not found` | cid 不存在/非法 |
| 409 | `read_only` | 非 schema v2 |
| 409 | `confirmation required` | 非严格 true |
| 409 | `client_refresh_required` + 中文 `message` | 精确旧版四键长视频请求；刷新页面，不产生付费任务 |
| 422 | `invalid_submit_request` | 出现未知键 |
| 422 | `invalid_client_request_id` | id 不合规 |
| 422 | `invalid_dialogue` | mode、lines 形状或台词内容不合规 |
| 422 | `invalid_aspect_ratio` / `invalid_resolution` | 语义画幅或清晰度缺失/越界；claim 和供应商 POST 前拒绝 |
| 422 | `long_video_audio_mode_unsupported` | 长链使用了非 keep 创建模式、edit/custom 或 lines |
| 422 | `invalid_plan_receipt` | 长链缺失或 plan receipt 格式非法 |
| 422 | `invalid_fast_mode` | 长链 fast_mode 不是 JSON boolean |
| 422 | `invalid_fit_mode` / `fit_mode_required` / `fit_mode_not_allowed` | 画幅选择不合规 |
| 409 | `artifacts not ready` | 输入准备 status 不是 done |
| 503 | `h3_credentials_missing` | AutoDL 凭据缺失 |
| 503 | `h3_configuration_invalid` | 冻结后无法构造合法 H3Request/timeout 配置 |
| 409 | `prepared_input_invalid` / `frame_fit_failed` | 冻结输入或画幅派生失败 |
| 409 | `fit_requirement_unknown` | 历史长会话的 plan/anchors 无法安全派生画幅要求；不提交付费任务 |
| 409 | `long_video_plan_changed` / `long_video_plan_invalid` | 长链 CAS 不匹配或 plan/文件绑定无效 |
| 409 | `generation in progress` / `already submitted` | active/succeeded 使用不同 id |
| 409 | `new client_request_id required` | H3 阶段确定 failed 后复用旧 id；长链 `stage=stitch` 除外 |
| 409 | `resume_request_id_mismatch` | resume_required 没有使用原 client_request_id |
| 409 | `resume_parameters_changed` | active/resume/retry/stitch-retry 的 mode、归一化 lines、画幅、清晰度或 fit 与冻结值不一致 |
| 409 | `submission_outcome_unknown` | 既有 generation 为 submission_unknown；任意 id 均拒绝 |
| 409 | `generation_state_invalid` | 已持久化 generation status/attempt 不满足安全状态形状 |

相同 id 在 `queued/running/succeeded` 时返回现有 `{status,attempt}`，不重复 POST。确定 `failed` 的短链或长链 H3 阶段只有新 id 才进入人工 retry。长链 retry 从冻结的 `generation.segments` 复用状态和文件均完整的 `succeeded` 段，把其余段重置为待生成；前端直接展示服务端 `retry_paid_segment_count`。若长链为 `failed + stage=stitch`，必须复用原 `client_request_id` 和冻结参数，只重新执行本地拼接，attempt 不递增且新增付费 H3 子任务数为 0。即使半发布留下可播放的会话级 `generated.mp4`，详情仍同时公开 `has_video=true` 与拼接恢复状态。

旧 Context IR 契约下未完成的 generation 对外固定映射为 `failed / generation_path_removed`；历史成片保持可读。用户用新 id 重试时重写为当前直接 H3 receipt，不再恢复或查询 MiniMax task。

`resume_required` 表示 H3 provider task 已知：只接受原 `client_request_id`，且 dialogue mode、标准化 lines、`fit_mode`、画幅、清晰度及长链 plan receipt 必须与冻结输入完全一致。合法继续返回 `202 {"status":"queued","attempt":<原值>}`，不重写 receipt、不递增 attempt。已知 task 错误包括 `h3_query_failed/h3_timeout/download_failed/download_dns_failed/download_peer_unverified/output_write_failed/output_probe_failed`；`h3_running` 也进入此状态。长链启动恢复只 GET 已持久化子任务，不会 POST 尚未开始的段。

确定性输出安全拒绝 `download_url_rejected/download_redirect_rejected/download_too_large/download_invalid_video` 映射为 `failed`，只有用户明确使用新 id 才创建 retry attempt。它们不属于会因同参数继续而消失的传输故障。

`submission_unknown` 是唯一完全锁死状态：前端隐藏操作，服务端对任何 id 返回 409，必须先在 AutoDL 侧核对原 POST 是否已创建任务。

### `PATCH /api/conversations/{cid}/image-optimization-prompt`

严格请求：

```json
{
  "confirm": true,
  "segment_index": 0,
  "expected_sha256": "64-hex",
  "prompt": "本段全部关键帧共享的图片优化提示词"
}
```

短视频只能使用逻辑段 `0`；长视频使用连续正整数段 `1..N`。接口只允许 schema v2 分析已完成且 generation/postprocess 都尚未创建时调用，以 `expected_sha256` CAS 保存；未知字段、空白或超过 32 KiB 的提示词、非法段号在写入前拒绝。摘要漂移返回结构化 409 `image_optimization_prompt_changed`；输入已冻结返回 `image_optimization_prompt_frozen`。成功返回该段新的 `{text,default_text,sha256}`。恢复默认由客户端把 `default_text` 放入草稿，仍需调用本接口保存。

视觉关键帧冻结后，隔离 Codex 按段执行 `skills/image-postprocess`，只接收本段关键帧和后端编辑模式；其输出原样成为默认提示词，不读取或复制 H3 提示词。项目在分析完成时冻结同一个 Seedream 模型和模式，但这些内部字段不会出现在 detail。短视频在顶层返回 `image_optimization_prompt`；长视频把对应对象放入每个 `segments[]`。

### `POST /api/conversations/{cid}/postprocess`

可选三阶段关键帧后处理：

```json
{
  "confirm": true,
  "options": {
    "remove_subtitle": true,
    "remove_brand": false,
    "optimize_image": true
  }
}
```

顶层 key 必须恰为 `confirm/options`，canonical options 必须恰为三个 bool；精确旧两字段请求兼容为 `optimize_image=false`，其余未知、缺失或非 bool 返回 422。至少一项为 true。已知旧页面 option `change_bg/face_hold` 返回纯文本中文刷新提示，不写状态、不调用供应商；若同时混入其他未知字段仍 fail closed。每项还必须由 detail 的 `postprocess_capabilities` 允许：文字和品牌能力取决于 MediaKit 开关，图片优化能力取决于 `ARK_API_KEY`。

接受后冻结选项、每段已审阅提示词与后端模型/模式，返回 `{"status":"running","frames":[]}`。短视频按段 0，长视频按 1..N 并行；段内严格执行已选的 `full_screen_text_erase → full_screen_icon_erase → Seedream` 阶段屏障，帧请求受 MediaKit/Seedream 独立并发上限控制。图片模式为 `anchor_consistency` 时先用本段全部清理帧生成第一张锚帧，再并行处理剩余帧；`independent_parallel` 则每帧独立并行。供应商返回图统一转为源图精确尺寸 PNG，整段完成后才原子发布同名 canonical 文件。

detail 的 `postprocess` 为 `{status,options,frames,segments,error}`；每段只公开 `{index,status,stage,completed_frames,total_frames,revision,error}`。任一段失败不取消其他段，但整体为 failed，生成返回 409 `postprocess_not_ready`；全部完成后优化帧进入 H3 冻结输入。禁用返回 501，旧会话返回 409 `read_only`，输入未 done/正在运行返回 409；generation 已创建返回 409 `generation_already_started`；done 后改变冻结选项返回结构化 409 `postprocess_options_locked`。既有 failed 状态拒绝普通 POST 并返回 `postprocess_segment_retry_required`，避免重置 revision 或绕过分段 CAS。

### `POST /api/conversations/{cid}/postprocess/segments/{index}/retry`

请求 key 必须恰为 `{"confirm":true,"expected_revision":N}`，仅允许重试当前 failed 段；revision 漂移返回结构化 409 `postprocess_revision_changed`。服务复用项目冻结的选项、提示词、模型、模式和已完成本地产物，不接受客户端覆盖。未知提交的公共真源是分段 `error=submission_unknown`，不得从 `status/stage` 推断；用户明确确认潜在重复计费后才能人工调用。旧 revision 的 attempt 保留，启动恢复不会替用户调用该接口。

Seedream 每个帧 POST 前先持久化输入/提示词/模型/模式摘要。自动重试硬上限为总计 3 次，并且只认完整 HTTP 429、精确 `QuotaExceeded`、响应无 `data`；网络、超时、取消及其他不明结果写为 `submission_unknown` 且不自动重发。确定性 4xx/协议错误不重试；成功响应 bytes 已落盘时，恢复只做本地 PNG 发布。

`/`、`/index.html`、`/app.js`、`/styles.css` 的 GET/HEAD 响应（含条件请求的 304）均带 `Cache-Control: no-store`，避免 HTML、脚本和样式跨版本组合。

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
| `duration_s` | float/null | ffprobe `v:0` 视觉时长；音频/容器时长不参与画面规划和门禁 |
| `fit_required` | bool/null | 当前 `aspect_ratio` 对应 profile 的兼容投影 |
| `fit_profiles` | object/null | `16:9/9:16` 各自的 `fit_required/default_fit_mode`；pipeline 按实际 H3 输入计算 |
| `aspect_ratio`, `resolution` | str/null | pipeline 推荐，提交后冻结；闭集为 `16:9|9:16`、`480p|768p` |
| `dialogue_mode` | str | 默认 auto；最终提交选择 |
| `generation` | object/null | coarse H3 attempt 状态 |
| `segments` | list | 长链分段计划；短链为空 |

准备/提交后可增加：

| 字段 | 语义 |
| --- | --- |
| `voice_lines` | 当前有效裸台词 `{text,start_s,end_s}` |
| `voice_line_provenance` | 自动识别行 + `classification/provenance/kept` 决策 |
| `voice_text_normalizations` | 保留为空数组的兼容留痕；台词不再做供应商敏感词替换 |
| `vocal_filter_enabled`, `voice_warnings`, `voice_lines_vocal_dropped` | 声学过滤留痕 |
| `prepared_dialogue` | 本次提交冻结的完整有效台词（含 classification/provenance） |
| `prepared_input_receipt` / `receipt_version` | `prepared_input.json` / 1 |
| `long_video_plan_receipt` | 长链固定为 `long_video_plan.json` |
| `frozen_plan_receipt` | 长链首次提交确认的 plan SHA-256 |
| `fit_mode` | `none/crop/pad`；随冻结画幅解释 |
| `generation` | `{status,error,attempt,client_request_id,stage}`；长链另含冻结 boolean `fast_mode`、内部 `segments` 与 `fit_layout`，failed 时公开 `retry_paid_segment_count`，status 含 resume_required；历史缺 fast_mode 等价 false |
| `_image_optimization` | 私有项目冻结 receipt：同一模型/模式及每段 Codex 产出的 default/current/SHA；只投影用户可编辑的提示词字段 |
| `_postprocess_receipt` | 私有执行 receipt：冻结三选项与图片优化设置，不进入公开 detail |
| `postprocess` | `{status,options,frames,segments,error}`；存在时生成必须等待全部段 done，并使用完整优化帧集合 |

`≤15s` 的新 schema v2 使用顶层 keyframes/prompt；`>15s` 使用 `segments` 与 `long_video_plan.json`，每段独立工作目录和生成状态。历史 11–15 秒长链仍按已冻结计划恢复，两种契约不能互相降级或混用 receipt。

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
    "h3": {
      "workflow": "minimax_h3_lightx2v_v5_15s",
      "duration": 10,
      "aspect_ratio": "9:16",
      "resolution": "768p",
      "provider_resolution": "768p竖"
    }
  }
}
```

严格要求顶层和子对象 key 集合完全匹配；绑定路径必须位于会话根内；关键帧 1–9 张且路径唯一；每个文件重读校验 SHA-256；台词 canonical JSON、provenance 和 hash 必须匹配；最终 prompt 必须等于当前视觉 prompt + 当前结构化发声块。任何偏差抛 `PreparedInputError`，HTTP 层归一为 409 `prepared_input_invalid`。

`normalized_audio=null` 是无音轨的合法表示。无需适配时，优化后的 keyframe binding 可直接指向 `work/postprocessed/`；crop/pad 时则指向由所选原图或优化图生成的 `work/h3_frames/<aspect>/<mode>/`。历史 receipt 缺少语义参数时仅按原 `ratio=9:16`、`resolution=768p竖` 精确恢复，loader 不重写文件。

## Long-video plan receipt v2（兼容只读恢复 v1）

`data/<cid>/long_video_plan.json` 是 canonical JSON。新计划固定 `schema=duet.long-video-plan`、`version=2`；顶层绑定完整 source 的路径/SHA-256、实际总时长、`workflow=minimax_h3_lightx2v_v5_15s` 和有序 segments。每段严格连续覆盖 `[0,duration_s]`，六位小数边界归一后的长度至少 1 秒，provider 整秒时长不得超过 14 秒，并绑定：

- `index/start_s/end_s/chain_id/join_mode`；首段必须 `hard_cut`，后续为 `hard_cut` 或 `continue`；
- 分段 source、1–9 张关键帧、`first/end` 两张锚点及其 SHA-256；
- `visual_prompt`、最终 `prompt` 的路径/SHA-256；
- 本段局部台词的 canonical 数量与 SHA-256。

detail 的 `plan_receipt` 是整个文件的 SHA-256，而不是 receipt 内字段。提交时服务同时比对该摘要、meta 分段和全部 artifact；`fit_mode=crop/pad` 时冻结请求使用由本段最终选中关键帧派生的 `16:9/9:16` 输入。每段工作目录为 `work/segments/<N>/`，其中 `.h3/` 只归该子任务所有。当前 generation 在任何供应商 POST 前持久化内部 `workflow` 与 `fit_layout=legacy-v0|aspect-v1`，恢复只按冻结值读取。旧 boundary attempt 没有 workflow marker 时按 plan 中的首尾帧模式恢复，不切换接口。

历史 `version=1` receipt 仍可读取并按其原始浮点换算重建最长 15 秒的已知 attempt；超过 10 秒的历史段只能 GET 恢复，绝不创建新 POST，也不改写原 receipt。

## H3 attempt state v1

短链路径 `.h3/attempts/<六位递增号>/attempt.json`；长链路径 `work/segments/<N>/.h3/attempts/<六位递增号>/attempt.json`。最小形状：

```json
{
  "schema_version": 1,
  "cid": "...",
  "attempt_id": "000001",
  "client_request_id": "request-123456",
  "input": {},
  "input_receipt": "sha256",
  "status": "h3_submitting",
  "retryable": false,
  "h3": {"status": "submitting"}
}
```

内部 status：`ready_to_submit/h3_submitting/h3_running/succeeded/retryable_failure/failed/submission_unknown`。`ready_to_submit + h3.status=ready` 证明 exact input receipt 已落盘且尚未进入 POST；POST 前先原子改为 `h3_submitting`，因此无 task id 的 submitting 仍必须判为未知。当前 input manifest 与 attempt receipt 同时冻结语义 `aspect_ratio/resolution` 和 provider `resolution`；唯一投影为 `480p横/480p竖/768p横/768p竖`，provider body 只使用投影值。任务 id 出现后必须同时存在对应 receipt；最终 output receipt 只能是 `{name:"generated.mp4",sha256,size}`。`result_url` 明确禁止落状态文件。

### start / inspect / resume / retry

- `h3.start(request)`：同 client id 幂等；新 attempt 允许一次 H3 POST，已有 attempt 只按已知状态推进。
- `h3.prepare(request)`：只创建/校验 unpaid attempt 与 input receipt，绝不联网；返回 `not_started` 表示可安全提交。
- `h3.submit(request)`：只接受已 prepare 的 exact attempt，最多跨越一次 POST 边界，task id/receipt 落盘后立即返回，不做 GET 轮询；无 prepare、输入漂移或无 task id 的 submitting 状态都 fail closed。
- `h3.inspect(request)`：纯读，不写、不联网；验证 session/attempt receipts。
- `h3.resume(request)`：获取会话 flock 后默认 GET-only 推进已有任务；只有完整确认的 `h3_provider_failed` 有额度，或其后已落盘的 `ready_to_submit/h3.ready` 自动 attempt，才允许跨越新的 POST 边界。
- `h3.retry(request,new_id)`：底层显式创建人工新逻辑请求的 attempt；短链 runtime 只在普通确定失败或 provider 自动额度耗尽后，由用户提交新 id 调用。长链同样要求新父 id并只重做未成功段；`stage=stitch` 失败复用原父 id，且只运行本地拼接。`resume_required` 调 `start` 续同 attempt，`submission_unknown` 不调用。

`start/resume/retry` 共用同一私有自动推进器；`submit` 始终只提交当前 prepared attempt 一次。自动额度只统计同 `client_request_id + input_receipt` 的有效 attempt 链，已创建的 ready attempt 已占额度，总上限为 `1 + retry_count`。每个新 attempt 先原子写 receipt 再 POST；配置降低只停止新增，提高后可从最新完整 provider failure 继续。`h3_submit_rejected/h3_result_missing`、输入与下载安全拒绝不创建新 attempt；查询、超时和下载传输失败继续同 task；`submission_unknown` 永不重复 POST。

短链供应商顺序固定：把 1–9 张冻结帧以 data URL 和冻结源提示词一起 `POST /api/v1/comfyui/comfyui_workflow/minimax_h3_lightx2v_v5_15s` → GET 结果 → 安全下载并原子写会话级 `generated.mp4`。新长链每段同样使用 `minimax_h3_lightx2v_v5_15s`，传该段冻结的 1–9 张 `ref_image_N`；历史 `minimax_h3_lightx2v_v5` 多图回执和 `minimax_h3_lightx2v` 首尾帧回执均保持原工作流恢复，不迁移付费提交。

默认长链同一 `chain_id` 严格串行，最多并发两条 chain；每段请求只使用本段冻结参考图。快速长链先预构造并 prepare 全部分段，随后最多 8 个短生命周期 worker fan-out POST，再最多 8 个 worker GET 轮询；队列可以超过 8，但不会创建无界长生命周期线程。单段确定失败只重做该段，已成功下游独立复用；任一子任务 `submission_unknown` 锁住整批且该段不再 POST，快速模式仍让已知兄弟 task 完成 GET。全部成功后 ffmpeg 移除 H3 子片段音轨、归一为 24fps H.264/yuv420p 并按顺序拼接；`continue` 边界去除后一段首个解码帧。`auto` 复用 source 音轨，以视频 presentation start 归零并由解码器处理 AAC/Opus priming，再按解码后的音频时间戳补前置静音或裁视频零点前音频，最后在画面终点裁剪或补静音，画面时长不变；`none` 为静音。拼接失败以同一父请求仅重跑本地拼接。

短链 reference 输出仍要求目标时长 ±0.5 秒。长链分段无论 reference 或历史 boundary，在 attempt receipt、输入和输出 hash 完整匹配后，都按供应商整秒请求验收：不得比源片段目标短超过一帧，也不得比整秒请求长超过 1 秒。该容差不改变最终时长契约，stitch 仍按每个 source segment 的 24fps 帧预算精确裁剪或补齐并验证全片。

输出下载门禁：只接受无 userinfo 的 HTTPS；hostname/IP 预解析必须全为公网地址，私网、loopback、local、reserved、multicast 均确定拒绝。owned httpx client 使用 `trust_env=false`；响应后在读取 status/body 前通过 `extensions.network_stream.get_extra_info("server_addr")` 验证实际 socket peer 也是公网地址。实际 peer 为私网仍是 `download_url_rejected`；DNS 临时失败为 `download_dns_failed`，缺失/异常/非 IP peer 为 `download_peer_unverified`，后二者同 id 恢复。

显式不跟随 3xx；Content-Length 和实际流都不得超过 `200 * 1024 * 1024` 字节。内容先以 0600 写同目录临时文件。ffprobe 缺失、OS 错误或超时为可恢复 `output_probe_failed`；ffprobe 正常执行但媒体无法解析、无 `v:0` 或其 `duration`/`duration_ts*time_base` 非正有限为确定失败 `download_invalid_video`。验证通过后才 `os.replace`、目录 fsync。状态只保存输出 name/SHA-256/size，不保存 URL。

## Python 公开接口

- `app.main.create_app(settings) -> FastAPI` — 应用工厂；启动时扫描可恢复 generation。
- `app.storage.new_conversation(...) -> dict` — 创建 schema v2 meta 和 `work/`。
- `app.pipeline.run(settings,cid,runner) -> None` — 输入准备；失败只写 meta，不向调用者抛。
- `app.prepared_input.prepare_dialogue(...) -> tuple[dict,...]` — 规范化 auto/edit/custom/none 来源。
- `app.prepared_input.write_prepared_input(...) -> PreparedInput` — 原子写 receipt 并复用 loader 验证。
- `app.prepared_input.load_prepared_input(...) -> PreparedInput` — 读取冻结 bytes，漂移即拒绝。
- `app.frame_fit.fit_frames(paths,output_dir,mode,aspect_ratio) -> tuple[Path,...]` — 调用方必须显式传 `16:9|9:16` 目标；crop/pad 都严格输出该比例，没有隐藏的固定 9:16 默认值。
- `app.h3.prepare/submit/start/inspect/resume/retry` — 可恢复状态机；prepare/submit 为快速 fan-out 分离 unpaid receipt 与单次 POST；resume 默认 GET-only，唯一新 POST 例外是严格验证且有额度的 provider terminal failure；runtime 对 resume_required 暴露同 id 继续，对其他确定 failed 暴露新 id retry，不对 submission_unknown 暴露操作。
- `app.long_video.plan_segments/write_plan_receipt` — 安全边界规划与 canonical plan 落盘。
- `app.long_generation.freeze_plan/run` — fail-closed 冻结、默认最多两链编排、可选快速 fan-out 和分段恢复。
- `app.stitch.stitch_video` — 本地确定性归一化、音轨选择、拼接与 receipt。
- `app.image_optimization.freeze_prompts/replace/public_prompts` — 冻结每段共享提示词并执行 SHA CAS，只公开安全投影。
- `app.seedream.edit` — 一输出图片编辑、精确尺寸 PNG、持久 attempt 与结果未知门控。
- `app.postprocess.start/run_task/retry_segment/generation_keyframes/recover_running` — 三阶段分段编排、定向重试、启动恢复与 H3 完整性门控。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ACCESS_TOKEN` | 无，必填 | 共享 Bearer 口令 |
| `ENABLE_PIPELINE` | `1` | 生产是否启动输入准备；直接构造 Settings 的测试默认 false |
| `DATA_DIR` | `data` | 会话根目录 |
| `MAX_UPLOAD_MB` | `500` | 上传/URL 下载上限 |
| `CODEX_TIMEOUT_S` | `1800` | 单次 codex 硬超时 |
| `AUTO_RETRY_COUNT` | `2` | 同 task 的瞬时操作重试次数、完整确认 `h3_provider_failed` 后的额外 attempt 数，以及 MediaKit 明确 HTTP 429 `RequestLimitExceeded` 的额外尝试数；默认最多 3 次 POST |
| `AUTO_RETRY_INTERVAL_S` | `15` | 上述瞬时操作、provider failure 或 MediaKit 明确限流重试前的固定等待秒数 |
| `CODEX_CONCURRENCY` | `10` | pipeline 并发闸 |
| `MAX_QUEUED` | `100` | queued 会话上限 |
| `VOCAL_FILTER` | `on` | off/false/0 才关闭 keep/drop；未知值保持开启 |
| `ASR_CLI` | `/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli` | 本地多语种 whisper.cpp 可执行文件 |
| `ASR_MODEL` | `/home/xy/.local/share/duet-asr/ggml-small.bin` | multilingual small 模型；运行时不得下载 |
| `ASR_TIMEOUT_S` | `600` | 单次本地听写超时 |
| `ASR_THREADS` | `4` | 单次本地听写 CPU 线程数 |
| `ENABLE_H3_SUBMIT` | false | H3 提交总开关 |
| `AUTODL_ART_TOKEN` | 空 | AutoDL H3 凭据 |
| `H3_REQUEST_TIMEOUT_S` | `30` | 普通供应商请求超时 |
| `H3_POLL_TIMEOUT_S` | `1500` | H3 轮询总时限 |
| `H3_DOWNLOAD_TIMEOUT_S` | `180` | 成片下载超时 |
| `H3_POLL_INTERVAL_S` | `3` | 两次查询间隔，可为 0 |
| `ENABLE_MEDIAKIT_ERASE` | false | 可选关键帧擦除开关 |
| `VOLC_MEDIAKIT_API_KEY` | 空 | AI MediaKit Bearer 凭据，不进入公开 API/meta/日志 |
| `MEDIAKIT_CONCURRENCY` | `4` | 帧级并发，最小钳制为 1 |
| `MEDIAKIT_TIMEOUT_S` | `180` | 上传、擦除和结果下载请求超时 |
| `ARK_API_KEY` | 空 | 火山方舟 Seedream Bearer 凭据；留空关闭图片优化 capability，不进入 Settings repr、公开 API/meta/日志 |
| `SEEDREAM_MODEL` | `doubao-seedream-5-0-260128` | 项目级图片模型；allowlist 另含 `doubao-seedream-4-5-251128`、`doubao-seedream-4-0-250828` |
| `SEEDREAM_EDIT_MODE` | `anchor_consistency` | `anchor_consistency` 或 `independent_parallel`；按项目冻结，不暴露前端 |
| `SEEDREAM_CONCURRENCY` | `4` | Seedream 帧级进程内并发上限，必须为正整数 |
| `SEEDREAM_TIMEOUT_S` | `180` | 单次 Seedream POST 超时秒数，必须为正有限数 |
| `TIKTOK_PROXY` | 空 | TikTok/DoH 下载代理 |
| `DOWNLOAD_TIMEOUT_S` | `120` | URL 下载整体时限 |
| `HOST` / `PORT` | `0.0.0.0` / `3211` | `run.sh` 默认；生产 unit 必须覆盖为 `127.0.0.1/3212` |

全部服务环境只放 `%h/.config/duet-ad1/service.env` 这一份 0600 EnvironmentFile，不写 unit、仓库或命令行。部署步骤见 [.deploy/runbook.md](../../../.deploy/runbook.md)。

## 依赖

- Python：FastAPI、uvicorn、httpx、OpenCV、ai-edge-litert 等（以 `requirements.txt` 为准）。
- 可执行：`ffmpeg`、`ffprobe`、已认证的 `codex` CLI。
- 外部服务：AutoDL Art H3；可选 AI MediaKit 图像擦除与火山方舟 Seedream 图片编辑。
- 运行约束：Linux `flock/fsync` 语义、单 uvicorn 进程、Caddy 本机反代。
