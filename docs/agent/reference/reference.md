---
name: h3-runtime
type: reference
status: done
owner: agent
updated: 2026-08-28
tdd: N/A
links: [conversation-task, app/main.py, app/h3.py, app/prepared_input.py, app/long_generation.py]
---

# H3 runtime · 接口（How/Now）

## HTTP 通则

- `/api/health`、`/api/login` 无需鉴权；其余 `/api/conversations*` 使用 `Authorization: Bearer <ACCESS_TOKEN>`。
- JSON 错误 detail 为安全字符串或 `{code,message}`。结构化冲突只公开固定 code/中文提示；供应商响应正文、URL、token 和本机路径不作为公开错误。
- H3 提交是 202 + 轮询；`POST /submit` 不等待最终视频。

### Current v4 single operation

current v4 的唯一 create path 是 `segments[N>=1] + Fusion v3 + backend Ref2VA compiler + Context local identity + H3 + EDL`。技术验收 A 接受后，所有阶段共用同一 `operation_id=cid` 并自动延续：

```text
202 {operation_id:<cid>,status:"running",stage:<current-stage>}
  -> same accepted claim, no refresh or second submit
200 {operation_id:<cid>,status:"succeeded",stage:"commit_b"}
```

内部 Fusion refresh/CAS、image acceptance 冲突和质量诊断不会成为 current 的公开 409/刷新步骤；重放只确保同一 operation。Fusion v3 每段只输出 `{index,visual[]}`，后端独占 Ref2VA provider prompt 的编译权。Context 将该 prompt 原样绑定为 `local:identity:<sha256>`，effective prompt 同 SHA，HTTP 调用数为 0。

current 每段固定 exact 3 个 Picture reference，允许极短连续 scene 重复最近合法源帧并绑定 provenance；H3 `reference_audios=()`，源音频只供 ASR/YAMNet 分析。成片音频只取 H3 原生音轨，缺音轨的 segment 在同一 EDL 补静音，源音频不回挂或 overlay。quality score/diagnostics 只用于测试和 Skill 迭代，不阻断、不重试、不产生 fallback。

Fusion v1、旧 Context HTTP、多模态 source-audio reference、旧 short/long、speaker visibility 与 quality-verdict receipt 均只读；历史已知 provider task 只按原 receipt GET 恢复，不能迁移为 current 或新建 fallback POST。

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

current v4 detail 统一投影 `segments[N>=1]`、current plan receipt、Fusion production binding、operation stage 和最终媒体状态。`N=1` 也不公开历史 short prompt 编辑合同。`plan_receipt` 是 canonical `long_video_plan.json` 的 SHA-256，`segment_count` 来自冻结计划；公开 segment 只含安全状态投影，不泄漏 task id、内部 child id 或文件路径。`read_only` 由历史 schema/receipt 派生；历史详情和成片可读，但不能创建或迁移为 current。成功页只从服务端冻结值生成摘要。

### `PATCH /api/conversations/{cid}/prompt`

该接口只属于历史 prepared-input/short receipt；current v4 不允许用它改写 Fusion visual 或 backend Ref2VA prompt。历史 CAS 结果只供诊断，不是 current A→B 的刷新推进步骤。

### `GET /api/conversations/{cid}/files/{name}`

白名单：`source.mp4`（映射唯一 `source.*`）、`preview.mp4`、`generated.mp4`、`contact_sheet.jpg`、`keyframes/<basename>`、`postprocessed/<basename>`，以及长视频 `segments/N/work/{keyframes|postprocessed}/<basename>`。路径穿越、非白名单或文件不存在均 404。

### `POST /api/conversations/{cid}/submit`

current v4 使用下方带 `expected_plan_receipt` 的统一分段请求；`N=1` 也不进入独立 short 实现。请求 shape、枚举和 receipt 在 A 前校验；A 一旦接受，响应投影为上文 single operation，不再以 refresh/409 要求客户端推动内部阶段。

以下 short prepared-input 请求仅用于历史只读合同说明，不是 current create path：

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

current v4 请求严格为：

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

统一分段请求允许上述八个键；`fast_mode` 缺失等价 `false`，显式值必须是 JSON boolean。`dialogue_mode` 当前只允许 `auto|none`：`auto` 只把冻结 spoken 文本/时间投影进 Ref2VA，`none` 使用空台词；两者都固定 `voice_references=[]`，不会复用源音轨或发送 source audio reference。`expected_plan_receipt` 是当前 canonical plan 的 64 位小写 SHA-256。服务在任何付费 POST 前重新校验 plan、meta、exact-3 图片、Fusion input/output、Ref2VA prompt 和文件哈希；所有段共用冻结画幅、清晰度和 fit。

旧标签页的四键请求和 `client_refresh_required` 只属于历史兼容诊断，不是 current 推进方式。历史项目/receipt 只读；current v4 的 accepted A 由同一后台 continuation 自动推进。

接受后返回 `202 {"operation_id":"<cid>","status":"running","stage":"..."}`。后台在同一 accepted claim 上继续；最终成片有效后同一接口读取得到 `200 succeeded / commit_b`。

下表是底层技术/历史诊断，不是 current A→B 的客户端推进状态；current operation 对内部 refresh/CAS 保持 `running`：

| HTTP | detail | 条件 |
| --- | --- | --- |
| 501 | `H3 submission is disabled.` | `ENABLE_H3_SUBMIT` 未开启；此门在会话查找前 |
| 404 | `not found` | cid 不存在/非法 |
| 409 | `read_only` | 非 schema v2 |
| 409 | `confirmation required` | 非严格 true |
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

旧 Context HTTP、旧 multimodal audio 与 Fusion v1 receipt 只读；历史成片保持可读，已有 task 只按原 receipt GET 恢复。它们不得用新 id 迁移成 current、覆盖 prompt 或创建 fallback POST。current v4 Context 始终是同字节 local identity。

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

该接口只保留给历史 v1 continuity receipt：短视频使用逻辑段 `0`，长视频使用连续正整数段 `1..N`，并继续以 `expected_sha256` CAS 保存而不迁移旧 receipt。存在有效 v2 双目标 plan 时提示词由后端确定性编译，接口固定返回 409 `image_optimization_prompt_compiled`；v2 plan 损坏返回 `image_optimization_plan_invalid`，绝不回退 v1。未知字段、空白或超过 32 KiB 的旧提示词、非法段号在写入前拒绝。

current v4 的 `skills/image-postprocess` 只以 `phase=plan` 输出通用人物、持久实体、场景和逐帧可见状态语义；后端补齐实体 ID、所有权关系图及其他结构字段，绑定 source scene/time/transition 与 SHA，并确定性编译逐帧 Seedream prompt。缺失语义使用 `source_preserve` 继续并写 diagnostics；semantic compiler 的 `score/issues/ignored_mechanical_fields` 只进入日志、测试断言和 Skill 迭代，不参与生产控制流。旧 `_image_continuity` 与 quality-verdict receipt 只读，不迁移或重写。

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

接受后冻结选项、每段已审阅提示词与后端模型/模式，返回 `{"status":"running","frames":[]}`。短视频按段 0，长视频按 1..N 并行；段内严格执行已选的 `full_screen_text_erase → full_screen_icon_erase → Seedream` 阶段屏障，帧请求受 MediaKit/Seedream 独立并发上限控制。图片模式为 `anchor_consistency` 时先用本段全部清理帧生成第一张锚帧，再并行处理剩余帧；锚帧只约束目标帧中重复元素的替换身份和外观，不得成为构图或内容模板。`independent_parallel` 则每帧独立并行。供应商返回图统一转为源图精确尺寸 PNG，整段完成后才原子发布同名 canonical 文件。

detail 的 `postprocess` 为 `{status,options,frames,segments,error}`；每段只公开 `{index,status,stage,completed_frames,total_frames,revision,error}`。任一段失败不取消其他段，但整体为 failed，生成返回 409 `postprocess_not_ready`；全部完成后优化帧进入 H3 冻结输入。禁用返回 501，旧会话返回 409 `read_only`，输入未 done/正在运行返回 409；generation 已创建返回 409 `generation_already_started`；done 后改变冻结选项返回结构化 409 `postprocess_options_locked`。既有 failed 状态拒绝普通 POST 并返回 `postprocess_segment_retry_required`，避免重置 revision 或绕过分段 CAS。

### `POST /api/conversations/{cid}/postprocess/segments/{index}/retry`

请求 key 必须恰为 `{"confirm":true,"expected_revision":N}`，仅允许重试当前 failed 段；revision 漂移返回结构化 409 `postprocess_revision_changed`。服务复用项目冻结的选项、提示词、模型、模式和已完成本地产物，不接受客户端覆盖。未知提交的公共真源是分段 `error=submission_unknown`，不得从 `status/stage` 推断；用户明确确认潜在重复计费后才能人工调用。旧 revision 的 attempt 保留，启动恢复不会替用户调用该接口。

Seedream 每个帧 POST 前先持久化输入/提示词/模型/模式摘要。每个冻结请求总计只允许一次 POST；HTTP 429 `QuotaExceeded`、确定性 4xx 和协议错误都不自动重试，网络、超时、取消及其他不明结果写为 `submission_unknown` 且不自动重发。成功响应 bytes 已落盘时，恢复只做本地 PNG 发布。

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
| `segments` | list | current v4 固定非空；`N=1` 与 `N>1` 共用统一分段计划 |

准备/提交后可增加：

| 字段 | 语义 |
| --- | --- |
| `voice_lines` | 当前有效裸台词 `{text,start_s,end_s}` |
| `voice_line_provenance` | 自动识别行 + `classification/provenance/kept` 决策 |
| `voice_text_normalizations` | 保留为空数组的兼容留痕；台词不再做供应商敏感词替换 |
| `vocal_filter_enabled`, `voice_warnings`, `voice_lines_vocal_dropped` | 声学过滤留痕 |
| `prepared_dialogue` | 本次提交冻结的完整有效台词（含 classification/provenance） |
| `prepared_input_receipt` / `receipt_version` | `prepared_input.json` / 1 |
| `long_video_plan_receipt` | current v4 固定为 `long_video_plan.json` |
| `frozen_plan_receipt` | accepted A 绑定的 current plan SHA-256 |
| `fit_mode` | `none/crop/pad`；随冻结画幅解释 |
| `generation` | `{status,error,attempt,client_request_id,stage}`；长链另含冻结 boolean `fast_mode`、内部 `segments` 与 `fit_layout`，failed 时公开 `retry_paid_segment_count`，status 含 resume_required；历史缺 fast_mode 等价 false |
| `_image_optimization` | 私有项目冻结 receipt：同一模型/模式及每段 Codex 产出的 default/current/SHA；只投影用户可编辑的提示词字段 |
| `_image_continuity` | 多段项目私有全局元素映射：版本、完整段号、跨段元素及确定性 SHA；不进入 detail，短视频不存在 |
| `_postprocess_receipt` | 私有执行 receipt：冻结三选项与图片优化设置，不进入公开 detail |
| `postprocess` | `{status,options,frames,segments,error}`；存在时生成必须等待全部段 done，并使用完整优化帧集合 |

current v4 不按 15 秒复制合同：`≤15s` 是 `segments.length=1`，更长输入是 `segments.length>1`，都使用 v5 `long_video_plan.json`。顶层 keyframes/prompt、prepared input 和 v1-v4 plan 都是历史只读，不能混用或降级。

## Prepared input receipt v1（历史只读）

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

## Long-video plan receipt v5（current；v1-v4 只读）

`data/<cid>/long_video_plan.json` 是 canonical JSON。current Fusion v3 promotion 固定 `schema=duet.long-video-plan`、`version=7`；顶层绑定完整 source、实际总时长、统一 H3 workflow、Fusion production manifest 和有序 segments。每段严格连续覆盖 `[0,duration_s]`，provider 整秒时长不得超过 14 秒，并绑定：

- `index/start_s/end_s/chain_id/join_mode`；首段必须 `hard_cut`，后续为 `hard_cut` 或 `continue`；
- 分段 source、exact 3 张关键帧及各自 source time/scene/transition、兼容锚点和 SHA-256；
- Fusion visual prose、后端编译的最终 Ref2VA prompt 及 production manifest 的路径/SHA-256；
- 本段局部台词的 canonical 数量与 SHA-256。

detail 的 `plan_receipt` 是整个文件的 SHA-256，而不是 receipt 内字段。提交时服务同时比对该摘要、meta 分段和全部 artifact；`fit_mode=crop/pad` 时冻结请求使用由本段最终选中关键帧派生的 `16:9/9:16` 输入。每段工作目录为 `work/segments/<N>/`，其中 `.h3/` 只归该子任务所有。当前 generation 在任何供应商 POST 前持久化内部 `workflow` 与 `fit_layout=legacy-v0|aspect-v1`，恢复只按冻结值读取。旧 boundary attempt 没有 workflow marker 时按 plan 中的首尾帧模式恢复，不切换接口。

`version=1..4` 及 Fusion v1 receipt 均只读；已有 task 只按原输入与 workflow GET 恢复，不创建、迁移或改写为 v5，也不作为 current fallback。

## H3 attempt state v1

current v4 路径为 `work/segments/<N>/.h3/attempts/<六位递增号>/attempt.json`；顶层 `.h3/` 是历史 short 布局。最小形状：

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
- `h3.resume(request)`：获取会话 flock 后仅以 GET/download 推进已有 task；失败或历史 ready 后续 attempt 都不允许跨越新的 POST 边界。
- `h3.retry(request,new_id)`：底层显式创建人工新逻辑请求的 attempt；短链 runtime 只在普通确定失败后，由用户提交新 id 调用。长链同样要求新父 id并只重做未成功段；`stage=stitch` 失败复用原父 id，且只运行本地拼接。`resume_required` 调 `start` 续同 attempt，`submission_unknown` 不调用。

`start/resume/retry` 共用单-attempt 推进器；`submit` 始终只提交当前 prepared attempt 一次。每个新 attempt 先原子写 receipt 再 POST，只有显式 `retry(request,new_id)` 或经精确证据授权的受控存储拒绝接口能创建后续 attempt。`retry_count` 仅用于同 task 的 GET、下载和本地可重试操作。`h3_provider_failed/h3_submit_rejected/h3_result_missing`、输入与下载安全拒绝不自动创建新 attempt；查询、超时和下载传输失败继续同 task；`submission_unknown` 永不重复 POST。

current v7 每段把 exact 3 张冻结 Picture 与后端编译的 Ref2VA prompt 提交同一 H3 workflow；`reference_audios=()`，Context receipt 是同字节 local identity。历史 1–9 图、多模态 source-audio 或首尾帧 receipt 只按原 workflow GET 恢复，不迁移付费提交。

默认同一 `chain_id` 严格串行，快速模式先预构造并 prepare 全部分段，再有界 fan-out POST/GET。成功兄弟独立复用，`submission_unknown` 段不再 POST。全部成功后按统一 EDL 归一 24fps H.264/yuv420p：有音轨段使用 H3 原生音频，无音轨段补有限静音；源音频、source reference 与 conditioning audio 从不回挂或 overlay。拼接失败只重做本地 EDL。

current 每个 segment 在 attempt/input/output receipt 完整匹配后按供应商整秒请求验收，并由 EDL 按 source segment 的 24fps 帧预算精确裁补；历史 short/boundary 输出只按其冻结容差读取。

输出下载门禁：只接受无 userinfo 的 HTTPS；hostname/IP 预解析必须全为公网地址，私网、loopback、local、reserved、multicast 均确定拒绝。owned httpx client 使用 `trust_env=false`；响应后在读取 status/body 前通过 `extensions.network_stream.get_extra_info("server_addr")` 验证实际 socket peer 也是公网地址。实际 peer 为私网仍是 `download_url_rejected`；DNS 临时失败为 `download_dns_failed`，缺失/异常/非 IP peer 为 `download_peer_unverified`，后二者同 id 恢复。

显式不跟随 3xx；Content-Length 和实际流都不得超过 `200 * 1024 * 1024` 字节。内容先以 0600 写同目录临时文件。ffprobe 缺失、OS 错误或超时为可恢复 `output_probe_failed`；ffprobe 正常执行但媒体无法解析、无 `v:0` 或其 `duration`/`duration_ts*time_base` 非正有限为确定失败 `download_invalid_video`。验证通过后才 `os.replace`、目录 fsync。状态只保存输出 name/SHA-256/size，不保存 URL。

## Python 公开接口

- `app.main.create_app(settings) -> FastAPI` — 应用工厂；启动时扫描可恢复 generation。
- `app.storage.new_conversation(...) -> dict` — 创建 schema v2 meta 和 `work/`。
- `app.pipeline.run(settings,cid,runner) -> None` — 输入准备；失败只写 meta，不向调用者抛。
- `app.prepared_input.*` — 历史 short receipt 只读接口，不参与 current v4 create。
- `app.frame_fit.fit_frames(paths,output_dir,mode,aspect_ratio) -> tuple[Path,...]` — 调用方必须显式传 `16:9|9:16` 目标；crop/pad 都严格输出该比例，没有隐藏的固定 9:16 默认值。
- `app.h3.prepare/submit/start/inspect/resume/retry` — 可恢复状态机；prepare/submit 为快速 fan-out 分离 unpaid receipt 与单次 POST；resume 默认 GET-only，唯一新 POST 例外是严格验证且有额度的 provider terminal failure；runtime 对 resume_required 暴露同 id 继续，对其他确定 failed 暴露新 id retry，不对 submission_unknown 暴露操作。
- `app.long_video.plan_segments/write_plan_receipt` — 安全边界规划与 canonical plan 落盘。
- `app.long_generation.freeze_plan/run` — fail-closed 冻结、默认最多两链编排、可选快速 fan-out 和分段恢复。
- `app.stitch.stitch_video` — 本地确定性归一化、音轨选择、拼接与 receipt。
- `app.image_optimization.compile_semantic_plan/compile_frame_prompts` — current v4 后端结构补齐与确定性 Seedream compiler；diagnostics 只供日志/测试迭代。
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
| `SEEDREAM_MODEL` | `doubao-seedream-5-0-pro-260628` | 项目级图片模型；allowlist 另含 Lite、4.5、4.0；Pro 请求省略 `sequential_image_generation`，其余固定为 `disabled` |
| `SEEDREAM_EDIT_MODE` | `independent_parallel` | `independent_parallel` 或 `anchor_consistency`；按项目冻结，不暴露前端；显式配置仍可选择 anchor |
| `SEEDREAM_CONCURRENCY` | `4` | Seedream 帧级进程内并发上限，必须为正整数 |
| `SEEDREAM_TIMEOUT_S` | `300` | 单次 Seedream 请求超时秒数；可显式覆盖为正有限值 |
| `TIKTOK_PROXY` | 空 | TikTok/DoH 下载代理 |
| `DOWNLOAD_TIMEOUT_S` | `120` | URL 下载整体时限 |
| `HOST` / `PORT` | `0.0.0.0` / `3211` | `run.sh` 默认；生产 unit 必须覆盖为 `127.0.0.1/3212` |

全部服务环境只放 `%h/.config/duet-ad1/service.env` 这一份 0600 EnvironmentFile，不写 unit、仓库或命令行。部署步骤见 [.deploy/runbook.md](../../../.deploy/runbook.md)。

## 依赖

- Python：FastAPI、uvicorn、httpx、OpenCV、ai-edge-litert 等（以 `requirements.txt` 为准）。
- 可执行：`ffmpeg`、`ffprobe`、已认证的 `codex` CLI。
- 外部服务：AutoDL Art H3；可选 AI MediaKit 图像擦除与火山方舟 Seedream 图片编辑。
- 运行约束：Linux `flock/fsync` 语义、单 uvicorn 进程、Caddy 本机反代。
