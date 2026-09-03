# 极简前端：后端适配合同（v1）

本文定义极简创建页依赖的后端 v1 合同。目标是把“来源、生成参数、目标语言、可选替换图”在一次请求中原子冻结，由后端自动 ASR，再用 DeepSeek 完成轻量台词微调与目标语言翻译；前端只依赖项目级进度，不感知分析、关键帧、台词生成或供应商任务等内部节点。客户端不提交台词文本。

## 1. 不变量与兼容边界

1. `GET /api/capabilities` 新增 `minimal_creation`；既有 capability 字段在迁移期保留。
2. `POST /api/conversations` 仍为 `multipart/form-data`，但 v1 创建必须在同一请求中携带来源、`client_request_id`、`generation_request`，以及成对出现的可选替换图和替换说明。
3. v1 创建一经接受，后端冻结 `effective_request` 和输入文件回执，直接进入无人值守的完整流水线；后端必须自动识别、微调并翻译原视频台词，不得要求用户提供台词或中途等待校对。
4. `GET /api/conversations` 与 `GET /api/conversations/{id}` 都返回同一语义的 `project_progress`。极简前端只展示该项目级投影，不从内部 stage、segment、node 或供应商状态拼装进度。
5. 老项目必须仍可读取。缺少 v1 冻结字段时返回 `null`，不得伪造请求或文件 hash；但仍须由后端把旧状态投影为有效的 `project_progress`。
6. capability 是启用新创建路径的唯一开关。前端只有取得完全匹配的 v1 capability 才能启用创建；缺失、形状漂移或其他版本都必须 fail closed，不得静默回退旧表单合同。
7. 创建响应、list 和 detail 必须保留布尔字段 `has_video`。当前前端仅在 detail 的 `has_video === true` 时从固定路径 `GET /api/conversations/{id}/files/generated.mp4` 读取成片。

## 2. 能力发现

### 2.1 响应样例

`GET /api/capabilities` 的顶层新增以下对象。示例中的两个 replacement 限制值是可发布配置；前端只要求它们是正整数并以响应值为准。除这些运行时限制外，下面标出的固定值、字段名及数组顺序均属于 v1 合同。

```json
{
  "minimal_creation": {
    "supported": true,
    "version": 1,
    "endpoint": "/api/conversations",
    "encoding": "multipart/form-data",
    "request_field": "generation_request",
    "replacement_image_field": "replacement_image",
    "aspect_ratios": ["16:9", "9:16"],
    "resolutions": ["480p", "768p"],
    "defaults": {
      "fit_mode": "auto",
      "optimize_image": true,
      "remove_subtitle": true,
      "remove_logo": true
    },
    "dialogue": {
      "mode": "auto_rewrite",
      "translation": true
    },
    "replacement": {
      "supported": true,
      "accept": ["image/jpeg", "image/png", "image/webp"],
      "max_bytes": 10485760,
      "max_instruction_chars": 1000
    }
  }
}
```

前端当前会严格核对以下内容，后端不得改名、给数组重排或用等价别名替代：

- `supported=true`、`version=1`；其他版本即使其余字段相似也不兼容；
- `endpoint=/api/conversations`；
- `encoding=multipart/form-data`；
- `request_field=generation_request`；
- `replacement_image_field=replacement_image`；
- `aspect_ratios` 精确按 `16:9, 9:16` 排列；
- `resolutions` 精确按 `480p, 768p` 排列；
- `defaults` 必须且只能含 `fit_mode/optimize_image/remove_subtitle/remove_logo`，值精确为 `auto/true/true/true`；
- `dialogue` 必须且只能为 `{"mode":"auto_rewrite","translation":true}`；`auto_rewrite` 表示后端自动 ASR、用 DeepSeek 微调并翻译，不表示客户端提交脚本；
- `replacement.supported=true`，`accept` 精确按 `image/jpeg, image/png, image/webp` 排列，`max_bytes` 与 `max_instruction_chars` 为正整数。

不得先发布一个形状不完整但 `supported=true` 的对象。暂未就绪时省略 `minimal_creation` 或返回 `supported=false`；完整后一次启用。

## 3. 原子创建请求

### 3.1 multipart 字段

v1 请求只接受下列字段，每个字段最多出现一次：

| 字段 | 必填 | 合同 |
| --- | --- | --- |
| `reference_url` | 条件必填 | 极简前端的默认来源；与 `file` 必须且只能提供一个；空白字符串按未提供处理 |
| `file` | 条件必填 | API 保留的上传来源；与 `reference_url` 必须且只能提供一个；内容是原视频 |
| `client_request_id` | 是 | `^[0-9A-Za-z-]{8,64}$`；整个创建动作的幂等键 |
| `generation_request` | 是 | UTF-8 JSON；结构见下节 |
| `replacement_image` | 条件必填 | 仅当 `replacement_guidance` 非 `null` 时提供，字段名必须与 `image_field` 一致 |

未知字段、重复字段、同时提供或同时不提供 `file/reference_url` 均 fail closed。来源的 UI 默认值不改变 API 的 XOR 约束。v1 不需要后续 submit，也不接受旧的 `voice_mode`、`dialogue_mode`、`script`、`lines`、`dialogue_review_policy` 或 `generation_config` 来覆盖 `generation_request`。

### 3.2 `generation_request` 精确结构

```json
{
  "version": 1,
  "output": {
    "aspect_ratio": "9:16",
    "resolution": "768p",
    "fit_mode": "auto"
  },
  "processing": {
    "optimize_image": true,
    "remove_subtitle": true,
    "remove_logo": true
  },
  "dialogue": {
    "mode": "auto_rewrite",
    "target_language": "日语"
  },
  "replacement_guidance": null
}
```

有指定替换时，`replacement_guidance` 为：

```json
{
  "instruction": "把画面中的白色水杯替换成参考图中的产品杯",
  "image_field": "replacement_image"
}
```

结构约束如下：

- 各对象按上述 key 白名单校验，未知、缺失、重复或类型错误均拒绝；`version` 只接受整数 `1`。
- `output.aspect_ratio`、`output.resolution` 必须分别来自 capability 数组，`fit_mode` 必须精确为 `auto`。
- `processing` 三项必须全部存在且都为布尔值 `true`。`false`、`remove_watermark`、`remove_brand` 等旧名都不是 v1 输入；内部如需映射，只能在合同边界之后完成。
- `dialogue` 必须且只能含 `mode/target_language`；`mode` 只能为 `auto_rewrite`。请求不得含 `script`、`lines`、`source`、`translate` 或其他用户台词字段/分支。
- `target_language` 是 trim 后非空的字符串，按 UTF-16 code unit 计数最多 80。它是最终视频目标语言；后端必须先对原视频自动 ASR，再由 DeepSeek 对识别台词做轻量微调并翻译为该语言。
- `replacement_guidance` 只能为 `null` 或精确的 `instruction/image_field` 对象。instruction 去除首尾空白后必须非空，`image_field` 必须等于 capability 的 `replacement_image_field`。
- 为与浏览器 `String.length` 一致，`max_instruction_chars` 按 instruction 去除首尾空白后的 UTF-16 code unit 数计数；后端不得改用 UTF-8 字节数解释该字段。

### 3.3 替换图成对规则

| JSON / 文件组合 | 结果 |
| --- | --- |
| `replacement_guidance=null` 且没有 `replacement_image` | 接受 |
| guidance 对象且恰有一个合法 `replacement_image` | 接受 |
| guidance 对象但没有图片 | 422 `replacement_image_required` |
| guidance 为 `null` 但携带图片 | 422 `replacement_guidance_required` |
| `image_field` 不是 `replacement_image` | 422 `invalid_replacement_image_field` |

后端必须同时校验声明 MIME、文件签名/实际解码结果和 `max_bytes`。只接受 capability 中按顺序列出的 JPEG、PNG、WebP；不能仅信任扩展名或浏览器提供的 `Content-Type`。

### 3.4 multipart 示例

默认 URL 来源、目标语言为中文：

```bash
curl --request POST 'https://example.invalid/api/conversations' \
  --header 'Authorization: Bearer REDACTED' \
  --form 'reference_url=https://media.example.invalid/source.mp4' \
  --form 'client_request_id=minimal-create-000001' \
  --form 'generation_request={"version":1,"output":{"aspect_ratio":"9:16","resolution":"768p","fit_mode":"auto"},"processing":{"optimize_image":true,"remove_subtitle":true,"remove_logo":true},"dialogue":{"mode":"auto_rewrite","target_language":"中文"},"replacement_guidance":null};type=application/json'
```

API 的上传来源、目标语言为日语并携带替换图：

```bash
curl --request POST 'https://example.invalid/api/conversations' \
  --header 'Authorization: Bearer REDACTED' \
  --form 'file=@/home/xy/example/source.mp4;type=video/mp4' \
  --form 'replacement_image=@/home/xy/example/product.png;type=image/png' \
  --form 'client_request_id=minimal-create-000002' \
  --form 'generation_request={"version":1,"output":{"aspect_ratio":"16:9","resolution":"480p","fit_mode":"auto"},"processing":{"optimize_image":true,"remove_subtitle":true,"remove_logo":true},"dialogue":{"mode":"auto_rewrite","target_language":"日语"},"replacement_guidance":{"instruction":"把白色水杯替换成参考图中的产品杯","image_field":"replacement_image"}};type=application/json'
```

## 4. 冻结、回执与幂等

### 4.1 冻结内容

后端先把上传或下载内容写入隔离的 staging 区，完成来源、媒体、大小、JSON 和替换图校验，再形成不可变的：

- `effective_request`：去除允许字段的首尾空白、完成白名单校验后的 canonical v1 请求；它只冻结用户选择，不含用户台词，也不保留可被后续默认值漂移改变的隐式选项。
- `generation_request_sha256`：对 `effective_request` 使用稳定 key 排序、无无意义空白的 UTF-8 JSON 序列化后计算 SHA-256，而不是对 multipart 中原始 JSON 字符串计算。
- `source.sha256`：实际落盘的原视频字节 SHA-256；URL 来源也必须对下载后的字节计算。
- `replacement_image.sha256`：实际替换图字节 SHA-256；未使用替换图时整个条目为 `null`。

ASR 和 DeepSeek 生成的改编台词不是创建输入，不能写回或改写 `effective_request`，也不参与 `generation_request_sha256`。它首次生成成功后必须作为同一项目的不可变、可恢复流水线产物原子持久化；后台恢复必须复用该产物，不能因为进程重启、轮询或幂等重放而再次生成出另一版台词。是否向前端公开生成后的台词不属于本 v1 合同。

创建提交点必须一次持久化项目、来源文件、可选替换图、`effective_request`、回执和 durable queued claim。提交前失败不应在 list/detail 留下项目，也不得调用任何付费供应商；提交后丢失后台任务时，启动恢复或同 id 重放只能认领同一 queued 项目，不能再创建一个项目。

### 4.2 幂等绑定

`client_request_id` 唯一绑定以下三元组：

```text
(generation_request_sha256, source.sha256, replacement_image.sha256|null)
```

规则：

- 首次成功提交返回 `201`。
- 相同 id 且三元组完全一致，返回已有项目的 `200`，不覆盖冻结字段、不重复排队、不重复调用付费供应商；即使已有项目正在运行、已成功或已失败，也只返回现有项目。
- 相同 id 但 effective request、原视频字节或替换图字节任一不同，返回 `409 client_request_id_conflict`，不修改已有项目。
- JSON key 顺序或无意义空白不同、但 canonical `effective_request` 相同，不应产生冲突。
- 对 `reference_url` 重放也要以本次实际取得的来源字节 hash 比较；同 URL 内容已经变化时必须冲突，不能把 URL 字符串相同当作内容相同。
- 并发相同 id 必须由唯一约束或等价 CAS 串行化；至多一个请求完成首次提交。

### 4.3 创建响应

首次创建和幂等命中使用同一响应形状，区别仅为 HTTP `201/200`：

```json
{
  "id": "a4fce59c6ae84bfa8673d2337e990001",
  "has_video": false,
  "project_progress": {
    "percent": 0,
    "status": "queued"
  },
  "effective_request": {
    "version": 1,
    "output": {
      "aspect_ratio": "9:16",
      "resolution": "768p",
      "fit_mode": "auto"
    },
    "processing": {
      "optimize_image": true,
      "remove_subtitle": true,
      "remove_logo": true
    },
    "dialogue": {
      "mode": "auto_rewrite",
      "target_language": "中文"
    },
    "replacement_guidance": null
  },
  "input_receipt": {
    "version": 1,
    "client_request_id": "minimal-create-000001",
    "generation_request_sha256": "fe4a87e2569cd84b7318db85be4fceac7618b59d17702628bbd03de831c7b86a",
    "source": {
      "sha256": "5392f39fe8c053958f6f390b712f0894e7ca1ee00c9f43baef6bb8d2a8efb901",
      "bytes": 24883200
    },
    "replacement_image": null
  }
}
```

SHA 字符串必须为 64 位小写十六进制。回执中的 `bytes` 是实际冻结文件大小；文件名、源 URL、staging 路径、供应商 asset id 等不是极简前端合同。

## 5. 项目级进度

`project_progress` 是后端对整个项目的权威投影：

```json
{
  "percent": 47,
  "status": "running"
}
```

约束：

- `percent` 是整数，闭区间 `0..100`。
- `status` 只能为 `queued/running/succeeded/failed`。
- 新项目的正常状态转移为 `queued -> running -> succeeded|failed`；恢复内部任务不得让已公开进度倒退。
- `queued` 固定为 `0`；`succeeded` 只在最终视频已写入并通过可读性校验后返回，固定为 `100`。
- `running` 返回 `1..99`。`failed` 保留失败前最后确认的百分比，范围 `0..99`；不能因内部节点数变化而跳到 100。
- 相同项目在 list 和 detail 中的对象必须来自同一聚合函数/持久化快照，不允许两套路由各自猜测。
- 前端不得通过 `analysis_status`、`navigation_status`、`generation.stage`、segments 数量或供应商任务状态覆盖它。

后端可以用内部节点和工作量权重计算进度，但公开对象不得包含节点名、segment 明细、模型、供应商、task id、重试次数或付费次数。需要向用户展示失败原因时，另给稳定、脱敏的项目级 `error`；不得把供应商原始响应塞进 `project_progress`。

极简前端只允许对无法提供 `project_progress` 的历史旧项目做临时、粗粒度兼容推导。所有通过 `minimal_creation` v1 创建的新项目必须以后端 `project_progress` 为唯一权威，不能长期依赖前端回退，也不能因字段暂时缺失而改读内部状态。

## 6. list 与 detail 响应

### 6.1 list 行示例

`GET /api/conversations` 的每一行至少增加：

```json
{
  "id": "a4fce59c6ae84bfa8673d2337e990001",
  "title": "source.mp4",
  "created_at": "2026-09-03T08:10:00Z",
  "has_video": false,
  "project_progress": {
    "percent": 47,
    "status": "running"
  }
}
```

迁移期可继续返回旧前端所需的顶层字段，但极简前端不消费这些字段来判断进度。

### 6.2 detail 响应示例

```json
{
  "id": "a4fce59c6ae84bfa8673d2337e990001",
  "title": "source.mp4",
  "created_at": "2026-09-03T08:10:00Z",
  "updated_at": "2026-09-03T08:11:42Z",
  "has_video": false,
  "project_progress": {
    "percent": 47,
    "status": "running"
  },
  "error": null,
  "effective_request": {
    "version": 1,
    "output": {
      "aspect_ratio": "9:16",
      "resolution": "768p",
      "fit_mode": "auto"
    },
    "processing": {
      "optimize_image": true,
      "remove_subtitle": true,
      "remove_logo": true
    },
    "dialogue": {
      "mode": "auto_rewrite",
      "target_language": "中文"
    },
    "replacement_guidance": null
  },
  "input_receipt": {
    "version": 1,
    "client_request_id": "minimal-create-000001",
    "generation_request_sha256": "fe4a87e2569cd84b7318db85be4fceac7618b59d17702628bbd03de831c7b86a",
    "source": {
      "sha256": "5392f39fe8c053958f6f390b712f0894e7ca1ee00c9f43baef6bb8d2a8efb901",
      "bytes": 24883200
    },
    "replacement_image": null
  }
}
```

新 v1 项目的 `effective_request` 与 `input_receipt` 必须始终非空且不可变。老项目缺少可信证据时返回：

```json
{
  "effective_request": null,
  "input_receipt": null,
  "project_progress": {
    "percent": 100,
    "status": "succeeded"
  }
}
```

老项目的成功只能由已有最终产物及其可读性校验支持；缺失 hash 时保持 `null`，不能事后对当前可能已变化的文件补算后冒充创建时回执。旧分析/生成状态可以在后端用于投影，但不应新增到 `project_progress`。

### 6.3 成片可用性与固定文件路径

- `has_video` 是当前极简前端判断成片是否可取的唯一开关。创建和运行中必须为 `false`；只有最终 `generated.mp4` 已原子写入且通过可读性校验后才变为 `true`。
- detail 返回 `has_video === true` 后，前端请求 `GET /api/conversations/{id}/files/generated.mp4`；该固定路径和现有 Bearer 鉴权语义必须保留。
- list 中的 `has_video` 用于列表展示，detail 中的同名字段用于实际取片门禁；两者必须来自同一已验证产物事实，不能仅由项目状态猜测。
- `project_progress.status=succeeded`、`project_progress.percent=100` 与 `has_video=true` 应在同一公开快照中出现，避免前端看到成功却无法取得文件。
- 后端未来可以附加 `output` 对象，但它只是可选扩展；当前前端不依赖 `output.available`、`output.file` 或其他动态文件名，新增字段不得替代 `has_video` 和固定文件接口。

## 7. 校验与错误码

错误响应统一使用稳定机器码；前端按 `detail.code` 分支，`message` 仅用于展示：

```json
{
  "detail": {
    "code": "target_language_required",
    "message": "请填写目标语言",
    "field": "generation_request.dialogue.target_language"
  }
}
```

| HTTP | `detail.code` | 条件与副作用 |
| --- | --- | --- |
| 400 | `source_exactly_one_required` | `file/reference_url` 不是恰好一个；不创建项目 |
| 400 | `invalid_client_request_id` | id 缺失或格式非法；不创建项目 |
| 401 | `unauthorized` | token 缺失/非法 |
| 409 | `client_request_id_conflict` | 同 id 的 canonical 请求或任一素材 hash 不同；已有项目不变 |
| 413 | `source_too_large` | 原视频超过服务端来源限制 |
| 413 | `replacement_image_too_large` | 替换图超过 capability 的 `max_bytes` |
| 415 | `unsupported_replacement_media_type` | 替换图实际类型不在 capability `accept` 中 |
| 422 | `invalid_create_request` | 未知/重复 multipart 字段或字段类型错误 |
| 422 | `invalid_generation_request_json` | JSON 无法解析、不是对象或编码非法 |
| 422 | `unsupported_generation_request_version` | `version` 不是整数 1 |
| 422 | `invalid_generation_request` | 未知/缺失 key 或对象结构不精确 |
| 422 | `invalid_output_config` | 画幅、清晰度或 `fit_mode` 不符合 capability |
| 422 | `processing_must_be_enabled` | 三个 processing 开关缺失、非 bool 或不全为 true |
| 422 | `target_language_required` | `target_language` 缺失、非字符串或 trim 后为空 |
| 422 | `target_language_too_long` | `target_language` 超过 80 个 UTF-16 code unit |
| 422 | `replacement_image_required` | 有 guidance、无图片 |
| 422 | `replacement_guidance_required` | 有图片、guidance 为 null |
| 422 | `invalid_replacement_image_field` | `image_field` 不等于 `replacement_image` |
| 422 | `replacement_instruction_required` | instruction trim 后为空 |
| 422 | `replacement_instruction_too_long` | 超过 capability 限制 |
| 422 | `invalid_replacement_image` | 文件签名、解码或安全校验失败 |
| 422 | `invalid_source_media` | 视频下载、探测、格式、时长或安全校验失败；staging 清理 |
| 429 | `rate_limited` / `queue_full` | 限流或容量不足；不创建项目 |

任何 4xx 校验失败都必须发生在 durable 项目提交和供应商调用之前。服务端内部错误使用 5xx 和脱敏机器码；不能把 provider body、凭据、内部路径或节点名返回前端。

## 8. 新旧路径隔离

后端可在迁移期继续读取旧项目，必要时也可暂时保留旧客户端创建入口，但 `generation_request.version=1` 必须走独立、确定的 v1 解析和冻结路径：

- 新项目的权威输入只包含 `target_language`，不包含用户脚本；`dialogue.mode=auto_rewrite` 表示后端自动 ASR、DeepSeek 微调并翻译，禁止要求客户端补传 `script` 或改走旧台词提交路径。
- `target_language` 必须贯穿后续提示词和台词产物；持久化结果必须满足冻结的目标语言，且不能把 ASR 文本、微调文本或译文写入 `effective_request`。
- 新项目不能产生用户可见的 dialogue review 等待态，不能要求调用旧的校对 commit 或旧 submit 才继续。
- 内部可以为完成任务执行分析、分段和恢复，但这些都是同一 durable 项目的后台实现细节。
- 极简前端如果 capability 不匹配必须 fail closed；“发旧字段试试看”不是兼容方案。

## 9. 迁移与发布顺序

1. **先加持久化和读取兼容**：支持 v1 `effective_request`、`input_receipt`、幂等唯一约束和 durable queued claim；所有新增读取代码都允许旧记录字段缺失。
2. **再统一项目进度投影**：detail/list 复用同一聚合器，给新旧项目都返回 `project_progress`；用测试证明无内部节点泄漏和进度不倒退。
3. **实现但暂不宣告 v1 创建**：加入 multipart v1 解析、staging、文件探测、canonical hash、原子提交和恢复；此时不要返回 `minimal_creation.supported=true`。
4. **完成幂等与失败注入测试**：覆盖上传、URL、替换图、并发、提交前崩溃、提交后丢 background task；证明没有重复项目和重复付费调用。
5. **部署后端兼容版本**：先以 capability 缺失或 `supported=false` 上线，验证旧项目 list/detail、旧前端和新字段读取。
6. **部署极简前端**：它读取并严格校验 capability；能力未开启时显示服务未就绪并禁用创建。
7. **最后开启 capability**：仅在同一生产实例的 POST/detail/list 全部就绪后返回完整 v1 对象，随后执行无付费的 capability/detail smoke；付费创建验收需单独授权。

回滚时先关闭/移除 `minimal_creation`，再回滚前端；已落盘的 v1 字段和已生成台词必须继续可读、可恢复，不能回滚成要求用户提交台词或中途校对。

## 10. 验收清单

### Capability

- [ ] `minimal_creation` 的固定字符串、版本、字段名、defaults 精确值通过契约测试。
- [ ] capability 与 generation request 的 `version` 均精确为整数 `1`；其他版本 fail closed。
- [ ] `aspect_ratios` 精确为 `["16:9","9:16"]`，`resolutions` 精确为 `["480p","768p"]`。
- [ ] capability 的 `dialogue` 精确为 `{"mode":"auto_rewrite","translation":true}`，不含脚本输入能力。
- [ ] replacement `accept` 精确为 `["image/jpeg","image/png","image/webp"]`，两个 replacement 限制值为正整数。
- [ ] 未完全就绪时不返回 `supported=true`；前端在 capability 缺失/漂移时禁用创建且不回退。

### 创建与校验

- [ ] file/reference_url 的 0 个、2 个、重复字段、未知字段均在创建前拒绝。
- [ ] 极简前端默认使用 `reference_url`；切换上传时仍只提交 `file`，两种来源都遵守同一 XOR API 合同。
- [ ] output 只接受 capability 值，`fit_mode` 只接受 `auto`。
- [ ] processing 三项始终为 true；false、漏项和旧别名均拒绝。
- [ ] dialogue 只接受精确的 `mode/target_language`；`script`、`lines`、`source`、`translate` 等字段作为未知 key 在创建前拒绝。
- [ ] `target_language` trim 后必须非空且不超过 80 个 UTF-16 code unit；缺失、空白和超限均在创建前拒绝。
- [ ] guidance/image 两者同时缺失和同时合法时通过；只给一个时拒绝。
- [ ] JPEG、PNG、WebP 按实际内容验真；伪 MIME、损坏图片和超限图片拒绝。
- [ ] 任一校验失败都不留下可见项目、staging 残留、队列任务或供应商请求。

### 冻结与幂等

- [ ] 新项目 detail 中 `effective_request`、request hash、source hash 永久稳定；使用替换图时其 hash 也稳定。
- [ ] 同 id、同 canonical JSON、同两个素材 hash 返回 200 同一 id，且不重复入队/调用供应商。
- [ ] 只改变 JSON、原视频或替换图中的任一项，都返回 409 且已有项目完全不变。
- [ ] JSON key 顺序/空白不同但 effective request 相同不会误报冲突。
- [ ] 相同 URL 内容变化、并发同 id、进程在原子提交前后崩溃均有覆盖。
- [ ] 未使用替换图时 receipt 明确为 null，而不是空 hash、全零 hash 或缺失语义。
- [ ] Agent 首次生成成功的改编台词被原子持久化；恢复、轮询和幂等重放均复用同一产物，不改变 `effective_request` 或 request hash。

### 进度与兼容

- [ ] list/detail 都返回合法的 `{percent,status}`，并在同一时刻语义一致。
- [ ] queued=0、running=1..99、succeeded=100，failed 不伪装成功；恢复不导致公开百分比倒退。
- [ ] `project_progress` 不含内部 stage、node、segment、模型、供应商、task id 或重试详情。
- [ ] 最终文件写入并验证前绝不返回 succeeded/100。
- [ ] 创建响应、list、detail 都保留布尔 `has_video`；运行中为 false，成功快照为 true。
- [ ] detail 的 `has_video === true` 时，固定 `GET /api/conversations/{id}/files/generated.mp4` 可通过现有鉴权取得已验证成片。
- [ ] 当前前端不依赖可选 `output` 对象，且不会用它替代 `has_video` 或固定文件路径。
- [ ] 旧 queued/running/succeeded/failed 项目均能读取；缺失冻结证据时返回 null，不伪造 hash。
- [ ] 前端粗粒度进度回退只覆盖历史旧项目；所有 minimal_creation v1 新项目都只使用后端 `project_progress`。
- [ ] 所有 v1 新创建项目都自动完成 ASR、DeepSeek 微调与目标语言翻译并运行到终态，绝不要求用户提交脚本或进入中途校对等待态。
