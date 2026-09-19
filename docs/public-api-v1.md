# Duet 视频生成 API v1

Duet API 用于提交视频生成任务，并在任务完成后返回最终视频和费用。

本合同版本 `1.2.0` 新增报价与确认接口，支持先展示预计积分、用户确认后再执行。原有直接创建、查询、下载和账户接口保持兼容。

本合同版本 `1.3.0` 新增多参考图：同一 `replacement_image` multipart 字段可重复传入 2～4 张图片；单图调用和其他既有行为保持兼容。

生成请求支持可选字幕开关 `add_subtitles`，默认关闭；开启后在最终成片中叠加台词字幕。直接创建和报价都支持该参数，已有调用省略它即可保持原行为。

视频生成通常需要数分钟，因此接口采用异步模式：创建任务后立即返回任务 ID；接入方使用任务 ID 查询状态；成功后再下载视频。

## 基本信息

| 项目 | 值 |
|---|---|
| 合同版本 | `1.3.0` |
| 文档更新日期 | `2026-09-15` |
| 联调环境 | `https://8.166.140.227:3211` |
| API 前缀 | `/api/v1` |
| 鉴权 | HTTP Bearer API Key |
| 创建请求格式 | `multipart/form-data` |
| 普通响应格式 | `application/json` |
| 视频格式 | `video/mp4` |
| 时间格式 | ISO 8601，UTC 时区 |
| OpenAPI | `/api/v1/openapi.json` |

## 鉴权

除 OpenAPI 文件外，所有接口都需要在请求头中提供 API Key：

```http
Authorization: Bearer YOUR_API_KEY
```

API Key 类似 `duet_live_example01.REPLACE_WITH_SECRET`。请只把它保存在服务端密钥系统中，不要写入网页、客户端安装包、日志或代码仓库。

API Key 对应一个账户。这个账户拥有自己的任务和积分；不同账户不能读取彼此的任务或视频。鉴权 scheme 大小写不敏感，例如 `Bearer` 与 `bearer` 等价。

## 接入流程

1. 上传原视频或提交 URL 到报价接口，由服务端读取时长并返回预计积分；此时不启动生成、不冻结或扣除积分。
2. 将 `duration_seconds`、`billable_seconds`、`estimated_credits` 和 `expires_at` 展示给用户。报价 = 原视频时长（秒，向上取整）× 60 积分，例如 24.37 秒按 25 秒计算，共 1,500 积分。
3. 用户同意后确认该报价，保存返回的任务 `id`；确认时才检查余额、冻结积分并进入原有生成流程。
4. 每隔 5～10 秒查询该任务；状态变成 `succeeded` 后，使用 `result.video.content_url` 下载视频。

报价有效期为 24 小时，不保证确认时账户仍有足够余额。用户不接受时无需确认或调用取消接口，报价会自动过期，也不会产生生成费用。旧的直接创建接口仍保留，调用它会立即执行，不经过用户确认。

创建请求如果遇到网络超时，必须使用原来的 `Idempotency-Key` 和完全相同的输入重试。不要直接换新 Key，否则可能创建两个收费任务。

## 接口一览

| 方法 | 路径 | 用途 | 版本变化 |
|---|---|---|---|
| `GET` | `/api/v1/video-generations/capabilities` | 查询当前参数限制和价格 | 保持兼容 |
| `POST` | `/api/v1/video-generations/quotes` | 校验并保存输入，读取时长和预计积分 | 1.2.0 新增 |
| `GET` | `/api/v1/video-generations/quotes/{quote_id}` | 查询报价及其确认状态 | 1.2.0 新增 |
| `POST` | `/api/v1/video-generations/quotes/{quote_id}/confirm` | 用户确认报价后创建异步任务 | 1.2.0 新增 |
| `POST` | `/api/v1/video-generations` | 创建异步生成任务 | 保持兼容 |
| `GET` | `/api/v1/video-generations/{job_id}` | 查询任务状态、结果和费用 | 保持兼容 |
| `GET` | `/api/v1/video-generations/{job_id}/content` | 下载视频，支持单段 Range | 保持兼容 |
| `HEAD` | `/api/v1/video-generations/{job_id}/content` | 读取视频元数据，不下载正文 | 保持兼容 |
| `GET` | `/api/v1/account/credits` | 查询积分余额 | 保持兼容 |
| `GET` | `/api/v1/account/credit-transactions` | 查询积分流水 | 保持兼容 |
| `GET` | `/api/v1/openapi.json` | 获取公共 API 的 OpenAPI 描述 | 1.2.0 加入报价与确认路径 |

---

## 查询生成能力

返回当前支持的参数、输入限制、价格和建议轮询间隔。建议接入方读取本接口，不要把价格和参数范围永久写死。

```http
GET /api/v1/video-generations/capabilities HTTP/1.1
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
Accept: application/json
```

### 请求参数

#### Header 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `Authorization` | string | 是 | `Bearer YOUR_API_KEY` | API 鉴权信息 |

本接口没有 Path、Query 或 Body 参数。

### 成功响应

HTTP `200 OK`

```json
{
  "version": "v1",
  "endpoint": "/api/v1/video-generations",
  "encoding": "multipart/form-data",
  "defaults": {
    "aspect_ratio": "9:16",
    "resolution": "768p",
    "target_language": null,
    "dialogue_mode": "auto_rewrite",
    "add_subtitles": false
  },
  "allowed_output_combinations": [
    {"aspect_ratio": "9:16", "resolution": "768p"},
    {"aspect_ratio": "9:16", "resolution": "480p"},
    {"aspect_ratio": "16:9", "resolution": "768p"},
    {"aspect_ratio": "16:9", "resolution": "480p"}
  ],
  "source_video": {
    "exactly_one_of": ["source_video", "source_video_url"],
    "extensions": [".mp4", ".mov", ".webm"],
    "max_bytes": 524288000,
    "max_duration_seconds": 300,
    "url_scheme": "https"
  },
  "replacement_image": {
    "paired_with": "replacement_instruction",
    "media_types": ["image/jpeg", "image/png", "image/webp"],
    "max_bytes": 10485760,
    "max_images": 4,
    "max_instruction_chars": 1000
  },
  "target_language": {
    "omitted_means": "same_as_source",
    "max_chars": 80
  },
  "dialogue": {
    "modes": ["auto_rewrite", "generate"],
    "default_mode": "auto_rewrite",
    "brief": {
      "required_for": "generate",
      "max_chars": 1000,
      "unit": "utf16_code_units"
    }
  },
  "pricing": {
    "credits_per_cny": 100,
    "credits_per_second": 60,
    "billed_unit": "source_video_second",
    "currency": "CNY",
    "price_version": "credits-per-second-60-v1"
  },
  "polling": {
    "recommended_seconds": 5,
    "maximum_backoff_seconds": 10
  }
}
```

### 响应字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `version` | string | API 能力版本，当前固定为 `v1` |
| `endpoint` | string | 创建任务的路径 |
| `encoding` | string | 创建任务使用的编码 |
| `defaults` | object | 未传输出参数时采用的默认值 |
| `defaults.add_subtitles` | boolean | 字幕开关默认值，当前为 `false` |
| `allowed_output_combinations` | array<object> | 当前允许的画幅与清晰度组合 |
| `source_video` | object | 原视频格式、大小、时长和 URL 限制 |
| `replacement_image` | object | 参考图格式、大小和说明文字限制 |
| `target_language` | object | 目标语言的省略语义和长度限制 |
| `dialogue` | object | 台词模式列表、默认模式和 `dialogue_brief` 的约束 |
| `pricing` | object | 当前积分换算、按秒单价（60 积分/秒，按原视频时长向上取整计费）和价格版本 |
| `polling` | object | 推荐轮询间隔和最大退避间隔，单位为秒 |

---

## 创建报价

```http
POST /api/v1/video-generations/quotes
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
Idempotency-Key: quote_order_20260909_000001
Content-Type: multipart/form-data
```

请求的 multipart 字段、输入限制和鉴权与下文“创建视频生成任务”完全相同，`Idempotency-Key` 必填。必须提交本次计划生成所需的完整输入：视频文件或 URL，以及需要的语言、画幅、清晰度、台词和替换参数。

需要字幕时，在报价 multipart 中添加文本字段 `add_subtitles=true`。该开关与媒体和其他参数一起冻结；确认接口仍只接受 JSON `{"confirm":true}`，不能在确认时添加或修改字幕开关。需要改变开关时用新 Key 重新报价。Quote 响应不新增字幕字段；确认后的 Job 通过 `parameters.add_subtitles` 返回实际冻结值。

服务端下载或接收视频、读取实际时长，并保存本次媒体与生成参数；不创建生成任务、不调用生成模型，也不产生 `reserve`、`capture` 或 `release` 积分流水。客户端无需事先知道视频时长。确认会复用报价保存的输入，不重新下载 URL 或重新上传文件；修改任何输入需要重新报价。

报价不占用生成任务的执行并发槽位，余额不足也可以取得报价。

上传文件示例（文件路径替换为实际绝对路径）：

```bash
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations/quotes' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Idempotency-Key: quote_order_20260909_000001' \
  --form 'source_video=@/home/xy/api-input/source.mp4' \
  --form 'target_language=中文'
```

URL 示例：

```bash
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations/quotes' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Idempotency-Key: quote_order_20260909_000002' \
  --form 'source_video_url=https://media.example.com/source.mp4'
```

首次成功返回 `201 Created`，相同 Key 和相同输入的重放返回 `200 OK` 和原报价的当前状态，不延长有效期；已确认或过期时也不会自动创建新报价。同一 Key 配不同输入返回 `409 idempotency_key_reused`。需要新报价时使用新 Key。报价与旧直接创建接口的 Key 独立判重；不要把同一业务同时提交到两个流程。本接口没有 Query 参数。

```json
{
  "quote_id": "vq_0123456789abcdef0123456789abcdef",
  "status": "quoted",
  "duration_seconds": 24.37,
  "billable_seconds": 25,
  "estimated_credits": 1500,
  "credits_per_second": 60,
  "price_version": "credits-per-second-60-v1",
  "expires_at": "2026-09-10T08:00:00+00:00",
  "job_id": null
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `quote_id` | string | `vq_` 加 32 位小写十六进制字符；查询和确认时原样使用 |
| `status` | string | `quoted` 待确认、`confirmed` 已创建任务、`expired` 未确认且已过期 |
| `duration_seconds` | number | 服务端读取的原视频实际时长，单位为秒 |
| `billable_seconds` | integer | 原视频时长向上取整后的计费秒数 |
| `estimated_credits` | integer | 本报价预计消费积分，即 `billable_seconds × credits_per_second` |
| `credits_per_second` | integer | 本报价每计费秒积分，当前为 `60` |
| `price_version` | string | 本报价冻结的价格版本 |
| `expires_at` | string | ISO 8601 UTC 截止时间；有效期从报价创建开始计算，共 24 小时 |
| `job_id` | string 或 null | 确认创建的任务 ID；未确认时为 `null` |

时长和价格均由服务端计算，客户端不能提交或覆盖 `duration_seconds`、`billable_seconds`、`estimated_credits`、单价或价格版本。

有效期内按该报价冻结的价格执行：确认所得 Job 的 `billing.quoted_credits` 等于 `estimated_credits`，`billing.price_version` 等于报价的 `price_version`。期间能力接口的价格变化仅影响新报价，不改变已创建报价。

## 查询报价

```http
GET /api/v1/video-generations/quotes/vq_0123456789abcdef0123456789abcdef
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
```

返回 `200 OK` 和上面的报价对象，不需要 Query 或 Body 参数。只有报价所属账户可访问；不存在或属于其他账户均返回 `404 quote_not_found`。未确认报价过期后返回 `status: "expired"`；已确认报价即使超过原截止时间，仍返回 `confirmed` 和原 `job_id`。

报价的 `confirmed` 只表示已创建任务，不代表视频已生成成功；生成结果必须继续查询 Job 的 `status`。

```bash
curl 'https://8.166.140.227:3211/api/v1/video-generations/quotes/vq_0123456789abcdef0123456789abcdef' \
  --header 'Authorization: Bearer YOUR_API_KEY'
```

## 确认报价并执行

```bash
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations/quotes/vq_0123456789abcdef0123456789abcdef/confirm' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{"confirm":true}'
```

路径携带报价 ID，JSON Body 必须严格为 `{"confirm":true}`。不重新提交媒体、参数或价格，也不需要 `Idempotency-Key`，不接受 Query 参数。只有用户同意预计积分后才调用此接口；`false`、字符串 `"true"`、缺失确认或额外字段不是有效确认。

首次确认成功返回 `201 Created` 和下文的 Job 对象，此时才检查当前余额、冻结报价积分并提交到原有生成流程。同一报价重复确认返回 `200 OK` 和原 Job；并发确认或网络超时后的重试不会重复创建、冻结积分或生成。已成功确认后，即使超过原报价截止时间，重复确认仍返回原 Job。

| HTTP | 错误码 | 处理 |
|---:|---|---|
| `402` | `insufficient_credits` | 当前余额不足以支付报价；充值后在报价有效期内确认原报价 |
| `404` | `quote_not_found` | 报价不存在或不属于当前账户；检查报价 ID 和 API Key |
| `410` | `quote_expired` | 未确认报价已过期；使用新 Key 重新报价并取得用户同意 |
| `415` | `unsupported_media_type` | 确认请求必须使用 `Content-Type: application/json` |
| `422` | `invalid_confirm_request` | JSON 损坏、重复字段，或 Body 不严格等于 `{"confirm":true}`；修正后重试原报价 |
| `429` | `rate_limit_exceeded` | 按 `Retry-After` 等待后重试原报价 |
| `503` | `quote_state_unavailable` | 报价状态暂不可读或不可用；保留原报价 ID，稍后查询或重试 |

确认连接失败或超时后，继续确认同一个 `quote_id`，或先查询该报价获取 `job_id`；不要换报价或改走直接创建接口。收到 `429` 时按 `Retry-After` 等待后重试原报价；收到 `503` 时保留原报价 ID 重试或查询状态。任务一旦创建，其状态、查询、下载与成功扣费/失败释放规则均沿用现有 API。

示例中的报价 ID 应替换为报价接口返回的 `quote_id`。确认后使用响应中的任务 `id` 轮询：

```bash
curl 'https://8.166.140.227:3211/api/v1/video-generations/vg_0123456789abcdef0123456789abcdef' \
  --header 'Authorization: Bearer YOUR_API_KEY'
```

每次间隔 5～10 秒；仅在 `succeeded` 后使用返回的 `content_url` 下载：

```bash
curl 'https://8.166.140.227:3211/api/v1/video-generations/vg_0123456789abcdef0123456789abcdef/content' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --output '/home/xy/api-output/final-video.mp4'
```

下载目录应预先存在，文件路径需替换为实际绝对路径。

---

## 创建视频生成任务

创建一个异步任务。此旧接口保留直接执行行为，不等待用户确认报价。接口在完成输入校验和积分冻结后立即返回，不会等待视频生成完成；需要先展示积分的接入方应使用前述报价和确认接口。

```http
POST /api/v1/video-generations HTTP/1.1
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
Idempotency-Key: order_20260903_000001
Content-Type: multipart/form-data; boundary=DuetBoundary
Accept: application/json

--DuetBoundary
Content-Disposition: form-data; name="source_video"; filename="source.mp4"
Content-Type: video/mp4

<source.mp4 的二进制内容>
--DuetBoundary
Content-Disposition: form-data; name="aspect_ratio"

9:16
--DuetBoundary
Content-Disposition: form-data; name="resolution"

480p
--DuetBoundary
Content-Disposition: form-data; name="target_language"

中文
--DuetBoundary
Content-Disposition: form-data; name="replacement_image"; filename="reference.webp"
Content-Type: image/webp

<reference.webp 的二进制内容>
--DuetBoundary
Content-Disposition: form-data; name="replacement_instruction"

把视频中的狗替换成参考图里的狗
--DuetBoundary--
```

### 请求参数

#### Header 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `Authorization` | string | 是 | `Bearer YOUR_API_KEY` | API 鉴权信息 |
| `Idempotency-Key` | string | 是 | 8～64 位；仅限 `A-Z`、`a-z`、`0-9`、`_`、`-` | 本次业务生成操作的唯一标识 |
| `Content-Type` | string | 是 | `multipart/form-data; boundary=...` | multipart 边界由所用 HTTP 客户端生成 |
| `Accept` | string | 否 | 建议 `application/json` | 期望的响应格式 |

#### Body 参数

| 参数 | 类型 | 必填 | 默认值 | 可选值或约束 | 说明 |
|---|---|---:|---|---|---|
| `source_video` | binary | 条件必填 | 无 | MP4、MOV、WebM；最大 500 MiB；最长 300 秒 | 上传原视频；与 `source_video_url` 必须二选一。文件名无可用扩展名时，multipart `Content-Type` 为 `video/mp4`、`video/quicktime` 或 `video/webm` 也可识别 |
| `source_video_url` | string | 条件必填 | 无 | 无用户名密码的公网 HTTPS URL | 由服务端下载原视频；与 `source_video` 必须二选一 |
| `aspect_ratio` | string | 否 | `9:16` | `9:16`、`16:9` | 输出画幅 |
| `resolution` | string | 否 | `768p` | `768p`、`480p` | 输出清晰度 |
| `target_language` | string | 否 | 与原视频相同 | 去除首尾空白后不能为空；最多 80 个 UTF-16 字符单元 | 对白目标语言，例如 `中文`、`English` |
| `add_subtitles` | boolean | 否 | `false` | multipart 文本 `true`、`false` | 在最终成片上叠加台词字幕；参数随任务冻结，幂等重试必须保持一致 |
| `dialogue_mode` | string | 否 | `auto_rewrite` | `auto_rewrite`、`generate` | 台词模式；`auto_rewrite` 翻译并自然改写原台词，`generate` 按 `dialogue_brief` 描述自动生成台词 |
| `dialogue_brief` | string | 条件必填 | 无 | 去除首尾空白后不能为空；最多 1,000 个 UTF-16 字符单元 | 口播台词的内容描述；`dialogue_mode` 为 `generate` 时必填，其余模式不允许提供 |
| `replacement_image` | binary，可重复 | 条件必填 | 无 | 1–4 张 JPEG、PNG、WebP，每张最大 10 MiB；按 multipart 出现顺序编号 | 参考图，不预设图片类型或用途；必须与 `replacement_instruction` 同时提供，单图客户端无需变更 |
| `replacement_instruction` | string | 条件必填 | 无 | 去除首尾空白后不能为空；最多 1,000 个 UTF-16 字符单元 | 描述如何使用参考图；一条说明对全部参考图生效，必须与 `replacement_image` 同时提供 |

本接口没有 Path 或 Query 参数。仅 `replacement_image` 允许重复 1–4 次；其他未知或重复字段、超过 4 张图、同时提供两个视频来源或只提供参考图组合中的一个字段，都会返回 `422`。不使用参考图时同时省略图片和说明。

多图按上传顺序称为图 1、图 2、图 3、图 4，说明文字可引用这些编号。报价冻结全部图片的原始字节和顺序；相同幂等键更换任一图片、数量、顺序或说明均返回 `409`。确认仍只发送 `{"confirm":true}`，不能临时换图。上传与 URL 视频来源均遵守相同规则，URL 报价重放无需重新访问已经冻结的视频链接。视频积分仍按原视频时长计算，不因图片数量改变。

普通中文、英文和数字通常各占 1 个 UTF-16 字符单元；部分 Emoji 占 2 个。

`add_subtitles` 的接口类型为 boolean，但上传与报价请求使用 multipart，应传表单文本 `true` 或 `false`，例如 `--form 'add_subtitles=true'`，不是将整个创建请求改成 JSON。省略或显式 `false` 均表示关闭；开启不会改变按原视频时长计算积分的规则。同一幂等 Key 从关闭改为开启会返回 `409 idempotency_key_reused`。

开启字幕的直接创建示例：

```bash
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Idempotency-Key: subtitle_order_20260910_000001' \
  --form 'source_video_url=https://media.example.com/source.mp4' \
  --form 'add_subtitles=true'
```

需要先报价时，仅将上面请求地址改为 `/api/v1/video-generations/quotes`，再按报价确认流程执行。

### Idempotency-Key 规则

| 情况 | HTTP 状态 | 结果 |
|---|---:|---|
| 第一次成功创建 | `201` | 创建并返回新任务 |
| 相同 Key、相同输入 | `200` | 返回原任务，不重复生成或扣费 |
| 相同 Key、不同输入 | `409` | 返回 `idempotency_key_reused`，不创建新任务 |
| 缺少或格式错误 | `400` | 返回 `invalid_idempotency_key` |

Key 应绑定你方的一次业务操作，而不是某一次 HTTP 尝试。创建请求超时后，应使用原 Key 和原输入重试。

### 成功响应

首次创建返回 HTTP `201 Created`；幂等重放返回 `200 OK`。响应体都是同一个 Job 对象。

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /api/v1/video-generations/vg_0123456789abcdef0123456789abcdef
Retry-After: 5
Cache-Control: private, no-store
X-Request-ID: req_0123456789abcdef0123456789abcdef
```

```json
{
  "id": "vg_0123456789abcdef0123456789abcdef",
  "status": "queued",
  "progress": {"percent": 0},
  "parameters": {
    "aspect_ratio": "9:16",
    "resolution": "480p",
    "target_language": "中文",
    "replacement_image": true,
    "add_subtitles": false
  },
  "billing": {
    "currency": "CNY",
    "credits_per_cny": 100,
    "quoted_credits": 2100,
    "quoted_amount_minor": 2100,
    "price_version": "credits-per-second-60-v1",
    "settlement_status": "pending",
    "settled_credits": null,
    "settled_amount_minor": null
  },
  "result": null,
  "error": null,
  "created_at": "2026-09-03T12:57:32.315082+00:00",
  "updated_at": "2026-09-03T12:57:32.315082+00:00",
  "poll_after_seconds": 5
}
```

完整 Job 字段定义见“Job 对象”一节。

---

## 查询任务

读取一个任务的状态、进度、费用和最终结果。

```http
GET /api/v1/video-generations/vg_0123456789abcdef0123456789abcdef HTTP/1.1
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
Accept: application/json
```

### 请求参数

#### Header 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `Authorization` | string | 是 | `Bearer YOUR_API_KEY` | API 鉴权信息 |
| `Accept` | string | 否 | 建议 `application/json` | 期望的响应格式 |

#### Path 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `job_id` | string | 是 | `vg_` 加 32 位小写十六进制字符 | 创建任务时返回的任务 ID |

本接口没有 Query 或 Body 参数。

### 成功响应

HTTP `200 OK`，响应体为 Job 对象。任务仍在处理时，响应头包含 `Retry-After: 5`。

```json
{
  "id": "vg_0123456789abcdef0123456789abcdef",
  "status": "succeeded",
  "progress": {"percent": 100},
  "parameters": {
    "aspect_ratio": "9:16",
    "resolution": "480p",
    "target_language": "中文",
    "replacement_image": true,
    "add_subtitles": false
  },
  "billing": {
    "currency": "CNY",
    "credits_per_cny": 100,
    "quoted_credits": 2100,
    "quoted_amount_minor": 2100,
    "price_version": "credits-per-second-60-v1",
    "settlement_status": "final",
    "settled_credits": 2100,
    "settled_amount_minor": 2100
  },
  "result": {
    "video": {
      "content_url": "/api/v1/video-generations/vg_0123456789abcdef0123456789abcdef/content",
      "content_type": "video/mp4",
      "size_bytes": 1601490,
      "sha256": "232a4ebe62f91fa4d9578c6485b37922122b4ee0e413eca7d74378b27c5803b3",
      "duration_seconds": 16.916667,
      "expires_at": null
    }
  },
  "error": null,
  "created_at": "2026-09-03T12:57:32.315082+00:00",
  "updated_at": "2026-09-03T13:11:53.939182+00:00",
  "poll_after_seconds": 5
}
```

### 状态定义

| `status` | 是否完成 | 含义 | 接入方操作 |
|---|---:|---|---|
| `queued` | 否 | 已接收，等待处理 | 按 `Retry-After` 继续查询 |
| `running` | 否 | 正在处理 | 每 5～10 秒继续查询 |
| `succeeded` | 是 | 已成功生成并完成扣费 | 使用 `content_url` 下载视频 |
| `failed` | 是 | 明确失败，冻结积分已释放 | 停止查询；如需重做，由业务明确创建新任务 |
| `submission_unknown` | 否 | 上游是否接收任务暂时无法确认 | 继续查询原任务并联系支持，绝对不要重新创建 |

进度百分比表示大致处理阶段，不是剩余时间预估。进度可能在某个值停留较长时间。

---

## 查询积分余额

返回账户当前可用、冻结和累计消费积分。

```http
GET /api/v1/account/credits HTTP/1.1
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
Accept: application/json
```

### 请求参数

#### Header 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `Authorization` | string | 是 | `Bearer YOUR_API_KEY` | API 鉴权信息 |
| `Accept` | string | 否 | 建议 `application/json` | 期望的响应格式 |

本接口没有 Path、Query 或 Body 参数。

### 成功响应

HTTP `200 OK`

```json
{
  "owner_id": "partner_example",
  "credits_per_cny": 100,
  "available_credits": 9000,
  "reserved_credits": 0,
  "spent_credits": 1000
}
```

### 响应字段

| 字段 | 类型 | 可为空 | 说明 |
|---|---|---:|---|
| `owner_id` | string | 否 | API Key 所属账户 ID |
| `credits_per_cny` | integer | 否 | 1 元对应的积分，当前固定为 `100` |
| `available_credits` | integer | 否 | 当前可用于创建任务的积分，最小为 `0` |
| `reserved_credits` | integer | 否 | 已被进行中任务冻结的积分，最小为 `0` |
| `spent_credits` | integer | 否 | 历史累计消费积分，最小为 `0` |

可用余额不足本次报价积分时，直接创建或确认报价返回 `402 insufficient_credits`，不会创建任务。仅创建报价不要求余额足够，也不会冻结积分。

---

## 查询积分流水

按时间从新到旧返回积分事件。

```http
GET /api/v1/account/credit-transactions?limit=50 HTTP/1.1
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
Accept: application/json
```

### 请求参数

#### Header 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `Authorization` | string | 是 | `Bearer YOUR_API_KEY` | API 鉴权信息 |
| `Accept` | string | 否 | 建议 `application/json` | 期望的响应格式 |

#### Query 参数

| 参数 | 类型 | 必填 | 默认值 | 可选值或约束 | 说明 |
|---|---|---:|---|---|---|
| `limit` | integer | 否 | `50` | 规范十进制整数 `1`～`100`；不接受符号、小数、前导零、空白或重复参数 | 最多返回多少条最近流水 |

本接口没有 Path 或 Body 参数。

### 成功响应

HTTP `200 OK`

```json
{
  "data": [
    {
      "id": "job:vg_0123456789abcdef0123456789abcdef:capture",
      "type": "capture",
      "credits": 2100,
      "direction": null,
      "job_id": "vg_0123456789abcdef0123456789abcdef",
      "reason": null,
      "created_at": "2026-09-03T13:11:53.939182+00:00"
    }
  ]
}
```

### 流水字段

| 字段 | 类型 | 可为空 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `id` | string | 否 | 事件唯一 ID | 可用于去重和审计 |
| `type` | string | 否 | `adjustment`、`reserve`、`capture`、`release` | 流水类型 |
| `credits` | integer | 否 | 正整数 | 本次事件涉及的积分数量 |
| `direction` | string | 是 | `credit`、`debit`；只有 `adjustment` 使用 | 调账方向；`credit` 为增加，`debit` 为减少 |
| `job_id` | string | 是 | Job ID；任务流水存在 | 对应任务 |
| `reason` | string | 是 | 调账说明；任务流水通常为空 | 本次调账原因 |
| `created_at` | string | 否 | ISO 8601 UTC 时间 | 事件创建时间 |

流水类型含义：

| 类型 | 含义 |
|---|---|
| `adjustment` | 充值或人工调账 |
| `reserve` | 创建任务并冻结积分 |
| `capture` | 任务成功，冻结积分转为正式消费 |
| `release` | 任务明确失败，冻结积分退回可用余额 |

---

## 下载视频

只有任务状态为 `succeeded` 时才能下载。

```http
GET /api/v1/video-generations/vg_0123456789abcdef0123456789abcdef/content HTTP/1.1
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
Accept: video/mp4
```

### 请求参数

#### Header 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `Authorization` | string | 是 | `Bearer YOUR_API_KEY` | API 鉴权信息 |
| `Accept` | string | 否 | 建议 `video/mp4` | 期望的视频格式 |
| `Range` | string | 否 | 单一字节范围，例如 `bytes=0-1048575`、`bytes=1048576-` 或 `bytes=-1048576` | 分段下载或断点续传；不支持多个范围 |

#### Path 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `job_id` | string | 是 | `vg_` 加 32 位小写十六进制字符 | 已成功任务的 ID |

本接口没有 Query 或 Body 参数。

### 完整下载响应

HTTP `200 OK`

```http
HTTP/1.1 200 OK
Content-Type: video/mp4
Content-Length: 1601490
Accept-Ranges: bytes
ETag: "232a4ebe62f91fa4d9578c6485b37922122b4ee0e413eca7d74378b27c5803b3"
Content-Disposition: attachment; filename="vg_0123456789abcdef0123456789abcdef.mp4"
Cache-Control: private, no-store

<MP4 二进制内容>
```

### Range 下载示例

```http
GET /api/v1/video-generations/vg_0123456789abcdef0123456789abcdef/content HTTP/1.1
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
Range: bytes=0-1023
```

HTTP `206 Partial Content`

```http
HTTP/1.1 206 Partial Content
Content-Type: video/mp4
Content-Length: 1024
Content-Range: bytes 0-1023/1601490
Accept-Ranges: bytes
ETag: "232a4ebe62f91fa4d9578c6485b37922122b4ee0e413eca7d74378b27c5803b3"

<前 1024 字节>
```

同一账户最多同时进行 2 个视频下载。超过限制返回 `429 download_limit_exceeded`，并携带 `Retry-After: 1`。

---

## 读取视频元数据

读取与下载接口相同的响应头，但不传输视频正文。可用于提前获得文件大小、类型、文件名和 ETag。

```http
HEAD /api/v1/video-generations/vg_0123456789abcdef0123456789abcdef/content HTTP/1.1
Host: 8.166.140.227:3211
Authorization: Bearer YOUR_API_KEY
```

### 请求参数

#### Header 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `Authorization` | string | 是 | `Bearer YOUR_API_KEY` | API 鉴权信息 |

#### Path 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `job_id` | string | 是 | `vg_` 加 32 位小写十六进制字符 | 已成功任务的 ID |

本接口没有 Query 或 Body 参数。`HEAD` 不处理 `Range`，成功时固定返回 `200`。

### 成功响应

```http
HTTP/1.1 200 OK
Content-Type: video/mp4
Content-Length: 1601490
Accept-Ranges: bytes
ETag: "232a4ebe62f91fa4d9578c6485b37922122b4ee0e413eca7d74378b27c5803b3"
Content-Disposition: attachment; filename="vg_0123456789abcdef0123456789abcdef.mp4"
```

---

## 获取 OpenAPI 文件

返回只包含公共 v1 接口的 OpenAPI 3 描述。该接口不需要鉴权。

```http
GET /api/v1/openapi.json HTTP/1.1
Host: 8.166.140.227:3211
Accept: application/json
```

### 请求参数

#### Header 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `Accept` | string | 否 | 建议 `application/json` | 期望的响应格式 |

本接口没有 Path、Query 或 Body 参数，也不需要 `Authorization`。

### 成功响应

HTTP `200 OK`，响应体为 OpenAPI JSON，标题为 `Duet Video Generation API`，`info.version` 为 `1.2.0`，包含报价与确认路径。

---

## Job 对象

创建任务和查询任务都会返回同一种 Job 对象。

### 顶层字段

| 字段 | 类型 | 可为空 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `id` | string | 否 | `vg_` 加 32 位小写十六进制字符 | 公开任务 ID |
| `status` | string | 否 | `queued`、`running`、`succeeded`、`failed`、`submission_unknown` | 当前任务状态 |
| `progress` | object | 否 | 见下表 | 进度信息 |
| `parameters` | object | 否 | 见下表 | 创建时冻结的公开参数 |
| `billing` | object | 否 | 见下表 | 报价和结算信息 |
| `result` | object | 是 | 只有成功后存在 | 最终视频信息 |
| `error` | object | 是 | 失败或提交结果未知时存在 | 任务级错误 |
| `created_at` | string | 否 | ISO 8601 UTC 时间 | 任务创建时间 |
| `updated_at` | string | 否 | ISO 8601 UTC 时间 | 任务最后更新时间 |
| `poll_after_seconds` | integer | 否 | 当前固定为 `5` | 建议下次查询前等待的秒数 |

### `progress`

| 字段 | 类型 | 可为空 | 约束 | 说明 |
|---|---|---:|---|---|
| `percent` | integer | 是 | `0`～`100` | 大致处理进度；为空表示暂时无法计算 |

### `parameters`

| 字段 | 类型 | 可为空 | 可选值 | 说明 |
|---|---|---:|---|---|
| `aspect_ratio` | string | 否 | `9:16`、`16:9` | 输出画幅 |
| `resolution` | string | 否 | `768p`、`480p` | 输出清晰度 |
| `target_language` | string | 是 | 用户提交的语言；`null` 表示沿用原语言 | 目标语言 |
| `replacement_image` | boolean | 否 | `true`、`false` | 本任务是否使用了参考图 |
| `add_subtitles` | boolean | 否 | `true`、`false` | 本任务是否叠加台词字幕；历史任务默认 `false` |

### `billing`

| 字段 | 类型 | 可为空 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `currency` | string | 否 | 当前固定为 `CNY` | 结算币种 |
| `credits_per_cny` | integer | 否 | 当前固定为 `100` | 1 元对应的积分 |
| `quoted_credits` | integer | 否 | 计费秒数 × 60 | 创建时冻结的报价积分 |
| `quoted_amount_minor` | integer | 否 | 当前与报价积分数值相同 | 报价金额，单位为人民币分 |
| `price_version` | string | 否 | 当前为 `credits-per-second-60-v1` | 本任务冻结的价格版本 |
| `settlement_status` | string | 否 | `pending`、`final` | 是否已经最终结算 |
| `settled_credits` | integer | 是 | 成功为 `quoted_credits`，明确失败为 `0`，未结算为 `null` | 最终消费积分 |
| `settled_amount_minor` | integer | 是 | 成功为 `quoted_amount_minor`，明确失败为 `0`，未结算为 `null` | 最终金额，单位为人民币分 |

### `result.video`

| 字段 | 类型 | 可为空 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `content_url` | string | 否 | `/api/v1/.../content` 相对路径 | 视频下载地址，使用时需加环境域名 |
| `content_type` | string | 否 | 当前固定为 `video/mp4` | 视频 MIME 类型 |
| `size_bytes` | integer | 否 | 正整数 | 文件大小，单位为字节 |
| `sha256` | string | 否 | 64 位小写十六进制字符 | 文件 SHA-256，可用于完整性校验 |
| `duration_seconds` | number | 否 | 大于 `0` | 最终视频时长，单位为秒 |
| `expires_at` | null | 是 | 当前固定为 `null` | 当前成片没有自动过期时间 |

### `error`

| 字段 | 类型 | 可为空 | 说明 |
|---|---|---:|---|
| `code` | string | 否 | 稳定的任务错误码 |
| `message` | string | 否 | 供人阅读的错误说明 |
| `retryable` | boolean | 否 | 当前任务是否能被客户端自动重试；当前返回均为 `false` |

任务失败只返回公开稳定错误码：调用方可处理的原因会明确分类，其余内部失败统一为 `generation_failed`；不返回内部节点名、供应商原始响应或堆栈。

| 错误码 | 含义 | 调用方处理 |
|---|---|---|
| `generation_failed` | 生成过程明确失败，原因不适合由调用方处理 | 停止轮询；需要重做时使用新 Key 创建任务 |
| `image_rejected_by_provider` | 参考图或处理后的图片未通过供应商审核 | 更换或调整图片后，使用新 Key 创建任务 |
| `video_rejected_by_provider` | 视频内容未通过供应商审核 | 调整视频素材后，使用新 Key 创建任务 |
| `video_audio_required` | 视频没有可用音轨 | 上传带口播音轨的视频后，使用新 Key 创建任务 |
| `video_audio_unsupported` | 视频音频模式不受支持 | 更换视频后，使用新 Key 创建任务 |
| `video_duration_exceeds_h3_limit` | 视频时长超过上限 | 裁剪视频后，使用新 Key 创建任务 |
| `video_duration_invalid` | 无法确认视频时长 | 更换视频后，使用新 Key 创建任务 |
| `submission_outcome_unknown` | 供应商是否接单仍待对账 | 继续查询原任务，禁止创建替代任务 |

## 积分与结算规则

- 100 积分 = 1 元。
- 每个任务报价为原视频时长向上取整秒数 × 60 积分；24.37 秒按 25 秒计算，为 1,500 积分，即 15 元。
- 创建报价只保存媒体、参数和报价；不冻结积分、不启动生成。报价有效期 24 小时，确认时重新检查可用余额。
- 创建任务时先从可用积分转入冻结积分。
- 成功后，冻结积分转为已消费积分。
- 明确失败后，冻结积分退回可用积分，最终消费为 0。
- `submission_unknown` 期间积分保持冻结，等待同一任务对账。

`submission_unknown` 不是失败，也不表示上游没有收到任务。接入方只能继续查询原任务，不能使用新 Key 创建替代任务。

## 错误响应

接口级错误使用统一 JSON 格式：

```http
HTTP/1.1 422 Unprocessable Entity
Content-Type: application/json
X-Request-ID: req_0123456789abcdef0123456789abcdef
```

```json
{
  "error": {
    "code": "invalid_target_language",
    "message": "target_language 不能为空",
    "field": "target_language",
    "request_id": "req_0123456789abcdef0123456789abcdef"
  }
}
```

| 字段 | 类型 | 必有 | 说明 |
|---|---|---:|---|
| `error.code` | string | 是 | 可供程序判断的稳定错误码 |
| `error.message` | string | 是 | 供人阅读的错误说明 |
| `error.field` | string | 否 | 与错误直接相关的请求参数 |
| `error.request_id` | string | 是 | 服务端请求 ID；联系支持时请提供 |

### 常见错误

| HTTP | 错误码 | 含义 | 接入方处理 |
|---:|---|---|---|
| `400` | `invalid_idempotency_key` | Key 缺失或格式错误 | 修正 Key 格式 |
| `400` | `invalid_multipart` | multipart 边界缺失或正文损坏 | 让 HTTP 客户端重新生成 multipart 请求 |
| `400` | `invalid_content_length` | 创建请求的 Content-Length 无效 | 让 HTTP 客户端生成合法请求头后重试 |
| `401` | `invalid_api_key` | API Key 无效或已停用 | 检查鉴权信息或联系支持 |
| `402` | `insufficient_credits` | 可用积分少于本次报价 | 充值后确认仍有效的原报价；直接创建则使用原 Key 和原输入重试 |
| `404` | `quote_not_found` | 报价不存在或不属于当前账户 | 检查报价 ID 和 API Key |
| `410` | `quote_expired` | 未确认报价已过期 | 用新 Key 重新报价并取得用户同意 |
| `404` | `job_not_found` | 任务不存在或不属于当前账户 | 检查任务 ID和所用 API Key |
| `404` | `endpoint_not_found` | 请求路径不是 v1 接口 | 修正请求路径 |
| `405` | `method_not_allowed` | HTTP 方法不受支持 | 按 `Allow` 响应头改用正确方法 |
| `409` | `idempotency_key_reused` | Key 已绑定到不同输入 | 原业务恢复原输入；新业务使用新 Key |
| `409` | `result_not_ready` | 任务尚未成功 | 继续查询任务，不要提前下载 |
| `413` | `source_too_large` | 上传或链接的视频超过大小限制 | 压缩或更换视频 |
| `413` | `request_too_large` | 整个 HTTP 请求正文超过网关限制 | 减小上传文件后重试 |
| `415` | `unsupported_source_media_type` | 视频文件格式不支持 | 使用 MP4、MOV 或 WebM |
| `415` | `unsupported_replacement_media_type` | 参考图声明的媒体类型不受支持 | 使用 JPEG、PNG 或 WebP |
| `415` | `unsupported_content_type` | 直接创建或报价请求不是 multipart | 使用 `multipart/form-data` 并让客户端生成 boundary |
| `415` | `unsupported_media_type` | 确认报价请求不是 JSON | 使用 `Content-Type: application/json` |
| `416` | `invalid_range` | Range 格式或范围无效 | 改为合法的单一字节范围 |
| `422` | `source_exactly_one_required` | 视频上传和 URL 没有正确二选一 | 只提供一个来源 |
| `422` | `replacement_pair_required` | 参考图和说明没有成对提供 | 两个字段同时提供或同时省略 |
| `422` | `invalid_target_language` | 目标语言为空 | 提供非空值或省略字段 |
| `422` | `invalid_aspect_ratio` | 输出画幅不受支持 | 使用能力接口列出的画幅 |
| `422` | `invalid_resolution` | 输出清晰度不受支持 | 使用能力接口列出的清晰度 |
| `422` | `target_language_too_long` | 目标语言超过 80 个 UTF-16 字符单元 | 缩短内容 |
| `422` | `invalid_dialogue_mode` | 台词模式不受支持 | 使用能力接口列出的模式 |
| `422` | `invalid_dialogue_brief` | `generate` 模式缺少台词描述，或非 `generate` 模式携带了 `dialogue_brief` | 按模式规则提供或省略 |
| `422` | `dialogue_brief_too_long` | 台词描述超过 1,000 个 UTF-16 字符单元 | 缩短内容 |
| `422` | `replacement_instruction_too_long` | 替换说明超过 1,000 个 UTF-16 字符单元 | 缩短内容 |
| `422` | `invalid_source_video_url` | URL 不符合公网 HTTPS 规则、无法访问，或解析到的内容不是视频（登录墙/错误页） | 修改 URL，或直接上传视频文件 |
| `422` | `invalid_create_request` | 包含未知或重复字段，或 `replacement_image` 超过 4 张 | 仅重复提交 `replacement_image`，且最多 4 张；其他字段各提交一次 |
| `422` | `invalid_generation_request` | 生成参数组合无效 | 根据能力接口重新提交参数 |
| `422` | `invalid_request` | 请求字段类型或结构不符合 v1 合同 | 按字段定义修正请求 |
| `422` | `invalid_confirm_request` | 确认 JSON 损坏、重复字段或不严格等于 `{"confirm":true}` | 修正后确认原报价 |
| `422` | `video_duration_invalid` | 报价无法取得合法时长，或时长超过 300 秒 | 更换或裁剪视频后使用新 Key 报价 |
| `422` | `invalid_limit` | 流水 `limit` 不在 1～100 | 修改查询参数 |
| `422` | `invalid_query_parameters` | 接口收到未知或重复 Query 参数 | 只提交该接口文档列出的参数一次 |
| `429` | `rate_limit_exceeded` | 请求频率过高 | 按 `Retry-After` 等待 |
| `429` | `download_limit_exceeded` | 同账户下载并发超过 2 | 按 `Retry-After` 等待 |
| `503` | `api_key_registry_unavailable` | API Key 注册表暂时不可用 | 保留原请求，稍后重试 |
| `503` | `result_temporarily_unavailable` | 成片正在校验或暂时不可读 | 稍后查询原任务 |
| `503` | `billing_state_unavailable` | 积分状态暂时不可用 | 确认流程保留原 quote_id；直接创建保留原 Key，稍后重试或联系支持 |
| `503` | `job_state_invalid` | 任务公开状态暂时不可用 | 继续查询原任务；持续发生时提供 `request_id` 联系支持 |
| `503` | `download_state_unavailable` | 下载并发状态暂时不可用 | 稍后重试下载 |
| `503` | `creation_failed` | 创建阶段发生未分类内部错误 | 确认流程保留原 quote_id；直接创建或报价保留原 Key 和原输入，稍后重试 |
| `503` | `quote_state_unavailable` | 报价查询或确认所需状态暂不可用 | 保留原 quote_id，稍后查询或重试确认 |

所有公共 API 响应都会携带 `X-Request-ID`。建议接入方记录它，但不要记录 `Authorization` 请求头或完整 API Key。

## 请求频率限制

限制按账户计算：

| 类型 | 限制 |
|---|---:|
| 创建任务、创建报价和确认报价共享此配额 | 每分钟最多 10 次 |
| 查询类请求 | 每分钟最多 120 次 |
| 视频下载 | 同时最多 2 个连接 |

收到 `429` 后读取 `Retry-After`，等待指定秒数再试。不要进行无间隔循环重试。

## 推荐的客户端逻辑

```text
生成一个与本次报价绑定的 Idempotency-Key

创建报价（上传文件或提交 URL，以及完整生成参数）
  如果连接失败或超时：
    使用相同 Key 和相同输入重试报价
  如果返回 201 或 200：
    保存 quote_id，展示时长、计费秒数、预计积分、有效期

用户不同意：不确认，结束
用户同意：确认同一个 quote_id，Body 为 {"confirm": true}
  如果连接失败或超时：重试确认同一个 quote_id
  如果 402：充值后在有效期内确认原报价
  如果 410：重新报价，再取得用户同意
  如果 201 或 200：
    保存 job_id = 响应的 id（Job 字段）；查询报价时对应字段为 job_id

循环查询同一个 job_id
  queued / running：等待 poll_after_seconds 后继续
  submission_unknown：继续查询并告警，禁止创建替代任务
  failed：结束；如业务决定重新生成，再使用新 Key 创建新任务
  succeeded：下载 content_url，校验 size_bytes 或 sha256，然后结束
```

## 当前 v1 边界

当前版本不提供任务列表、取消任务、手工重试和 webhook。接入方必须在创建成功后持久化任务 ID，并使用查询接口轮询。

## 上线前检查清单

- API Key 只保存在服务端密钥系统中；
- 每个业务生成操作都有稳定且唯一的 `Idempotency-Key`；
- 报价流程已展示预计积分，且仅在用户同意后确认；
- 已持久化报价 ID，确认重试始终使用同一报价 ID；
- 网络超时会使用原 Key 和原输入重试；
- 已持久化创建响应中的任务 ID；
- 轮询间隔不短于 5 秒；
- 已正确处理 `submission_unknown`，不会自动新建任务；
- 只在 `succeeded` 后下载视频；
- 下载后会校验 `size_bytes` 或 `sha256`；
- 能区分可用、冻结和已消费积分；
- 日志记录 `X-Request-ID`，但不记录 API Key。
