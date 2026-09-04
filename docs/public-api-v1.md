# Duet 视频生成 API v1

Duet API 用于提交视频生成任务，并在任务完成后返回最终视频和费用。

视频生成通常需要数分钟，因此接口采用异步模式：创建任务后立即返回任务 ID；接入方使用任务 ID 查询状态；成功后再下载视频。

## 基本信息

| 项目 | 值 |
|---|---|
| 合同版本 | `1.0.0` |
| 文档更新日期 | `2026-09-04` |
| 联调环境 | `https://8.166.140.227:3213` |
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

1. 查询积分，确认 `available_credits` 不少于 1,000。
2. 创建任务，并保存返回的任务 `id`。
3. 每隔 5～10 秒查询该任务。
4. 状态变成 `succeeded` 后，使用 `result.video.content_url` 下载视频。

创建请求如果遇到网络超时，必须使用原来的 `Idempotency-Key` 和完全相同的输入重试。不要直接换新 Key，否则可能创建两个收费任务。

## 接口一览

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/v1/video-generations/capabilities` | 查询当前参数限制和价格 |
| `POST` | `/api/v1/video-generations` | 创建异步生成任务 |
| `GET` | `/api/v1/video-generations/{job_id}` | 查询任务状态、结果和费用 |
| `GET` | `/api/v1/video-generations/{job_id}/content` | 下载视频，支持单段 Range |
| `HEAD` | `/api/v1/video-generations/{job_id}/content` | 读取视频元数据，不下载正文 |
| `GET` | `/api/v1/account/credits` | 查询积分余额 |
| `GET` | `/api/v1/account/credit-transactions` | 查询积分流水 |
| `GET` | `/api/v1/openapi.json` | 获取公共 API 的 OpenAPI 描述 |

---

## 查询生成能力

返回当前支持的参数、输入限制、价格和建议轮询间隔。建议接入方读取本接口，不要把价格和参数范围永久写死。

```http
GET /api/v1/video-generations/capabilities HTTP/1.1
Host: 8.166.140.227:3213
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
    "target_language": null
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
    "max_instruction_chars": 1000
  },
  "target_language": {
    "omitted_means": "same_as_source",
    "max_chars": 80
  },
  "pricing": {
    "credits_per_cny": 100,
    "job_price_credits": 1000,
    "job_price_amount_minor": 1000,
    "currency": "CNY",
    "price_version": "credits-fixed-1000-v1"
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
| `allowed_output_combinations` | array<object> | 当前允许的画幅与清晰度组合 |
| `source_video` | object | 原视频格式、大小、时长和 URL 限制 |
| `replacement_image` | object | 参考图格式、大小和说明文字限制 |
| `target_language` | object | 目标语言的省略语义和长度限制 |
| `pricing` | object | 当前积分换算、任务价格和价格版本 |
| `polling` | object | 推荐轮询间隔和最大退避间隔，单位为秒 |

---

## 创建视频生成任务

创建一个异步任务。接口在完成输入校验和积分冻结后立即返回，不会等待视频生成完成。

```http
POST /api/v1/video-generations HTTP/1.1
Host: 8.166.140.227:3213
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
| `source_video` | binary | 条件必填 | 无 | MP4、MOV、WebM；最大 500 MiB；最长 300 秒 | 上传原视频；与 `source_video_url` 必须二选一 |
| `source_video_url` | string | 条件必填 | 无 | 无用户名密码的公网 HTTPS URL | 由服务端下载原视频；与 `source_video` 必须二选一 |
| `aspect_ratio` | string | 否 | `9:16` | `9:16`、`16:9` | 输出画幅 |
| `resolution` | string | 否 | `768p` | `768p`、`480p` | 输出清晰度 |
| `target_language` | string | 否 | 与原视频相同 | 去除首尾空白后不能为空；最多 80 个 UTF-16 字符单元 | 对白目标语言，例如 `中文`、`English` |
| `replacement_image` | binary | 条件必填 | 无 | JPEG、PNG、WebP；最大 10 MiB | 主体替换参考图；必须与 `replacement_instruction` 同时提供 |
| `replacement_instruction` | string | 条件必填 | 无 | 去除首尾空白后不能为空；最多 1,000 个 UTF-16 字符单元 | 描述如何使用参考图；必须与 `replacement_image` 同时提供 |

本接口没有 Path 或 Query 参数。未知字段、重复字段、同时提供两个视频来源或只提供参考图组合中的一个字段，都会返回 `422`。

普通中文、英文和数字通常各占 1 个 UTF-16 字符单元；部分 Emoji 占 2 个。

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
    "replacement_image": true
  },
  "billing": {
    "currency": "CNY",
    "credits_per_cny": 100,
    "quoted_credits": 1000,
    "quoted_amount_minor": 1000,
    "price_version": "credits-fixed-1000-v1",
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
Host: 8.166.140.227:3213
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
    "replacement_image": true
  },
  "billing": {
    "currency": "CNY",
    "credits_per_cny": 100,
    "quoted_credits": 1000,
    "quoted_amount_minor": 1000,
    "price_version": "credits-fixed-1000-v1",
    "settlement_status": "final",
    "settled_credits": 1000,
    "settled_amount_minor": 1000
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
Host: 8.166.140.227:3213
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

余额不足 1,000 积分时，创建任务返回 `402 insufficient_credits`，不会创建任务。

---

## 查询积分流水

按时间从新到旧返回积分事件。

```http
GET /api/v1/account/credit-transactions?limit=50 HTTP/1.1
Host: 8.166.140.227:3213
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
      "credits": 1000,
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
Host: 8.166.140.227:3213
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
Host: 8.166.140.227:3213
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
Host: 8.166.140.227:3213
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
Host: 8.166.140.227:3213
Accept: application/json
```

### 请求参数

#### Header 参数

| 参数 | 类型 | 必填 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `Accept` | string | 否 | 建议 `application/json` | 期望的响应格式 |

本接口没有 Path、Query 或 Body 参数，也不需要 `Authorization`。

### 成功响应

HTTP `200 OK`，响应体为 OpenAPI JSON，版本字段为 `1.0.0`，标题为 `Duet Video Generation API`。

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

### `billing`

| 字段 | 类型 | 可为空 | 可选值或约束 | 说明 |
|---|---|---:|---|---|
| `currency` | string | 否 | 当前固定为 `CNY` | 结算币种 |
| `credits_per_cny` | integer | 否 | 当前固定为 `100` | 1 元对应的积分 |
| `quoted_credits` | integer | 否 | 当前固定为 `1000` | 创建时冻结的报价积分 |
| `quoted_amount_minor` | integer | 否 | 当前固定为 `1000` | 报价金额，单位为人民币分，即 10.00 元 |
| `price_version` | string | 否 | 当前为 `credits-fixed-1000-v1` | 本任务冻结的价格版本 |
| `settlement_status` | string | 否 | `pending`、`final` | 是否已经最终结算 |
| `settled_credits` | integer | 是 | 成功为 `1000`，明确失败为 `0`，未结算为 `null` | 最终消费积分 |
| `settled_amount_minor` | integer | 是 | 成功为 `1000`，明确失败为 `0`，未结算为 `null` | 最终金额，单位为人民币分 |

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
- 每个任务报价 1,000 积分，即 10 元。
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
| `401` | `invalid_api_key` | API Key 无效或已停用 | 检查鉴权信息或联系支持 |
| `402` | `insufficient_credits` | 可用积分少于 1,000 | 充值后使用原 Key 和原输入重试 |
| `404` | `job_not_found` | 任务不存在或不属于当前账户 | 检查任务 ID和所用 API Key |
| `404` | `endpoint_not_found` | 请求路径不是 v1 接口 | 修正请求路径 |
| `405` | `method_not_allowed` | HTTP 方法不受支持 | 按 `Allow` 响应头改用正确方法 |
| `409` | `idempotency_key_reused` | Key 已绑定到不同输入 | 原业务恢复原输入；新业务使用新 Key |
| `409` | `result_not_ready` | 任务尚未成功 | 继续查询任务，不要提前下载 |
| `413` | `source_too_large` | 上传视频超过大小限制 | 压缩或更换视频 |
| `413` | `request_too_large` | 整个 HTTP 请求正文超过网关限制 | 减小上传文件后重试 |
| `415` | `unsupported_source_media_type` | 视频文件格式不支持 | 使用 MP4、MOV 或 WebM |
| `415` | `unsupported_content_type` | 创建请求不是 `multipart/form-data` | 使用 multipart 并让客户端生成 boundary |
| `416` | `invalid_range` | Range 格式或范围无效 | 改为合法的单一字节范围 |
| `422` | `source_exactly_one_required` | 视频上传和 URL 没有正确二选一 | 只提供一个来源 |
| `422` | `replacement_pair_required` | 参考图和说明没有成对提供 | 两个字段同时提供或同时省略 |
| `422` | `invalid_target_language` | 目标语言为空 | 提供非空值或省略字段 |
| `422` | `invalid_aspect_ratio` | 输出画幅不受支持 | 使用能力接口列出的画幅 |
| `422` | `invalid_resolution` | 输出清晰度不受支持 | 使用能力接口列出的清晰度 |
| `422` | `target_language_too_long` | 目标语言超过 80 个 UTF-16 字符单元 | 缩短内容 |
| `422` | `replacement_instruction_too_long` | 替换说明超过 1,000 个 UTF-16 字符单元 | 缩短内容 |
| `422` | `invalid_source_video_url` | URL 不符合公网 HTTPS 规则 | 修改 URL |
| `422` | `invalid_source_media` | 下载或上传的内容不是可读取视频 | 更换视频或检查直链内容 |
| `422` | `invalid_replacement_image` | 参考图格式、内容或解码结果无效 | 更换为可正常打开的 JPEG、PNG 或 WebP 图片 |
| `422` | `video_duration_exceeds_h3_limit` | 视频超过最长 300 秒 | 裁剪视频后重试 |
| `422` | `invalid_create_request` | 包含未知或重复字段 | 只提交文档列出的字段一次 |
| `422` | `invalid_generation_request` | 生成参数组合无效 | 根据能力接口重新提交参数 |
| `422` | `invalid_request` | 请求字段类型或结构不符合 v1 合同 | 按字段定义修正请求 |
| `422` | `invalid_limit` | 流水 `limit` 不在 1～100 | 修改查询参数 |
| `422` | `invalid_query_parameters` | 接口收到未知或重复 Query 参数 | 只提交该接口文档列出的参数一次 |
| `429` | `rate_limit_exceeded` | 请求频率过高 | 按 `Retry-After` 等待 |
| `429` | `queue_full` | 当前生成队列已满 | 稍后使用原 Key 和原输入重试 |
| `429` | `download_limit_exceeded` | 同账户下载并发超过 2 | 按 `Retry-After` 等待 |
| `503` | `api_key_registry_unavailable` | API Key 注册表暂时不可用 | 保留原请求，稍后重试 |
| `503` | `result_temporarily_unavailable` | 成片正在校验或暂时不可读 | 稍后查询原任务 |
| `503` | `billing_state_unavailable` | 积分状态暂时不可用 | 保留原 Key，稍后重试或联系支持 |
| `503` | `job_state_invalid` | 任务公开状态暂时不可用 | 继续查询原任务；持续发生时提供 `request_id` 联系支持 |
| `503` | `download_state_unavailable` | 下载并发状态暂时不可用 | 稍后重试下载 |
| `503` | `creation_failed` | 创建阶段发生未分类内部错误 | 保留原 Key 和原输入，稍后重试 |

所有公共 API 响应都会携带 `X-Request-ID`。建议接入方记录它，但不要记录 `Authorization` 请求头或完整 API Key。

## 请求频率限制

限制按账户计算：

| 类型 | 限制 |
|---|---:|
| 创建任务 | 每分钟最多 10 次 |
| 查询类请求 | 每分钟最多 120 次 |
| 视频下载 | 同时最多 2 个连接 |

收到 `429` 后读取 `Retry-After`，等待指定秒数再试。不要进行无间隔循环重试。

## 推荐的客户端逻辑

```text
生成一个与业务订单绑定的 Idempotency-Key

创建任务
  如果连接失败或超时：
    使用相同 Key 和相同输入重试创建
  如果返回 201 或 200：
    保存 job_id

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
- 网络超时会使用原 Key 和原输入重试；
- 已持久化创建响应中的任务 ID；
- 轮询间隔不短于 5 秒；
- 已正确处理 `submission_unknown`，不会自动新建任务；
- 只在 `succeeded` 后下载视频；
- 下载后会校验 `size_bytes` 或 `sha256`；
- 能区分可用、冻结和已消费积分；
- 日志记录 `X-Request-ID`，但不记录 API Key。
