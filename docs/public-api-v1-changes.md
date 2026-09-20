# 对外 API v1 客户快速迁移指南与差异清单

## 提示词审核（2026-09-21，合同版本 1.4.0）

本次新增可选提示词审核开关 `prompt_review`，默认 `false`（直通，行为与之前完全一致），旧调用无需修改。开启后任务在付费生成前暂停，由接入方审核/编辑 AI 生成的提示词主体并确认后继续。

| 位置 | 客户适配 |
|---|---|
| `POST /api/v1/video-generations` 与 `POST .../quotes` | 需要审核时添加 multipart 文本字段 `prompt_review=true`；省略或 `false` 即保持原直通行为 |
| `GET /api/v1/video-generations/{job_id}` | `status` 枚举新增 `awaiting_prompt_review`（非终态；不确认则任务静止，不发生后续付费生成） |
| `GET /api/v1/video-generations/{job_id}/prompt-review` | 新增：读取每段提示词的可编辑部分（不含后端注入内容和主体末尾的权威替换计划块）；仅 `awaiting_prompt_review` 与 `succeeded`（只读展示）可读，其余状态 `409 prompt_review_not_available` |
| `POST /api/v1/video-generations/{job_id}/prompt-review/confirm` | 新增：JSON `{"segments":[{"segment_index":1,"prompt":"..."}]}`，段集合须与读取结果一致、`prompt` 只含可编辑部分（固定部分由后端用冻结原文拼接，请求文本无法改动）；成功 `202` 回到 `queued` 继续生成；非审核暂停状态 `409 prompt_review_not_pending` |

最小审核示例：

```bash
# 1. 创建（或报价）时开启审核
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Idempotency-Key: review_order_20260921_000001' \
  --form 'source_video_url=https://media.example.com/source.mp4' \
  --form 'prompt_review=true'

# 2. 轮询任务直到 status == awaiting_prompt_review，然后读取提示词主体
curl 'https://8.166.140.227:3211/api/v1/video-generations/vg_xxx/prompt-review' \
  --header 'Authorization: Bearer YOUR_API_KEY'

# 3. 编辑后确认继续
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations/vg_xxx/prompt-review/confirm' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{"segments":[{"segment_index":1,"prompt":"编辑后的第 1 段主体"}]}'
```

## 内部管线更新（2026-09-16，合同版本保持 1.3.0）

本次部署只更新服务端内部生成管线（人脸与物品统一检测遮盖、网络错误自动重试、出口链路代理），接口路径、请求字段、响应结构、错误码、幂等与计费规则全部不变；客户无需任何适配，1.3.0 合同继续有效。

## 多参考图（2026-09-15，合同版本 1.3.0）

本次新增多参考图能力：同一 multipart 字段 `replacement_image` 可重复传入 2～4 张图片，配合一条 `replacement_instruction` 对全部参考图生效。默认行为不变；单图调用一字不改即可继续使用，旧调用无需修改。

| 位置 | 客户适配 |
|---|---|
| `POST /api/v1/video-generations` | 需要多图时重复传入 `replacement_image` 字段，共 2～4 张；单图请求保持原写法 |
| `POST /api/v1/video-generations/quotes` | 同样支持重复图片字段；报价冻结全部图片及其顺序，更改任一图片、数量、顺序或说明时使用新 Key 重新报价 |
| `POST /api/v1/video-generations/quotes/{quote_id}/confirm` | 仍只发送 JSON `{"confirm":true}`，不重传或修改参考图和说明 |
| `GET /api/v1/video-generations/capabilities` | 读取 `replacement_image.max_images`，当前为 `4`；每张仍限 JPEG、PNG、WebP 和 10 MiB |
| 创建和报价请求校验 | 图片与说明仍须成对提供；超过 4 张返回 `422 invalid_create_request` |

最小多图创建示例：

```bash
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Idempotency-Key: multiref_order_20260915_000001' \
  --form 'source_video=@/home/xy/api-input/source.mp4' \
  --form 'replacement_image=@/home/xy/api-input/reference-1.jpg' \
  --form 'replacement_image=@/home/xy/api-input/reference-2.png' \
  --form 'replacement_instruction=使用图 1 和图 2 中的商品作为替换参考'
```

## 字幕开关补充（2026-09-10，合同版本保持 1.2.0）

本次增加可选 `add_subtitles` 参数，用于在最终成片叠加台词字幕；默认 `false`，旧调用无需修改。接口路径、报价确认流程、积分算法及轮询下载方式不变。

| 位置 | 客户适配 |
|---|---|
| `POST /api/v1/video-generations` | 需要字幕时添加 multipart 文本字段 `add_subtitles=true` |
| `POST /api/v1/video-generations/quotes` | 同样支持该字段，并在报价时冻结；更改开关需要新 Key 重新报价 |
| `POST /api/v1/video-generations/quotes/{quote_id}/confirm` | 仍只发送 JSON `{"confirm":true}`，不添加字幕参数 |
| `GET /api/v1/video-generations/capabilities` | `defaults.add_subtitles` 为 JSON 布尔值 `false` |
| 创建、确认和查询任务的 Job 响应 | `parameters.add_subtitles` 返回 JSON 布尔值；历史任务默认 `false` |
| Quote 响应 | 字段不变，不单独返回字幕开关；创建的 Job 返回冻结值 |

上传/报价的字段逻辑类型为 boolean，但请求编码仍为 multipart，使用表单文本 `true` 或 `false`；不是将创建请求改成 JSON。最小报价示例：

```bash
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations/quotes' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Idempotency-Key: subtitle_quote_20260910_000001' \
  --form 'source_video_url=https://media.example.com/source.mp4' \
  --form 'add_subtitles=true'
```

用户同意积分后按下文确认流程执行。省略或显式 `false` 均关闭字幕；同一幂等 Key 改成 `true` 返回 `409 idempotency_key_reused`。字幕不额外改变报价公式，仍为 `ceil(原视频时长秒数) × 60`。

## 1.1.0 → 1.2.0（2026-09-09）

本节说明 1.2.0 新增接口与客户适配方式。现有 1.1.0 接口保持兼容。

本版本新增可选的“报价 → 用户确认 → 执行”流程，生成管线和旧的直接创建接口保持原有行为。

### 客户最少需要改什么

| 流程 | 调用顺序 |
|---|---|
| 原流程，继续支持 | 上传文件或提交 URL → `POST /api/v1/video-generations` → 获得 Job → 轮询与下载 |
| 新流程，1.2.0 新增 | 上传文件或提交 URL → `POST /api/v1/video-generations/quotes` → 展示预计积分 → 用户同意 → `POST /api/v1/video-generations/quotes/{quote_id}/confirm` → 获得 Job → 原有轮询与下载 |

不需要“执行前确认积分”的接入方无需修改现有调用。接入新流程只需调整以下四处：

1. 将原上传请求的目标改为 `/api/v1/video-generations/quotes`，复用原 multipart 字段、API Key 和创建幂等 Key 规则。
2. 将首次响应改按 Quote 读取：持久化 `quote_id`，展示时长 `duration_seconds`、计费秒数 `billable_seconds`、预计积分 `estimated_credits` 和有效期 `expires_at`。
3. 用户同意后用 `quote_id` 调用确认接口，仅发送 JSON `{"confirm":true}`；不再上传文件或提交生成参数。不接受报价时不调用确认即可，无需取消接口。
4. 将确认响应的 `id` 交给原有 Job 轮询和下载代码；这部分接口与状态处理保持原样。

同一业务选择一种创建流程，不能报价确认后再调用旧的直接创建接口，否则会另外创建任务。

### 输入与响应对照

| 项目 | 原直接创建 | 新报价与确认 |
|---|---|---|
| 视频来源 | `source_video` 或 `source_video_url` 二选一 | 报价时原样复用，确认时不重传、不重新下载 URL |
| 生成参数 | `aspect_ratio`、`resolution`、`target_language`、`dialogue_mode`、`dialogue_brief` | 报价时按原默认值与约束提交，之后冻结；修改需要重新报价 |
| 替换输入 | `replacement_image` 与 `replacement_instruction` 成对提供 | 报价时原样复用，确认时不重传 |
| 首次响应 | Job，任务 ID 字段为 `id` | Quote，报价 ID 字段为 `quote_id`，此时 `job_id` 为 `null` |
| 确认响应 | 无 | 原有完整 Job，任务 ID 字段为 `id`；首次 `201`，重放 `200` |
| 查询报价 | 无 | `GET /api/v1/video-generations/quotes/{quote_id}` 返回 Quote；确认后 `job_id` 指向已创建任务 |
| 任务轮询与下载 | 使用 Job 的 `id` | 无变化；报价的 `confirmed` 仅代表已建任务，不能视为 Job 的 `succeeded` |

### 最小请求示例

接口地址为 `https://8.166.140.227:3211`。示例 API Key 和 ID 均为占位值。报价请求示例使用 URL；上传文件时，将 `source_video_url` 字段替换为 `source_video` 文件字段即可，完整限制见主合同。

```bash
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations/quotes' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Idempotency-Key: quote_order_20260909_000001' \
  --form 'source_video_url=https://media.example.com/source.mp4' \
  --form 'target_language=中文'
```

首次返回 `201 Created`，相同 Key 和输入重放返回 `200 OK`。Quote 响应示例：

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

用户同意后确认，路径换成响应中的真实 `quote_id`。确认不需要 `Idempotency-Key`；JSON 必须严格为布尔值 `true`，不接受价格、生成参数或其他字段：

```bash
curl --request POST 'https://8.166.140.227:3211/api/v1/video-generations/quotes/vq_0123456789abcdef0123456789abcdef/confirm' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{"confirm":true}'
```

首次返回 `201 Created`，重复确认返回 `200 OK` 和同一任务。以下仅摘录完整 Job 响应中的任务 ID 与状态，实际响应还包含主合同定义的 `billing`、`parameters` 等字段：

```json
{
  "id": "vg_0123456789abcdef0123456789abcdef",
  "status": "queued"
}
```

接入方保存 `job_id = 确认响应.id`，随后调用原查询接口 `GET /api/v1/video-generations/{job_id}`，每 5～10 秒轮询，成功后使用原下载接口。

### 价格、错误与重试要点

报价不启动生成、不占用生成执行槽位、不冻结或扣除积分；余额不足也可报价。计价继续采用 `ceil(原视频时长秒数) × 60`：24.37 秒按 25 秒计，为 1,500 积分。报价 24 小时内锁定积分和价格版本，确认所得 `Job.billing.quoted_credits` 等于 `Quote.estimated_credits`；未来价格变化只影响新报价，但报价不保证确认时余额足够。

| 情况 | 返回或行为 | 客户处理 |
|---|---|---|
| 报价网络超时 | 可能已保存报价 | 用原 Key、原输入重试；报价和旧创建的 Key 各自独立判重 |
| 报价 Key 搭配不同输入 | `409 idempotency_key_reused` | 原业务恢复原输入；修改输入用新 Key 报价 |
| 确认网络超时或并发重复确认 | 原报价只创建一个任务 | 重试同一 `quote_id`，或 GET 原报价读取 `job_id`；不改用新报价或旧创建接口 |
| 余额不足 | `402 insufficient_credits` | 充值后在有效期内确认原报价 |
| 报价不存在或不属于账户 | `404 quote_not_found` | 检查报价 ID 和 API Key；所有新增接口都需要 Bearer 鉴权 |
| 未确认报价过期 | GET 返回 `status: "expired"`；确认返回 `410 quote_expired` | 用新 Key 重新报价，再取得用户同意；原 Key 重放不延长有效期 |
| 已确认报价超过截止时间 | 仍为 `confirmed`，保留同一 `job_id` | 可继续 GET 或确认重放；生成结果仍查 Job 状态 |
| 确认编码或正文错误 | `415 unsupported_media_type` 或 `422 invalid_confirm_request` | 使用 `application/json` 和严格 `{"confirm":true}`，不添加其他字段 |
| 报价时长无效或超过 300 秒 | `422 video_duration_invalid` | 更换或裁剪视频后用新 Key 报价 |
| 请求限流 | `429 rate_limit_exceeded` | 按 `Retry-After` 等待；报价、确认、旧创建共享账户 10 次/分钟，报价 GET 共享查询 120 次/分钟 |
| 报价状态暂不可用 | `503 quote_state_unavailable` | 保留原 `quote_id`，稍后 GET 或重试确认 |
| 其他暂时性创建或积分错误 | `503 creation_failed`、`503 billing_state_unavailable` | 确认保留原 `quote_id`；报价或旧创建保留原 Key 和输入重试 |

上述新接口不接受 Query 参数；错误响应继续使用主合同的 `error.code`、`error.message`、`error.request_id` 格式。完整字段、输入约束及错误表见 [主 API 合同](/home/xy/duet-ad1-h3-max-r2v-quotes/docs/public-api-v1.md)。

### 本次完整差异

| 接口/行为 | 1.1.0 | 1.2.0 |
|---|---|---|
| `POST /api/v1/video-generations/quotes` | 无 | 使用与直接创建相同的 multipart 输入和必填 `Idempotency-Key`，接收文件或下载 URL、读取时长、保存媒体和参数；首次 `201`，同 Key 同输入重放 `200`，不同输入 `409 idempotency_key_reused` |
| 报价内容 | 只能在创建任务后读取任务报价 | 返回 `quote_id`、`status`、`duration_seconds`、`billable_seconds`、`estimated_credits`、`credits_per_second`、`price_version`、`expires_at`、`job_id` |
| `GET /api/v1/video-generations/quotes/{quote_id}` | 无 | 查询报价和确认状态；状态为 `quoted`、`confirmed` 或 `expired`；已确认时返回原 `job_id` |
| `POST /api/v1/video-generations/quotes/{quote_id}/confirm` | 无 | 严格 JSON `{"confirm":true}`；复用已保存的媒体和参数，检查余额后冻结积分并提交到原有生成流程；不重新上传、下载 URL 或接收客户端价格 |
| 确认幂等 | 无 | 报价 ID 本身保证幂等，无需幂等头；首次 `201`，重复确认 `200`，均返回同一 Job，不重复冻结或生成 |
| 报价积分影响 | 无独立报价流程 | 报价阶段不创建任务、不生成、不冻结、不扣积分；不要求余额足够，不保证确认时余额足够 |
| 有效期与账户隔离 | 无 | 待确认报价有效期 24 小时；确认过期报价返回 `410 quote_expired`；跨账户或不存在返回 `404 quote_not_found`；已确认报价过期后仍可查询和重放确认 |
| 锁价 | 无独立报价 | 有效期内沿用报价积分和价格版本；后续价格变化仅影响新报价，不保证未来余额 |
| 确认格式错误 | 无 | 非 JSON 返回 `415 unsupported_media_type`；JSON 损坏、重复字段、非严格 `{"confirm":true}` 返回 `422 invalid_confirm_request`，使用现有错误响应格式 |
| 报价状态暂不可用 | 无 | 查询/确认返回 `503 quote_state_unavailable`；保留原报价 ID 重试 |
| 请求频率 | 创建 10 次/分钟，查询 120 次/分钟 | 报价、确认与直接创建共享账户每分钟 10 次；报价查询共享查询额度；超限沿用 `429 rate_limit_exceeded` 与 `Retry-After` |
| 直接创建与生成流程 | 创建即执行 | 保留兼容行为；旧调用不被强制改为两阶段 |

计费算法没有改变：`ceil(原视频时长秒数) × 60`，例如 `24.37 → 25 → 1500` 积分。此次同步修正文档中遗留的“固定 1,000 积分”和“余额不足 1,000”的旧说明，包括 Job billing 字段、余额说明与结算规则。

接入方应持久化报价 ID，向用户展示时长、计费秒数、预计积分及有效期。确认超时应重试原报价，不能创建替代任务；未确认报价过期后需要用新 Key 重新报价并取得用户同意。报价 Key 重放不延长有效期；已确认或过期时仍返回原报价的当前状态。确认返回 Job 的 `id`，报价查询字段为 `job_id`，二者指向同一任务；报价 `confirmed` 不代表生成成功。

## 历史附录：1.0.0 → 1.1.0（非本次变更）

对比基准：旧文档 `docs/public-api-v1.md` 合同版本 `1.0.0`（2026-09-04，duet-ad1 仓库已发布版本）；
当时的新文档合同版本 `1.1.0`（2026-09-08），以下为该次发布的历史差异，不属于本次报价与确认改动。

### 历史接口与字段

| 接口/字段/行为 | 旧（1.0.0） | 新（1.1.0） |
|---|---|---|
| `GET capabilities` 响应 `defaults` | 仅 `aspect_ratio`、`resolution`、`target_language` | 新增 `dialogue_mode: "auto_rewrite"` |
| `GET capabilities` 响应 | 无台词约束块 | 新增 `dialogue` 块：`modes`（`auto_rewrite`、`generate`）、`default_mode`、`brief` 约束（`generate` 必填、最多 1,000 个 UTF-16 字符单元） |
| `POST /video-generations` Body | 无台词字段，台词固定按原视频自动改写 | 新增可选 `dialogue_mode`（`auto_rewrite`/`generate`，默认 `auto_rewrite`） |
| `POST /video-generations` Body | 无 | 新增条件必填 `dialogue_brief`：`dialogue_mode=generate` 时必填且非空（最多 1,000 个 UTF-16 字符单元）；其余模式不允许提供 |
| 台词生成行为 | 只支持翻译并自然改写原台词 | `auto_rewrite` 保持原行为；`generate` 按 `dialogue_brief` 描述自动生成全新口播台词 |
| 幂等判重输入 | 不含台词参数 | 相同 `Idempotency-Key` 的输入指纹包含 `dialogue_mode`、`dialogue_brief`；台词参数不同视为不同输入，返回 `409 idempotency_key_reused` |
| 任务定价 | 固定 1,000 积分/任务（`price_version=credits-fixed-1000-v1`） | 按原视频时长计费：向上取整秒数 × 60 积分（`price_version=credits-per-second-60-v1`）；`capabilities.pricing` 改为 `credits_per_second` + `billed_unit`；`Billing.quoted_credits` 等字段随任务时长变化，不再固定 1,000 |
| 联调环境 | `https://8.166.140.227:3213` | `https://8.166.140.227:3211`（文档内所有示例 Host 统一为 3211） |

### 历史错误码

| 错误码 | 旧（1.0.0） | 新（1.1.0） |
|---|---|---|
| `422 invalid_dialogue_mode` | 无 | 新增：`dialogue_mode` 取值不受支持 |
| `422 invalid_dialogue_brief` | 无 | 新增：`generate` 模式缺少非空 `dialogue_brief`，或非 `generate` 模式携带了 `dialogue_brief` |
| `422 dialogue_brief_too_long` | 无 | 新增：`dialogue_brief` 超过 1,000 个 UTF-16 字符单元 |
| `400 invalid_content_length` | 无 | 新增：创建请求的 `Content-Length` 缺失或无法解析为有效值时由网关层返回 |
| `415 unsupported_replacement_media_type` | 未列出 | 补充列出：参考图声明的媒体类型不是 JPEG/PNG/WebP（实现原已返回，旧文档遗漏） |
| `429 queue_full` | 创建时可能返回"生成队列已满" | 移除：当前实现不再有此错误码 |
| `422 video_duration_exceeds_h3_limit`（创建时） | 创建任务时可同步返回 | 移除：时长超限改为任务级失败，任务进入 `failed`，通过 `Job.error.code = video_duration_exceeds_h3_limit` 返回 |
| `422 invalid_source_media` | 创建任务时可能返回 | 移除：创建阶段不再单独返回；未分类内部失败统一收敛为 `503 creation_failed` |
| `422 invalid_replacement_image` | 创建任务时可能返回 | 移除：创建阶段只校验参考图声明的媒体类型（见 `unsupported_replacement_media_type`）；解码类内部失败统一收敛为 `503 creation_failed` |

### 历史无变化部分

以下接口和规则 1.0.0 → 1.1.0 完全兼容，未做任何调整：

- 鉴权方式（HTTP Bearer API Key）与 `Idempotency-Key` 规则；
- `GET /video-generations/{job_id}`、`GET/HEAD .../content`、下载 Range 与并发限制；
- `GET /account/credits`、`GET /account/credit-transactions` 的结构及 reserve/capture/release 结算机制（单任务报价算法的变化见上表）；
- Job 对象结构、`status` 状态机（含 `submission_unknown` 语义）和任务级错误码表；
- 请求频率限制（创建 10 次/分钟、查询 120 次/分钟、下载并发 2）；
- `/api/v1/openapi.json`。
