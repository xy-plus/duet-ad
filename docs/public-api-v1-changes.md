# 对外 API v1 文档差异清单

## 1.1.0 → 1.2.0（2026-09-09）

本节为已定设计、待上线合同，供客户提前适配；新增接口尚未在 3211 发布，启用时间另行通知。现有 1.1.0 接口继续可用。

拟新增可选的“报价 → 用户确认 → 执行”流程，生成管线和旧的直接创建接口保持原有行为。

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

## 1.0.0 → 1.1.0（历史记录）

对比基准：旧文档 `docs/public-api-v1.md` 合同版本 `1.0.0`（2026-09-04，duet-ad1 仓库已发布版本）；
新文档合同版本 `1.1.0`（2026-09-08），逐条以当前实现 `app/public_api.py`、`app/main.py`、`app/public_credits.py` 核实。

## 接口与字段

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

## 错误码

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

## 无变化部分

以下接口和规则 1.0.0 → 1.1.0 完全兼容，未做任何调整：

- 鉴权方式（HTTP Bearer API Key）与 `Idempotency-Key` 规则；
- `GET /video-generations/{job_id}`、`GET/HEAD .../content`、下载 Range 与并发限制；
- `GET /account/credits`、`GET /account/credit-transactions` 的结构及 reserve/capture/release 结算机制（单任务报价算法的变化见上表）；
- Job 对象结构、`status` 状态机（含 `submission_unknown` 语义）和任务级错误码表；
- 请求频率限制（创建 10 次/分钟、查询 120 次/分钟、下载并发 2）；
- `/api/v1/openapi.json`。
