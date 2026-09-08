# 对外 API v1 文档差异清单（1.0.0 → 1.1.0）

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
- `GET /account/credits`、`GET /account/credit-transactions` 及全部积分与结算规则（1,000 积分/任务、reserve/capture/release）；
- Job 对象结构、`status` 状态机（含 `submission_unknown` 语义）和任务级错误码表；
- 请求频率限制（创建 10 次/分钟、查询 120 次/分钟、下载并发 2）；
- `/api/v1/openapi.json`。
