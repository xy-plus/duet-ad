---
name: generation-safety
type: behavior
status: done
owner: human
updated: 2026-08-25
tdd: N/A
links: [antd-x-frontend, conversation-task]
---

# 提示词、生成参数与付费安全

## 规则

| 当 | 则 |
| --- | --- |
| 分析完成、会话可操作且尚无 generation | 可编辑源提示词；保存必须确认并带当前 SHA-256 |
| 保存提示词遇到 `prompt_changed` | 拉取并显示服务端最新版，禁止自动覆盖或自动重发 |
| 配置短视频生成 | 可选自动/编辑识别/自定义/无台词、`16:9/9:16`、`480p/768p`、`none/crop/pad` |
| 配置长视频生成 | 台词只允许自动或无台词；快速模式默认开启但不显示开关或结果摘要 |
| 服务端建议或冻结字段缺失、越界或相互矛盾 | fail closed，提示刷新；不猜默认值，不展示付费按钮 |
| 用户首次确认生成 | 显示精确付费任务数；长视频同时提交 64 位 `expected_plan_receipt`，任务数未知时禁用确认 |
| generation 已存在 | 编辑器替换为服务端冻结的台词、画幅、清晰度和适配方式；提示词也锁定 |
| 状态为 `resume_required` | “继续原任务”复用旧 `client_request_id` 和全部冻结参数，新增付费数为 0 |
| 长视频失败阶段为 `stitch` | “继续拼接”复用旧 id，只做本地拼接，新增付费数为 0 |
| 服务端确认 generation 失败 | 重试使用新 id；长视频只按 `retry_paid_segment_count` 展示新增付费数，未知时禁用按钮 |
| 状态为 `submission_unknown` 或正在运行 | 不提供提交、继续或重试动作，避免重复付费 |

## 边界

- 前端同一会话同时最多发出一个 submit；TanStack mutation 不自动重试。
- provider-facing POST 前先持久化 reconciliation lease；network/5xx/响应无效/409 `submission_outcome_unknown` 时跨 reload/logout 只做 GET 核对，直到明确响应或同 request id 相对提交前 baseline 出现权威推进才解锁。
- `read_only !== false` 或 `submit_enabled !== true` 时，提示词、生成和后处理全部不可操作，但已有详情与媒体仍可查看。
- `crop` 会裁掉画面边缘，`pad` 保留完整画面并可能留边；界面必须同时暴露两者，不能把 pad 描述为铺满。
- H3 自动补交、attempt receipt 与 `submission_unknown` 的供应商核对规则见 [长视频行为](../../conversation-task/behaviors/long-video.md)，前端不重新解释或放宽。

## 例子

- 输入：30 秒长视频显示 4 个新付费分段 → 输出：确认按钮明确显示 4 个；若 `segment_count` 或 plan receipt 无效则按钮不可用。
- 输入：原任务已全部生成但拼接失败 → 输出：只出现“继续拼接”，沿用旧请求 id，不产生新 H3 子任务。
