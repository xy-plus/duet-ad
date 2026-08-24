---
name: processing-state
type: behavior
status: done
owner: human
updated: 2026-08-21
tdd: N/A
links: [conversation-task, submit-gate]
---

# 准备与生成状态

## 规则

| 阶段 | 可见状态 | 行为 |
| --- | --- | --- |
| 输入准备 | `queued → processing → done/failed` | 抽帧、台词准备、视觉 prompt 和 receipt 完成后才允许生成 |
| 首次人工提交 | `generation: queued → running` | 冻结的 H3 源提示词直接提交 AutoDL H3 |
| 长视频分段生成 | `generation.stage=h3`，每段独立状态 | 同链串行、最多两链并发；成功段保留，确定失败只开放失败段及下游继续 |
| 长视频本地拼接 | `generation.stage=stitch` | 全段成功后去除子片段音轨并拼接；失败用原 request id 只重跑本地拼接，新增付费 H3 子任务为 0 |
| 生成成功 | `generation: succeeded` | 原子落盘 `generated.mp4`，详情 `has_video=true` |
| 已知 task 查询/超时、下载传输/DNS/peer 验证或输出写入/探测基础设施失败 | `generation: resume_required` | 包括 `download_dns_failed/download_peer_unverified/output_probe_failed`；保留原 request id、receipt 和 attempt，只允许原参数继续 |
| 成片 URL、重定向、体积或媒体内容确定拒绝 | `generation: failed` | 安全拒绝码为 `download_url_rejected/download_redirect_rejected/download_too_large/download_invalid_video`；只有人工新 id retry |
| 可确定失败 | `generation: failed` | 展示安全错误码；用户确认后必须用新 request id 创建 retry attempt |
| 供应商 POST 结果未知 | `generation: submission_unknown` | 不猜测是否扣费；隐藏重试入口，所有再次提交返回 409，必须先到供应商侧核对 |
| 服务重启时存在 `queued/running` | 仍读取同一冻结输入，仅执行 H3 `resume` | 恢复只查询已持久化任务并下载已有结果，不创建供应商任务 |
| 用户在 H3 阶段确定的 `failed` 后点“重试生成” | 新 attempt，再次 `queued → running` | 必须生成新的 `client_request_id`；长链复用成功段，只重做失败段及同链下游 |
| 用户在长链 `failed + stage=stitch` 后点“重试拼接” | 原 attempt，再次 `queued → running` | 必须复用原 `client_request_id` 和冻结参数，只重跑本地拼接 |

会话导航不再自行拼接这些底层状态。列表和详情均返回同一个
`navigation_status`：分析阶段、生成阶段、输出缺失、完成以及后处理状态都有独立枚举。
只有 generation 明确 `succeeded` 且最终输出通过服务端验收时才是 `completed`；孤立成片
不能把 `analysis_complete` 冒充为完成。已有有效成片时，后处理的进行中、失败、完成状态
进行中或失败时优先显示为 `postprocessing/postprocess_failed`；后处理完成且最终视频有效时统一显示 `completed`（“已完成”）。

## 边界

- `resume_required` 使用新 id 返回 409 `resume_request_id_mismatch`，台词/画幅漂移返回 409 `resume_parameters_changed`；合法继续仍返回原 attempt 数字。
- 相同请求 id 在 active/succeeded 状态只返回既有状态；确定失败后复用旧 id 返回 409；`submission_unknown` 使用任何 id 都返回 409 `submission_outcome_unknown`。
- active 或 succeeded 会话不接受不同 id 的并发提交。
- 没有自动付费重试、定时重试或 Seedance 回退。
- 长视频任一子任务 `submission_unknown` 会锁住整批；启动恢复只 GET 已知子任务，绝不提交尚未开始的段。
- 已生成的历史视频保持可读；旧 Context IR 契约下未完成的 generation 映射为 `failed / generation_path_removed`，必须用新请求 id 按直接 H3 链路创建 attempt，不再查询 MiniMax。
- `.h3/session.lock` 防止同一会话被并发推进；生产仍要求单 uvicorn 进程。

## 例子

- H3 已提交后服务重启：恢复只 GET；查询或下载失败时显示 `resume_required`，用户确认后原 attempt 继续。
- H3 查询超时：显示 `resume_required / h3_timeout`；页面不自动 POST，用户点击“继续既有任务”仍是原 attempt。
