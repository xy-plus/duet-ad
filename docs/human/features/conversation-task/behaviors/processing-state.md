---
name: processing-state
type: behavior
status: done
owner: human
updated: 2026-08-20
tdd: N/A
links: [conversation-task, submit-gate]
---

# 准备与生成状态

## 规则

| 阶段 | 可见状态 | 行为 |
| --- | --- | --- |
| 输入准备 | `queued → processing → done/failed` | 抽帧、台词准备、视觉 prompt 和 receipt 完成后才允许生成 |
| 首次人工提交 | `generation: queued → running` | 返回 202 后后台调用 H3 `start`，先 Context IR，成功后再提交 AutoDL H3 |
| 生成成功 | `generation: succeeded` | 原子落盘 `generated.mp4`，详情 `has_video=true` |
| 已知 task 查询/超时、下载传输/DNS/peer 验证或输出写入/探测基础设施失败 | `generation: resume_required` | 包括 `download_dns_failed/download_peer_unverified/output_probe_failed`；保留原 request id、receipt 和 attempt，只允许原参数继续 |
| 成片 URL、重定向、体积或媒体内容确定拒绝 | `generation: failed` | 安全拒绝码为 `download_url_rejected/download_redirect_rejected/download_too_large/download_invalid_video`；只有人工新 id retry |
| 可确定失败 | `generation: failed` | 展示安全错误码；用户确认后必须用新 request id 创建 retry attempt |
| 供应商 POST 结果未知 | `generation: submission_unknown` | 不猜测是否扣费；隐藏重试入口，所有再次提交返回 409，必须先到供应商侧核对 |
| 服务重启时存在 `queued/running` | 仍读取同一冻结输入，仅执行 H3 `resume` | 恢复只查询已持久化任务并下载已有结果，不创建供应商任务 |
| GET-only 恢复到 `ready_for_h3` | 映射为 `resume_required / ready_for_h3` | Context IR 已完成但 H3 尚未提交；必须用原 id 和冻结参数确认继续同一 attempt |
| 用户在确定的 `failed` 后点重试 | 新 attempt，再次 `queued → running` | 必须生成新的 `client_request_id`；后端调用显式 `retry` |

## 边界

- `resume_required` 使用新 id 返回 409 `resume_request_id_mismatch`，台词/画幅漂移返回 409 `resume_parameters_changed`；合法继续仍返回原 attempt 数字。
- 相同请求 id 在 active/succeeded 状态只返回既有状态；确定失败后复用旧 id 返回 409；`submission_unknown` 使用任何 id 都返回 409 `submission_outcome_unknown`。
- active 或 succeeded 会话不接受不同 id 的并发提交。
- 没有自动付费重试、定时重试或 Seedance 回退。
- 新 attempt 不再产生 `ir_dialogue_mismatch`；历史 attempt 若已有该错误，仍按确定失败展示并允许用户用新请求重试。
- `.h3/session.lock` 防止同一会话被并发推进；生产仍要求单 uvicorn 进程。

## 例子

- Context IR 已完成后服务重启：恢复只 GET；到达 H3 付费边界时显示 `resume_required`，用户确认后原 attempt 继续。
- H3 查询超时：显示 `resume_required / h3_timeout`；页面不自动 POST，用户点击“继续既有任务”仍是原 attempt。
