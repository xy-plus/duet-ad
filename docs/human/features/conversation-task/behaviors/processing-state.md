---
name: processing-state
type: behavior
status: done
owner: human
updated: 2026-08-17
links: []
---

# 处理状态机

## 规则

| 当 | 则 |
| --- | --- |
| 会话刚创建 | 状态 `queued`，界面显示「排队中」 |
| 后台任务开始处理 | 状态转 `processing`，界面显示「处理中」，每 2 秒轮询刷新 |
| 4fps 抽帧 → codex 沙箱处理 → 产物校验全部成功 | 状态转 `done`，停止轮询，展示结果 |
| 任一步骤失败（含 codex 超时 600s、产物校验不过、抽帧失败） | 状态转 `failed`，`error` 展示截断后的可读原因（≤500 字） |
| 状态到达 `done`/`failed` | 终态，不再自动刷新；`failed` 只展示错误，无重试按钮 |

## 边界

- 状态只前进不回退：`queued → processing → done|failed`，无取消、无重跑
- 后端默认串行处理（CODEX_CONCURRENCY=1），并发上传排队等待
- 测试配置下 `enable_pipeline=False`：会话停在 `queued`，不启动处理
- 进程重启后 `processing` 中的会话不会自动续跑（内存后台任务）

## 例子

- 输入：上传 20s 合规视频 → 输出：约数分钟内状态 `queued → processing → done`
- 输入：codex 未安装（PATH 找不到）→ 输出：`failed`，`error` 含 `codex executable not found on PATH`
