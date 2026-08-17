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
| 所有视频：抽帧后先做场景检测（scenes.json）；仅时长 >20 秒才拆段（每段 4~15 秒），拆段后各段独立切视频 + 抽帧 + codex 并行处理 | 逐段产物聚合进 meta.segments（每段关键帧、提示词、该段台词）；台词按句子时间落入段区间归属；分段模式下该段提示词由后端加「不要生成背景音乐」一行 |
| 视频 ≤20 秒（segments 为空）或场景检测失败/结果非法 | 按单段流程处理（不拆段，现有行为不变；检测失败落 meta 内部字段 scenes_note 留痕） |
| 选了口播转换时：抽帧后先做口播听写（codex 听 work/voice.mp3，按模式保持/洗稿/翻译，输出带时间戳台词） | 台词校验后随结果落盘（进 meta.voice_lines，detail 响应含 voice_lines 字段）；失败则整个会话 `failed` |
| 任一步骤失败（含 codex 超时 1800s、产物校验不过、抽帧失败）；拆段模式下任一段失败 | 状态转 `failed`，`error` 展示截断后的可读原因（≤500 字；段失败会指明段号） |
| 状态到达 `done`/`failed` | 终态，不再自动刷新；`failed` 只展示错误，无重试按钮 |

## 边界

- 状态只前进不回退：`queued → processing → done|failed`，无取消、无重跑
- 后端默认并发 10（CODEX_CONCURRENCY，部署未覆盖），并发上传排队等待信号量
- 测试配置下 `enable_pipeline=False`：会话停在 `queued`，不启动处理
- 进程重启后 `processing` 中的会话不会自动续跑（内存后台任务）

## 例子

- 输入：上传 20s 合规视频 → 输出：约数分钟内状态 `queued → processing → done`
- 输入：codex 未安装（PATH 找不到）→ 输出：`failed`，`error` 含 `codex executable not found on PATH`
