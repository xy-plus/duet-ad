---
name: result-display
type: behavior
status: done
owner: human
updated: 2026-08-21
tdd: N/A
links: [conversation-task, processing-state]
---

# 结果评审与 H3 参数

## 规则

| 当 | 则 |
| --- | --- |
| 输入准备为 `done` | 展示源视频、1–9 张关键帧、最终 H3 prompt、台词选择、画幅选择和最终视频区 |
| 短视频 H3 attempt 尚未创建 | 自动生成的 H3 源提示词可二次修改；保存使用当前 SHA-256 防止覆盖新版本 |
| 长视频准备完成 | 展示每段 H3 提示词但不提供顶层编辑；各段内容已由 plan receipt 逐段绑定 |
| H3 attempt 已创建 | H3 源提示词锁定；不能让页面展示内容与已冻结输入发生漂移 |
| 选择 `auto` | 使用详情 `dialogue.auto_lines` 中的自动有效台词，不允许随请求上传 `lines` |
| 选择 `edit` | 以自动台词预填，提交至少一行 `{text,start_s,end_s}` |
| 选择 `custom` | 提交至少一行人工台词，不依赖自动识别结果 |
| 选择 `none` | 发声块明确写“无台词”，不允许上传 `lines` |
| 用户确认台词与画幅 | 按钮为“生成最终视频”；H3 源提示词直接作为实际生成输入 |
| 长视频生成确认 | 展示冻结的子任务数量；提交时绑定当前 plan receipt，只允许 `auto/none` |
| 长视频生成中 | 展示各段的 chain、join、状态和 attempt；不公开供应商 task id |
| 准备完成后的全部实际 H3 输入帧都是 9:16 | 画幅固定 `none`，不能选择 crop/pad；长视频检查每个硬切段 first 与全部 end anchors，续接段 first 由上游尾帧替换 |
| 任一实际 H3 输入帧不是 9:16 | 必须选择居中 `crop` 或黑边 `pad`，不提供静默默认值；历史未冻结长会话也从 plan anchors 派生 |
| generation active | 禁用参数和提交按钮，2 秒轮询状态 |
| generation 为 `resume_required` | 台词、画幅和请求 id 锁定，显示“继续既有任务”；确认后继续原 attempt |
| 长链 H3 阶段确定失败 | 展示错误、“重试生成”和服务端给出的本次新增付费子任务数；状态成功但分段文件缺失时仍计入；点击才创建新请求 id |
| 长链本地拼接失败 | 即使已有可播放的半发布成片，也同时展示成片和恢复区；“重试拼接”显示“本次新增 0 个付费 H3 子任务”，点击复用原请求 id，只执行本地重拼 |
| 短链 generation 确定失败 | 展示错误和“重试生成”；点击才创建新请求 id |
| generation 为 `submission_unknown` | 展示“先到供应商核对”的阻断说明，不显示重试按钮 |
| `generated.mp4` 存在 | 内嵌播放 H3 最终视频 |
| 旧会话 | 展示只读提示；不能修改台词、画幅、后处理或再次生成 |

## 边界

- `duration_s` 展示实际 ffprobe 时长；上传门禁确保其不超过总输入上限 300 秒。
- 所有媒体经带 Bearer 鉴权的 files API 获取，页面使用 blob URL，不暴露目录直链。
- 画面 OCR 只在视觉 prompt 中展示；唯一发声块只由结构化台词机械生成。
