---
name: upload-create
type: behavior
status: done
owner: human
updated: 2026-08-21
tdd: N/A
links: [conversation-task]
---

# 上传与输入准备

## 规则

| 当 | 则 |
| --- | --- |
| 上传 `.mp4/.mov/.webm` 或提供受支持的公网视频链接 | 创建 schema v2 会话，立即进入 `queued`；文件与链接必须且只能选一个 |
| 源视频可读且实际时长为正有限数 | ffprobe 校验视频流尺寸；时长超过 300 秒时删除刚创建的会话并返回结构化 422，前端提示裁剪后重新上传 |
| 源视频超过 10 秒 | 创建时只接受 `voice_mode=keep`；准备阶段形成 1–15 秒的安全分段、首尾锚点和冻结 plan receipt |
| 源视频没有音轨 | 合法；自动台词、声学证据和 normalized audio 均为空，不伪造台词 |
| 短视频选择原文保持/改编/翻译 | 只决定自动 ASR 的准备方式；翻译必须填目标语言。最终仍在 H3 提交前选择 `auto/edit/custom/none` |
| 自动台词 agent 开始听写 | 后端把 `voice.mp3` 与仅含必要时长的 `manifest.json` 复制到 `/tmp` 音频专用工作区，并以外层文件系统沙箱运行；agent 不能读取源视频、抽帧、contact sheet、视觉 prompt、会话目录或仓库。缺少沙箱能力或路径校验异常时准备失败，不降级为提示词禁令 |
| 自动台词 agent 返回结果 | 后端只从音频专用工作区读取 `voice_lines.json`，白名单校验通过后才写入会话；重试前删除上次输出，超时但完整有效的输出仍可收养 |
| MP3 编码尾部使音频时长略长于视频 | 听写先按真实音频时长校验并做声学分类，再把最终台词机械裁到视频时间轴；跨尾部句的 `end_s` 截到视频时长，完全从视频结束后开始的句子丢弃并留 provenance/warning |
| 自动听写有结果 | 每句做声学分类；默认只保留 `spoken` 和 `sung`。若整段只有一句、句级时间戳偏离真实口播，但全轨存在强人声证据，则按全轨 `spoken/sung` 兜底；纯 BGM 不触发。无声学人声证据的假转录丢弃并留 provenance |
| 自动听写返回 `[无法辨识]`、`[inaudible]` 等占位符 | 占位符不属于源视频台词，按空听写处理；音轨有人声时只重试一次，仍无法听写则展示“未识别到可用台词”，允许用户编辑、自定义或选择无台词，绝不把占位符写进最终提示词 |
| 音轨有人声证据但听写为空 | 只重试一次听写；仍为空则记录 warning 并按无台词继续 |
| 视觉 agent 生成提示词 | 看不到结构化台词；OCR、字幕、画面文字和备注只可作为视觉内容，不能写成角色发声 |
| 短视频准备完成 | 产出 1–9 张关键帧、`visual_prompt.txt`、机械组合的 `prompt.txt` 和 `prepared_input.json`，状态变为 `done` |
| 长视频准备完成 | 每段独立产出关键帧、首尾锚点和提示词，并以 `long_video_plan.json` 绑定完整计划，状态变为 `done` |
| 短视频用户在首次 H3 attempt 前二次修改源提示词 | 以当前 SHA-256 做并发保护；保存后重写并复核 prepared receipt |
| 用户确认台词和画幅后点击“生成最终视频” | 短链冻结 prepared input；长链用当前 plan receipt 做 CAS 后冻结各段 FL2VA 输入 |

## 边界

- 同一 IP 每分钟最多创建 10 次；排队会话数由 `MAX_QUEUED` 限制。
- `client_request_id` 可用于创建幂等；同 id 命中返回既有会话。
- `VOCAL_FILTER=off` 可保留未分类为人声的句子，但仍记录分类；未知值按启用处理。
- `≤10s` 是原 Ref2VA 单段契约；`>10s` 是 FL2VA 分段契约。二者都属于 schema v2，但使用不同冻结 receipt。

## 例子

- 9.2 秒、无音轨、9:16 视频：准备成功，自动台词为空，引擎提交时长为 10 秒。
- 15 秒、`voice_mode=keep` 视频：准备 1 个 FL2VA 首尾帧子任务。
- 300.01 秒视频：上传后返回 `video_duration_exceeds_h3_limit`，不创建可见会话、不运行准备流水线。
