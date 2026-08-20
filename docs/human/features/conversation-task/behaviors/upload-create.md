---
name: upload-create
type: behavior
status: done
owner: human
updated: 2026-08-20
tdd: N/A
links: [conversation-task]
---

# 上传与输入准备

## 规则

| 当 | 则 |
| --- | --- |
| 上传 `.mp4/.mov/.webm` 或提供受支持的公网视频链接 | 创建 schema v2 会话，立即进入 `queued`；文件与链接必须且只能选一个 |
| 源视频可读且实际时长 `0 < duration_s <= min(MAX_DURATION_S, 15)` | ffprobe 校验视频流尺寸并保存实际浮点时长；画幅只在准备完成后按实际选中关键帧判断，超过时长上限返回 422 并删除会话目录 |
| 源视频没有音轨 | 合法；自动台词、声学证据和 normalized audio 均为空，不伪造台词 |
| 选择原文保持/改编/翻译 | 只决定自动 ASR 的准备方式；翻译必须填目标语言。最终仍在 H3 提交前选择 `auto/edit/custom/none` |
| 自动台词 agent 开始听写 | 后端把 `voice.mp3` 与仅含必要时长的 `manifest.json` 复制到 `/tmp` 音频专用工作区，并以外层文件系统沙箱运行；agent 不能读取源视频、抽帧、contact sheet、视觉 prompt、会话目录或仓库。缺少沙箱能力或路径校验异常时准备失败，不降级为提示词禁令 |
| 自动台词 agent 返回结果 | 后端只从音频专用工作区读取 `voice_lines.json`，白名单校验通过后才写入会话；重试前删除上次输出，超时但完整有效的输出仍可收养 |
| MP3 编码尾部使音频时长略长于视频 | 听写先按真实音频时长校验并做声学分类，再把最终台词机械裁到视频时间轴；跨尾部句的 `end_s` 截到视频时长，完全从视频结束后开始的句子丢弃并留 provenance/warning |
| 自动听写有结果 | 每句做声学分类；默认只保留 `spoken` 和 `sung`，无声学人声证据的假转录丢弃并留 provenance |
| 音轨有人声证据但听写为空 | 只重试一次听写；仍为空则记录 warning 并按无台词继续 |
| 视觉 agent 生成提示词 | 看不到结构化台词；OCR、字幕、画面文字和备注只可作为视觉内容，不能写成角色发声 |
| 准备完成 | 产出 1–9 张关键帧、`visual_prompt.txt`、机械组合的 `prompt.txt` 和 `prepared_input.json`，状态变为 `done` |

## 边界

- 同一 IP 每分钟最多创建 10 次；排队会话数由 `MAX_QUEUED` 限制。
- `client_request_id` 可用于创建幂等；同 id 命中返回既有会话。
- `VOCAL_FILTER=off` 可保留未分类为人声的句子，但仍记录分类；未知值按启用处理。
- 新 schema v2 输入是单段契约；不会走旧长视频拆段生成。

## 例子

- 9.2 秒、无音轨、9:16 视频：准备成功，自动台词为空，引擎提交时长为 10 秒。
- 15.01 秒视频：422，不留下会话。
