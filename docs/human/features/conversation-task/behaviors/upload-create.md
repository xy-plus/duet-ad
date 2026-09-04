---
name: upload-create
type: behavior
status: done
owner: human
updated: 2026-08-28
tdd: N/A
links: [conversation-task]
---

# 上传与输入准备

## 规则

| 当 | 则 |
| --- | --- |
| 上传 `.mp4/.mov/.webm` 或提供受支持的公网视频链接 | 创建 schema v2 会话，立即进入 `queued`；文件与链接必须且只能选一个 |
| 源视频可读且实际时长为正有限数 | ffprobe 校验视频流尺寸；时长超过 300 秒时删除刚创建的会话并返回结构化 422，前端提示裁剪后重新上传 |
| 任意 current v4 源视频 | 准备为 `segments[N>=1]`；每段 provider 整秒时长不超过 8 秒并冻结 exact 9 张关键帧及 source scene/time/transition |
| 源视频没有音轨 | 合法；自动台词、声学证据和 normalized audio 均为空，不伪造台词 |
| 历史 short 选择原文保持/改编/翻译 | 只属于历史 prepared-input；current v4 不从该合同创建或迁移 |
| 自动台词 agent 开始听写 | 后端把 `voice.mp3` 与仅含必要时长的 `manifest.json` 复制到 `/tmp` 音频专用工作区，并以外层文件系统沙箱运行；agent 不能读取源视频、抽帧、contact sheet、视觉 prompt、会话目录或仓库。缺少沙箱能力或路径校验异常时准备失败，不降级为提示词禁令 |
| 自动台词 agent 返回结果 | 后端只从音频专用工作区读取 `voice_lines.json`，白名单校验通过后才写入会话；重试前删除上次输出，超时但完整有效的输出仍可收养 |
| MP3 编码尾部使音频时长略长于视频 | 听写先按真实音频时长校验并做声学分类，再把最终台词机械裁到视频时间轴；跨尾部句的 `end_s` 截到视频时长，完全从视频结束后开始的句子丢弃并留 provenance/warning |
| 自动听写有结果 | 每句做声学分类；默认只保留 `spoken` 和 `sung`。若整段只有一句、句级时间戳偏离真实口播，但全轨存在强人声证据，则按全轨 `spoken/sung` 兜底；纯 BGM 不触发。无声学人声证据的假转录丢弃并留 provenance |
| 自动听写返回 `[无法辨识]`、`[inaudible]` 等占位符 | 占位符不属于源视频台词，按空听写处理；音轨有人声时只重试一次，仍无法听写则展示“未识别到可用台词”，允许用户编辑、自定义或选择无台词，绝不把占位符写进最终提示词 |
| 音轨有人声证据但听写为空 | 只重试一次听写；仍为空则记录 warning 并按无台词继续 |
| 视觉 agent 生成提示词 | 看不到结构化台词；OCR、字幕、画面文字和备注只可作为视觉内容，不能写成角色发声；时长只写“与源片段时长一致”，不写具体秒数 |
| current v4 准备完成 | 每段产出 exact 9 张关键帧和旧视频动态骨架，以 `long_video_plan.json` 绑定完整 segment 计划；短 scene 可用有 provenance 的重复帧满足 exact-9 |
| 技术验收 A 接受 | 同一 operation 自动继续 image-postprocess、Fusion、backend Ref2VA、Context local identity、H3 和 EDL，不等待修改旧 prompt 或第二次提交 |

## 边界

- 同一 IP 每分钟最多创建 10 次；排队会话数由 `MAX_QUEUED` 限制。
- `client_request_id` 可用于创建幂等；同 id 命中返回既有会话。
- `VOCAL_FILTER=off` 可保留未分类为人声的句子，但仍记录分类；未知值按启用处理。
- current v4 的 `≤8s` 是 `segments.length=1`，`>8s` 是 `segments.length>1`；两者共用同一冻结、Fusion、Context、H3 和 stitch 合同。旧 prepared-input/short receipt 只读。

## 例子

- 8 秒、无音轨、9:16 视频：准备一个 exact-9 segment，自动台词为空，H3 零 source audio reference。
- 8.1 秒视频：准备至少 2 个多图参考子任务，每个整秒请求不超过 8 秒。
- 300.01 秒视频：上传后返回 `video_duration_exceeds_h3_limit`，不创建可见会话、不运行准备流水线。
