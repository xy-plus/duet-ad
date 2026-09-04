---
name: long-video
type: behavior
status: done
owner: human
updated: 2026-08-28
links: [conversation-task, submit-gate, result-display]
---

# 统一分段生成

当前 v4 只使用 `segments[N>=1]`。`N=1` 是单元素列表，`N>1` 是多元素列表；两者进入同一个 A→B operation 和同一个 segment coordinator。

## 规则

| 当 | 则 |
| --- | --- |
| `video-maker` 完成分析 | 为每个 segment 冻结 exact 9 张原始关键帧及其 source time、source scene、transition 和旧视频动态骨架 |
| scene 很短、可用解码帧不足 9 张 | 重复最近的合法源帧并保留 provenance；仍产出 exact 9 个有序槽位。相同 scene 的相同 PTS 合法，不把 exact-9 误写成 9 个唯一源帧 |
| source scene 改变 | 冻结为精确 `hard_cut`；切后帧开始新视觉区间并成为新 anchor。scene 不变机械映射为 `continuous` |
| MediaKit 被选择 | 在图片优化前按去字幕、去 Logo 顺序执行；不得改变 exact-9、帧序或 segment 归属 |
| `image-postprocess` 执行 | Skill 只给替换视觉语义，后端确定性编译 Seedream prompts；每段发布 exact 9 张优化图 |
| 图片技术验收 A 落盘 | 同一 operation 自动继续 Fusion、Ref2VA、Context、H3 和拼接；不等待刷新、第二次确认或另一个 POST |
| `video-prompt-fusion` 执行 | 一次项目级调用读取四类冻结输入，每个 hard-cut 区间只输出一条 visual prose；不输出 provider prompt |
| Fusion visual prose 可读 | 后端机械编译唯一 Ref2VA prompt：Picture 1…9、scene/cut 时间、冻结 spoken 台词、`non_diegetic_music: N/A` 均由 receipt 真源写入 |
| Context 执行 | 当前 v2 Ref2VA prompt 走 local identity：effective prompt 同字节、task id 为 `local:identity:<sha256>`、HTTP 0 |
| H3 执行 | 每段发送 exact 9 张 Picture reference、零 source audio reference；H3 prompt 和图片 bytes 均受 receipt 绑定 |
| 某段 H3 返回音轨 | 保留 H3 原生音频并按同一 segment EDL 裁补 |
| 某段 H3 无音轨 | 在同一 EDL 为该段补有限静音；不读取、回挂或 overlay 源音频 |
| 全部分段成功 | 按冻结顺序、帧预算和 PTS 用同一拼接器输出 `generated.mp4`；`N=1` 也处理单元素 EDL |
| 质量 score 或 diagnostics 较低 | 仅记录给测试和下一轮 Skill 迭代；不阻断、重试、选择备用 prompt、回退旧图片或切换 workflow |
| 历史 v1 / old multimodal receipt | 只读或对既有 task 做原 receipt GET 恢复；不得新建、迁移或作为 current fallback |

## 不变量

- 全链只有 `video-maker`、`image-postprocess`、`video-prompt-fusion` 三个 Skill；Ref2VA compiler、Context、H3 和 EDL 都属于后端。
- Fusion v2 输出只有 `{index,visual[]}`；Picture、Shot、时间码、台词和 music policy 只由后端编译。
- 技术合同要求 schema、数量、顺序、路径和 SHA 完整；视觉质量和语义评分不是生产门禁。
- 每段原始图、优化图、Fusion 输入和 H3 Picture 都是 exact 9 个有序槽位，并逐值绑定 source scene/time/transition。
- 当前音频始终 `voice_references=[]`、逐行 `voice_ref=null`；源音频只供 ASR/YAMNet 分析。
- H3 原生音频与同 EDL 静音是唯二成片音频来源；源混音和任何 conditioning audio 都不得 overlay。
- A 接受后只存在一个 operation；内部阶段失败保留同一 durable work item，不向用户暴露 refresh/409 作为推进步骤。
- `submission_unknown` 维持 GET-only，不因此创建第二个 operation 或另一条 workflow。

## 例子

- 2 秒连续 scene 只有少量不同解码帧：为当前 segment 重复最近合法帧填满 Picture 1…9，并绑定重复 provenance；仍走同一 H3 task。
- 8 秒视频在 2.267 秒存在 source hard cut：仍是一个 exact-9 segment；后端按冻结 timeline 把 visual prose 编译成两个 Shot，第二个 Shot 在 `00:02.267` 切到对应 Picture。
- 30 秒视频规划为四个 segment：每段 exact 9 Picture、零 source audio reference；四段 H3 成功后按同一 EDL 拼接。
