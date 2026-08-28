---
name: conversation-task
type: feature
status: done
owner: human
updated: 2026-08-28
tdd: N/A
links: []
---

# 会话式 H3 视频复刻

## 唯一权威链路

本节是当前 v4 产品链路的唯一权威说明。当前只有一条从技术验收 A 到成片提交 B 的 operation；`N=1` 只是单元素 segment 列表，不是另一条短链。

```text
原视频
  -> 既有 ASR + YAMNet 只做音频语义分析（非 Skill）
  -> video-maker Skill
  -> segments[N>=1]；每段 exact 9 张原始关键帧，并绑定 source scene / time / transition
  -> 可选 MediaKit 去字幕和/或去 Logo；不改变 exact-9、顺序或 segment 归属
  -> image-postprocess Skill 只输出视觉替换语义
  -> 后端确定性编译 Seedream 提示词，生成每段 exact 9 张优化关键帧
  -> 技术验收 A 落盘；同一 operation 自动继续，不等待刷新、第二次确认或第二个 POST
  -> video-prompt-fusion Skill 一次项目级调用，只输出每个冻结 hard-cut 区间的视觉 prose
  -> 后端把视觉 prose + exact-9 Picture 时间轴 + 冻结台词 + music policy 编译为唯一 Ref2VA prompt
  -> Context 写本地 identity receipt，effective prompt 与 Ref2VA prompt 同字节，HTTP 调用数为 0
  -> H3 每段使用 exact 9 张 Picture reference、零 source audio reference，生成原生音视频
  -> 同一 EDL 按 segment 拼接；H3 无音轨的区间补同时间轴静音
  -> 原子发布 generated.mp4，提交 B
```

当前 operation 只有两个公开结果：A 已接受后返回 `202 {operation_id,status:"running",stage}`；receipt 绑定的成片有效后返回 `200 {operation_id,status:"succeeded",stage:"commit_b"}`。内部 Fusion、计划或恢复状态都在同一 CID 上继续，不能要求用户刷新、消费 409 后重提，或用另一组参数创建备用生成。

## 硬约束

- 全链只调用三个 Skill：`video-maker` 负责分段分析和 exact-9 原始关键帧；`image-postprocess` 只负责图片替换语义；`video-prompt-fusion` 只负责视觉融合。Audio、Binding、Speaker、Context、Ref2VA compiler、H3 和拼接都不是 Skill。
- 当前 tree 三个 Skill 的权威 SHA-256 分别是：
  - `video-maker`：`0bbb22baeb8f14fef737b279e2ab2e8f70bf8965d41b182f1987537e1e3e4785`
  - `image-postprocess`：`126508628d8923fa2b179bc347c2cacf6973a6ed5e96ed62d8007042f6743e8c`
  - `video-prompt-fusion`：`34145f90532ec65a45d029d62b09e9ef60516721df47d6aa83ea649699a8e16c`
- 每个当前 segment 的原始关键帧、优化关键帧、Fusion `new_keyframes` 和 H3 `<Picture 1>`…`<Picture 9>` 都必须恰好为 9 张并保持顺序。极短连续 scene 可重复最近的已解码源帧来满足 exact-9；同 scene 的相同 PTS 是有 provenance 的合法 receipt，不得把“9 张”误写为“9 个唯一源帧”。
- 每张帧同时绑定 segment-local time、source scene 和 transition。scene 改变必须是冻结时点的 `hard_cut`，scene 不变必须是 `continuous`；切后帧是新 anchor。Fusion 的视觉文字无权改写这些机械字段。
- `image-postprocess` 当前只执行 `phase=plan`。后端补齐并编译结构字段；图片质量 score、compiler diagnostics、Fusion 语义检查和 A/B 观察只用于测试与下一轮 Skill 迭代，不阻断生产发布、不触发重试、不选择备用 prompt、旧视觉或另一 workflow。schema、数量、顺序、路径和 SHA 不匹配仍属于技术输入无效，不是质量评分。
- `video-prompt-fusion` v2 每段只输出 `{index,visual[]}`。它不能写 `[Shot]`、时间戳、Picture/Audio/Subject 标签、台词、music policy 或任何 provider 字段；后端是 Ref2VA prompt 的唯一 compiler 和发送权威。
- Fusion 的四类冻结输入仍是 exact-9 新关键帧、旧视频动态骨架、逐帧图片优化提示词和音频语义。`audio_content` 只含冻结 spoken 文本/时间/呈现方式与 `music_policy=forbid`；`voice_references=[]`，每行 `voice_ref=null`。
- 当前 Context 不优化或改写提示词。它为后端编译的 Ref2VA prompt 生成 `local:identity:<sha256>` receipt，effective prompt 与 source prompt 同 SHA，同路径 HTTP 调用数为 0；语义 score 固定为 identity 证明，不是生产门禁。
- 源音频只供 ASR/YAMNet 分析，不进入 Skill、Context 或 H3 reference，不在拼接时回挂、覆盖、混音或 overlay。成片音频只来自 H3 原生输出；某段缺音轨时，同一 EDL 为该段补有限静音。
- `N=1` 与 `N>1` 共用同一 plan、Fusion、Ref2VA compiler、Context identity、H3 attempt、恢复、EDL 拼接和验收实现。
- v1 Fusion、旧 multimodal/voice-reference、旧 short/long、speaker-visibility 与 quality-verdict receipt 都只读；已知 provider task 只允许按原 receipt GET 恢复。它们不得创建、迁移、覆盖或成为 current fallback。
- `submission_unknown` 仍服从 receipt 驱动的 GET-only 付费安全边界，但不会派生第二条 current operation。

## 变更前对照门

每个 current v4 变更必须证明：

1. A 被接受后是否在同一 CID、同一 operation 自动推进到 B，没有刷新、二次确认或备用提交？
2. 是否仍只调用三个 Skill，且 Fusion 只输出视觉 prose、Ref2VA 只由后端编译？
3. 是否对 `N=1` 与 `N>1` 使用同一实现？
4. 是否保持每段 exact 9 张图、scene/time/transition 和 Picture 1…9 的一一映射，并允许有 provenance 的短 scene 重复帧？
5. 是否确保质量 score/diagnostics 只进入测试与迭代，不阻断、重试或回退生产？
6. 是否保持 Context local identity、同字节 prompt 和零 HTTP？
7. 是否保持零 source audio reference、零 overlay，以及 H3 native audio / 同 EDL静音的唯一成片策略？
8. 是否把旧版本与历史 receipt 保持为只读，而不是兼容写入或备用路径？

任一回答为否，变更即偏离唯一链路。

## 验收

- [x] 当前 v4 技术验收 A 自动延续到 B；进行中统一公开为 `202 running`，成功统一为 `200 succeeded / commit_b`
- [x] 每段 exact 9 张原始图、优化图和 H3 Picture reference；source scene 改变机械映射为 hard cut
- [x] Fusion v2 只输出 visual prose；后端确定性编译唯一 Ref2VA prompt
- [x] Context current path 为 local identity，同字节 receipt，HTTP 0
- [x] 质量 score/diagnostics 只用于测试与迭代，不阻断生产、不触发重试或 fallback
- [x] 当前 H3 固定零 source audio reference；成片只使用 H3 原生音频或同一 EDL 静音，源音频不 overlay
- [x] 单段和多段共用统一 segment coordinator 与 stitch
- [x] 历史 v1/旧多模态/旧质量 verdict 只读，不进入 current create path
- [x] H3 付费 attempt、task id、输入/输出 SHA 与 `submission_unknown` GET-only 安全边界保持不变

## 边界

- H3 是模型名，不代表服务启用了 HTTP/3。
- 当前 Ref2VA 请求逐段使用 exact 9 张冻结 Picture；provider 整秒时长不超过 14 秒，总输入不超过 300 秒。
- 同镜头普通动作不臆造秒数；source hard cut 的 segment-local 精确时点只由冻结时间轴和后端 compiler 负责。
- 分段链路不是供应商原生 extend；每段是独立、receipt 绑定的 H3 task，再由本地统一 EDL 拼接。
- 完整可用链只交付到 Web `3211`；生产发布与回滚不属于本文档变更。

> 表现规格见本目录 `behaviors/`；纯文档变更不适用 TDD。
