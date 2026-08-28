---
name: long-video
type: behavior
status: done
owner: human
updated: 2026-08-28
links: [conversation-task, submit-gate, result-display]
---

# 统一分段生成

本文件不得定义独立的“短链”和“长链”。当前项目统一表示为 `segments[N>=1]`；单段视频只是 `N=1`，多段视频只是 `N>1`。

## 规则

| 当 | 则 |
| --- | --- |
| 原视频完成分析 | `video-maker` 规划连续 `segments[N>=1]`；源 scene 的硬切必须成为 segment 边界，每段固定冻结 9 张原始关键帧、源时间、scene、视频提示词、动作/镜头/时间关系 |
| `N=1` | 使用与 `N>1` 完全相同的 plan、提交、Context、H3、attempt、恢复、拼接和验收代码；不得转入独立 short 实现 |
| `N>1` | 各段使用同一合同独立冻结输入，可在服务端并发预算内并行提交，最终按 segment 顺序和源时间轴拼接 |
| 两段属于同一镜头 | `continue` 只描述连续关系；两段仍各自使用 9 张冻结关键帧 |
| 边界是硬切 | 开始新的 chain，但仍使用同一 segment 数据结构和执行器；切点两侧参考图不得进入同一 H3 请求 |
| 用户选择去字幕和/或去 Logo | 在图片优化前复用既有 MediaKit 预处理；顺序固定为去字幕后去 Logo，未选择的阶段跳过。预处理不得改变每段 9 张帧的数量、顺序或 segment 归属，也不得派生新 Skill |
| 用户确认图片 | 每段必须恰有 9 张已确认优化图；确认后冻结其顺序和 bytes，缺帧、重排或漂移均在 H3 前拒绝，不回退原图 |
| 图片确认完成 | 调用 `video-prompt-fusion`：输入有序新关键帧及其源时间/scene/transition、旧视频提示词、图片优化提示词和音频内容；新视觉元素以优化图为准，动作与相对节奏沿用旧提示词，源硬切时间以冻结分析为准；输出最终视频提示词并绑定全部输入 SHA |
| 自动台词分析 | 只保留 `spoken`；`sung/chant/rap/humming` 不进入 dialogue。仅有歌词时等同无台词，不发送音频 reference |
| 用户选择画外声音 | 纯后端确定性编译现有真实口播、时间和已证明为 clean voice 的唯一 reference；不要求嘴型，不调用任何 Skill；完整源混音不具备该资格 |
| 用户选择画内声音 | 只消费 `video-maker` 主分析已经冻结的画内人物/时间证据；证据缺失时付费前明确不可用，不额外调用 Skill 补证 |
| 用户选择无台词，或自动分析没有真实口播 | 不传台词或声音 reference；丢弃 H3 音轨并由现有拼接器发布静音成片 |
| Context IR 完成 | 只优化 `video-prompt-fusion` 产生的最终视频提示词并直接交给 H3；不得恢复旧提示词中的旧视觉元素，也不得改变帧序、台词、时间、声音呈现或 voice reference |
| 开始生成 | 所有 segment 共用冻结的画幅、清晰度和适配方式；每段读取自身 9 张图、提示词、台词和声音 reference |
| 快速模式开启且 `N>1` | 所有分段输入先冻结，再并行提交；`N=1` 经过相同代码，实际只有一个任务 |
| 某段供应商明确失败 | 完整持久化失败 attempt；成功兄弟不重提，后续是否重试服从统一预算和用户确认 |
| 某段提交结果未知 | 整个项目停止新 POST，只查询已有任务；自动和人工均不得重提 |
| 所有分段成功 | 使用同一拼接器按冻结逻辑时长和顺序处理：有合法口播的段保留 H3 原生音频，无真实口播的段丢弃 H3 音轨并静音；`N=1` 的输入列表长度为一，不调用另一个短片拼接器 |
| 拼接失败 | 只重做本地拼接，不重新提交 H3 |
| 最终输出 | 校验 receipt、输入/输出 SHA、帧数、PTS、音画起止和总时长后原子写入会话级 `generated.mp4` |

## 不变量

- 全链只有 `video-maker`、`image-postprocess` 与 `video-prompt-fusion` 三个 Skill；音频、Context、H3 和拼接都不是 Skill。
- `image-postprocess` 已经用户验收并冻结，不再迭代；它只接收关键帧并输出优化图，不拒绝合法可解码素材。
- segment 索引在统一 plan 内连续；API、Web 和内部实现不得用索引规则选择另一套业务逻辑。
- 每段原始关键帧和优化关键帧都固定为 9 张，顺序进入 frozen receipt 和 H3 请求。
- 每张关键帧的源时间、source scene 与 transition 都进入 frozen receipt；源硬切时点不得由 Fusion 或 Context 改写，供应商请求不得跨硬切复用参考图。
- segment 的逻辑时长与供应商请求时长分别冻结；逻辑时长不足供应商最短值时按合法最短值请求，拼接仍按逻辑帧预算裁切。
- H3 最终 prompt 必须绑定 `video-prompt-fusion` 输出及四类输入 SHA；旧视觉 prompt 只能作为融合输入，不得直接进入 Context 或 H3。
- 有合法 clean voice reference 的段只使用 H3 原生输出音轨；源混音和 conditioning voice 不得在拼接时回挂或 overlay。无真实口播的段必须静音，不能发布 H3 新生音乐。
- 历史 short/long receipt 只允许兼容读取和安全恢复；所有新 v4 项目必须进入统一 segment coordinator。

## 例子

- 输入：14.5 秒且没有源硬切的视频，规划为一个 segment → `segments.length=1`，执行一次统一 H3 segment 任务，再由统一拼接器处理单元素列表。
- 输入：14.5 秒视频在 2.267 秒存在源硬切 → 规划为两个现有 segment；前段按供应商合法最短时长生成但按 2.267 秒逻辑帧预算裁切，切点两侧各自使用 9 张参考图并硬拼。
- 输入：30 秒视频，规划为三个 segment → 三段使用同一输入合同并行生成，最后按冻结顺序拼接。
