---
name: video-prompt-fusion
description: Fuse frozen keyframes, segment dynamics, replacement bindings, relations, and audio boundaries into ordered visual prose for Context IR and H3.
---

只读 `work/multimodal_input.json` 及其中 SHA 绑定的新关键帧；一次处理全部 ordered segments，不选帧、不改音频、不写 provider 字段。输入、输出均为 version 2，注入的 JSON Schema 是输出结构的唯一权威。

每段按 order 提供 `new_keyframes`、`old_video_prompt`、`image_optimization_prompt`、`relation_occurrences` 和 `audio_content`。关键帧、文本、台词和关系字段均冻结，局部时间与 transition 定义 hard cut。`audio_content` 仅用于避免冲突，不进入 visual，也不生成声音、台词、口型或音乐。

## 融合

每个 hard-cut 区间独立取证，不能跨界传播场景、动作或关系。

1. 新关键帧是人物、物体、场景、构图、机位、裁切、接触和遮挡等静态事实的唯一权威；不可见、模糊或边缘内容不补全。
2. 旧提示词只可贡献同一区间起止帧支持的动作顺序、因果、机位运动和节奏；不得恢复旧静态事实，也不得新增切点、morph、方向反转或帧后终态。
3. 同 order 的 `image_optimization_prompt` 是替换素材绑定权威；`relation_occurrences` 是逐帧关系状态和全局 replacement system 的唯一结构化权威。只写当前帧有证据的元素和关系；不得互换主客体、合并/改写关系、从邻帧补关系或凭旧提示词修复消失关系。

跨段只保持同 stable element 的设计和同一 relation system；动作、因果、镜头和剧情仍限于本段。`preserve` 的 replacement system 继续约束全项目设计；末帧仅写可见状态。

## 输出

向 `work/h3_prompt_plan.json` 输出：每段按 hard-cut 区间写一条简洁英文 `visual` prose，第一帧开始第一区间，每个 `hard_cut` 开始新区间，`continuous` 留在当前区间。每条只写该区间图片和已有动态证据；不输出时间戳、图片标记、stable key、tile、relation key、音频字段或 provider 语法。`segments` 与输入一一对应，每段只填 `index`、`visual`；不输出 `relation_states`。后端依据冻结 occurrence、transition 和该输出编译 Context IR/H3。
