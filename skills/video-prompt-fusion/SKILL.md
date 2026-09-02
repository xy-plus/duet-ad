---
name: video-prompt-fusion
description: Fuse frozen optimized keyframes, old segment dynamics, frame replacement and relation bindings, and audio timing into ordered visual prose for Context IR and H3. Preserve project identities and relation systems while keeping actions and hard-cut evidence local.
---

# Video Prompt Fusion

只读取 `work/multimodal_input.json` 及其中列出的新关键帧。一次调用处理全部 ordered segments；不选帧、不改音频、不写 provider 字段。四类既有模态输入不变，不新增第五类输入；`relation_occurrences` 是 `image_optimization_prompt` 的后端结构化关系 sidecar。输入输出 schema 保持 version 2。

输入合同由 `multimodal_input.json` 的 `schema/version` 和后端校验器冻结。每段依次提供 `index/new_keyframes/old_video_prompt/image_optimization_prompt/relation_occurrences/audio_content`；关键帧、文本、台词均带 SHA，关系 occurrence 带逐帧主客体、predicate、state、geometry、preserve 和 replace_together。不要重述或改写输入结构。

每段 9 张图片和 9 条 frame prompt 按 order 一一对应；segment 索引连续。核对列出文件及文本 exact bytes SHA。局部时间严格递增，order 1 为 `segment_time_s=0/type=start/at_segment_s=0`；continuous 的 `at_segment_s=null`；hard_cut 时间位于前后关键帧局部时间之间。`audio_content` 只作为冻结台词冲突边界，`voice_references=[]`、`music_policy="forbid"`，不生成声音、台词、口型或音乐。

## 融合

每个 hard-cut 区间独立建立内部证据账本：

1. 新关键帧独占静态事实权威，包括人物、对象、场景、材质、数量、构图、机位、裁切、接触和遮挡；边缘、局部、模糊或不可见内容不补全。
2. 旧提示词只贡献同一区间内、起止帧都支持的动作顺序、因果、camera movement type 和相对节奏；不贡献旧静态事实，不新增切点、morph、方向反转或末帧之后的终态。
3. 同 order 的 `image_optimization_prompt[].text` 消费共享替换参考板绑定和结构化 `relation_occurrences`。前者是素材绑定权威；后者同时携带全局 `replacement_system` 与逐帧关系状态，是关系的唯一结构化权威。只绑定当前新关键帧直接有记录的元素和关系，空数组表示当前帧没有关系证据。
4. 关系 ID、主客体、predicate、state、geometry、preserve 和 replace_together 由后端逐字段机械冻结；模型不得互换主客体、合并关系、改写状态、给同一 subject 添加无输入证据的 object，或从邻帧补关系。每个 hard cut 开始新账本，上一 interval 的关系不传播。
5. `audio_content` 不进入 visual 文本，只用于避免视觉动作与冻结台词时间冲突。

跨段只共享 stable element design 和 relation system；动作阶段、因果、camera movement、hard-cut 和剧情仍严格段内。若图片已经不呈现某关系，不得凭旧提示词修复；删除最小无证据片段，继续融合其他内容。质量评分不触发拒绝、重试、回退或新 workflow。

`relation_occurrences.preserve` 中冻结的 `replacement_system=` 条目继续约束全项目关系设计；`relation_id` 对应该关系的逐帧证据。主客体和功能角色全项目不变，不能互换主客体；状态生命周期覆盖连接、装载、作用、释放、分离。末帧若仍在运动，只写可见状态，不授权跨 hard cut 或无证据传播。

## 输出

每段严格按 order 为 9 张关键帧分别输出 9 条简洁英文 `visual` prose，一帧一条，不合并、不缺省、不增加。每条只描述对应图片及同区间已有动态证据；不输出时间戳、图片标记、stable key、tile、relation key、音频字段或 provider 语法。后端按 transition 将九条描述机械归入 hard-cut 区间，再与关系块共同编译到现有 H3 transport；这不是质量评分。模型不输出 `relation_states`，关系状态完全由后端按冻结输入生成。

输出格式以本次调用注入的 JSON Schema 为唯一权威：固定 `schema/version/input_sha256`，`segments` 与输入一一对应，每段只填写 `index/visual`，visual 固定为 9 条并逐条对应 frame order 1 至 9。后端完全依据冻结 occurrence 机械生成关系状态、按 hard cut 组合九条视觉描述并编译进 Context IR/H3 effective prompt，再原子发布 `work/h3_prompt_plan.json`。
