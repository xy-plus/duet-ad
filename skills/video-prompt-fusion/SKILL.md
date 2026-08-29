---
name: video-prompt-fusion
description: Fuse frozen optimized keyframes, old segment dynamics, frame replacement and relation bindings, and audio timing into ordered visual prose for Context IR and H3. Preserve project identities and relation systems while keeping actions and hard-cut evidence local.
---

# Video Prompt Fusion

只读取 `work/multimodal_input.json` 及其中列出的新关键帧。一次调用处理全部 ordered segments；不选帧、不改音频、不写 provider 字段。输入输出 schema 保持 version 2，不新增第五类输入。

```text
VideoPromptFusionInput = { schema: "duet.video-prompt-fusion-input"; version: 2; segments: NonEmptyArray<SegmentInput> }
SegmentInput = { index: Int1; new_keyframes: NineOrdered<KeyframeReceipt>; old_video_prompt: FrozenText; image_optimization_prompt: NineOrdered<FrozenFramePrompt>; audio_content: FrozenAudioContent }
KeyframeReceipt = { order: Int1; path: NonEmpty; sha256: Sha256; segment_time_s: Number; source_scene_id: NonEmpty; transition: { type: "start" | "continuous" | "hard_cut"; at_segment_s: Number | null } }
FrozenText = { text: NonEmpty; sha256: Sha256 }
FrozenFramePrompt = { order: Int1; text: NonEmpty; sha256: Sha256 }
FrozenAudioContent = { lines_json: NonEmptyJsonText; lines_sha256: Sha256; voice_references: []; music_policy: "forbid" }
```

每段 9 张图片和 9 条 frame prompt 按 order 一一对应；segment 索引连续。核对列出文件及文本 exact bytes SHA。局部时间严格递增，order 1 为 `segment_time_s=0/type=start/at_segment_s=0`；continuous 的 `at_segment_s=null`；hard_cut 时间位于前后关键帧局部时间之间。`audio_content` 只作为冻结台词冲突边界，`voice_references=[]`、`music_policy="forbid"`，不生成声音、台词、口型或音乐。version 1 只读，不创建输出。

## 融合

每个 hard-cut 区间独立建立内部证据账本：

1. 新关键帧独占静态事实权威，包括人物、对象、场景、材质、数量、构图、机位、裁切、接触和遮挡；边缘、局部、模糊或不可见内容不补全。
2. 旧提示词只贡献同一区间内、起止帧都支持的动作顺序、因果、camera movement type 和相对节奏；不贡献旧静态事实，不新增切点、morph、方向反转或末帧之后的终态。
3. 同 order 的 `image_optimization_prompt[].text` 解释替换绑定。消费两个固定块：`全项目共享替换参考板绑定：stable_key -> TILE_XX -> replacement_description` 和 `全项目共享关系绑定：relation_key -> subject_key -> predicate -> object_key -> replacement_system`。只绑定当前新关键帧可见的元素和关系。
4. 关系 ID 的主客体和功能角色全项目不变；在同一 hard-cut 区间内按逐帧证据维持其状态生命周期，例如连接、装载、作用、释放、分离。每个状态必须有当前帧支持，不能互换主客体、混用不同成员或从邻段补阶段。末帧若仍在运动，只写可见状态，不补完成结果。
5. `audio_content` 不进入 visual 文本，只用于避免视觉动作与冻结台词时间冲突。

跨段只共享 stable element design 和 relation system；动作阶段、因果、camera movement、hard-cut 和剧情仍严格段内。若图片已经不呈现某关系，不得凭旧提示词修复；删除最小无证据片段，继续融合其他内容。质量评分不触发拒绝、重试、回退或新 workflow。

## 输出

后端按 transition 建区间：第一帧开始区间，hard_cut 当前帧开始新区间，continuous 留在当前区间。每个区间输出一条英文 `visual` prose，只引用本区间图片；不输出时间戳、图片标记、stable key、tile、relation key、音频字段或 provider 语法。

只写 `work/h3_prompt_plan.json`：

```text
VideoPromptFusionOutput = { schema: "duet.video-prompt-fusion-output"; version: 2; input_sha256: Sha256; segments: NonEmptyArray<{ index: Int1; visual: NonEmptyArray<NonEmptyText> }> }
```

segments 与输入一一对应，visual 数量等于 hard-cut 区间数。先写临时文件，原子替换目标后立即退出。输出供 Context IR 优化，再由后端按冻结素材逐段独立生成视频。
