---
name: video-maker
description: 从预先抽取的参考视频帧理解叙事、选择最多 9 张有序关键帧并生成纯视觉叙事；图片优化完成后，把原始动态事实与 receipt 绑定的新人物、新场景协调成统一视觉 IR；对后端冻结的真实 PTS 样本逐帧判定 PERSON 可见性与嘴部可验性；当后端另行提供冻结的多模态输入时，只规划 H3 所需的人物、图片、现有台词行、语言、画内或画外发声、声线参考、环境声和音效绑定。用于视频分析、图片优化后的视觉协调、发声人物可见性取证或 H3 音画联合生成前的音画阶段；不提交视频、不调用供应商。
---

# video-maker

## 1. 选择阶段

只按文件选择一种阶段，不混用：

- `work/speaker_visibility_input.json` 存在时严格优先执行 **speaker_visibility**；它在 production 中可与既有 `work/multimodal_input.json` 共存，但本阶段不得读取 `work/multimodal_input.json` 或 `work/reconcile_after_image_optimization_input.json`，也不执行其他阶段。
- `work/speaker_visibility_input.json`、`work/reconcile_after_image_optimization_input.json` 与 `work/multimodal_input.json` 都不存在：执行**视觉阶段**。
- 只有 `work/reconcile_after_image_optimization_input.json` 存在：执行第三阶段 **reconcile_after_image_optimization**。
- 只有 `work/multimodal_input.json` 存在：执行**音画阶段**。
- speaker visibility 输入不存在但后两个阶段输入同时存在：只写 `work/unified_visual_ir.json` 的 `phase_input_conflict` 固定失败结构，不执行任一正常阶段，也不写 `work/h3_prompt_plan.json`。

视觉阶段不得读取任何音频文件，只处理图片与文字。第三阶段不得读取音频或台词，只协调已经选择的帧；音画阶段只规划冻结输入的语义绑定，不重新分析原视频。speaker_visibility 只读取描述符逐字绑定的采样帧、联系表和 PERSON identity refs，不读取音频、台词或其他阶段输入。

## 2. 视觉阶段

### 输入与输出

输入均在 `work/`：

- `NN_frame_*.png`：外部程序按时间抽取的帧。
- `contact_sheet_*.jpg`：分页联系表。
- `manifest.json`：源视频时长、尺寸、帧率和帧数。
- `scenes.json`：辅助场景分组，不是事实真源。

输出固定为：

- `work/keyframes/01.png` 至 `09.png`：按源时间排序的关键帧。
- `work/prompt.txt`：纯视觉叙事提示词。

### 约束

- 眼见为实；每个细节都能对应检查过的帧。
- 最多选择 9 张；一帧一个主导状态，不凑数。
- 每个结果状态都有可见的原因状态；保持动作、接触、遮挡和因果关系。
- 选择清晰稳定、主体可见、遮挡少的帧；字幕或水印无法避开时才使用 `scripts/crop_image.py`。
- 提示词使用用户语言，只描述画面、动作、镜头、顺序和因果。
- 时长只写“与源片段时长一致”，不写具体秒数，不截断后半段内容。
- 不从画面文字推断台词，不生成 `voice_lines.json`，不写嘴型、声线、配乐或音效要求。

### 步骤

1. 按场景查看联系表，必要时打开原始帧，理清初始状态、动作和结果。
2. 按时间选择关键帧，复制到 `work/keyframes/` 并从 `01.png` 连续编号。
3. 写入 `work/prompt.txt`：

```text
生成一支与源片段时长一致、采用源视频[比例]、[分辨率，默认 720p]的[主题]短视频。

拍摄风格：[已观察到的机位、运动、光线与质感]。无字幕、贴纸或水印等叠加元素。

叙事：从图片1的状态开始，[动作与镜头如何依次推进]；最后到达图片N的[结果状态]。

因果：[先看到动作，再看到变化；动作到哪里，变化到哪里；未受作用的内容保持原样。]
```

## 3. 音画阶段

### 边界

只读取后端冻结的 `work/multimodal_input.json`；逐字保留其视觉提示和 `dialogue_source_sha256`，不得改写视觉事实、不得重新选关键帧、不得复制或补造台词文本、时间窗、语言或声音事件。

只写 `work/h3_prompt_plan.json`。不得直接写供应商最终提示词；后端确定性编译该计划为 H3 Context-IR。

### 资格门

只有全部成立才输出 `"eligible": true`：

1. 有 1–3 段冻结且可读取的参考音频；每段有唯一稳定编号，并且只承担声线参考、环境声或音效一种用途。
2. 每个画内发声人物都有上游确认的说话人与人物映射、图片引用和唯一声线参考；同画面的静默人物可以保留，但 `voice_ref` 必须为 `null`。
3. 每条现有 dialogue 行都有唯一的 1-based `line_index`，并已确认 `delivery/language`；画内发声绑定 `subject_id`，画外发声不绑定人物且 `voice_ref` 可为明确的声线引用或 `null`。
4. `speech_bindings` 按现有 dialogue 行顺序排列，`line_index` 从 1 开始连续无缺号；不得另写 `order/text/start_s/end_s/narration text`，重叠发声已由上游拆分或确认。
5. 每个环境声和音效都有用途匹配的唯一音频引用，以及输入提供的非空逐字描述。

任何必填事实缺失、未知、冲突或无法确认时，输出 `"eligible": false` 和稳定错误码，其余字段固定置空。不得根据时间重叠、嘴部外观、同框或素材顺序猜测关系。

### 严格结构

所有字段必填，禁止额外字段。成功计划固定写 `"version": 2`、`"phase": "multimodal_audio"`、`"eligible": true`。

```text
Int1 = 从 1 开始的整数
NonEmpty = 冻结输入中的非空字符串
SubjectId = 冻结输入中的稳定人物编号
ErrorCode = 稳定非空错误码

Plan = {
  version: 2;
  phase: "multimodal_audio";
  eligible: true;
  reason: null;
  visual_prompt: NonEmpty;
  dialogue_source_sha256: NonEmpty;
  subjects: Array<{ subject_id: SubjectId; picture_refs: NonEmptyArray<Int1>; voice_ref: Int1 | null }>;
  audio_refs: Array<{ audio_index: Int1; purpose: "voice" | "ambience" | "effect"; subject_id: SubjectId | null }>;
  speech_bindings: Array<{
    line_index: Int1;
    delivery: "on_screen" | "off_screen_voiceover";
    subject_id: SubjectId | null;
    language: NonEmpty;
    voice_ref: Int1 | null;
  }>;
  sound_design: {
    ambience_refs: Array<{ audio_index: Int1; description: NonEmpty }>;
    effects: Array<{ audio_index: Int1; description: NonEmpty }>;
  };
}
```

图片、音频外部编号从 1 开始并沿用输入；不得按数组位置重编号，`audio_index` 必须按输入顺序连续无缺号。`picture_refs` 是非空、升序、无重复的多值数组；引用必须存在，不同 `subject_id` 不得复用同一图片编号。冻结输入必须已给出连续的 `S1…Sn`，Skill 不得重命名；否则失败关闭。

`dialogue_source_sha256` 必须逐字复制冻结输入提供的现有 dialogue receipt SHA；不得由 Skill 重算或替换。`speech_bindings` 的长度必须与现有 dialogue 行数完全一致，数组第 N 项只能绑定 `line_index: N`。`delivery: "on_screen"` 时 `subject_id` 必须存在、`voice_ref` 必须为 `null`，实际声线由对应 `subjects[].voice_ref` 决定；`delivery: "off_screen_voiceover"` 时 `subject_id` 必须为 `null`，`voice_ref` 可引用画外声线或为 `null` 使用供应商默认声线。

同一 `audio_index` 只能有一种 `purpose`，并且必须恰好绑定一个对应语义：

- `voice`：人物声线的 `audio_refs.subject_id` 为对应人物；画外声线为 `null`。非空 `subjects.voice_ref` 与画外 `speech_bindings.voice_ref` 只能引用 `voice`。
- `ambience` 与 `effect`：`audio_refs.subject_id` 为 `null`；分别由同编号的 `ambience_refs` 或 `effects` 项引用。`description` 逐字来自输入。

画外发声不得出现 `subject_id`。现有 dialogue 的文本、时间窗、模式和来源证明始终由后端 receipt 冻结；Skill 只做逐行映射。环境声和音效不得混入 `speech_bindings`。

失败结构固定为：

```text
{
  "version": 2, "phase": "multimodal_audio", "eligible": false, "reason": ErrorCode,
  "visual_prompt": "", "dialogue_source_sha256": "", "subjects": [],
  "audio_refs": [], "speech_bindings": [],
  "sound_design": {"ambience_refs": [], "effects": []}
}
```

按上述结构写 UTF-8 严格 JSON。`line_index` 只绑定上游既有顺序；精确时间窗由后端从同一 dialogue receipt 投影，不是供应商 PTS 硬锁。人物、图片和声线关系不是未经确认的 speaker-face 推断。参考音频不是时间锁，只按 reference 语义使用，不是最终音轨；不得使用 `fully_copy`、`partially_copy` 或 `audio reuse`，不承诺完整或逐样本复制。媒体字节、哈希、格式、模式和供应商参数由后端处理，不进入语义计划。生成后仍须验收音轨、精确台词、发声顺序和人物发声关系。

## 4. 优化后语义协调阶段

### 边界与输入

只读取 `work/reconcile_after_image_optimization_input.json` 指向的原始源帧、`work/original_visual_prompt.txt`、`work/frozen_image_plan.json`、`work/image_verification.json` 和已选优化图；只写 `work/unified_visual_ir.json`。这些文件由后端冻结并校验路径，Skill 不查找其他候选图。

这是同一 Skill 的第三阶段，固定写 `"phase": "reconcile_after_image_optimization"`。existing dialogue 不得进入本阶段；输入出现 dialogue、speech、audio 或台词文件即以 `unexpected_dialogue_input` 失败关闭。不得读取音频，不得复制或推断台词、嘴型、发声人、语言或声音事件。不得重新选择、增删或重排关键帧，也不得改写帧号和 PTS。

输入描述符所有字段必填，禁止额外字段：

```text
Sha256 = 64 位小写十六进制 SHA-256
Integer = 整数
NonNegativeInt = 大于等于 0 的整数
Int1 = 从 1 开始的整数
SegmentIndex = NonNegativeInt
NonEmpty = 非空字符串
TimeBase = { numerator: Int1; denominator: Int1 }

ReconcileInput = {
  version: 1;
  phase: "reconcile_after_image_optimization";
  source_evidence: {
    frame_manifest_sha256: Sha256;
    old_visual_prompt_sha256: Sha256;
  };
  target_static_plan: {
    image_plan_sha256: Sha256;
    image_verification_sha256: Sha256;
  };
  frames: NonEmptyArray<{
    segment_index: SegmentIndex;
    frame_index: Int1;
    source_file: NonEmpty;
    source_pts: Integer;
    source_time_base: TimeBase;
    source_frame_sha256: Sha256;
    optimized_file: NonEmpty;
    optimized_image_sha256: Sha256;
    output_receipt_file: NonEmpty;
    output_receipt_sha256: Sha256;
    image_plan_sha256: Sha256;
  }>;
}
```

数组按 `(segment_index, frame_index)` 升序，必须与原始帧 manifest、canonical 计划、通过的 image verification 和已选输出 receipt 一一覆盖且哈希互相闭合。`source_pts` 是未换算的源时间戳整数；`source_time_base` 恰含互质的正整数 `numerator/denominator`，禁止使用浮点、字符串或换算为毫秒后再写回。只接受 verification 已整体通过的输入：output receipt 只绑定 image plan、源帧和优化图；后产生的 image verification receipt 绑定完整且有序的 output receipt SHA 集合，并逐字绑定同一 `image_plan_sha256`。

### 事实权威

- 原始源帧像素与 PTS 是动作、机位和时序的事实权威。旧视觉提示词仅是动作、机位和时序的低优先级检索证据；冲突时以原始源帧像素与 PTS 为准。
- canonical 图片计划是新人物和新场景的语义权威。已选优化图及其输出 receipt 是该计划已实现结果的权威；两者冲突时失败，不能挑一个继续。
- 优化图不得反向改写动作、机位、时序或物理关系。它改变这些动态事实时，判图片结果不合格；不得改写动态事实去迎合错误优化图。
- 旧人物、旧服装、旧场景、旧材质和旧静态外观只作负证据，不得进入统一视觉 IR。不得复制 canonical 计划中的目标描述；人物、服装、场景、材质与光色只通过 target_static_plan_binding 和稳定 ID 引用，由后端从同一 receipt 确定性投影。

### 资格门

只有全部成立才输出 `eligible=true`：

1. 每个原始帧恰有一个同段、同帧号、同 PTS 的已选优化图与输出 receipt；任一源帧、PTS、优化图或输出 receipt 映射缺失、重复、错序或哈希不闭合，失败为 `frame_mapping_missing`。
2. 每张优化图保持原始源帧的可见身体部位、姿态、动作状态、尺度、接触、遮挡、裁切和因果结果；优化图改变姿态、动作状态、接触、遮挡或因果结果，失败为 `optimized_action_changed`。
3. canonical 新场景和优化图保留原动作需要的功能等价支撑面、可操作物、间隙、通路、接触链和可达范围；新场景不能闭合原动作依赖的支撑、接触或可达关系，失败为 `physical_support_unclosed`，不能补造不可见过渡。
4. 每个 beat 只引用 canonical 计划中的 `PERSON_*`、`SCENE_*` 和该帧 `ENTITY_*` ID；动作字段只描述姿态、运动、状态变化与因果，镜头字段只描述拍摄几何，时序字段只描述 PTS、节奏与转场。无法去除旧静态描述时失败为 `source_static_semantics_leaked`。
5. 所有事实都能从列明证据唯一确认；任何未知或其他冲突失败为 `reconciliation_unknown`。

错误码按 `phase_input_conflict`、`unexpected_dialogue_input`、`receipt_binding_mismatch`、`frame_mapping_missing`、`optimized_action_changed`、`physical_support_unclosed`、`source_static_semantics_leaked`、`reconciliation_unknown` 的顺序取首个成立项。receipt 未通过、未绑定同一计划或内容互相矛盾均为 `receipt_binding_mismatch`。

### 严格统一视觉 IR

禁止二次自由文本真源。只描述动态事实；不输出最终视频提示词，不复制新旧人物、服装、场景、材质或光色描述。统一 IR 的所有字段必填，禁止额外字段，成功固定为 `"version": 1`、`"eligible": true`、`"reason": null`、`"conflicts": []`：

```text
PersonId = canonical 计划中的 PERSON_* ID
SceneId = canonical 计划中的 SCENE_* ID
EntityId = canonical 计划对应帧中的 ENTITY_* ID
ReconcileErrorCode = "phase_input_conflict" | "unexpected_dialogue_input" | "receipt_binding_mismatch" | "frame_mapping_missing" | "optimized_action_changed" | "physical_support_unclosed" | "source_static_semantics_leaked" | "reconciliation_unknown"

Plan = {
  version: 1;
  phase: "reconcile_after_image_optimization";
  eligible: true;
  reason: null;
  source_evidence_binding: {
    frame_manifest_sha256: Sha256;
    old_visual_prompt_sha256: Sha256;
  };
  target_static_plan_binding: {
    image_plan_sha256: Sha256;
    image_verification_sha256: Sha256;
  };
  frame_bindings: NonEmptyArray<{
    segment_index: SegmentIndex;
    frame_index: Int1;
    source_frame_sha256: Sha256;
    source_pts: Integer;
    source_time_base: TimeBase;
    optimized_image_sha256: Sha256;
    output_receipt_sha256: Sha256;
  }>;
  preserved_beats: NonEmptyArray<{
    beat_index: Int1;
    segment_index: SegmentIndex;
    frame_refs: NonEmptyArray<Int1>;
    actor_refs: Array<PersonId>;
    scene_ref: SceneId;
    entity_refs: Array<{ frame_index: Int1; entity_id: EntityId }>;
    action: { initial_state: NonEmpty; motion: NonEmpty; result_state: NonEmpty };
    camera: { shot_scale: NonEmpty; angle: NonEmpty; movement: NonEmpty; composition: NonEmpty; focus: NonEmpty };
    timing: { start_source_pts: Integer; end_source_pts: Integer; source_time_base: TimeBase; pace: NonEmpty; transition: NonEmpty };
  }>;
  conflicts: [];
}
```

`frame_bindings` 与输入帧逐项相同并按同一顺序复制。`preserved_beats` 按源 PTS 和 `beat_index` 升序，覆盖原始可见叙事的初始状态、动作和结果；`frame_refs` 升序无重复且只能引用同段 `frame_bindings`。每个 beat 的起止 PTS 必须逐字取自首尾 frame_refs，同段使用同一 `source_time_base`，相邻 beat 不得颠倒源 PTS。所有人物、场景和物体只能以稳定 ID 出现在动态字段中，动作、camera 和 timing 文本不得承载任何静态目标设计。

eligible plan 的 conflicts 必须为空。失败结构固定字段不变，两个 binding 为 `null`、两个正常数组为空；`conflicts` 至少一项且只写稳定错误码、可空帧定位和证据 receipt/SHA/ID 引用，不复制新旧静态描述：

```text
Conflict = {
  code: ReconcileErrorCode;
  segment_index: SegmentIndex | null;
  frame_index: Int1 | null;
  evidence_refs: NonEmptyArray<NonEmpty>;
}

FailurePlan = {
  version: 1;
  phase: "reconcile_after_image_optimization";
  eligible: false;
  reason: ReconcileErrorCode;
  source_evidence_binding: null;
  target_static_plan_binding: null;
  frame_bindings: [];
  preserved_beats: [];
  conflicts: NonEmptyArray<Conflict>;
}
```

按上述结构写 UTF-8 严格 JSON，无 Markdown 围栏或解释。该 IR 是后端将动态事实与 canonical 静态计划确定性合并的唯一协调输入；本阶段不编写供应商提示词、不调用 Context IR 或供应商。

## 5. 发声人物可见性阶段

### 边界

`work/speaker_visibility_input.json` 存在时，只读取它以及描述符逐字绑定的 cut source、采样帧、联系表和 PERSON identity refs；只写 `work/speaker_visibility_output.json`。不得读取 `work/multimodal_input.json`；不得读取 `work/reconcile_after_image_optimization_input.json`；不得读取音频、dialogue、speech、台词文本或台词时间窗，也不得读取未绑定的相邻帧或其他身份素材。

本阶段只做冻结样本的身份可见性和嘴部可验性分类，不判断谁正在说话。none 或全部 offscreen 时，后端不得创建 `work/speaker_visibility_input.json`，不得调用本阶段 Skill。输入结构、媒体字节、哈希、顺序或证据不闭合时不得写 `work/speaker_visibility_output.json`；本阶段没有可猜测的失败输出。

### 严格输入

输入 JSON 所有字段必填，禁止额外字段：

```text
Sha256 = 64 位小写十六进制 SHA-256
Pts = 大于等于 0 的整数
PositivePts = 大于 0 的整数
Int1 = 从 1 开始的整数
NonEmpty = 非空字符串
Boolean = true | false
TimeBase = { numerator: Int1; denominator: Int1 }
PersonId = PERSON_* 稳定 ID
SubjectId = 上游冻结的非空 on-screen subject ID

VisibilityInput = {
  schema: "duet.speaker-visibility-input";
  version: 1;
  phase: "speaker_visibility";
  source: { sha256: Sha256; duration_pts: PositivePts; time_base: TimeBase };
  sampling: {
    algorithm: "decoded_pts_nearest_v1";
    cadence_fps: 8;
    max_unobserved_gap_pts: PositivePts;
    endpoint_shrink_intervals: 1;
  };
  decoded_frame_pts: NonEmptyArray<Pts>;
  cut_pts: Array<PositivePts>;
  cut_source: { path: "scenes.json"; sha256: Sha256 };
  frames: NonEmptyArray<{ order: Int1; path: NonEmpty; sha256: Sha256; pts: Pts; cut_before: Boolean }>;
  contact_sheets: NonEmptyArray<{ order: Int1; path: NonEmpty; sha256: Sha256; frame_orders: NonEmptyArray<Int1> }>;
  persons: NonEmptyArray<{ person_id: PersonId; identity_refs: NonEmptyArray<{ path: NonEmpty; sha256: Sha256 }> }>;
  on_screen_subjects: NonEmptyArray<SubjectId>;
}
```

`source.sha256` 绑定源视频原始字节；`duration_pts`、所有 PTS 和互质正整数 `time_base` 都由后端冻结，禁止换算成浮点秒或毫秒。`decoded_frame_pts` 严格递增、无重复并位于 `[0,duration_pts)`；`cut_pts` 严格递增、无重复并位于 `(0,duration_pts)`。`cut_source` 逐字绑定产生这些 cut 的 `work/scenes.json` 原始字节，禁止另猜切镜。

`frames` 至少 4 项，按 `order=1..N` 连续排列；`pts` 严格递增，且恰好是 `decoded_pts_nearest_v1` 按 8 FPS 目标节拍选中的真实 decoded frame PTS，不是合成时间戳。每项 `path/sha256` 必须与 `speaker-visibility-frames/` 中实际可读字节匹配。`cut_before` 必须逐字匹配从上一 sample PTS 到当前 sample PTS 之间是否存在 `cut_pts`。

`contact_sheets` 按 `order=1..M` 连续排列，文件字节与 SHA 必须匹配；所有 `frame_orders` 串联后恰为 `1..N`，不重不漏。`persons` 的 `person_id` 唯一，每个 `identity_refs` 非空且绑定 `keyframes/` 中互不复用的实际字节与 SHA。`on_screen_subjects` 非空、升序、无重复。描述符绑定的 sample、contact sheet 和 identity ref 文件必须恰好闭合，不接受额外媒体。

### 映射与逐样本判定

subject_person_mapping 必须与 on_screen_subjects 等长且同序，PERSON 映射必须一对一。只有 subject 与 PERSON 身份能从冻结输入唯一证明时才允许映射；单一 on-screen subject 且 roster 只有一个 PERSON 是可机械唯一证明的情形。多人映射不能从冻结 identity refs 与采样证据唯一证明时，不得写 `work/speaker_visibility_output.json`。

不得按 subject/PERSON 数组位置、命名序号或 PERSON 在画面中的位置推断映射；不得按嘴部运动、同框关系或出现频率推断映射；不得按台词文本或台词时间窗反推。`lip_verifiable_person_ids` 只表示该 sample 中 PERSON 身份、面部和完整嘴唇边界可直接验看，不表示正在发声。

对 `frames` 逐样本穷举：每个输入 sample 恰有一个同 order 的输出项。visible_person_ids 只列该 sample 像素中可由 identity refs 唯一确认的 PERSON；lip_verifiable_person_ids 必须是 visible_person_ids 的子集，只列嘴唇边界在该 sample 自身清楚可验的 PERSON。两个数组均按 ID 升序、无重复；无法唯一确认时写空数组，不写 `unknown/maybe` 占位值。

联系表只用于导航；只能以该 sample 自身图像字节作事实。不得用相邻 sample 补证，不得用同一 contact sheet 的其他格、identity ref 或 source 中未绑定的帧补证，不得跨 `cut_before=true` 补证，不得在未知空洞之间插值，也不得复制上一帧分类。

### 严格输出

成功输出所有字段必填、禁止额外字段：

```text
VisibilityOutput = {
  schema: "duet.speaker-visibility-output";
  version: 1;
  phase: "speaker_visibility";
  input_sha256: Sha256;
  subject_person_mapping: NonEmptyArray<{ subject_id: SubjectId; person_id: PersonId }>;
  frames: NonEmptyArray<{ order: Int1; visible_person_ids: Array<PersonId>; lip_verifiable_person_ids: Array<PersonId> }>;
}
```

`input_sha256` 必须是 `work/speaker_visibility_input.json` 原始文件字节的逐字节 SHA-256，不得重排或重序列化 JSON 后再计算。后端另行冻结本次使用的 Skill 原始字节及 SHA；Skill 不复制或生成 `skill_sha256`。

按上述结构写 UTF-8 严格 JSON，无 Markdown 围栏或解释。Skill 不生成时间窗、不合并区间、不收缩端点、不生成 timing 或 receipt。后端机械合并相邻 verified samples、按 cut/空洞断开并收缩窗端点；`max_unobserved_gap_pts` 与 `endpoint_shrink_intervals` 只供后端验证和投影，Skill 不执行这些规则。
