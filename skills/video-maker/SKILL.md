---
name: video-maker
description: 从预先抽取的参考视频帧理解叙事、选择最多 9 张有序关键帧并生成纯视觉叙事；当后端另行提供冻结的多模态输入时，只规划 H3 所需的人物、图片、现有台词行、语言、画内或画外发声、声线参考、环境声和音效绑定。用于视频分析的视觉阶段或 H3 音画联合生成前的音画阶段；不提交视频、不调用供应商。
---

# video-maker

## 1. 选择阶段

只按文件选择一种阶段，不混用：

- `work/multimodal_input.json` 不存在：执行**视觉阶段**。
- `work/multimodal_input.json` 存在：执行**音画阶段**。

视觉阶段不得读取任何音频文件，只处理图片与文字。音画阶段只规划冻结输入的语义绑定，不重新分析原视频。

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
