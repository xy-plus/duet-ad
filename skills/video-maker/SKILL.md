---
name: video-maker
description: 从预先抽取的参考视频帧理解叙事、选择最多 9 张有序关键帧并生成纯视觉叙事；当后端另行提供冻结的多模态输入时，规划 H3 所需的人物、图片、声线参考、精确台词、语言、旁白、环境声和音效绑定。用于视频分析的视觉阶段或 H3 音画联合生成前的音画阶段；不提交视频、不调用供应商。
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

只读取后端冻结的 `work/multimodal_input.json`；逐字保留其视觉提示，不得改写视觉事实、不得重新选关键帧、不得补造未提供的人物、台词、语言或声音事件。

只写 `work/h3_prompt_plan.json`。不得直接写供应商最终提示词；后端确定性编译该计划为 H3 Context-IR。

### 资格门

只有全部成立才输出 `"eligible": true`：

1. 有 1–3 段冻结且可读取的参考音频；每段有唯一稳定编号，并且只承担声线参考、环境声或音效一种用途。
2. 每个发声人物都有上游确认的说话人与人物映射、图片引用和唯一声线参考；不得输出没有人物台词的静默 subject。
3. 每句人物精确台词都有非空 `subject_id/language/text/order`；每句旁白都有非空 `language/text/order`，不绑定人物，`voice_ref` 可为明确的声线引用或 `null`。
4. `dialogue[].order` 与 `narration[].order` 共用从 1 开始、连续无缺号的全局发声顺序；重叠发声已由上游拆分或确认。
5. 每个环境声和音效都有用途匹配的唯一音频引用，以及输入提供的非空逐字描述。

任何必填事实缺失、未知、冲突或无法确认时，输出 `"eligible": false` 和稳定错误码，其余字段固定置空。不得根据时间重叠、嘴部外观、同框或素材顺序猜测关系。

### 严格结构

所有字段必填，禁止额外字段。成功计划固定写 `"version": 1`、`"phase": "multimodal_audio"`、`"eligible": true`。

```text
Int1 = 从 1 开始的整数
NonEmpty = 冻结输入中的非空字符串
SubjectId = 冻结输入中的稳定人物编号
ErrorCode = 稳定非空错误码

Plan = {
  version: 1;
  phase: "multimodal_audio";
  eligible: true;
  reason: null;
  visual_prompt: NonEmpty;
  subjects: Array<{ subject_id: SubjectId; picture_refs: NonEmptyArray<Int1>; voice_ref: Int1 }>;
  audio_refs: Array<{ audio_index: Int1; purpose: "voice" | "ambience" | "effect"; subject_id: SubjectId | null }>;
  dialogue: Array<{ order: Int1; subject_id: SubjectId; language: NonEmpty; text: NonEmpty }>;
  sound_design: {
    narration: Array<{ order: Int1; language: NonEmpty; text: NonEmpty; voice_ref: Int1 | null }>;
    ambience_refs: Array<{ audio_index: Int1; description: NonEmpty }>;
    effects: Array<{ audio_index: Int1; description: NonEmpty }>;
  };
}
```

图片、音频外部编号从 1 开始并沿用输入；不得按数组位置重编号，`audio_index` 必须按输入顺序连续无缺号。`picture_refs` 是非空、升序、无重复的多值数组；引用必须存在，不同 `subject_id` 不得复用同一图片编号。冻结输入必须已给出连续的 `S1…Sn`，Skill 不得重命名；否则失败关闭。

同一 `audio_index` 只能有一种 `purpose`，并且必须恰好绑定一个对应语义：

- `voice`：人物声线的 `audio_refs.subject_id` 为对应人物；旁白声线为 `null`。`subjects.voice_ref` 与非空 `narration.voice_ref` 只能引用 `voice`。
- `ambience` 与 `effect`：`audio_refs.subject_id` 为 `null`；分别由同编号的 `ambience_refs` 或 `effects` 项引用。`description` 逐字来自输入。

旁白不得出现 `subject_id`。`dialogue` 与 `narration` 两个数组各自按 `order` 升序，合并后必须唯一、连续并覆盖全部发声事件；环境声和音效没有 `order`，不得混入全局发声顺序。

失败结构固定为：

```text
{
  "version": 1, "phase": "multimodal_audio", "eligible": false, "reason": ErrorCode,
  "visual_prompt": "", "subjects": [], "audio_refs": [], "dialogue": [],
  "sound_design": {"narration": [], "ambience_refs": [], "effects": []}
}
```

按上述结构写 UTF-8 严格 JSON。`order` 只表示语义先后，不是精确 PTS；人物、图片和声线关系不是 speaker-face 绑定。参考音频不是时间锁，只按 reference 语义使用，不是最终音轨；不得使用 `fully_copy`、`partially_copy` 或 `audio reuse`，不承诺完整或逐样本复制。媒体字节、哈希、格式、模式和供应商参数由后端处理，不进入语义计划。生成后仍须验收音轨、精确台词、发声顺序和人物发声关系。
