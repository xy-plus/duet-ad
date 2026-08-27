---
name: video-maker
description: 从参考帧选择关键帧并生成纯视觉叙事，或把后端冻结的 H3 多模态事实编排成可确定编译的音画计划；不提交任务、不调用供应商。
---

# video-maker

## 阶段路由

- `work/multimodal_input.json` 不存在：执行**视觉阶段**。
- `work/multimodal_input.json` 存在：执行**音画阶段**。

两阶段不得混合。视觉阶段不得读取任何音频文件；音画阶段只读取该 JSON，不得重新分析媒体。

## 视觉阶段

输入是 `work/NN_frame_*.png`、`contact_sheet_*.jpg`、`manifest.json`；`scenes.json` 仅辅助分组。输出仅为：

- `work/keyframes/01.png` 至 `09.png`：按源时间连续编号。
- `work/prompt.txt`：用户语言的纯视觉叙事。

1. 查看联系表，必要时查看原始帧；只写已检查帧支持的事实。
2. 选最多 9 张清晰稳定、主体可见、遮挡少的帧；每帧一个主导状态，每个结果有可见原因，不凑数。
3. 按画面顺序描述构图、动作、镜头、结果和因果；保持接触、遮挡和未受作用内容。字幕或水印无法避开时才用 `scripts/crop_image.py`。
4. 时长只写“与源片段时长一致”，不写具体秒数或截断后半段；不从画面文字推断台词，不生成 `voice_lines.json`，不写嘴型、声线、配乐或音效。

## 音画阶段

### 边界

后端已在 `work/multimodal_input.json` 冻结事实。逐字保留视觉提示；不得改写视觉事实、不得重新选关键帧、不得补造未提供的人物、台词、语言或声音事件。

只写 `work/h3_prompt_plan.json`。不得直接写供应商最终提示词；后端确定性编译为 H3 Context-IR。

### 资格门

仅当全部成立才输出 `"eligible": true`：

1. 有 1–3 段冻结参考音频；每个 `audio_index` 唯一且只有声线参考、环境声或音效一种用途。
2. 每个发声人物都有上游确认的说话人与人物映射、稳定 `subject_id` 和唯一声线参考。
3. 每句人物台词都有非空 `subject_id`、`language`、`text`、`order`；每句旁白都有非空 `voice_ref`、`language`、`text`、`order`，且不绑定画面人物。精确台词逐字来自输入。
4. 图片、音频、人物、台词和声音事件无重复、悬空或跨用途引用；重叠说话已由上游拆分或确认。

任何必填事实缺失、未知、冲突或无法确认时，输出 `"eligible": false` 和稳定错误码。不得根据时间重叠、嘴部外观、同框或素材顺序猜测说话人、声音用途或素材关系。

### 严格结构

下列类型是输出契约：所有字段必填，禁止额外字段。成功计划固定写 `"version": 1`、`"phase": "multimodal_audio"`、`"eligible": true`。

```text
Int1 = 从 1 开始的整数
NonEmpty = 冻结输入中的非空字符串
SubjectId = 冻结输入中的非空稳定人物编号
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
    narration: Array<{ order: Int1; voice_ref: Int1; language: NonEmpty; text: NonEmpty }>;
    ambience_refs: Array<{ order: Int1; audio_ref: Int1 }>;
    effects: Array<{ order: Int1; audio_ref: Int1 | null; subject_id: SubjectId | null; text: NonEmpty }>;
  };
}
```

图片、音频沿用输入的 1-based 外部编号；`picture_refs` 是非空多值数组。所有 `order` 在 `dialogue` 和三个 `sound_design` 数组间构成唯一、连续的全局播放顺序。

同一 `audio_index` 只能有一种 `purpose`：

- `purpose=voice`：绑定人物时 `audio_refs.subject_id` 为该人物，供旁白时为 `null`；`subjects.voice_ref`、`narration.voice_ref` 只能引用此类音频。
- `purpose=ambience`：`audio_refs.subject_id` 为 `null`；`ambience_refs` 只引用 `purpose=ambience`。
- `purpose=effect`：`audio_refs.subject_id` 为 `null`；`effects.audio_ref` 只引用 `purpose=effect`。

旁白用 voice_ref 明确发声者，不得出现 subject_id。`effects.audio_ref` 或 `effects.subject_id` 为 `null` 只表示输入明确“不绑定”，不表示未知。环境声、音效和声线不得互换或复用用途。

失败结构固定为：

```text
{
  "version": 1, "phase": "multimodal_audio", "eligible": false, "reason": ErrorCode,
  "visual_prompt": "", "subjects": [], "audio_refs": [], "dialogue": [],
  "sound_design": {"narration": [], "ambience_refs": [], "effects": []}
}
```

按上述结构写 UTF-8 严格 JSON。参考音频不是时间锁。参考音频不是最终音轨；只表达已声明的参考语义，不保证时间对齐、逐样本复用或供应商服从。生成后仍须验收输出音轨、精确台词和人物发声关系。
