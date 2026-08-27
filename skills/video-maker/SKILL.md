---
name: video-maker
description: 从预先抽取的参考视频帧理解叙事、选择最多 9 张有序关键帧并生成纯视觉叙事；当后端另行提供冻结的多模态输入时，规划 H3 所需的人物、图片、声线参考、精确台词、语言、旁白、环境声和音效绑定。用于视频分析的视觉阶段或 H3 音画联合生成前的音画阶段；不提交视频、不调用供应商。
---

# video-maker

## 1. 选择阶段

只按文件选择一种阶段，不混用：

- `work/multimodal_input.json` 不存在：执行**视觉阶段**。
- `work/multimodal_input.json` 存在：执行**音画阶段**。

视觉阶段只处理图片与文字，不读取、推断或编排音频。音画阶段只规划冻结输入的语义绑定，不重新分析原视频。

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

后端已把视觉事实和音频事实冻结在 `work/multimodal_input.json`。只读取该文件；不得改写视觉事实、不得重新选关键帧、不得补造未提供的人物、台词或声音事件。

音画阶段只写 `work/h3_prompt_plan.json`。不得直接写供应商最终提示词；后端确定性编译该计划为 H3 Context-IR，并冻结编译器版本和哈希。

计划只表达素材之间的语义关系。不得向计划加入媒体路径、字节、哈希、格式、模式或供应商参数；这些事实继续由后端冻结、校验和提交。

### 稳定编号

- 图片与音频的外部编号都从 1 开始，逐字沿用输入给出的稳定编号；不得出现 0，不得按数组位置重新编号，也不得把后端的零基参数名抄进计划。
- `subject_id` 逐字沿用输入；不得按人物出场或发声顺序重命名。
- `picture_refs` 是非空、升序、无重复的外部图片编号。一个人物可引用多张图片；同一张图片也可为多个人物提供事实。
- `audio_index` 必须唯一引用输入中的一段音频。每段音频只承担 `voice`、`ambience`、`effect` 中一个用途，不能悬空或跨用途复用。

### 资格门

只有同时满足下列条件才可令 `eligible=true`：

- 有 1–3 段已冻结、有序且可读取的参考音频；每段的用途只能是声线参考、环境声或音效。
- 每句人物台词都有明确的 `subject_id`、语言、逐字文本和顺序；每个发声人物都有经过上游确认的说话人与人物映射，以及唯一声线参考。
- 每句旁白都有语言、逐字文本和顺序，不绑定画面人物；只有输入明确提供旁白声线时才填写其 `voice_ref`，否则写 `null`。
- `dialogue[].order` 与 `sound_design.narration[].order` 共用一条从 1 开始且无缺号的全局发声顺序。二者合并后必须唯一、升序并覆盖全部发声事件；重叠发声未被上游拆分成明确顺序时不合格。
- 图片、音频、人物和台词使用输入给出的稳定编号；没有缺号、重复或悬空引用。
- 环境声与音效均有输入提供的非空语义描述，并且引用用途相符的唯一音频。

任一条件不成立时，输出 `eligible=false`、稳定 `reason`，其余绑定数组置空。不得根据时间重叠、嘴部外观或同框关系猜测说话人。

### 字段契约

成功计划的顶层字段严格只有 `version/phase/eligible/reason/visual_prompt/subjects/audio_refs/dialogue/sound_design`，各数组保持冻结输入的稳定顺序：

- `subjects[]` 每项严格只有 `subject_id/picture_refs/voice_ref`。`voice_ref` 为正整数；没有人物台词的静默主体才可为 `null`。
- `audio_refs[]` 每项严格只有 `audio_index/purpose/subject_id`。`purpose` 只能是 `voice/ambience/effect`；人物声线写对应 `subject_id`，旁白声线、环境声和音效写 `null`。
- `dialogue[]` 每项严格只有 `order/subject_id/language/text`。
- `sound_design.narration[]` 每项严格只有 `order/language/text/voice_ref`；不得出现 `subject_id`。
- `sound_design.ambience_refs[]` 每项严格只有 `audio_index/description`。
- `sound_design.effects[]` 每项严格只有 `audio_index/description`。

`language`、`text` 和 `description` 逐字复制输入，包括语言标签、用字和标点；不得从画面文字、参考音频原词或常识补写。`order` 只表达语义先后，不是精确 PTS，也不生成起止时间字段。

`subject_id + picture_refs + voice_ref` 是供 Context-IR 编译的声明式关系，不是供应商的 speaker-face API，也不证明人脸、嘴型或声线会被模型强绑定。参考音频不是时间锁，也不等于最终音轨；所有参考音频都按 `reference` 语义处理：声线只参考音色与表达，环境声和音效只参考其声音语义；不得使用 `fully_copy`、`partially_copy` 或 `audio reuse`，不得声称逐样本复用原音或保留原时间戳。后续必须验收输出音轨、逐字台词、发声顺序和人物发声关系。

### 输出结构

成功时输出严格 JSON。下列占位文字只定义来源，不提供可复用的台词或声音样本；实际值必须逐字来自输入：

```json
{
  "version": 1,
  "phase": "multimodal_audio",
  "eligible": true,
  "reason": null,
  "visual_prompt": "逐字保留输入的视觉事实",
  "subjects": [
    {
      "subject_id": "S1",
      "picture_refs": [1, 2],
      "voice_ref": 1
    }
  ],
  "audio_refs": [
    {
      "audio_index": 1,
      "purpose": "voice",
      "subject_id": "S1"
    },
    {
      "audio_index": 2,
      "purpose": "ambience",
      "subject_id": null
    },
    {
      "audio_index": 3,
      "purpose": "effect",
      "subject_id": null
    }
  ],
  "dialogue": [
    {
      "order": 1,
      "subject_id": "S1",
      "language": "输入语言标签",
      "text": "输入逐字原文"
    }
  ],
  "sound_design": {
    "narration": [
      {
        "order": 2,
        "language": "输入语言标签",
        "text": "输入逐字原文",
        "voice_ref": null
      }
    ],
    "ambience_refs": [
      {
        "audio_index": 2,
        "description": "逐字保留输入的声音描述"
      }
    ],
    "effects": [
      {
        "audio_index": 3,
        "description": "逐字保留输入的声音描述"
      }
    ]
  }
}
```

失败时只输出：

```json
{
  "version": 1,
  "phase": "multimodal_audio",
  "eligible": false,
  "reason": "稳定错误码",
  "visual_prompt": "",
  "subjects": [],
  "audio_refs": [],
  "dialogue": [],
  "sound_design": {"narration": [], "ambience_refs": [], "effects": []}
}
```

`reason` 只使用下列稳定错误码，并按从上到下的优先级选择第一个：`input_schema_invalid`、`audio_count_invalid`、`reference_unreadable`、`reference_index_invalid`、`speaker_mapping_invalid`、`voice_reference_invalid`、`vocal_order_invalid`、`sound_binding_invalid`。

### 自检

- `visual_prompt` 与输入逐字一致。
- `subject_id` 不变；picture/audio 引用均为 1-based、存在、用途匹配，且没有多余字段。
- 每个发声人物只有一个声线参考；旁白没有 `subject_id`；环境声和音效不冒充声线。
- 人物台词与旁白合并后的 `order` 从 1 连续递增；文本、语言、声音描述逐字来自输入。
- 没有新增视觉动作、人物、台词、语言、声音事件、时间字段或供应商参数。
- JSON 以 UTF-8 写入唯一输出路径，未写其他文件。
