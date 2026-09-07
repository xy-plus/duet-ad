---
name: h3-max-r2v-prompt
description: 根据原片段描述、Object Replace 增强后的全局对象索引、项目级新形象图版映射和翻译台词，为 MiniMax H3 Max Reference-to-Video 编写逐段最终提示词。
---

# H3 Max R2V Prompt

只生成可直接提交给视频模型的一段提示词，不解释过程。传输格式以本次调用注入的 JSON Schema 为唯一权威：逐字回填固定的 `schema`、`version`、`input_sha256` 和 `segment_index`，把完整提示词放入 `prompt`；不得增删字段，不输出 Markdown。

冻结的 `global_element_index` 是本项目唯一对象身份、替换与人物音色锚点。不得重新识别、重命名、合并或拆分 stable key，不得重新决定替换目标，不得要求读取用户替换说明，也不得根据常识新增替换。逐项读取当前 segment 的 occurrences、replacement、voice、preserve 和 relations。

`replacement_sheet` 是按同一冻结索引生成的项目级单张新形象图版；`h3_reference_images` 给出 H3 图片的有序角色。后端生成的 `stable key -> TILE_XX -> 实际 Image N` 映射是唯一图片引用权威：图版固定为 `Image 1`（身份锚点占主参考位），存在唯一有效用户图绑定时用户图是 `Image 2`；没有该绑定时不发送原始用户图，图版仍是 `Image 1`。用户图只约束增强索引中与其对应的那个 stable key，不得把用户图外观扩散给其他 stable key；所有替换对象最终都使用各自在图版中的 tile。同一项目的每个 segment 必须沿用同一张图版和同一套映射，不得重新编号或自行推断图片顺序。

`authoritative_constraints` 是后端从增强索引和图版 spec 机械编译的本段替换、`stable key -> TILE_XX -> Image N`、关系和人物音色/台词轨权威块，后端会在你的 `prompt` 尾部原样附加。你必须以它规划主体提示；主体中不得出现稳定 ID、TILE、Image N、Audio N 或 BEGIN/END 块，不得另写对象、图片或音色映射，也不得输出与它冲突的要求。后端只校验输出结构和输入绑定，不会二次判断你是否完成这些语义要求；完整表达替换、音色和连续性是本节点自身职责。

1. 对当前片段出现且 `replacement` 非 null 的每个 stable key，让动作、遮挡和连续性适配已冻结的新设计；模型主体只写自然的画面和动作衔接，不写 stable key。源对象、目标对象及其图版 tile 由输出尾部的权威块逐项明确。
2. 指示 H3 从权威块指定的图版 tile 取得该 stable key 的新形象，并替换 Video 1 中对应的旧形象；不得从整张图版任选相似对象，不得跨 tile 借用外观。即使 `replacement.source="image_1"`，也以该 stable key 在新形象图版中的最终 tile 为下游外观权威，而不是把原始用户图直接套给其他对象。人物 tile 是头肩面部特写：它是人脸身份的唯一权威，必须明确要求 H3 把 Video 1 中人物的脸换成该 tile 的脸（脸型、五官、肤色、发型），不能只换衣服或配饰；人物的服装与全身外观以权威块的文字描述为准。
3. `replacement=null` 的实体保持原样，不得把它加入替换清单。人物和场景按索引中的非空 replacement 完整执行，不得遗漏。
4. 对当前片段出现的每个人物，读取其 `voice.audio_reference` 和 `voice.features`。本段唯一的音频输入 Audio 1 是后端预生成的台词语音轨：各人物的台词已按各自音色合成在内，人物只需精确对 Audio 1 的口型与节奏，不得让人物说出台词轨之外的任何语音。
5. 保持原片段动作、因果、速度、机位、构图、镜头运动、焦段、光线和转场，并遵守逐帧 `relations`，尤其是 `replace_together=true` 的接口、尺度和功能配合。对 `replacement` 非 null 的记录，只服从 `authoritative_constraints` 编译出的安全不变量，不得用 raw `preserve` 恢复源身份或外观；raw `preserve` 只直接用于 `replacement=null` 的 KEEP 项。
6. 原片段在下游模型输入中称为 Video 1。只使用后端已经给出的实际 Image N 和 Audio N 映射；不得假定 `Image 1` 永远是同一种角色。
7. 不改写台词。台词文本和对口型要求由后端在你的输出之后机械附加。
