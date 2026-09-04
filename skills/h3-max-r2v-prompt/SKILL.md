---
name: h3-max-r2v-prompt
description: 根据原片段描述、Object Replace 增强后的全局对象索引和翻译台词，为 MiniMax H3 Max Reference-to-Video 编写逐段最终提示词。
---

# H3 Max R2V Prompt

只生成可直接提交给视频模型的一段提示词，不解释过程。传输格式以本次调用注入的 JSON Schema 为唯一权威：逐字回填固定的 `schema`、`version`、`input_sha256` 和 `segment_index`，把完整提示词放入 `prompt`；不得增删字段，不输出 Markdown。

`global_element_index` 是本项目唯一替换与人物音色权威。不得重新决定替换目标，不得要求读取用户替换说明，也不得根据常识新增替换。逐项读取当前 segment 的 occurrences、replacement、voice、preserve 和 relations：

`authoritative_constraints` 是后端从增强索引机械编译的本段替换、关系和 `person -> Audio N` 权威块，后端会在你的 `prompt` 尾部原样附加。你必须以它规划主体提示；主体中不得出现稳定 ID、Image 1、Audio N 或 BEGIN/END 块，不得另写对象与音色映射，也不得输出与它冲突的要求。后端只校验输出结构和输入绑定，不会二次判断你是否完成这些语义要求；完整表达替换、音色和连续性是本节点自身职责。

1. 对当前片段出现且 `replacement` 非 null 的每个稳定 ID，让动作、遮挡和连续性适配已冻结的目标；模型主体只写自然的画面和动作衔接，不写稳定 ID。源对象、目标对象和替换方式由输出尾部的权威块逐项明确。
2. 当 `replacement.source="image_1"` 时，按冻结目标的外观规划动作与遮挡，但模型主体不写 `Image 1`；后端权威块会明确引用下游收到的 Image 1。没有任何当前元素引用 `image_1` 时，不得暗示存在图片参考。
3. `replacement=null` 的实体保持原样，不得把它加入替换清单。人物和场景按索引中的非空 replacement 完整执行，不得遗漏。
4. 对当前片段出现的每个人物，读取其 `voice.audio_reference` 和 `voice.features`；人物到 Audio N 的精确约束由权威块给出。不得交换 Audio 序号，不得把一个人物的音色套给另一个人物。
5. 保持原片段动作、因果、速度、机位、构图、镜头运动、焦段、光线和转场，并遵守逐帧 `relations`，尤其是 `replace_together=true` 的接口、尺度和功能配合。对 `replacement` 非 null 的记录，只服从 `authoritative_constraints` 编译出的安全不变量，不得用 raw `preserve` 恢复源身份或外观；raw `preserve` 只直接用于 `replacement=null` 的 KEEP 项。
6. 原片段在下游模型输入中称为 Video 1。只使用增强索引已经给出的 Image 1 和 Audio N 映射。
7. 不改写台词。台词和时间窗由后端在你的输出之后机械附加。
