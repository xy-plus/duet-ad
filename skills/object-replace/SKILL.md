---
name: object-replace
description: 基于 video-maker 的稳定全局索引、全部分段视频描述、用户替换要求和 Image 1，规划项目级元素替换、人物音色与一张全项目合并新形象图的 Seedream 基础提示词。用于在逐段 Fusion 前冻结唯一的替换与人物音色权威，不生成最终视频提示词。
---

# Object Replace

只规划替换、人物音色和合并新形象图的项目级基础提示，不重写原视频事实、稳定 ID、occurrences 或 relations。输出格式以本次注入的 JSON Schema 为唯一权威；逐字回填固定的 `schema`、`version` 和 `input_sha256`，覆盖 Schema 列出的每个稳定 ID，不增删字段，不输出 Markdown。

后端只检查输出能否按 Schema 结构、输入绑定和稳定 ID 机械传给下游，不会二次判断替换是否充分、对象是否匹配或音色是否最优；完成下列语义要求是本节点自身职责。

输入中的 `replacement_set` 是后端从 video-maker 索引确定性生成的必换集合，按 `people/entities/scenes` 列出全部 stable key。`global_element_index` 是只供本节点规划的事实视图：后端已从人物、实体和场景记录中移除容易与替换目标冲突的 `preserve/replaceable` 提示，但会在下游继续保留完整原始索引。`segment_visual_prompts` 用于理解动作和跨段语境。用户上传的替换图片在模型输入中称为 Image 1。`voice_catalog_lines` 每行格式为 `voice_id - file - language - gender - 短特征`；这些标签和文件对应关系不可改写或补猜。

## 替换规划

1. `replacement_set` 中的每个 stable key 都必须设置非空 `replacement`，不得因没有 `preserve/replaceable`、没有用户参考图或不是用户点名对象而保留、复制或忽略源外观。
2. 用户替换要求与 Image 1 同时存在时，只把 Image 1 约束到文字要求唯一语义对应的一个既有 stable key：该记录设 `matches_user_request=true`、`replacement.source=image_1`，并具体描述 Image 1 中的目标对象；不得把参考图背景或无关对象扩散到其他 stable key。其他必换 key 仍各自使用 `generated_variant`，不能因为 Image 1 只指向一个对象而停止替换。
3. `people` 中所有可见人物都换成同类型但不同身份的新人物。只保留索引明确给出的性别、种族、年龄段、整体风格和服装主色；必须改变人物自身的人脸或可见身体，服装保持颜色和风格但改变款式。只有身体局部可见时，只针对该局部的形态、比例、肤色、纹理或标记生成统一的新设计，不推断画面外的人脸、服装或受保护属性，也不为具体身体部位编写特例。
4. `scenes` 中所有真实场景都换成同类但不同实例。保持场景类型、用途、布局、色调、构图、机位、镜头运动、焦段和光线；通过主要陈设、表面材质或图案、门窗或柜体结构等可见设计形成新实例，不得只复述源场景或只改变其关联物。
5. `entities` 中所有可见实体都必须替换，包括用户指定元素及其联动组件。保持功能、动作接口、尺度、接触和相对关系，同时明确改变材质、颜色、品种、轮廓、结构或造型中的一个或多个主要维度；只添加附件、只改关联物或只做近邻色相变化不算替换。
6. 同一个 stable key 在不同图片、不同分段和所有 occurrence 中必须逐字复用同一套新设计。联动元素的设计必须兼容，但仍各自输出自己的 replacement；`replace_together` 不会把 Image 1 或 `matches_user_request` 自动传播给无关元素。
7. `target_visual_description` 先简短写清完整的新目标，再点明相对源素材的可观察差异；描述必须可直接供 Fusion 使用。保持动作、姿态、位置、尺度、接触、遮挡、前后关系、角色功能和跨段连续性，不得保留与新身份、新款式、新材质、新造型或新场景实例冲突的源外观。

## 合并新形象图

1. `reference_sheet_prompt` 只是供 Seedream 生成一张全项目合并新形象图的项目级基础提示词，不是 Fusion 或 H3 的最终提示词。
2. 基础提示词必须描述 `people`、`entities` 和 `scenes` 中所有实际非空 `replacement`，逐个写出完整 stable key 与对应的 `target_visual_description`；不得增加索引外元素，也不得合并、改名或重复 stable key。
3. 全局索引是唯一锚点。同一 stable key 在 `reference_sheet_prompt`、所有分段和后续 Fusion 中必须复用同一套新设计，不得在生图提示中另行改造或补猜。
4. 该字段只描述新形象内容，不决定、声明或重排 TILE、网格、槽位、图片顺序或 Image N。后端必须根据冻结的全局索引机械分配这些定位信息，本节点不得覆盖。
5. Image 1 只约束 `replacement.source=image_1` 的对应 stable key。其他 stable key 的新形象不得继承 Image 1 的身份、外观、背景、风格或无关元素。

## 人物音色

1. 为 `people` 中每个人物选择一个目录内存在的 `voice_id`；即使人物未匹配用户替换要求，也不得省略音色。
2. 只依据索引明确给出的语言、性别、种族、年龄和表达风格匹配；缺失属性不得推断。多人可以复用 voice_id，全项目最多使用三个不同 voice_id。
3. 人物超过三个时，按相近语言与表达风格复用最合适的已有 voice_id。不要输出文件路径、Audio 序号、目录标签副本或解释；这些由后端根据稳定人物顺序机械补入增强索引。

本次 `replacement_set` 已覆盖全部人物、实体和场景，因此本节点不应输出 `replacement=null`。后端仍只校验结构与输入绑定，不把这条语义职责变成质量门禁。关系不在本阶段重新生成。
