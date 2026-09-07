---
name: object-replace
description: 基于 video-maker 的稳定全局索引、全部分段视频描述、用户替换要求和 Image 1，规划项目级元素替换、人物音色与每个替换元素的隔离新外观。用于在逐段 Fusion 前冻结唯一的替换与人物音色权威，不生成最终视频提示词。
---

# Object Replace

只规划替换、人物音色和每个新形象的隔离外观，不重写原视频事实、稳定 ID、occurrences 或 relations。输出格式以本次注入的 JSON Schema 为唯一权威；逐字回填固定的 `schema`、`version` 和 `input_sha256`，覆盖 Schema 列出的每个稳定 ID，不增删字段，不输出 Markdown。

后端只检查输出能否按 Schema 结构、输入绑定和稳定 ID 机械传给下游，不会二次判断替换是否充分、对象是否匹配或音色是否最优；完成下列语义要求是本节点自身职责。

输入中的 `replacement_set` 是后端从 video-maker 索引确定性生成的必换集合，按 `people/entities/scenes` 列出全部 stable key。`global_element_index` 中完整且逐字一致的 stable key 是跨节点、跨图片、跨分段识别同一元素的唯一锚点；名称、相似外观、空间邻近或关系都不能替代 stable key。`global_element_index` 是只供本节点规划的事实视图：后端已从人物、实体和场景记录中移除容易与替换目标冲突的 `preserve/replaceable` 提示，但会在下游继续保留完整原始索引。`segment_visual_prompts` 用于理解动作和跨段语境。用户上传的替换图片在模型输入中称为 Image 1。`voice_catalog_lines` 每行格式为 `voice_id - file - language - gender - 短特征`；这些标签和文件对应关系不可改写或补猜。

## 替换规划

1. `replacement_set` 中的每个 stable key 都必须设置非空 `replacement`，不得因没有 `preserve/replaceable`、没有用户参考图或不是用户点名对象而保留、复制或忽略源外观。
2. 非空用户替换要求与 Image 1 同时存在时，必须且只能把 Image 1 约束到文字要求唯一语义对应的一个既有 stable key：该记录设 `matches_user_request=true` 且 `replacement.source=image_1`，其余记录必须 `matches_user_request=false` 且不得使用 `image_1`。这两个字段必须双向一致，不能只设置其中一个；不得任意挑选、遗漏绑定或绑定多个 key。没有 Image 1 或用户替换要求为空时，所有记录都必须 `matches_user_request=false` 且不得使用 `image_1`。不得把参考图背景或无关对象扩散到其他 stable key；其他必换 key 仍各自使用 `generated_variant`，不能因为 Image 1 只指向一个对象而停止替换。
3. `people` 中所有可见人物都换成同类型但不同身份的新人物。只强制保留索引明确给出的性别和年龄段；新身份必须在至少两个可见维度上与源人物有明确差异（从发型长短、卷直、发色、脸型轮廓、有无胡须/明显面部标记中选择），不得只改变细节。服装保持颜色和风格但改变款式。只有身体局部可见时，只针对该局部的形态、比例、肤色、纹理或标记生成统一的新设计，不推断画面外的人脸、服装或受保护属性，也不为具体身体部位编写特例。
4. `scenes` 中所有真实场景都换成同类但不同实例。保持场景类型、用途、布局、色调、构图、机位、镜头运动、焦段和光线；通过主要陈设、表面材质或图案、门窗或柜体结构等可见设计形成新实例，不得只复述源场景或只改变其关联物。
5. `entities` 中所有可见实体都必须替换，包括用户指定元素及其联动组件。保持功能、动作接口、尺度、接触和相对关系，同时明确改变材质、颜色、品种、轮廓、结构或造型中的一个或多个主要维度；只添加附件、只改关联物或只做近邻色相变化不算替换。
6. 同一个 stable key 在不同图片、不同分段和所有 occurrence 中必须逐字复用同一套新设计。联动元素的设计必须兼容，但仍各自输出自己的 replacement；`replace_together` 不会把 Image 1 或 `matches_user_request` 自动传播给无关元素。
7. 每个 `replacement` 同时输出 `target_visual_description` 和 `appearance_description`，两者描述同一个 stable key、同一套新设计，但职责不可混用。
8. `target_visual_description` 先简短写清完整的新目标，再点明相对源素材的可观察差异；描述必须可直接供 Fusion 使用。保持动作、姿态、位置、尺度、接触、遮挡、前后关系、角色功能和跨段连续性，不得保留与新身份、新款式、新材质、新造型或新场景实例冲突的源外观。动作和关系只能写在此字段，不得转写进 `appearance_description`。

## 隔离新外观

1. `appearance_description` 只写当前 stable key 自身内在、可见的新外观，是后端生成合并新形象图的单元素输入；它不是分段视频叙事，也不是 Fusion 或 H3 的最终提示词。
2. 每条 `appearance_description` 必须能够脱离原帧独立绘制，只出现该 stable key 本身。严禁写动作、姿态、接触、交互、空间位置、前后关系、镜头内构图、所在场景、其他 stable key 或其他前景对象；不得用关系描述间接带入别的元素。
3. `people`：只描述该人物自身的身份外观。人物格在图版中以头肩面部特写绘制，因此面部身份特征必须写得最充分、最具体：脸型、五官（眼、鼻、嘴、眉）、肤色与肤质、雀斑/痣/皱纹等标记、发型发色发色线；服装只保留主色和风格的一句概述。新身份必须在至少两个可见维度上与源人物明确不同（发型长短/卷直/发色、脸型轮廓、有无胡须或明显面部标记），仅保留性别与年龄段；只改五官细节而发型脸型不变属于撞脸，必须拒绝。若索引只看见身体局部，只描述该可见局部的形态、比例、肤色、纹理或标记，不推断画外人脸、服装或身体。
4. `entities`：只描述该实体自身的类别、轮廓、结构、材质、颜色、品种、纹理或标记；不得出现持有者、操作者、配套对象、被作用对象、环境或接触关系。
5. `scenes`：描述同类不同实例的空置环境自身，可包含布局、材质、色调、建筑结构、表面和固定陈设；必须排除索引人物、动物、可移动实体、前景商品、动作和接触关系。
6. 完整 stable key 是唯一锚点。同一 stable key 的 `appearance_description`、`target_visual_description`、所有分段和后续 Fusion 必须复用同一套新设计，不得另行改造或补猜；不同 stable key 不得合并、改名或共享一条描述。
7. 本节点不得决定、声明或重排 TILE、网格、槽位、图片顺序或 Image N；这些定位信息由后端依据冻结的全局索引机械分配。
8. Image 1 只约束 `replacement.source=image_1` 的对应 stable key。其 `appearance_description` 只提取参考图中对应对象自身的可见外观；其他 stable key 的新形象不得继承 Image 1 的身份、外观、背景、风格或无关元素。

## 人物音色

1. 为 `people` 中每个人物选择一个目录内存在的 `voice_id`；即使人物未匹配用户替换要求，也不得省略音色。
2. 只依据索引明确给出的语言、性别、种族、年龄和表达风格匹配；缺失属性不得推断。多人可以复用 voice_id，全项目最多使用三个不同 voice_id。
3. 人物超过三个时，按相近语言与表达风格复用最合适的已有 voice_id。不要输出文件路径、Audio 序号、目录标签副本或解释；这些由后端根据稳定人物顺序机械补入增强索引。

## 台词归属

1. `dialogue_voices` 与输入 `translated_dialogue` 逐行一一对应，长度必须相等；每项是该行台词的说话人 stable key（必须来自 `people`）或 `null`。
2. 依据台词文本、时间窗和各段人物 occurrences 判断说话人；台词时间窗内出现且有语境支撑的人物才能作为归属。
3. 旁白、画外音或无法确定说话人的行一律给 `null`，不得猜测归属。

本次 `replacement_set` 已覆盖全部人物、实体和场景，JSON Schema 因此要求每个 stable key 的 `replacement` 都是非空对象；缺失或 `null` 属于结构不完整，必须拒绝并重试。后端不对描述做审美判断。关系不在本阶段重新生成。
