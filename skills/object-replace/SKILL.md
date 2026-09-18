---
name: object-replace
description: 基于 video-maker 的稳定全局索引、全部分段视频描述、用户替换要求和有序用户参考图，规划项目级元素替换、人物音色与每个替换元素的隔离新外观。用于在逐段 Fusion 前冻结唯一的替换与人物音色权威，不生成最终视频提示词。
---

# Object Replace

只规划替换、人物音色和每个新形象的隔离外观，不重写原视频事实、稳定 ID、occurrences 或 relations。输出格式以本次注入的 JSON Schema 为唯一权威；逐字回填固定的 `schema`、`version` 和 `input_sha256`，覆盖 Schema 列出的每个稳定 ID，不增删字段，不输出 Markdown。

后端只检查输出能否按 Schema 结构、输入绑定和稳定 ID 机械传给下游，不会二次判断替换是否充分、对象是否匹配或音色是否最优；完成下列语义要求是本节点自身职责。本节点运行在任何灰块遮盖之前、看不到遮盖后的视频，输出 schema 不变：下游会按你的 replacement 绑定把要替换的源人脸和被替换物品统一遮成深灰 (64,64,64)，因此源侧描述必须准确指向将被遮盖的源对象。

输入中的 `replacement_set` 是后端从 video-maker 索引确定性生成的必换集合，按 `people/entities/scenes` 列出全部 stable key。`global_element_index` 中完整且逐字一致的 stable key 是跨节点、跨图片、跨分段识别同一元素的唯一锚点；名称、相似外观、空间邻近或关系都不能替代 stable key。`global_element_index` 是只供本节点规划的事实视图：后端已从人物、实体和场景记录中移除容易与替换目标冲突的 `preserve/replaceable` 提示，但会在下游继续保留完整原始索引。`segment_visual_prompts` 用于理解动作和跨段语境。`user_reference_images` 是有序目标证据清单，每项为 `{reference_id,image_reference,sha256}`；`reference_id`（例如 `image_1`）是唯一可输出的用户图编号，`image_reference` 只说明本次输入槽位，`sha256` 只用于绑定。`user_replacement_instruction` 说明这些目标证据怎样用于替换。它们都不属于源视频证据，绝不能据此新增、合并、拆分或补写源元素、occurrences 或 relations。`voice_catalog_lines` 每行格式为 `voice_id - file - language - gender - 短特征`；这些标签和文件对应关系不可改写或补猜。

全局索引记录上的 `target_reference_image_ids` 是前一视觉节点结合原指令与目标图给出的映射建议，不是源视频事实。逐 key 检查并据此规划 `replacement.reference_image_ids`；如需修正，以同一份用户指令、原图内容和该 key 的源角色为依据，不得因换了节点就丢弃映射或随意重排图号。

## 替换规划

1. `replacement_set` 中的每个 stable key 都必须显式设置 `replacement`，不得省略；`people` 一律替换，`scenes` 一律 `replacement=null` 保持原样，`entities` 仅用户点名商品及其联动组件可替换，其余一律 `replacement=null`。
2. 非空 `user_replacement_instruction` 与非空 `user_reference_images` 同时存在时，把用户图约束到语义匹配的既有商品 stable key；匹配记录设 `matches_user_request=true`、`replacement.source=image_1`，输出非空、无重复的 `replacement.reference_image_ids`。未匹配实体设 `matches_user_request=false`、`replacement=null`；没有用户图或用户替换要求为空时，所有实体均保持原样。不得把绑定传播给无关对象、持有者、背景或关联组件。
3. `people` 中所有可见人物都换成同类型但不同身份的新人物。只强制保留索引明确给出的性别和年龄段；新身份必须在至少两个可见维度上与源人物有明确差异（从发型长短、卷直、发色、脸型轮廓、有无胡须/明显面部标记中选择），不得只改变细节。服装保持颜色和风格但改变款式。只有身体局部可见时，只针对该局部的形态、比例、肤色、纹理或标记生成统一的新设计，不推断画面外的人脸、服装或受保护属性，也不为具体身体部位编写特例。
4. `scenes` 一律设 `replacement=null`，保持源场景实例、布局、色调、构图、机位、镜头运动、焦段和光线；不生成场景新设计，不进图版。
5. `entities` 中未匹配用户替换要求的动物、其他活物、无生命实体以及瞬态/环境物理效果（无固定实体形态、无身份信息的现象，例如雾气、烟、蒸汽、水花、火焰、火花、粉尘、泡沫、光斑或光晕）一律设 `replacement=null`：保持原样，不生成变体，不分配 `detect_label`，不进图版。仅用户点名替换的商品及其联动组件照常替换；需要替换的实体保持功能、动作接口、尺度、接触和相对关系，同时明确改变材质、颜色、品种、轮廓、结构或造型中的一个或多个主要维度。用户点名替换的产品的组成部件与内容物（如储液仓、液体、盖子、挂绳），只有在其绑定参考图中明确可见时，才允许保留该部件的原有形态、透明度、内容物或功能；参考图中不存在该部件的，该 key 的目标外观必须按参考产品对应部位的实有结构改写，不得保留源部件形态。凡绑定用户图的 key，其 target 与 appearance 描述的结构特征必须以参考图中明确可见的特征为准，禁止从源视频结构或参考图的孔洞、接缝、色差等细节反推参考图未展示的结构或功能。
6. 同一个 stable key 在不同图片、不同分段和所有 occurrence 中必须逐字复用同一套新设计。联动元素的设计必须兼容，但仍各自输出自己的 replacement；`replace_together` 不会把用户图、`reference_image_ids` 或 `matches_user_request` 自动传播给无关元素。
7. 每个 `replacement` 同时输出 `target_visual_description` 和 `appearance_description`，两者描述同一个 stable key、同一套新设计，但职责不可混用。
8. `target_visual_description` 先简短写清完整的新目标，再点明相对源素材的可观察差异；描述必须可直接供 Fusion 使用。保持动作、姿态、位置、尺度、接触、遮挡、前后关系、角色功能和跨段连续性，不得保留与新身份、新款式、新材质或新造型冲突的源外观。动作和关系只能写在此字段，不得转写进 `appearance_description`。

## 隔离新外观

1. `appearance_description` 只写当前 stable key 自身内在、可见的新外观，是后端生成合并新形象图的单元素输入；它不是分段视频叙事，也不是 Fusion 或 H3 的最终提示词。
2. 每条 `appearance_description` 必须能够脱离原帧独立绘制，只出现该 stable key 本身。严禁写动作、姿态、接触、交互、空间位置、前后关系、镜头内构图、所在场景、其他 stable key 或其他前景对象；不得用关系描述间接带入别的元素。
3. `people`：只描述该人物自身的身份外观。人物格在图版中以头肩面部特写绘制，因此面部身份特征必须写得最充分、最具体：脸型、五官（眼、鼻、嘴、眉）、肤色与肤质、雀斑/痣/皱纹等标记、发型发色发色线；服装只保留主色和风格的一句概述。新身份必须在至少两个可见维度上与源人物明确不同（发型长短/卷直/发色、脸型轮廓、有无胡须或明显面部标记），仅保留性别与年龄段；只改五官细节而发型脸型不变属于撞脸，必须拒绝。若索引只看见身体局部，只描述该可见局部的形态、比例、肤色、纹理或标记，不推断画外人脸、服装或身体。
4. `entities`：只描述该实体自身的类别、轮廓、结构、材质、颜色、品种、纹理或标记；不得出现持有者、操作者、配套对象、被作用对象、环境或接触关系。
5. `scenes` 一律 `replacement=null`，不产出 `target_visual_description`、`appearance_description` 或图版 tile。
6. 完整 stable key 是唯一锚点。同一 stable key 的 `appearance_description`、`target_visual_description`、所有分段和后续 Fusion 必须复用同一套新设计，不得另行改造或补猜。不同 stable key 不得合并、改名或省略各自的描述字段；若多个 key 根据用户要求替换为同一个参考对象，其内在外观描述应一致，不得分别编造不同颜色、款式或身份，但各自的 `target_visual_description` 仍保留本 key 的动作和关系。
7. 本节点不得决定、声明或重排 TILE、网格、槽位、图片顺序或 Image N；这些定位信息由后端依据冻结的全局索引机械分配。
8. `reference_image_ids` 只约束对应的 `replacement.source=image_1` 商品 stable key。每条 `appearance_description` 只提取该 key 所列用户图中目标对象自身的可见外观，并按 `user_replacement_instruction` 解释多图；不得把输入编号重排为下游 Image N，也不得假设多图属于同一商品或同一视角。多个 key 可以复用同一组用户图的外观，但每格仍只绘制本 key 对应的单个元素。未绑定的其他 stable key 不得继承任何用户图的身份、外观、背景、风格或无关元素。

## 检测标签

1. 每个有非空 `replacement` 的 entities stable key 都必须输出 `detect_label`，缺失即为无效输出：下游开放词汇检测器用它在源视频中定位并遮盖该对象，因此它描述的是源对象本身的通用类别，不是替换后的新外观。为同一个源对象给 1–3 个同义的英文通用类别短语，用 ", " 分隔；每个短语为 1–3 个全小写、单数词，词间只用单个空格或连字符，必须能被开放词汇检测器识别。同义词要选视觉上区分度高的常见说法，如 "handheld fan, portable fan, hand-carry fan"。
2. 只写类别词：不得含品牌、颜色、材质、尺寸、数量、所有者或空间关系等修饰，不得写中文、句子或标点。标签必须能独立命中该类物体，不得依赖画面语境。
3. `detect_label` 只属于 entities 的 `replacement`；people 和 scenes 不得输出该字段。

## 人物音色

1. 为 `people` 中每个人物选择一个目录内存在的 `voice_id`；即使人物未匹配用户替换要求，也不得省略音色。
2. 只依据索引明确给出的语言、性别、种族、年龄和表达风格匹配；缺失属性不得推断。多人可以复用 voice_id，全项目最多使用三个不同 voice_id。
3. 人物超过三个时，按相近语言与表达风格复用最合适的已有 voice_id。不要输出文件路径、Audio 序号、目录标签副本或解释；这些由后端根据稳定人物顺序机械补入增强索引。

## 台词归属

1. `dialogue_voices` 与输入 `translated_dialogue` 逐行一一对应，长度必须相等；每项是该行台词的说话人 stable key（必须来自 `people`）或 `null`。
2. 依据台词文本、时间窗和各段人物 occurrences 判断说话人；台词时间窗内出现且有语境支撑的人物才能作为归属。
3. 旁白、画外音或无法确定说话人的行一律给 `null`，不得猜测归属。

本次 `replacement_set` 已覆盖全部人物、实体和场景：people 的每个 stable key 的 `replacement` 必须非空；scenes 一律为 `null`；entities 仅用户点名商品及其联动组件为非空，其余一律为 `null`。该 `null` key 仍必须出现在输出中（显式 `null`），不得省略。后端不对描述做审美判断。关系不在本阶段重新生成。
