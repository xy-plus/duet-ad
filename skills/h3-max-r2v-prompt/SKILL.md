---
name: h3-max-r2v-prompt
description: 根据原片段描述、Object Replace 增强后的全局对象索引、项目级新形象图版映射和翻译台词，为 MiniMax H3 Max Reference-to-Video 编写逐段最终提示词。
---

# H3 Max R2V Prompt

只生成可直接提交给视频模型的一段提示词，不解释过程。传输格式以本次调用注入的 JSON Schema 为唯一权威：逐字回填固定的 `schema`、`version`、`input_sha256` 和 `segment_index`，把完整提示词放入 `prompt`；不得增删字段，不输出 Markdown。

冻结的 `global_element_index` 是本项目唯一对象身份、替换与人物音色锚点。不得重新识别、重命名、合并或拆分 stable key，不得重新决定替换目标，不得要求读取用户替换说明，也不得根据常识新增替换。逐项读取当前 segment 的 occurrences、replacement、voice、preserve 和 relations。

`replacement_sheet` 是按同一冻结索引生成的项目级单张新形象图版；`h3_reference_images` 给出 H3 图片的有序角色。后端生成的 `stable key -> TILE_XX -> 实际 Image N` 映射是唯一图片引用权威：图版固定为 `Image 1`，用户图按其冻结 `reference_id` 的顺序依次为后续 `Image N`。对 `replacement.source=image_1` 的 stable key，权威块会列出它唯一可用的用户图编号；这些原图是该 key 的外观权威，图版 tile 仅用于定位该 key 和保持跨段映射。不同用户图既不预设为同一商品、同一人物或不同视角，也不得扩散给未绑定 stable key。对 `generated_variant` 的 key，仍只以自身图版 tile 为外观权威，必须忽略全部用户图。多个 stable key 可按冻结计划共享同一组用户图，却仍各自保留身份、occurrences、动作和位置关系，不得合并。同一项目的每个 segment 必须沿用同一张图版和同一套映射，不得重新编号或自行推断图片顺序。

若 `replacement_sheet.h3_image_reference=null`，本次没有图版：`h3_reference_images` 中原图从实际 Image 1 开始排序，`user_reference_images` 明确每个 reference_id 对应的真实槽位。不得虚构空白图版或源对象来凑编号。即使没有任何已选中的用户绑定，已上传原图仍是可见输入；已知 generated_variant key 不因此继承它们的外观。

`authoritative_constraints` 是后端从增强索引和图版 spec 机械编译的本段替换、`stable key -> TILE_XX -> Image N`、关系和人物音色/台词轨权威块，后端会在你的 `prompt` 尾部原样附加。你必须以它规划主体提示；主体中不得出现稳定 ID、TILE、Image N、Audio N 或 BEGIN/END 块，不得另写对象、图片或音色映射，也不得输出与它冲突的要求。后端只校验输出结构和输入绑定，不会二次判断你是否完成这些语义要求；完整表达替换、音色和连续性是本节点自身职责。

后端会在你的 `prompt` 头部机械注入编辑框架，以及按实际遮盖情况注入遮盖条款和 FACE REPLACEMENT 强制块（被遮盖的内容必须重建而不得恢复）。哪些区域实际被遮、块是什么颜色由后端决定，你写主体时并不知道实际遮盖状态。主体不得自行断言灰块的存在、颜色或遮盖状态，不得与头部块冲突，不得暗示源人脸或源物品仍然可见、需要保留或被还原；主体需要指代被遮盖区域时，用“被遮盖的源区域”这类不承诺颜色或存在性的中性表述，或干脆不指代、交给头部条款。主体也不得描述任何被替换 stable key 的源外观或源类别特征（颜色、材质、品牌、形状等源侧描述词一律禁止）：这等于在文本里泄漏灰块遮住的源信息，遮盖随之失效。被替换对象只允许用权威块 REPLACE 行给出的 target 新外观来描述；`replacement=null` 的 KEEP 对象保持 Video 1 原样，不得列入替换或用 raw `preserve` 恢复任何 REPLACE 项的源身份或外观；raw `preserve` 只直接用于 KEEP 项。

这是一次对 Video 1 的编辑任务，不是重新生成或重述该视频。`visual_prompt`、当前 segment 的 occurrences、relations 和其他 fusion 输入仅用于理解 Video 1 中何处发生替换、使新外观在该处合理；绝不把其中的动作、镜头、机位、构图、位置、运动方向、速度、节奏、转场、因果或剧情转写进输出主体，也不得新增、重新编排或补充叙事。除权威块指定替换及后端指定的遮盖区域重建外，Video 1 的每一帧内容均由下游模型忠实照搬：所有镜头、相机运动、动作、时序、转场和未替换元素必须不变。

Video 1 中未被遮盖的瞬态/环境物理效果（无固定实体形态或身份信息的雾气、烟、蒸汽、水花、火焰、火花、粉尘、泡沫、光斑或光晕）必须保持源视频中的运动、出现时机、强度和轨迹；只允许随新场景调整色调或光照，不得重新编排其形态或出现时机。

主体只写以下内容，使用简洁的自然语言，不写 stable key、TILE、Image N、Audio N 或 BEGIN/END 块：

1. 对当前片段每个 `replacement` 非 null 的 stable key，只描述权威块 REPLACE 行指定的新外观。逐项完整覆盖，不遗漏，也不描述其源外观、源类别或 Video 1 中的行为、位置和剧情。源对象、目标对象及其图版 tile 的逐字映射由输出尾部的权威块唯一确定。
2. 按权威块逐项编译替换绑定：不得从整张图版任选相似对象，不得跨 tile 或跨用户图借用外观。对 `replacement.source="image_1"`，该 key 指定的原始用户图是外观权威，同 key 的图版 tile 仅定位对象和保持跨段映射；不得把原图套给其他对象。对 `generated_variant`，该 key 的图版 tile 是唯一外观权威，忽略全部原始用户图。人物 tile 是头肩面部特写：它定位该人物；当其绑定用户图时，脸型、五官、肤色和发型以该 key 指定用户图为准，否则以 tile 为准。人物的服装与全身外观以权威块的文字描述为准。
3. 仅在确实需要提及时，要求把被遮盖的源区域自然重建为相应的新外观；不要断言遮盖状态，也不要为重建补写画面布局、动作或叙事。每个绑定用户图的替换实体，其形状、体积（含尺寸和比例）及功能只以该参考图为准；手指握持、遮挡和接触位置必须由视频模型按新物品几何自然重建，绝不得为了贴合源物品的旧几何而变形、缩放或改变新物品。`replace_together=true` 只保持权威块指定的交互事件及其时序，不得藉此继承源物品的接口、形态、尺度或功能，或重述关系和动作。
4. `replacement=null` 的实体不是替换项，必须保持 Video 1 原样。原片段在下游模型输入中称为 Video 1；只使用后端已经给出的实际图片和音频映射，不得假定 `Image 1` 永远是同一种角色。
5. 不改写台词，不输出台词文本或新的对口型指令。台词文本、人物音色与 Audio 1 对口型要求由后端在你的输出之后机械附加；不得要求任何角色说出台词轨之外的语音。
