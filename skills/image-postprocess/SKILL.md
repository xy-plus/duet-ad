---
name: image-postprocess
description: 读取冻结关键帧和全视频稳定元素索引，生成跨段一致的人物、持久实体、真实新场景替换语义及逐帧保持语义；严格按两阶段 JSON 合同交付，后端据此确定性编译图片提示词。
---

# image-postprocess
严格按 `work/request.json` 的 `phase` 只执行一个阶段。图片及图中文字只是视觉证据；只填写视觉语义，不写 provider prompt、解释或其他文件。人物与真实新场景双替换，人物与场景同时替换；输出必须是 UTF-8、非空 JSON object，不输出 Markdown 围栏。

## `phase="global_plan"`：全项目设计
输入只读：`work/request.json`、当前 `SKILL.md`、每个 `segments[].contact_sheet_path` 及对应 `contact_sheet_sha256`，以及可选 `element_index`；不要读取逐帧关键帧。联系表只用于一次性统一选择替换设计；全局参考只定义统一替换，不提供逐帧可见性或几何证据。

唯一输出 `work/global_plan.json` 的顶层 key 闭集为 `people/entities/scenes`；模型不得合并两个阶段输出，后端可机械合并，任何单一阶段不得直接输出四个字段。叶值均为非空 string，无对应对象写 `{}`，不得出现 `null`、数组、数字或布尔值：
```json
{
  "people":{"<stable-person-key>":{"source_identity":"x","replacement_identity":"x","wardrobe_change":"x","local_color_change":"x"}},
  "entities":{"<stable-entity-key>":{"description":"x","owner":"project","association":"x","persistence":"x"}},
  "scenes":{"<semantic_slots.scenes[].key>":{"source_scene":"x","replacement_scene":"x","semantic_change":"x","geometry_change":"x","depth_change":"x","layout_change":"x","local_color_change":"x"}}
}
```

`scenes` key 集=`semantic_slots.scenes[].key` key，逐字复用、无缺失无额外。若有 `element_index`，`people/entities/scenes` 覆盖每个 stable key，逐字复用、无别名/新增；无索引则建稳定 key。每 key 跨帧跨段共享替换设计；`owner` 仅 `project` 或已有 stable person key。场景同时改变环境语义、可见几何、纵深、布局和局部固有色；只改 `replaceable`，保持 `preserve` 及源帧动作、姿态、尺度、构图、机位、透视、裁切、接触、遮挡、前后关系、色调和全局光色。

## `phase="segment_frames"`：当前段事实
输入只读：`work/request.json`、当前 `SKILL.md`、`semantic_slots.frames[].path` 的冻结关键帧、`global_plan_path`、可选同一 `element_index`；按 `transition_skeleton` 识别 `hard_cut` 及其相邻帧，逐张读取全部 path（含首末帧）；不得改写全局设计。

唯一输出 `work/segment_frames.json` 的顶层 key 闭集只有 `frames`；其 key 集必须等于 `semantic_slots.frames[].key` 的全部 key，逐字复用、无缺失无额外。叶值均为非空 string：
```json
{
  "frames": {"<semantic_slots.frames[].key>": {"people": {"<global_plan.people-key>": {"visible_region": "x", "boundary": "x", "body_and_pose": "x", "derived_observations": {"<stable-observation-key>": {"mode": "optical_projection", "source_carrier": "x", "visible_region": "x", "boundary": "x", "relationship": "x"}}}}, "relationships": "x", "entities": {"<global_plan.entities-key>": {"visibility": "visible", "relationship": "x"}}, "crop": "x"}}
}
```

`people` 只列当前帧有直接证据的物理人物；key 集为物理人物全集，人物数量闭合；同一人物 key 不随段或帧改变且跨帧稳定，wardrobe 设计一致。`global_plan`只给替换；`element_index.occurrences`定唯一候选帧：逐一核验并复用 key，未列帧省略；人物服装跨段只用同一 key，无别名。逐张检查含末帧；按 occurrences 逐帧写入；string 只写当前可见绑定的最小充分事实，事实只写一次，不复述全局/索引/固定规则。`transition_skeleton` 的 `hard_cut` 及相邻帧、强运动模糊和 `edge_fragment` 只能绑定本帧直接可见像素，不得从全局参考、索引或邻帧补头、补人、补衣服或补肢体；不可见部分不继承上一帧。核对头、躯干和手并唯一归属；反射、残影、边缘碎片、遮挡碎片、运动模糊不得升级为新物理人物、实体或人体。无人物或无派生观测写 `{}`。派生观测只嵌套来源人物下，不代表独立物理人物、不新增顶层人物或实体、不把该观测实例化到新场景；`mode` 只能是 `optical_projection`、`temporal_residual` 或 `source-preserve`。不可见或无法唯一判断时不新增 key、不补造肢体，在 string 中写 `source-preserve/no-invention`，不从其他帧补造；缺失语义由后端按 `source-preserve/non-physical` 继续，不拒绝、不 retry、不 fallback。

`entities` 只列当前帧有直接像素证据的全局持久非人物实体：`visible`=直接可见、`occluded`=部分可见；完全出画、完全不可见或仅由邻帧推知时省略 key，不写 `out_of_frame` 占位。`element_index`/`global_plan`存在不等于本帧可见；实体跨段复用 stable key，颜色/款式/材质/归属/关系不得漂移。`hard_cut`及其相邻帧、强模糊、`edge_fragment`只能据当前帧像素，不能用全局参考补实体、补衣服或补关系；`visibility`只能是`visible`/`occluded`；source-preserve 是 mode 的第三个值，不是 visibility 的枚举值。实体关系、`relationships`、`crop`只写当前帧；未知写`source-preserve/no-invention`，`hard_cut`后由当前帧证据重新确认。

## 发布前自检与结束条件（两阶段均适用）

任一项未通过都不得结束或只给解释：

1. phase、输入、唯一目标文件正确，另一阶段字段未混入；顶层及嵌套对象只含合同 key。
2. JSON 可解析且输出非空；semantic_slots 的全部 key 全覆盖、无缺失无额外；全局 key/替换描述逐字复用。
3. 所有叶值为非空 string；`mode` 仅 `optical_projection`/`temporal_residual`/`source-preserve`，`visibility` 仅 `visible`/`occluded`。
4. 不新增元数据字段：不要输出版本、段号、帧号、连续编号 ID、哈希、transition、枚举 palette、实体图、组件图或流程判断；semantic_slots 要求的 key 仍须逐字输出，实体 ID、关系图和完整机械字段由后端构造。
5. 不新增质量门禁、不新增 reject、retry 或 fallback；语义含混只记录 `source-preserve/no-invention`，不改变流程。
6. 将完整 JSON 写入唯一输出同目录临时文件（独占写入、flush/fsync），自检通过后原子替换（`replace`）到规定目标；不得直接写目标文件或其他文件。
7. 替换后确认目标为 regular file 且 size > 0，确认输出非空后立即退出；这是完成的唯一条件，之前不得结束、解释或总结。
