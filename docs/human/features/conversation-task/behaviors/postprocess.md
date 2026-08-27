---
name: postprocess
type: behavior
status: done
owner: human
updated: 2026-08-27
tdd: N/A
links: [conversation-task, result-display]
---

# 可选关键帧后处理

## 规则

| 当 | 则 |
| --- | --- |
| schema v2 会话已 `done` | 按 `postprocess_capabilities` 分别显示“移除文字/字幕”“移除常见 Logo/图标”“进行图片优化”；不可用项不展示 |
| 首次询问“是否优化素材？” | “否”初始高亮且仅在页面内记忆，不写后端；“是”打开弹窗，去字幕/去品牌默认选中，图片优化默认不选中 |
| 至少选一项并严格确认 | `POST /postprocess` 返回 running，逐帧并行编辑并以 2 秒轮询展示进度 |
| 双目标图片计划已冻结 | 页面可查看后端确定性编译的提示词，但不能用自由文本改写；PATCH 返回 `image_optimization_prompt_compiled` |
| 全部帧完成 | `postprocess.status=done`，展示 `postprocessed/` 对比图；外部评测结果不回收或隐藏这些已持久化图片 |
| 任一分段失败 | 保留成功帧；该段显示“重试本段”，请求携带 `confirm/expected_revision`，点击后立即禁用以防双击 |
| 旧会话 | 409 `read_only` |

## 图片优化计划

- 当前 `image-postprocess` 的生成效果、目标视觉和 A→B 提示词已冻结；本节只描述它从原始关键帧 A 到优化关键帧 B 的计划职责。Skill 只有 `phase=plan`：读取冻结关键帧，输出 v4 结构化人物与真实新场景双替换计划。
- 所有项目统一为 `segments[N>=1]`，每段固定读取 9 张原始关键帧并输出 9 张优化关键帧。`N=1` 不选择独立 short 逻辑，segment 索引也不得用于切换业务实现。
- 计划始终要求人物与真实新场景同时替换，不能成功 no-op。新人物与新场景跨帧/跨段复用各自冻结目标；场景必须真实改变语义、形状与空间结构、纵深、布局及局部材质或固有色，不得仅调色、换纹理或给原结构换皮；人物与场景局部固有色必须明显不同。
- `scene_plans[].continuity_graph` 是同一目标场景跨段的唯一组件 registry，独立于每帧 source ENTITY；图内不得引用逐帧 source ENTITY_ID。`COMPONENT_01` 起连续编号，topology 只用 `supports/contacts/separate_from`，view relations 只用 `in_front_of/occludes`，visibility 只用 `full/partial/edge_fragment/occluded/out_of_view`。views 覆盖每个冻结帧并逐字采用 `transition_skeleton`；每段 scene 保留 `layout_reference_frame_index`。`partial` 或 `occluded` 只用于目标场景组件间已知的遮挡；same_camera 的 `occluded`→`out_of_view` 只按当前源帧可见性保留。人物遮挡、未知遮挡和画外裁切改用 `edge_fragment` 或当前帧 source-preserve/no-invention。
- 逐帧保留可见身体部位数量、面部拓扑、姿态骨架、尺度、服装边界、接触点、遮挡前后顺序、画外裁切，以及当前可见非人物实体、独立物理面和画边碎片；只以当前源帧为事实，不同边界、法向、深度层或支撑链的物理面不得合并；不从相邻帧、reference 或编辑结果补全。
- `contact_points`、`contacts`、`supports` 与 `occludes` 只记录双方边界和层次同帧直接可见的确定事实。不可见、遮挡或画外裁切区域使用已有 `out_of_frame_crop`、`occlusion_order`、实体 visibility 和 `protected_relations` 的 `source-preserve/no-invention` 指令：只保留可见构图和边界，未见部分不补造、不猜测，也不输出候选关系。
- v4 的 exact `frame_constraints` 仍是 `frame_index/visible_body_parts/pose_skeleton/contact_points/occlusion_order/out_of_frame_crop/non_person_entity_ledger/dominant_palette_contract`，逐帧一一覆盖。`partial/cropped` 不得写成 `absent/fully-in-frame`；ledger 的实体/关系 ID、端点、排序和枚举遵循既有 schema。
- 固定保持画幅、裁切、机位、镜头、透视、构图、焦点、景深及全局光源方向、软硬、强度、曝光、白平衡、色温、整体色调、全局对比与 tone curve。`dominant_palette_contract` 的 exact 字段是 `area_weighted_warm_cool_family/saturation_style`；局部固有色可变，但大面积新区域不得翻转整帧冷暖感知。
