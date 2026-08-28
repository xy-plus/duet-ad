---
name: postprocess
type: behavior
status: done
owner: human
updated: 2026-08-28
tdd: N/A
links: [conversation-task, result-display]
---

# 当前关键帧后处理

## 规则

| 当 | 则 |
| --- | --- |
| schema v2 会话已 `done` | 按 `postprocess_capabilities` 分别显示“移除文字/字幕”“移除常见 Logo/图标”“进行图片优化”；不可用项不展示 |
| 首次询问“是否优化素材？” | “否”初始高亮且仅在页面内记忆，不写后端；“是”打开弹窗，去字幕/去品牌默认选中，图片优化默认不选中 |
| 至少选一项并严格确认 | `POST /postprocess` 返回 running，逐帧并行编辑并以 2 秒轮询展示进度 |
| 双目标图片计划已冻结 | 页面可查看后端确定性编译的提示词，但不能用自由文本改写；PATCH 返回 `image_optimization_prompt_compiled` |
| 全部帧完成 | `postprocess.status=done`，发布 `postprocessed/`；技术验收 A 后同一 operation 自动继续 Fusion/H3，不等待刷新或第二次确认 |
| score/diagnostics 或外部 A/B 质量较低 | 只记录给测试与下一轮 Skill 迭代；不回收图片、不阻断生产、不重试或选择备用 prompt/workflow |
| 任一分段失败 | 保留成功帧；该段显示“重试本段”，请求携带 `confirm/expected_revision`，点击后立即禁用以防双击 |
| 旧会话 | 409 `read_only` |

## 图片优化计划

- 当前 `image-postprocess` 只有 `phase=plan`：读取冻结关键帧并输出人物/场景替换的视觉语义；后端补齐 v4 结构字段、确定性编译逐帧 Seedream prompt。Skill 不负责流程控制、质量验收、发布或 fallback。
- 所有项目统一为 `segments[N>=1]`，每段读取和输出 exact 9 个有序关键帧槽位。短 scene 可以在同一 scene 内重复最近合法源帧并绑定 provenance；`N=1` 不选择独立 short 逻辑。
- semantic compiler 的 `score/issues/ignored_mechanical_fields` 仅写日志、测试断言和迭代分析。当前 v4 不运行 plan audit/verify phase，不持久化 `_image_verification`，也不因语义质量 fail/unknown 阻断 A→B。
- schema、exact-9、索引、普通文件路径和 SHA 属于技术完整性合同；这些字段无效与质量分数低是两类不同问题，不能用备用图片、旧计划或另一 workflow 混淆处理。
- 计划始终要求人物与真实新场景同时替换，不能成功 no-op。新人物与新场景跨帧/跨段复用各自冻结目标；场景必须真实改变语义、形状与空间结构、纵深、布局及局部材质或固有色，不得仅调色、换纹理或给原结构换皮；人物与场景局部固有色必须明显不同。
- `scene_plans[].continuity_graph` 是同一目标场景跨段的唯一组件 registry，独立于每帧 source ENTITY；图内不得引用逐帧 source ENTITY_ID。`COMPONENT_01` 起连续编号，topology 只用 `supports/contacts/separate_from`，view relations 只用 `in_front_of/occludes`，visibility 只用 `full/partial/edge_fragment/occluded/out_of_view`。views 覆盖每个冻结帧并逐字采用后端计算的 transition_skeleton；每段 scene 保留 `layout_reference_frame_index`。same_camera 的 `occluded`→`out_of_view` 只记录当前 source 可见性并进入 source-preserve/no-invention，不得作为内容 schema 拒绝。每帧 prompt 只消费所属 SCENE 的同一 target graph core 与该帧唯一 view，不能由自由文本删除人物、场景、光色、关系或连续性要求。
- 逐帧保留可见身体部位数量、面部拓扑、姿态骨架、尺度、服装边界、接触点、遮挡前后顺序、画外裁切，以及当前可见非人物实体、独立物理面和画边碎片；只以当前源帧为事实，不同边界、法向、深度层或支撑链的物理面不得合并；不从相邻帧、reference 或编辑结果补全。
- `contact_points`、`contacts`、`supports` 与 `occludes` 只记录双方边界和层次同帧直接可见的确定事实。不可见、遮挡、画外裁切或无法唯一判定的区域必须使用已有 `out_of_frame_crop`、`occlusion_order`、实体 visibility 和 `protected_relations` 的 `source-preserve/no-invention` 指令：只保留可见构图和边界，未见部分不补造、不猜测，也不输出候选关系；这些情况不会把计划改成不合格。
- v4 的 exact `frame_constraints` 仍是 `frame_index/visible_body_parts/pose_skeleton/contact_points/occlusion_order/out_of_frame_crop/non_person_entity_ledger/dominant_palette_contract`，逐帧一一覆盖。`partial/cropped` 不得写成 `absent/fully-in-frame`；ledger 的实体/关系 ID、端点、排序和枚举遵循既有 schema。
- 固定保持画幅、裁切、机位、镜头、透视、构图、焦点、景深及全局光源方向、软硬、强度、曝光、白平衡、色温、整体色调、全局对比与 tone curve。每帧 `area_weighted_warm_cool_family` 与 `saturation_style` 的整帧面积加权主色盘合同由后端从冻结 source 像素计算并覆盖，模型不得自报或决定精确 Lab 合同；局部固有色可变，但大面积新区域不得翻转整帧冷暖感知。
