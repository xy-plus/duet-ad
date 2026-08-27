---
name: postprocess
type: behavior
status: done
owner: human
updated: 2026-08-28
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
| 用户确认优化图片 | 每段恰好 9 张优化图按冻结顺序进入统一 segment plan；确认后才允许开始生成，缺帧或漂移不回退原图 |
| H3 计划包含画内台词 | 只消费 `video-maker` 主分析已经冻结的画内人物与时间证据；不存在时明确不可用，不再次调用 Skill 补证 |
| dialogue 为 none 或全部 off-screen | 不探测视频、不抽样，也不创建任何额外 Skill 输入 |
| 任一分段失败 | 保留成功帧；该段显示“重试本段”，请求携带 `confirm/expected_revision`，点击后立即禁用以防双击 |
| 旧会话 | 409 `read_only` |

## 图片优化计划

- 当前 `image-postprocess` Skill 已由用户预览并验收，自 2026-08-28 起冻结；不得继续修改 Skill 文件、提示词、规则、测试或职责。后续只允许把现有输出接入完整视频链路。
- 当前 `image-postprocess` Skill 只有 `phase=plan`：读取冻结关键帧，输出 v4 结构化人物与真实新场景双替换计划。它不做素材准入、供应商调用、发布、H3 门禁或图片验收；历史 receipt 由后端只读兼容，Skill 不迁移或重写。
- 所有项目统一为 `segments[N>=1]`，每段固定读取 9 张原始关键帧并输出 9 张优化关键帧。`N=1` 不选择独立 short 逻辑，segment 索引也不得用于切换业务实现。
- 有效冻结输入固定输出 `eligible=true`、`reason=null`。人物、场景、关系或不可见区域的不确定性不是素材准入条件；技术输入、schema、哈希和冻结 transition skeleton 的错误由调用 Skill 前的后端处理。历史 v2/old `eligible=false` 响应只交给 runtime protocol correction，不是图片计划的内容失败。
- 计划始终要求人物与真实新场景同时替换，不能成功 no-op。新人物与新场景跨帧/跨段复用各自冻结目标；场景必须真实改变语义、形状与空间结构、纵深、布局及局部材质或固有色，人物与场景局部固有色必须明显不同。
- `scene_plans[].continuity_graph` 是同一目标场景跨段的唯一组件 registry，独立于每帧 source ENTITY；图内不得引用逐帧 source ENTITY_ID。`COMPONENT_01` 起连续编号，topology 只用 `supports/contacts/separate_from`，view relations 只用 `in_front_of/occludes`，visibility 只用 `full/partial/edge_fragment/occluded/out_of_view`。views 覆盖每个冻结帧并逐字采用后端计算的 transition_skeleton；每段 scene 保留 `layout_reference_frame_index`。same_camera 的 `occluded`→`out_of_view` 只记录当前 source 可见性并进入 source-preserve/no-invention，不得作为内容 schema 拒绝。每帧 prompt 只消费所属 SCENE 的同一 target graph core 与该帧唯一 view，不能由自由文本删除人物、场景、光色、关系或连续性要求。
- 逐帧保留可见身体部位数量、面部拓扑、姿态骨架、尺度、服装边界、接触点、遮挡前后顺序、画外裁切，以及当前可见非人物实体、独立物理面和画边碎片；只以当前源帧为事实，不同边界、法向、深度层或支撑链的物理面不得合并；不从相邻帧、reference 或编辑结果补全。
- `contact_points`、`contacts`、`supports` 与 `occludes` 只记录双方边界和层次同帧直接可见的确定事实。不可见、遮挡、画外裁切或无法唯一判定的区域必须使用已有 `out_of_frame_crop`、`occlusion_order`、实体 visibility 和 `protected_relations` 的 `source-preserve/no-invention` 指令：只保留可见构图和边界，未见部分不补造、不猜测，也不输出候选关系；这些情况不会把计划改成不合格。
- v4 的 exact `frame_constraints` 仍是 `frame_index/visible_body_parts/pose_skeleton/contact_points/occlusion_order/out_of_frame_crop/non_person_entity_ledger/dominant_palette_contract`，逐帧一一覆盖。`partial/cropped` 不得写成 `absent/fully-in-frame`；ledger 的实体/关系 ID、端点、排序和枚举遵循既有 schema。
- 固定保持画幅、裁切、机位、镜头、透视、构图、焦点、景深及全局光源方向、软硬、强度、曝光、白平衡、色温、整体色调、全局对比与 tone curve。每帧 `area_weighted_warm_cool_family` 与 `saturation_style` 的整帧面积加权主色盘合同由后端从冻结 source 像素计算并覆盖，模型不得自报或决定精确 Lab 合同；局部固有色可变，但大面积新区域不得翻转整帧冷暖感知。

## 生成与后续门

- 图片优化主链只负责技术 freeze、计划/提示词编译、图片生成和持久化展示。图片生成后由用户在 Web 查看原图/优化图并明确确认；不得再增加模型评测 Skill 或隐藏已经生成的图片。
- 用户确认后必须调用 `video-prompt-fusion`：输入有序新关键帧、旧视频提示词、图片优化提示词和音频内容；新人物、新场景、新对象、服装、材质和空间结构以优化图为视觉事实，动作、镜头、构图、节奏和 segment 时间轴沿用旧视频提示词。融合输出及四类输入 SHA 一起冻结，旧视觉 prompt 不得直接进入 Context 或 H3。
- H3 只接受完整、匹配冻结输入且经用户确认的后处理 authority；缺帧、状态异常或确认漂移时 fail closed，不回退原图。
- 全链只允许 `video-maker`、`image-postprocess` 与 `video-prompt-fusion` 三个 Skill。音频呈现是后端对冻结台词、Web 画内/画外选择和唯一 voice reference 的确定性投影，不存在 Binding、Audio、Speaker 或 speaker-visibility Skill/phase。
- `dialogue_delivery=on_screen` 只消费 `video-maker` 主阶段已经产出的画内人物/时间证据；证据缺失时付费前明确不可用。`off_screen` 不需要嘴型证据；`none` 不发送台词或声音 reference。
- 请求顶层严格为 `confirm/options`；running 时不能重复提交，done 后改变选项返回结构化 409。页面只展示用户可理解的提示词、能力和分段进度，不展示内部模型、模板、供应商响应或堆栈。
