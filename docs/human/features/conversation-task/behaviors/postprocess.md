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
| 历史 v1 图片提示词被编辑 | `PATCH /image-optimization-prompt` 仍以 SHA CAS 保存，不迁移或重写旧 continuity receipt |
| 全部帧完成 | `postprocess.status=done`，展示 `postprocessed/` 对比图 |
| 图片优化验收通过且已选输出完整 | 同一 `video-maker` Skill 执行第三阶段 `reconcile_after_image_optimization`，输出 receipt 绑定的统一视觉 IR；未通过协调资格门时不进入 H3 |
| 任一分段失败 | 保留成功帧；该段显示“重试本段”，请求携带 `confirm/expected_revision`，点击后立即禁用以防双击 |
| 失败分段的 `error=submission_unknown` | 明确警告重复操作可能重复计费；用户再次明确确认后可按 `expected_revision` 人工重试本段；不得从 `status/stage` 推断 |
| 旧会话 | 409 `read_only` |
| 旧页面仍提交 `change_bg/face_hold` | 提示刷新页面；不写状态、不产生 MediaKit 费用 |

## 边界

- R11 当前为 `PREPARED_ONLY`：只准备 v4 schema、compile、freeze/receipt、plan_audit 与 verify 合同；不改变 postprocess 运行控制，也不启用 runtime anchor。
- 版本形状严格隔离：历史 v3 scene plan 无 graph、plan_audit 每帧 exact 四项 closure、verify 每帧 exact 九字段；新 v4 才增加 `continuity_graph`、audit 第五项 `scene_continuity_closure`，verify 每帧只增加 `scene_continuity_view`。v3 请求直接回归旧形状，不从 v4 示例补字段。
- 视觉关键帧冻结后，隔离 Codex 以同一个 `image-postprocess` Skill 执行 `phase=plan`：短视频读取逻辑段 `0`，多段项目读取全部分段关键帧。新输出只含结构化 v4 设计；后端确定性编译每个冻结帧的 Seedream 提示词，Skill 不读取或复制 H3 提示词。历史 v3 receipt 保持 exact 只读兼容，更旧 receipt 不迁移。
- 页面将单段/多段提示词收敛为同一个三态工作区：“展开生成提示词 / 展开段台词 / 展开图片优化”。三个按钮等宽并排且窄屏允许文字换行；工作区共用一个文本区域。短视频生成提示词与图片优化可编辑，长视频的逐段生成提示词和段台词只读。
- 图片优化编辑必须同时满足 `postprocess_capabilities.optimize_image=true`；即使异常详情带回提示词，能力为 false 时也只读且不能发 PATCH。短视频逻辑段号为 0，长视频段号必须为正整数，长视频出现 0 或非法段号时 fail closed。
- v4 双目标计划的编译提示词只提供查看和复制，且每个冻结帧各有独立提示词；每帧只消费所属 SCENE 的同一 target graph core 和该帧唯一 view，不能由自由文本删除人物、场景、光色、关系或场景连续性硬约束，供应商调用只能取自身帧的提示词。历史 v1/v2/v3 提示词继续原 CAS 行为，receipt 不迁移、不重写。
- v4 在任何 provider 提交前以同一 `image-postprocess` 执行 `phase=plan_audit`：只读取 canonical plan、冻结 audit receipt 与对应 source 帧，逐帧验 `body_closure/scene_closure/entity_closure/relation_closure/scene_continuity_closure`；最后一项只审计当前 view/transition 对 source 可见证据是否可实现、是否冲突，并核对冻结的 source transition evidence。target component 身份/topology 由 canonical graph 定义，尚未生成的 target 不由 source 证明，source ENTITY_ID 不得冒充 COMPONENT_ID。历史 v3 verdict 保持原四项 exact 形状。审计输入、verdict 和每帧检查均绑定 plan SHA、continuity SHA、audit input SHA、source transition evidence SHA 与每帧 source SHA；任一 fail/unknown 为 `passed=false`，调用方不得提交 provider。该阶段不修图、不补写 plan、不自动 replan；若后续启用 replan，只能在 audit failed 后显式执行一次，并保留原 plan、原 source 和 audit receipt。
- 当前选项为 `remove_subtitle`、`remove_brand`、`optimize_image`；`change_bg/face_hold` 已删除。旧页面请求只得到纯文本刷新提示，不会静默采用或自动重试。
- `remove_subtitle` 映射 `full_screen_text_erase`；`remove_brand` 作为兼容字段映射 `full_screen_icon_erase`，只承诺清理常见 Logo/图标。双选会执行两个独立付费阶段。
- 三个阶段在每段内严格按“文字/字幕 → Logo/图标 → 图片优化”执行。v4 `plan` 以连续性与现实合理性优先于替换幅度；人物与真实新场景仍是不可降级的双目标，任一目标与可见事实冲突时在付费前 `eligible=false`。短视频 `[0]` 使用相同结构，不能成功 no-op。
- 每个稳定主人物和物理场景各有一个冻结目标包；目标包逐段复用，不逐帧重设计，也不从编辑结果递推。每段按 ID 完整枚举全部主人物；不可见者标 `not_observable`，不得依据相邻段或 reference 补造人物或身体部分。
- v4 把目标场景连续性的唯一权威放在 `scene_plans[].continuity_graph`，同一 SCENE 跨 segment 共享同一图，图内不得引用逐帧 source ENTITY_ID。components 使用连续排序的 `COMPONENT_01...`，target_spec 非空；不同物理 component 可以同规格，稳定可见的孤立 component 可以没有 topology。topology 只允许 `supports/contacts/separate_from`，端点闭合，同一无序 pair 最多一项，contacts 与 separate_from 互斥，关系无自指/重复/环。views 一一覆盖该 SCENE 的所有冻结帧；每帧 observations 对全部 component 恰好一次，visibility 只允许 `full/partial/edge_fragment/occluded/out_of_view`，每个 component 至少一帧为 full/partial/edge_fragment。occluded 表示完全遮挡，partial 表示部分遮挡，edge_fragment 表示触边并优先；partial/occluded 必须且只能由当前可见遮挡证据对应 incoming occludes，incoming occludes 的 object 不得为 full，可以为 partial/occluded/edge_fragment。view relations 只允许 `in_front_of/occludes`，out_of_view component 不得作为端点，同一有向 pair 最多一项，occludes subject 必须当前可见，关系无自指/重复/环。same_camera 下任何非 out_of_view component 不得无相机运动突然出画；仅全项目首帧为 start，其余 transition 必须匹配冻结的 source transition/join evidence，缺证据或不一致 fail closed，hard_cut/camera_motion 不能由模型自报，hard_cut 不继承另一场景。
- `hard_cut` 是场景证据边界：切后场景只依据切后证据，不能继承无切后证据的切前设计。逐帧先做全画面人体像素、服装和肢体碎片账本；当前帧全部可观察的人体、面部拓扑、服装边界与裁切碎片形成闭包，任何可见碎片必须写入 `visible_body_parts`，可见人体裁切碎片必须写入 `visible_body_parts`。人物自身服装属于 PERSON target domain；容器式边界仅在可见时作为非人物实体。`contact_points` 只能写当前帧唯一可观察关系；接触双方边界都在当前帧可见；contact_points/contacts 只记录双方边界同帧直接可见的接触，禁止从人体在服装边界消失推断衣内接触；occlusion_order + occludes 冻结可见的覆盖、开口、穿入穿出拓扑，拓扑无法唯一确定且影响替换则 person_replacement_unsafe。再做全画面非人物实体及其边缘碎片账本；不同物理实体不得合并，每个画边碎片都必须登记并归属其物理实体；不同边界、法向、深度层或支撑链的物理面不得合并；概括性总称不能覆盖可独立消失、融合或错位的内部可见子区域；遮挡关系不能替代可见的支撑、接触或分离关系。逐一枚举与人物或目标操作相关的可见非人物实体及其支撑、接触、分离和遮挡链，作为逐帧事实写入既有约束，段级 `protected_relations` 不能替代逐帧事实。逐帧保留可见身体部位数量、姿态骨架、尺度、手脚/道具/绳索/支撑面的接触点、遮挡前后顺序与画外裁切。每一帧只以当前源帧为事实，不从相邻帧、reference 或编辑结果补全。支撑、接触或遮挡只有双方边界与层次都能在当前帧无歧义观察时才可写，禁止候选表述（“或”“可能”“接触或间隙”等）。任何遮挡或裁切使接触双方边界不可同帧观察，或证据不足则 `eligible=false/person_replacement_unsafe`，不得补证。不得补造画外身体或工具，不得删除或新增肢体，不得改变接触图；`continue` 才优先作为连续画面分析。
- 输出前逐字段自校验：eligible plan 的顶层、每段、每帧与嵌套项都恰含既有 schema 字段且按既有顺序；scene 恰含 `scene_id`、`target_region`、`boundary`、`layout_reference_frame_index`，每个 scene plan 恰多含一个 v4 continuity_graph。contacts/separate_from 的 subject_id 小于 object_id，并按 `(subject_id,predicate,object_id)` 升序；eligible 前逐帧完成 source-visible entity/relationship coverage closure，任何未归属可见像素区域或缺关系都输出空计划。人物域闭包失败则 person_replacement_unsafe；物理场景实体身份或必须关系不能闭合则 scene_components_ambiguous；已识别场景与 boundary/五维变化冲突则 scene_structure_replacement_unsafe；任一缺键、多键、反序、漏帧、漏碎片、合并实体或不可观察关系也输出空计划。
- v4 沿用历史 v3 exact `frame_constraints`：按帧号一一覆盖全部冻结帧，字段固定为 `frame_index/visible_body_parts/pose_skeleton/contact_points/occlusion_order/out_of_frame_crop/non_person_entity_ledger/dominant_palette_contract`，五个字段必须相互一致；`partial/cropped` 不得写成 `absent/fully-in-frame`。ledger 固定为 `entities/relations`：实体项固定为 `entity_id/description/visibility`，当前帧从 `ENTITY_01` 连续编号、至少一项最多 30 项且 description 同帧唯一；description 必须写当前帧可见形态和画面位置，edge_fragment 还须写明触及或被截断的画边及可见碎片形态；full 表示完整边界在画内且未被遮挡或画边截断，partial 表示被可见前景遮挡但不触画边，edge_fragment 表示任何可见部分触及或被画边截断并优先于 partial；同一物理实体只能一条记录，不能另建碎片实体。关系项固定为 `subject_id/predicate/object_id`、至少一项最多 60 项且每个实体至少参与一项，端点只许为当前帧实体或当前帧可观察人物、至少一端为实体，predicate 只能为 `supports/contacts/separate_from/occludes`；supports 表示 subject 支撑 object，occludes 表示 subject 位于前方并遮挡 object，contacts/separate_from 为无向关系。关系按 `(subject_id,predicate,object_id)` 升序；contacts/separate_from 的端点按字典序，同一无序端点对的 supports/contacts/separate_from 最多一项，occludes 同对最多一项，supports/occludes 不成有向环。`dominant_palette_contract` 固定为 `area_weighted_warm_cool_family/saturation_style`：前者只允许 `warm/cool/balanced`，后者只允许 `muted/natural/vivid`，均只由当前 source 帧的整帧面积加权主色盘判定。人物域闭包失败则 `eligible=false/person_replacement_unsafe`；物理场景实体身份或必须关系不能闭合则 `eligible=false/scene_components_ambiguous`；已识别场景与 boundary/五维变化冲突则 `eligible=false/scene_structure_replacement_unsafe`；不得输出 unknown/maybe 或跨帧补证。结束前逐帧自校验，任一矛盾返回空计划。段级 `photometric_contract` 固定为 `light_direction/light_quality/exposure_or_intensity/wb_cct/global_contrast/tone_curve`。canonical execution-input SHA 绑定 plan/continuity/profile/revision/model、typed slots、逐帧 source/transition evidence、frame constraint、photometric contract 和唯一 view；freeze/receipt 从 canonical plan 与 inventory 重算，合法但与 plan 不同的 target_spec/view 也失败，不靠逐帧复制 graph 文本代替绑定。
- `scene_plans` 对全部段无重叠全覆盖；每个所属段必须同时改变环境语义、可见形状与空间结构、纵深、空间布局和局部材质/固有色。scene boundary 与五维变化逐项相容，新增结构不得越过声明边界；冲突则 `eligible=false/scene_structure_replacement_unsafe` 并输出空计划。纯调色、纯纹理、只换材质或原结构换皮均不算换景。
- source identity、source scene、source reference 和逐帧 ENTITY_ID 只作定位旧设计及验收其消失的负样本证据，不能成为 target pack；人物目标由 frozen plan 的 replacement 字段定义，场景目标组件身份与 topology 只由 v4 continuity graph 定义。后端继续绑定对应帧 SHA、模型、profile、revision 和 plan SHA。
- 所有段固定保持画幅、裁切、机位、镜头、透视、构图、焦点、景深及全局光源方向、软硬、强度、曝光、白平衡、色温、整体色调、全局对比与 tone curve。除非输入有明确的全局调色要求，每帧整帧面积加权主色盘的 `area_weighted_warm_cool_family` 与 `saturation_style` 必须保持；目标人物和新场景的局部固有色必须明显不同，材质也可改变，但大面积新区域必须选源全局色彩家族兼容的颜色，不得以局部色变化翻转整帧冷暖感知；新几何只产生与原光源一致的局部阴影或反射，禁止全局重布光；交互、接触、持握、支撑、遮挡、前后顺序、视线、姿态、动作目的、数量、尺度与叙事关系不可破坏。
- 人物和场景目标包生成后、任何逐帧付费 POST 前，同一 Skill 执行 `phase=verify_pack`：每个人物验身份变更、旧身份消失、双视图一致和局部颜色；每个场景验语义、几何、纵深、布局和局部颜色；项目级验全局光向、曝光、白平衡/CCT 和 tone curve。v4 不增加 pack 字段，但每个场景既有 geometry/depth/layout evidence 合计必须逐 component 和 topology 覆盖 frozen graph。这一阶段只做语义判定，任一 `fail/unknown` 均 fail closed。
- 生成结果进入 H3 前，同一 Skill 执行 `phase=verify`：逐人物、逐可观察帧验证新身份和源身份消失；逐场景、逐所属段验证五类真实变化；v4 `frame_checks` 一一对照每帧的身体、姿态、接触、遮挡、裁切、`non_person_entity_ledger`、`dominant_palette_contract` 与光色合同，并且只增加结构化 `scene_continuity_view` 检查；稳定 graph 身份/topology 由既有项目级 `scene_continuity` 验收，其 evidence 必须逐 component/topology 覆盖，禁止每帧复制整 graph check。历史 v3 保持原九字段 exact 形状。`dominant_palette_contract` 逐帧以 source/output 的整帧面积加权主色盘验收冷暖家族与饱和度风格；局部固有色变化不得翻转整帧冷暖感知。所有输出先留在 staging，完整 verdict=`passed` 后才 publish；任何 `fail/unknown` 都使 `passed=false`，任一帧 unknown/fail 都使整项目 fail-closed、零 publish，`postprocess` 不会变为 `done`，H3 不可见。
- 图片 `verify` 通过后，同一 `video-maker` Skill 执行第三阶段 `reconcile_after_image_optimization`，不得重新选帧。原始源帧像素与 PTS 决定动作、机位和时序；旧视觉提示词只提供这些动态事实的低优先级检索证据；canonical 图片计划及其已选优化图 output receipts 决定新人物与新场景。existing dialogue、音频和台词 receipt 不进入该阶段。
- 第三阶段只输出 exact unified visual IR：动态 beat 结构化记录 action/camera/timing，PTS 固定为整数源时间戳并配套互质正整数 `time_base.numerator/denominator`，不接受浮点、字符串或毫秒换算；静态人物、服装、场景、材质和光色只绑定 canonical plan/image verification/output receipts 的 SHA 与稳定 ID，不再生成一份可漂移的新旧混合自由文本。优化图改变动作/姿态/接触/遮挡/因果、原始帧与优化图映射缺失，或新场景无法闭合原动作所需的支撑/接触/可达关系时一律 `eligible=false`；不能改写视频动态去迎合错误图片。
- 每个付费 POST 前持久化私有 receipt。只有完整 HTTP 429、`success=false`、精确 `RequestLimitExceeded` 时，才按 `AUTO_RETRY_COUNT/AUTO_RETRY_INTERVAL_S` 自动退避并追加新 attempt；网络异常、5xx、无效/不完整响应仍视为结果未知并禁止重发。已收到成功响应但下载失败时只恢复 GET。MediaKit WebP 结果经解码、尺寸校验和 PNG 转码后才进入 `frames`。
- Seedream 每帧总计最多 3 次 POST，且只有完整 HTTP 429、精确 `QuotaExceeded`、响应无 `data` 才自动重试；网络/超时/取消一律记为 `submission_unknown`。重启只恢复可证明安全的本地阶段，不自动重发未知 POST。
- 后处理一旦开始，H3 提交必须等待其 `done`；完成后每张 `postprocessed/` 优化帧都会替代同名原关键帧，进入冻结输入 receipt 和实际 H3 请求。缺帧或状态异常时 fail closed，不回退原图。
- “生成最终视频”的渲染门禁与 `/submit` 请求门禁共用同一判定：未开始后处理时允许使用原关键帧；后处理一旦存在，顶层必须为 `done`，且所有预期分段结构合法并逐段完整结束才放行。长视频 contract 必须 ready，段号必须严格连续为 `[1..segment_count]`，detail 与后处理段数也必须等于 `segment_count`；每段必须满足 `status=done`、`stage=done`、`error=null`、`revision>=1`、`total_frames>0` 且 `completed_frames=total_frames`。任一分段 running/failed、缺段、重复段、跳号、段数不符、残留错误或未完成帧均不显示生成操作且禁止请求。
- H3 generation 已创建后禁止再启动后处理，避免已冻结的付费输入与页面展示分叉。
- 请求顶层严格为 `confirm/options`，未知字段在写状态或调用供应商前拒绝。running 时不能重复提交；done 后改变选项返回结构化 409；failed 后普通 POST 一律拒绝，避免绕过分段 revision-CAS。
- 该能力可关闭；关闭不影响直接 H3 主链路。
- 页面 HTML、JS 和 CSS 均禁止缓存；遇到版本提示后刷新会取得同一版契约与样式。
- 页面只展示用户可理解的提示词、能力和分段进度，不展示模型 ID、模板 ID 或内部执行模式。
- 分段 `status/stage/error` 只通过有限中文白名单映射展示；未知阶段统一显示“处理中”，未知错误统一显示安全通用文案，绝不把后端原始值、供应商响应或堆栈透传到页面。
- “重试本段”同样按会话长短 fail closed：短视频只允许逻辑段 0，长视频只允许正整数段号；请求失败也只显示安全映射文案，不透传原始 `message`。
- 后处理整体进入 `failed` 后不再显示“是否优化素材？”或整项目普通 POST 重试；页面只提供各失败段的 revision-CAS 重试。若详情缺少 `segments`，只安全提示刷新。
- 分段阶段只接受公共值 `queued/text/brand/seedream/publishing/done` 并映射为供应商无关中文；短视频逻辑段 0 显示为“当前视频”。

## 例子

- 用户先做去字幕后再生成：H3 receipt 绑定优化图；若还需 crop/pad，则绑定由优化图产生的画幅派生帧。
