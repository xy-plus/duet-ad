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
| v2 双目标图片计划已冻结 | 页面可查看后端确定性编译的提示词，但不能用自由文本改写；PATCH 返回 `image_optimization_prompt_compiled` |
| 历史 v1 图片提示词被编辑 | `PATCH /image-optimization-prompt` 仍以 SHA CAS 保存，不迁移或重写旧 continuity receipt |
| 全部帧完成 | `postprocess.status=done`，展示 `postprocessed/` 对比图 |
| 任一分段失败 | 保留成功帧；该段显示“重试本段”，请求携带 `confirm/expected_revision`，点击后立即禁用以防双击 |
| 失败分段的 `error=submission_unknown` | 明确警告重复操作可能重复计费；用户再次明确确认后可按 `expected_revision` 人工重试本段；不得从 `status/stage` 推断 |
| 旧会话 | 409 `read_only` |
| 旧页面仍提交 `change_bg/face_hold` | 提示刷新页面；不写状态、不产生 MediaKit 费用 |

## 边界

- 视觉关键帧冻结后，隔离 Codex 以同一个 `image-postprocess` Skill 执行 `phase=plan`：短视频读取逻辑段 `0`，多段项目读取全部分段关键帧。输出只含结构化 v2 设计；后端确定性编译每段 Seedream 提示词，Skill 不读取或复制 H3 提示词。
- 页面将单段/多段提示词收敛为同一个三态工作区：“展开生成提示词 / 展开段台词 / 展开图片优化”。三个按钮等宽并排且窄屏允许文字换行；工作区共用一个文本区域。短视频生成提示词与图片优化可编辑，长视频的逐段生成提示词和段台词只读。
- 图片优化编辑必须同时满足 `postprocess_capabilities.optimize_image=true`；即使异常详情带回提示词，能力为 false 时也只读且不能发 PATCH。短视频逻辑段号为 0，长视频段号必须为正整数，长视频出现 0 或非法段号时 fail closed。
- v2 双目标计划的编译提示词只提供查看和复制，不能由自由文本删除人物、场景、光色或关系硬约束；历史 v1 提示词继续原 CAS 行为，receipt 不迁移、不重写。
- 当前选项为 `remove_subtitle`、`remove_brand`、`optimize_image`；`change_bg/face_hold` 已删除。旧页面请求只得到纯文本刷新提示，不会静默采用或自动重试。
- `remove_subtitle` 映射 `full_screen_text_erase`；`remove_brand` 作为兼容字段映射 `full_screen_icon_erase`，只承诺清理常见 Logo/图标。双选会执行两个独立付费阶段。
- 三个阶段在每段内严格按“文字/字幕 → Logo/图标 → 图片优化”执行。v2 `plan` 以连续性与现实合理性优先于替换幅度；人物与真实新场景仍是不可降级的双目标，任一目标与可见事实冲突时在付费前 `eligible=false`。短视频 `[0]` 使用相同结构，不能成功 no-op。
- 每个稳定主人物和物理场景各有一个冻结目标包；目标包逐段复用，不逐帧重设计，也不从编辑结果递推。每段按 ID 完整枚举全部主人物；不可见者标 `not_observable`，不得依据相邻段或 reference 补造人物或身体部分。
- `hard_cut` 是场景证据边界：切后场景只依据切后证据，不能继承无切后证据的切前设计。每帧姿态、边界与关系只以当前源帧为事实；`continue` 才优先作为连续画面分析。
- `scene_plans` 对全部段无重叠全覆盖；每个所属段必须同时改变环境语义、可见形状与空间结构、纵深、空间布局和局部材质/固有色。纯调色、纯纹理、只换材质或原结构换皮均不算换景。
- source identity、source scene 和 source reference 只作定位旧设计及验收其消失的负样本证据，不能成为 target pack；人物/场景目标由 frozen plan 的 replacement 与变化字段定义。后端继续绑定对应帧 SHA、模型、profile、revision 和 plan SHA。
- 所有段固定保持画幅、裁切、机位、镜头、透视、构图、焦点、景深及全局光源方向、曝光、白平衡/CCT、tone curve。目标局部固有色按计划变化，新几何只产生物理正确的局部阴影和反射；交互、接触、持握、支撑、遮挡、前后顺序、视线、姿态、动作目的、数量、尺度与叙事关系不可破坏。
- 人物和场景目标包生成后、任何逐帧付费 POST 前，同一 Skill 执行 `phase=verify_pack`：每个人物验身份变更、旧身份消失、双视图一致和局部颜色；每个场景验语义、几何、纵深、布局和局部颜色；项目级验全局光向、曝光、白平衡/CCT 和 tone curve。这一阶段只做语义判定，任一 `fail/unknown` 均 fail closed。
- 生成结果进入 H3 前，同一 Skill 执行 `phase=verify`：逐人物、逐可观察帧验证新身份和源身份消失；逐场景、逐所属段验证五类真实变化；再验证相机、全局光色、可见关系及连续性。任何 `fail/unknown` 都使 `passed=false` 且不发布。
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
