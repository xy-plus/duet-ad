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
- 三个阶段在每段内严格按“文字/字幕 → Logo/图标 → 图片优化”执行。v2 图片优化对每个稳定叙事主人物建立独立 `person_plans`，每段完整枚举人物的 `replace/not_observable` 与可观察帧；所有可观察主人物都必须替换，背景路人只保护。漏主人物、人物轨道混淆或不安全时项目在付费前 `eligible=false`，不能成功 no-op。
- v2 的 `scene_plans` 对全部段无重叠全覆盖；每个场景必须同时更换环境语义、可见形状、纵深、空间布局和局部材质/固有色。纯调色、纯纹理或原结构换皮不算换景。短视频 `[0]` 使用完全相同的双目标结构。
- 所有段固定保持画幅、裁切、机位、镜头、透视、构图、焦点、景深，以及全局光源方向、曝光、白平衡/CCT、tone curve；允许新几何产生物理正确的局部阴影。持握、接触、遮挡、数量、姿态、动作目的与叙事关系不可破坏。
- 每个主人物、场景组件和段布局都冻结来源明确的 reference slot；后端绑定对应帧 SHA、模型、profile、revision 和 plan SHA。执行层按当前源图、可观察主人物 identity refs、scene ref、layout ref 的固定顺序构造输入，不自行猜测。
- 生成结果进入 H3 前，同一 Skill 执行 `phase=verify`，只读取源/输出帧、冻结 v2 plan 和确定性指标；逐人物验证身份已换且源身份无残留，并验证场景语义/结构、局部颜色、光色、互动关系、段内与跨段连续性。任一 `fail/unknown` 都不发布。
- 每个付费 POST 前持久化私有 receipt。只有完整 HTTP 429、`success=false`、精确 `RequestLimitExceeded` 时，才按 `AUTO_RETRY_COUNT/AUTO_RETRY_INTERVAL_S` 自动退避并追加新 attempt；网络异常、5xx、无效/不完整响应仍视为结果未知并禁止重发。已收到成功响应但下载失败时只恢复 GET。MediaKit WebP 结果经解码、尺寸校验和 PNG 转码后才进入 `frames`。
- Seedream 每帧总计最多 3 次 POST，且只有完整 HTTP 429、精确 `QuotaExceeded`、响应无 `data` 才自动重试；网络/超时/取消一律记为 `submission_unknown`。重启只恢复可证明安全的本地阶段，不自动重发未知 POST。
- 后处理一旦开始，H3 提交必须等待其 `done`；完成后每张 `postprocessed/` 优化帧都会替代同名原关键帧，进入冻结输入 receipt 和实际 H3 请求。缺帧或状态异常时 fail closed，不回退原图。
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
