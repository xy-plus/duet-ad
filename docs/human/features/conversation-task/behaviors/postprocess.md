---
name: postprocess
type: behavior
status: done
owner: human
updated: 2026-08-26
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
| 图片优化提示词被编辑 | `PATCH /image-optimization-prompt` 携带 `confirm/segment_index/expected_sha256/prompt`，以 SHA CAS 保存 |
| 图片优化提示词恢复默认 | 仅把 `default_text` 载入编辑区并标记为未保存，仍需用户保存 |
| 全部帧完成 | `postprocess.status=done`，展示 `postprocessed/` 对比图 |
| 任一分段失败 | 保留成功帧；该段显示“重试本段”，请求携带 `confirm/expected_revision`，点击后立即禁用以防双击 |
| 失败分段的 `error=submission_unknown` | 明确警告重复操作可能重复计费；用户再次明确确认后可按 `expected_revision` 人工重试本段；不得从 `status/stage` 推断 |
| 旧会话 | 409 `read_only` |
| 旧页面仍提交 `change_bg/face_hold` | 提示刷新页面；不写状态、不产生 MediaKit 费用 |

## 边界

- 视觉关键帧冻结后，后端按段在隔离工作区执行独立 `image-postprocess` Skill；Codex 只看到本段关键帧与编辑模式，输出就是页面展示的默认 Seedream 提示词，不读取或复制视频生成提示词。该提示词只为用户可能选择的非确定性图片二次编辑做准备，不代表编辑一定执行、成功或被成片采用。
- 页面将单段/多段提示词收敛为同一个三态工作区：“展开生成提示词 / 展开段台词 / 展开图片优化”。三个按钮等宽并排且窄屏允许文字换行；工作区共用一个文本区域。短视频生成提示词与图片优化可编辑，长视频的逐段生成提示词和段台词只读。
- 图片优化编辑必须同时满足 `postprocess_capabilities.optimize_image=true`；即使异常详情带回提示词，能力为 false 时也只读且不能发 PATCH。短视频逻辑段号为 0，长视频段号必须为正整数，长视频出现 0 或非法段号时 fail closed。
- 图片优化提示词提供保存、恢复默认、复制；短视频逻辑段固定使用 `segment_index=0`，长视频使用各自正整数段号。存在 dirty 草稿时，切换文本、分段、会话、新建会话、开始生成、打开或提交后处理、退出应用都会先要求“保存 / 丢弃 / 取消”；保存失败留在原处。刷新或关闭页面由 `beforeunload` 拦截，2 秒轮询不得覆盖 dirty 文本。
- 当前选项为 `remove_subtitle`、`remove_brand`、`optimize_image`；`change_bg/face_hold` 已删除。旧页面请求只得到纯文本刷新提示，不会静默采用或自动重试。
- `remove_subtitle` 映射 `full_screen_text_erase`；`remove_brand` 作为兼容字段映射 `full_screen_icon_erase`，只承诺清理常见 Logo/图标。双选会执行两个独立付费阶段。
- 三个阶段在每段内严格按“文字/字幕 → Logo/图标 → 图片优化”执行；段之间并行，段内每个阶段的帧并行受服务端并发上限约束。锚帧一致性模式会先生成首张新锚帧，再以它约束后续帧中重复人物、服装、道具和场景的替换结果；锚帧不能覆盖当前目标帧的内容、构图、机位、动作、光线或元素位置。共享提示词不得复述单帧布局。两种内部模式都不显示在页面。
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
