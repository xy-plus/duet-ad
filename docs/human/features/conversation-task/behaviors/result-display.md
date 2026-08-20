---
name: result-display
type: behavior
status: done
owner: human
updated: 2026-08-20
tdd: N/A
links: [conversation-task, processing-state]
---

# 结果评审与 H3 参数

## 规则

| 当 | 则 |
| --- | --- |
| 输入准备为 `done` | 展示源视频、1–9 张关键帧、最终 H3 prompt、台词选择、画幅选择和最终视频区 |
| Context IR 尚未开始 | 自动生成的 H3 源提示词可二次修改；保存后的版本才进入 Context IR，保存使用当前 SHA-256 防止覆盖新版本 |
| Context IR 已开始 | H3 源提示词锁定；不能让页面展示内容与已冻结的 Context IR 输入发生漂移 |
| `10 < duration_s <= 15` | 显示“仍可生成，但稳定性可能下降”的 warning；不阻止提交 |
| 选择 `auto` | 使用详情 `dialogue.auto_lines` 中的自动有效台词，不允许随请求上传 `lines` |
| 选择 `edit` | 以自动台词预填，提交至少一行 `{text,start_s,end_s}` |
| 选择 `custom` | 提交至少一行人工台词，不依赖自动识别结果 |
| 选择 `none` | 发声块明确写“无台词”，不允许上传 `lines` |
| 准备完成后的全部实际关键帧都是 9:16 | 画幅固定 `none`，不能选择 crop/pad |
| 任一实际关键帧不是 9:16 | 必须选择居中 `crop` 或黑边 `pad`，不提供静默默认值；即使源视频是 9:16，关键帧被裁成其他比例也适用 |
| generation active | 禁用参数和提交按钮，2 秒轮询状态 |
| generation 为 `resume_required` | 台词、画幅和请求 id 锁定，显示“继续既有任务”；确认后继续原 attempt |
| Context IR 已校验成功、成片尚未生成 | 在 IR 审阅编辑器中提供“翻译为中文”；译文只替换当前查看内容，不修改或提交 IR 原文 |
| Context IR 已校验成功且成片已生成 | “Context IR 优化提示词”紧跟在 H3 源提示词下方，复用相同提示词卡、全文和复制效果，并提供“翻译为中文” |
| Context IR 尚未开始或仍在运行 | 不渲染 IR 正文卡；正文接口分别返回 404 或 409，不能从轮询详情获得大 prompt |
| Context IR 状态、正文或 SHA-256 不一致 | 按损坏状态关闭访问，不展示正文或摘要 hash |
| 历史 Context IR attempt 已有 `ir_dialogue_mismatch` | 兼容展示 `H3 生成失败 / ir_dialogue_mismatch` 和可编辑的台词参数；同请求 id 不能继续，用户修改为 edit/custom/none 并点击重试时创建新请求 id、新 receipt 和新 attempt；新 attempt 不再产生该错误 |
| generation 确定失败 | 展示错误和“重试生成”；点击才创建新请求 id |
| generation 为 `submission_unknown` | 展示“先到供应商核对”的阻断说明，不显示重试按钮 |
| `generated.mp4` 存在 | 内嵌播放 H3 最终视频 |
| 旧会话 | 展示只读提示；不能修改台词、画幅、后处理或再次生成 |

## 边界

- `duration_s` 展示实际 ffprobe 时长；Context IR/H3 请求使用 `ceil(duration_s)` 的整数秒，范围 1–15。
- 所有媒体经带 Bearer 鉴权的 files API 获取，页面使用 blob URL，不暴露目录直链。
- Context IR 正文只经带 Bearer 鉴权的只读按需 API 获取；详情轮询不得包含正文、中文译文、provider task id、result URL、凭据或其他供应商原始数据。
- Context IR 展开只懒加载一次；切换会话后各会话独立缓存，hash 变化才允许重新读取。
- 中文翻译必须由用户点击后按当前 IR SHA-256 请求；单独缓存且只用于显示，不能写回 IR state、prepared receipt、prompt 或 H3 请求。
- 画面 OCR 只在视觉 prompt 中展示；唯一发声块只由结构化台词机械生成。
