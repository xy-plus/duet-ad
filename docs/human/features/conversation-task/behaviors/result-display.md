---
name: result-display
type: behavior
status: done
owner: human
updated: 2026-08-18
links: [postprocess]
---

# 结果展示

## 规则

| 当 | 则 |
| --- | --- |
| 会话 `done`（单段模式，`segments` 为空） | 依次展示：用户气泡（标题/备注）→ 关键帧网格 → Seedance prompt → 最终视频区 |
| 会话 `done`（多段模式，`segments` 非空） | 依次展示：用户气泡 → 「分段产物」逐段「第 N 段」卡片（段关键帧网格 + 段提示词卡片 + 段台词列表）→ 最终视频区；不重复展示顶层关键帧/prompt |
| 关键帧 1..9 张 | 按时间序网格展示，图片经鉴权接口逐张取回（fetch blob → ObjectURL） |
| prompt 存在 | 全文展示，提供复制按钮（clipboard API，失败时降级 execCommand）；多段模式每段独立卡片 |
| 段台词（`seg.lines`） | 该段卡片内以列表展示；空列表不展示台词区 |
| 后处理（`postprocess`） | 关键帧区标题旁显示「后处理」按钮与弹窗、优化后对比网格、失败提示——见 `postprocess` behavior |
| `has_video` 为真 | 最终视频区内嵌播放 `generated.mp4` 成片 |
| `has_video` 为假 | 最终视频区显示「待提交生成」，按钮恒禁用 |
| 侧栏列表 | 每项显示标题 + 状态徽章，按创建时间倒序 |

## 边界

- `has_video` 由后端按磁盘实况探测（`generated.mp4` 是否存在）
- 分页联系表（contact_sheet_01.jpg…）不进 files 白名单；白名单仅保留旧版单页 contact_sheet.jpg 映射以兼容存量会话；前端不展示
- 文件直链全部需要 Bearer 鉴权，不能直接 `<img src>`，故一律 blob 化
- 「生成最终视频」按钮当前恒禁用，提示「待提交生成（接口预留，当前阶段未开放）」
- 段关键帧取图路径 `/files/segments/N/work/keyframes/<name>`；优化图 `/files/postprocessed/<name>`（多段 `/files/segments/N/work/postprocessed/<name>`）

## 例子

- 输入：打开一个 `done` 会话（单段）→ 输出：关键帧网格 + prompt 卡片（复制按钮）+ 最终视频区「待提交生成」
- 输入：打开一个多段 `done` 会话 → 输出：逐段「第 N 段」卡片（段关键帧 + 段提示词 + 段台词）
