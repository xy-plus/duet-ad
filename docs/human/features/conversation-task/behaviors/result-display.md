---
name: result-display
type: behavior
status: done
owner: human
updated: 2026-08-17
links: []
---

# 结果展示

## 规则

| 当 | 则 |
| --- | --- |
| 会话 `done` | 依次展示：用户气泡（标题/备注）→ 关键帧网格 → Seedance prompt → 最终视频区 |
| 关键帧 1..9 张 | 按时间序网格展示，图片经鉴权接口逐张取回（fetch blob → ObjectURL） |
| prompt 存在 | 全文展示，提供复制按钮（clipboard API，失败时降级 execCommand） |
| `has_video` 为真 | 最终视频区内嵌播放 `generated.mp4` 成片 |
| `has_video` 为假 | 最终视频区显示「待提交生成」，按钮恒禁用 |
| 侧栏列表 | 每项显示标题 + 状态徽章，按创建时间倒序 |

## 边界

- `has_video` 由后端按磁盘实况探测（`generated.mp4` 是否存在）
- 接触表（contact_sheet）经 files 接口可取，但前端当前不展示
- 文件直链全部需要 Bearer 鉴权，不能直接 `<img src>`，故一律 blob 化
- 「生成最终视频」按钮当前恒禁用，提示「待提交生成（接口预留，当前阶段未开放）」

## 例子

- 输入：打开一个 `done` 会话 → 输出：关键帧网格 + prompt 卡片（复制按钮）+ 最终视频区「待提交生成」
