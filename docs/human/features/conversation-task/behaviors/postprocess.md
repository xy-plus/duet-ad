---
name: postprocess
type: behavior
status: done
owner: human
updated: 2026-08-24
tdd: N/A
links: [conversation-task, result-display]
---

# 可选关键帧后处理

## 规则

| 当 | 则 |
| --- | --- |
| `ENABLE_MEDIAKIT_ERASE=true` 且 schema v2 会话已 `done` | 显示“移除文字/字幕”和“移除常见 Logo/图标”两个 MediaKit 选项 |
| 至少选一项并严格确认 | `POST /postprocess` 返回 running，逐帧并行编辑并以 2 秒轮询展示进度 |
| 全部帧完成 | `postprocess.status=done`，展示 `postprocessed/` 对比图 |
| 任一帧失败 | `postprocess.status=failed`，保留成功帧；相同选项可人工重试 |
| 旧会话 | 409 `read_only` |
| 旧页面仍提交 `change_bg/face_hold` | 提示刷新页面；不写状态、不产生 MediaKit 费用 |

## 边界

- 当前只保留 `remove_subtitle`、`remove_brand`；`change_bg/face_hold` 已删除。旧页面请求只得到纯文本刷新提示，不会静默采用或自动重试。
- `remove_subtitle` 映射 `full_screen_text_erase`；`remove_brand` 作为兼容字段映射 `full_screen_icon_erase`，只承诺清理常见 Logo/图标。双选会执行两个独立付费阶段。
- 每个付费 POST 前持久化私有 receipt。只有完整 HTTP 429、`success=false`、精确 `RequestLimitExceeded` 时，才按 `AUTO_RETRY_COUNT/AUTO_RETRY_INTERVAL_S` 自动退避并追加新 attempt；网络异常、5xx、无效/不完整响应仍视为结果未知并禁止重发。已收到成功响应但下载失败时只恢复 GET。MediaKit WebP 结果经解码、尺寸校验和 PNG 转码后才进入 `frames`。
- 后处理一旦开始，H3 提交必须等待其 `done`；完成后每张 `postprocessed/` 优化帧都会替代同名原关键帧，进入冻结输入 receipt 和实际 H3 请求。缺帧或状态异常时 fail closed，不回退原图。
- H3 generation 已创建后禁止再启动后处理，避免已冻结的付费输入与页面展示分叉。
- 请求顶层严格为 `confirm/options`，未知字段在写状态或调用供应商前拒绝。running 时不能重复提交；done/failed 后改变选项返回结构化 409，避免旧产物贴上新标签。
- 该能力可关闭；关闭不影响直接 H3 主链路。
- 页面 HTML、JS 和 CSS 均禁止缓存；遇到版本提示后刷新会取得同一版契约与样式。

## 例子

- 用户先做去字幕后再生成：H3 receipt 绑定优化图；若还需 crop/pad，则绑定由优化图产生的画幅派生帧。
