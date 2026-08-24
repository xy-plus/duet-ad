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
| `ENABLE_SEEDREAM_EDIT=true` 且 schema v2 会话已 `done` | 显示“去字幕水印”和“去版权/品牌物品”两个 Seedream 选项 |
| 至少选一项并严格确认 | `POST /postprocess` 返回 running，逐帧并行编辑并以 2 秒轮询展示进度 |
| 全部帧完成 | `postprocess.status=done`，展示 `postprocessed/` 对比图 |
| 任一帧失败 | `postprocess.status=failed`，保留成功帧；相同选项可人工重试 |
| 旧会话 | 409 `read_only` |
| 旧页面仍提交 `change_bg/face_hold` | 提示刷新页面；不写状态、不产生 Seedream 费用 |

## 边界

- 当前只保留 `remove_subtitle`、`remove_brand`；`change_bg/face_hold` 已删除。旧页面请求只得到纯文本刷新提示，不会静默采用或自动重试。
- 后处理不改变输入准备状态，也不进入冻结 H3 receipt。无论先后顺序如何，H3 都不会读取 `postprocessed/`。
- 请求顶层严格为 `confirm/options`，未知字段在写状态或调用供应商前拒绝。running 时不能重复提交；done/failed 后改变选项返回结构化 409，避免旧产物贴上新标签。
- 该能力可关闭；关闭不影响直接 H3 主链路。
- 页面 HTML、JS 和 CSS 均禁止缓存；遇到版本提示后刷新会取得同一版契约与样式。

## 例子

- 用户先做去字幕后再生成：页面展示优化图，但 H3 receipt 仍绑定原始或 crop/pad 派生关键帧。
