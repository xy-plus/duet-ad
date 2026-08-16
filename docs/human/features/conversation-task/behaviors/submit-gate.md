---
name: submit-gate
type: behavior
status: done
owner: human
updated: 2026-08-17
links: []
---

# 提交 Seedance 门控

## 规则

| 当 | 则 |
| --- | --- |
| `ENABLE_SEEDANCE_SUBMIT` 未开（默认） | 501 `Seedance submission is disabled.` |
| 会话不存在 | 404 `not found` |
| 请求体缺 `"confirm": true` | 409 `confirmation required` |
| 会话状态不是 `done` | 409 `artifacts not ready` |
| 已提交过（`has_video`） | 409 `already submitted` |
| 评审产物被改动（prompt.txt 缺失/为空、关键帧缺失、dry-run 预检构建失败） | 409 `payload changed since review` |
| 服务进程无 `ARK_API_KEY` | 503 `ARK_API_KEY not configured` |
| 提交执行失败/超时（1800s） | 502，detail 经脱敏（≤300 字） |
| 全部通过 | 200 `{"status":"succeeded","video":"generated.mp4"}`，成片落盘并可下载 |

门控按上表固定顺序执行；每会话一把锁，锁内复查 `has_video` 防并发重复扣费；提交请求在提交时由 `work/prompt.txt` + `work/keyframes/*.png` 现构建（建模固定），用户提交体只接受 `confirm`，不接受任何 prompt/参数覆盖。

## 边界

- 前端按钮当前恒禁用：即使后端开关打开，也需前端跟进才可用（预留）
- 重复提交/双击/并发：幂等，`already submitted`
- 密钥 `ARK_API_KEY` 只存在于服务进程环境：不进日志、不进响应、不进 meta.json；报错一律脱敏
- 提交后 meta 记录 `has_video/submitted_at/task_id`，这些字段不出现在 API 响应里

## 例子

- 输入：默认配置 `POST /api/conversations/<id>/submit {"confirm":true}` → 输出：501
- 输入：开关开启、会话 `done`、带 `confirm:true`、密钥齐备 → 输出：200，`generated.mp4` 可经 files 接口下载
