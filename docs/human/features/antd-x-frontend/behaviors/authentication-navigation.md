---
name: authentication-navigation
type: behavior
status: done
owner: human
updated: 2026-08-25
tdd: N/A
links: [antd-x-frontend]
---

# 登录、登出与会话导航

## 规则

| 当 | 则 |
| --- | --- |
| 浏览器首次打开 3213，当前 origin 没有 `cvs_token` | 只显示 token 登录页，不请求会话与媒体 |
| 用户提交 token | 前端调用 `POST /api/login`；成功后才把 token 存入当前 origin 的 localStorage 并加载会话列表 |
| 3213 登录成功 | 只代表 3213 origin 已登录；3211 的登录状态不会被读取或覆盖 |
| 任一 JSON 或文件请求返回 401 | 清除当前 token、TanStack Query cache 和会话 key，返回登录页 |
| 用户点击退出 | 执行与 401 相同的本地清理，不保留旧账号查询数据供下一次登录复用；唯一例外是防重复付费的 reconciliation lease，它不含 token 或秘密并按 [generation-safety](generation-safety.md) 契约跨登出保留 |
| 会话列表加载完成且用户没有主动新建 | 默认打开列表第一项；切换会话不停止其他运行中会话的后台详情轮询 |
| 用户点击“新建会话” | 保留侧栏，主区域切换到创建页；创建成功后打开真实返回的会话 id |
| 屏幕为移动尺寸 | 侧栏放入 Drawer；会话选择、新建和退出语义不变 |

## 边界

- token 会持久保存在当前 origin 的 localStorage；不要把 token 放进 URL、截图、日志或文档。
- 登录失败或网络失败只展示错误，不写入 token，也不伪造空工作区。
- 会话 badge 优先使用 API 的 `navigation_status`；字段缺失时只兼容 API 的 `generation/status/has_video`，未知值显示异常，不用本地计时推算成功。
- 3213 使用相对 `/api`，没有跨 origin CORS fallback 或写死的 3212 地址。

## 例子

- 输入：用户已在 3211 登录，首次打开 3213 → 输出：3213 仍要求单独输入 token。
- 输入：查看会话 A 时会话 B 仍在生成 → 输出：切到 A 后 B 继续后台刷新，侧栏状态随服务端详情更新。
