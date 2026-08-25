---
name: antd-x-frontend
type: architecture
status: building
owner: agent
updated: 2026-08-25
tdd: N/A
links: [antd-x-frontend]
---

# Ant Design X 前端架构（How/Now）

## 状态与运行边界

`web-next`、双入口 Caddy 配置和自动化测试已集成；现有生产 3213 由 Caddy 提供服务，本次视觉对齐仍处于发布前验证阶段，发布后必须重新执行本机与公网只读 smoke。

```mermaid
flowchart LR
  B1[Browser :3211] -->|all paths| C[Caddy h1/h2]
  B2[Browser :3213] -->|same origin| C
  C -->|3213 HTML/assets| S[web-next/dist]
  C -->|3211 all or 3213 /api/*| F[FastAPI 127.0.0.1:3212]
  F --> D[data/cid]
  F --> H[AutoDL Art H3]
```

- 3211 保持整站反代到 uvicorn，由 `web/` 提供旧 UI。
- 3213 的 `/api/*` 保留路径并反代同一 `127.0.0.1:3212`；其余路径由同一 Caddy unit 从 `web-next/dist` 提供，未知客户端路由回落 `index.html`。
- 不新增 CORS、前端 service、第二个 uvicorn 或前端专用数据存储。3211/3213 是不同 origin，因此 `cvs_token` 和 Query session 独立。

## 模块

| 模块 | 职责 | 实现的 feature |
| --- | --- | --- |
| `src/main.tsx` | 创建唯一 runtime，把 QueryClient、XProvider/AntD App 和 App 组合到 React root | antd-x-frontend |
| `src/api/` | 相对 `/api` 的 JSON、XHR 上传和认证 Blob transport；401 清 session，submit 单飞 | antd-x-frontend |
| `src/state/` | TanStack Query keys、2 秒条件轮询、mutation invalidation、session 隔离、Object URL lease | antd-x-frontend |
| `src/domain/` | 纯函数校验 detail、服务端建议/冻结值、CAS 与 generation/recovery payload；无 React 和网络 | antd-x-frontend |
| `src/app/` | 把 API/state/domain 与页面组合；唯一拥有跨 feature 的工作流状态 | antd-x-frontend |
| `src/features/shell` | token 登录、桌面侧栏、移动 Drawer、会话导航 | antd-x-frontend |
| `src/features/create` | URL/文件、说明、台词处理和上传进度的受控创建器 | antd-x-frontend |
| `src/features/conversation` | 服务端状态驱动的 Bubble/空态/错误态 | antd-x-frontend |
| `src/features/media` | 提示词、摘要、视频、关键帧和长视频分段的纯展示 | antd-x-frontend |
| `src/features/generation` | 生成参数、冻结证据、付费任务数、分段进度和安全动作 | antd-x-frontend |
| `src/features/postprocess` | 选项 Modal、后台状态卡、冻结选项重试与结果预览 | antd-x-frontend |
| `src/ui/antd.ts` | AntD、Ant Design X、icons 和唯一 Video 的单一组件门面 | antd-x-frontend |
| `src/ui/theme.tsx` | 唯一 Token、中文 locale、XProvider、AntD App 与 QueryClientProvider 边界 | antd-x-frontend |
| `tests/app.spec.ts` | 真实 Chromium API 合同、付费安全、后台行为、移动端和截图基线 | antd-x-frontend |

## 数据流

```mermaid
flowchart LR
  UI[features controlled UI] --> APP[app composition]
  APP --> DOMAIN[domain fail-closed builders]
  APP --> QUERY[state TanStack Query]
  QUERY --> API[ApiClient /api]
  API --> FASTAPI[existing FastAPI]
  FASTAPI --> QUERY
  QUERY --> APP
  API -->|Bearer Blob| LEASE[ObjectUrlLease]
  LEASE --> MEDIA[Image or governed Video]
```

关键不变量：

- API 返回是唯一业务状态源。组件不制造聊天回复、计时器、分段数、媒体参数或恢复能力。
- Query key 固定含 `sessionKey`；登录、退出和 401 都旋转 key，401/退出同时清空同一个 QueryClient。
- 已选中会话由页面 detail observer 在 queued/processing、generation queued/running、postprocess running 时按 2 秒轮询；列表中标记为后台运行且未选中的会话才挂载 background poller。`refetchIntervalInBackground=false` 只表示浏览器切到后台标签页时停止轮询。
- `read_only === false && submit_enabled === true` 才允许操作；缺字段即不可操作。
- prompt 保存使用 `expected_sha256`，长视频 submit 使用 `expected_plan_receipt`；冲突只刷新，不自动重发。
- 所有 TanStack mutation `retry=false`；会话 submit 另有 ApiClient 单飞门禁。
- 认证文件不直接放进 `<img>/<video>` URL；Bearer fetch 得到 Blob 后租约化 Object URL，切换 key 或卸载即 revoke。

## 视觉组合边界

页面由三层组合，不能由 feature 各自重新定义壳或宽度：

```text
WorkspaceShell (100dvh)
├─ ConversationSidebar (272px, independent scroll)
└─ Workspace
   ├─ Header (fixed in shell)
   └─ Content (independent scroll)
      └─ ConversationDetail (max 900px)
         ├─ analysis summary + compact source video
         ├─ keyframes + dialogue + segments
         ├─ prompt Collapse + generation settings/status
         └─ postprocess + final media + secondary timestamps
```

- `src/ui/theme.tsx` 决定全局色彩、圆角、阴影、控件高度和 Conversations creation 状态；feature CSS 不能建立第二套视觉变量。
- shell/app CSS 决定唯一页面几何；media/generation CSS 只决定该业务组合在既定内容流内的排列。业务 CSS 禁止建立背景、边框、圆角、阴影、字体、颜色和自定义变量，视觉表面必须由 facade 暴露的官方组件承担。
- 真实媒体、服务端字段和付费门禁优先于原型外观；原型的假 Sender、假指标、固定分片和占位媒体不属于架构。
- Playwright 同时冻结首屏、生成区、移动 Drawer 和移动正文，防止只优化单一截图。

## 对外接口

- 浏览器入口：`https://8.166.140.227:3213/`；SPA 不使用 React Router，当前页面选择保存在 App 内存。
- 后端入口：只使用同源相对 `/api`；完整清单见 [web-next reference](../reference/web-next.md)。
- 现有后端 schema、receipt 与付费状态机不变，见 [后端 reference](../reference/reference.md)。
- 发布与回滚只执行 [.deploy/runbook.md](../../../.deploy/runbook.md) 的 3213 合同；`vite dev/preview` 均不是生产入口。
