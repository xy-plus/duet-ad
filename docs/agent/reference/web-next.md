---
name: web-next
type: reference
status: building
owner: agent
updated: 2026-08-25
tdd: N/A
links: [antd-x-frontend]
---

# web-next · 接口与治理（How/Now）

## 版本与入口

| 项 | 固定值 |
| --- | --- |
| Node | `>=20.19` |
| React / React DOM | `19.1.1` |
| Ant Design / Ant Design X / icons | `6.6.1` / `2.9.0` / `6.1.0` |
| TanStack Query | `5.102.3` |
| ESLint / Stylelint / Playwright | `9.39.5` / `17.14.1` / `1.62.1` |
| API base | 相对 `/api`，无 dev proxy、CORS fallback 或写死端口 |
| token key | 当前 origin localStorage 的 `cvs_token` |
| detail poll | 仅运行态 `2_000ms`，后台标签页不轮询 |

公开启动入口：

- `src/main.tsx` — 创建 `createApiRuntime()`，把同一个 QueryClient 交给 ApiClient cache 清理和 `AppThemeProvider`。
- `createApiRuntime(options?) -> { apiClient, queryClient }` — 测试可注入 storage/fetch/XHR；生产使用相对 `/api`。
- `App({ apiClient })` — 无 token 显示 Login；有 token 显示 Workspace。没有前端 route 或第二套 store。

## API 状态

以下均由现有 FastAPI 实现；`web-next` 没有新增或修改后端接口。请求 mutation 不自动重试。

| 方法与路径 | 鉴权 | 前端用途与关键请求 |
| --- | --- | --- |
| `POST /api/login` | 否 | `{token}`；成功后才持久化 token |
| `GET /api/conversations` | Bearer | 会话侧栏；保留已知 generation/navigation 字段直到新列表明确覆盖 |
| `POST /api/conversations` | Bearer | multipart `file` 或 `reference_url`，另带 `note/client_request_id/voice_mode[/target_language]`；XHR 暴露进度 |
| `GET /api/conversations/{cid}` | Bearer | 权威 detail；决定轮询、只读能力、参数、媒体、generation 与 postprocess |
| `PATCH /api/conversations/{cid}/prompt` | Bearer | `{confirm:true,expected_sha256,prompt}`；`prompt_changed` 后只刷新 |
| `POST /api/conversations/{cid}/submit` | Bearer | 冻结 generation payload；ApiClient 同 cid 单飞 |
| `POST /api/conversations/{cid}/postprocess` | Bearer | `{confirm:true,options:{remove_subtitle,remove_brand}}`；锁定冲突后只刷新 |
| `GET /api/conversations/{cid}/files/{name}` | Bearer | Blob 媒体；每个 path segment 编码，Object URL 由 hook 管理 |
| `GET /api/health` | 否 | 只用于部署 smoke，不参与 App 状态 |

generation payload 固定包含：

```json
{
  "confirm": true,
  "client_request_id": "uuid-or-existing-id",
  "dialogue_mode": "auto|edit|custom|none",
  "fit_mode": "none|crop|pad",
  "aspect_ratio": "16:9|9:16",
  "resolution": "480p|768p"
}
```

- `edit/custom` 追加结构化 `lines`。
- 长视频追加 64 位 `expected_plan_receipt` 和冻结 `fast_mode`；新长视频默认 true，但 UI 不显示该开关或摘要。
- resume 与 stitch retry 从服务端冻结 detail 重建，并复用旧 `client_request_id`；确定 failed retry 使用新 id。
- `submission_unknown/running/succeeded` 不构建 action；长视频付费任务数未知时不允许 submit。

完整 response schema、HTTP 错误和 provider 状态机见 [后端 reference](reference.md)，这里不复制。

## Query 与资源所有权

Query key 形状：

```text
['session', sessionKey, 'conversations', 'list']
['session', sessionKey, 'conversations', 'detail', cid]
['session', sessionKey, 'conversations', 'detail', cid, 'file', name]
```

- query 默认 `retry=false`、`staleTime=30s`、不在 window focus 自动刷新；文件 `staleTime=Infinity`。
- 已选中会话由页面 detail observer 轮询；只有未选中且 `navigation_status` 表明仍在后台运行的会话才挂 background poller。浏览器页面进入后台时两者都停止 interval。
- mutation 成功/结束只 invalidate 当前 session 的 list/detail。
- provider-facing POST 前必须持久化 reconciliation lease；network/5xx/invalid response/409 `submission_outcome_unknown` 时跨 reload/logout 保持 GET-only，只有明确响应或同 request id 相对 baseline 的权威推进才清除。
- detail 回写 list 的 `status/has_video/navigation_status/generation`，侧栏不另建状态机。
- `ObjectUrlLease` 同 key+Blob 复用 URL，替换、清空和 dispose 都先 revoke；sessionKey 是资源 key 一部分。
- 任一 JSON/Blob 401 执行 `clearSession()`：删 token、清 QueryClient、旋转 sessionKey。

## 导入和组件 hard gates

业务 `src/**/*.{ts,tsx}`：

- 可见组件、X 原语和图标只能从 `src/ui/antd.ts` 导入；直接 import `antd`、`antd/*`、`@ant-design/x`、`@ant-design/icons` 由 ESLint 拒绝。
- 禁止原生 `button/form/input/select/textarea/img/svg/video` 等视觉/交互元素、inline `style`、业务色值和 `.svg` import。
- 语义 `main/section/header/div/figure/figcaption` 允许。
- 唯一原生 `<video>` 白名单是 `src/ui/video.tsx`，对外仍从 `src/ui/antd.ts` 导出；props 排除 inline style 并要求 `label`。
- 测试可直接 import 测试依赖；`src/ui/theme.tsx` 是颜色常量例外，不是业务逃生口。

`src/ui/theme.tsx` 是唯一视觉真源：

- `AppThemeProvider` 顺序为 `QueryClientProvider → XProvider(theme=appTheme, locale=zhCN) → AntApp`；禁止另建 ConfigProvider/XProvider。
- `cssVar.key=duet-next`、`hashed=false`；颜色、字体、圆角、阴影和控件尺寸集中在 `appTheme`。
- AntD reset 只由组件门面载入。

## CSS hard gates

Stylelint 对 `src/**/*.css` 强制：

- 禁止 `.ant-*` 内部 selector、hex/named color、`rgb/hsl/.../color()`、`!important`。
- `margin/padding/gap/row-gap/column-gap` 禁止裸 `px/rem`，必须使用 Ant Token CSS variable。
- feature CSS 只负责布局、滚动、响应式断点和视频宽高比；组件视觉由 Token/component props 表达。
- AGENTS 另禁止渐变、玻璃拟态和组件式通用 CSS；当前自动规则不应被描述成能识别全部审美语义。

`src/governance.test.ts` 用反例验证 import、原生元素、video 白名单、inline style、SVG、颜色、`.ant-*`、`!important` 和 spacing gates，不能靠 disable 注释绕过。

## 运行、测试与 3213 发布契约

```bash
cd web-next
npm ci
npm run check
npm run test:e2e
```

- `npm run check` = TypeScript + ESLint + Stylelint + 全部 Vitest + production build。
- `npm run test:e2e` 启动本机 Vite `127.0.0.1:4173`，Chromium 拦截 `/api` 验证真实浏览器交互，并核对 desktop/mobile screenshot。它不是线上 API E2E。
- `npm run dev` 不代理真实 `/api`；`npm run preview` 也只供本地静态检查。
- 生产发布只使用构建后的 `web-next/dist` 和同一个 Caddy unit：3213 `/api/*` 反代 3212，其余路径 file server + SPA fallback；HTML/no-route `no-store`，hash asset `public,max-age=31536000,immutable`。
- 发布验证必须包含后端全量测试、`tests/test_caddy_dual_frontend.py`、Caddy validate、构建产物检查及 [.deploy/runbook.md](../../../.deploy/runbook.md) 第 6 节 GET/HEAD smoke。
- 3213 smoke 禁止 POST/PUT/PATCH/DELETE、创建会话或运行付费 H3 smoke。真实付费验收必须另行授权。
- 仓库候选配置、监听 socket 或本机 200 都不能单独证明公网发布；线上只读 smoke 通过后才把相关文档 `status` 改为 `done`。
