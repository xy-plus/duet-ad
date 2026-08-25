---
name: antd-x-frontend
type: feature
status: building
owner: agent
updated: 2026-08-25
tdd: required
links: []
---

# Ant Design X 前端重构

## 要什么

用 Ant Design 与 Ant Design X 重新实现现有会话式 H3 前端。3212 保留旧 UI，HTTPS 3213 提供新 UI；两者共享同一个后端和会话数据。

## 为什么

把视觉语言、交互状态和组件边界固化到主题、组件入口和自动检查中，避免本次及后续功能继续手写视觉组件或偏离既定设计。

## 验收

- [ ] 现有登录、创建、分析、提示词、短长视频生成、恢复/重试、媒体和后处理功能完整保留
- [ ] 所有可见控件和表面由 AntD/AntD X 提供，仅原生视频控件例外
- [ ] 主题 Token、组件导入、CSS 和视觉回归门禁可阻止后续显示风格漂移
- [ ] 旧 `web/` 和后端不修改，3212/3213 两套前端共享同一个 3212 后端
- [ ] 3213 使用现有 Caddy 与 HTTPS，不使用 Vite preview 作为生产服务器

## 边界

- 不修改 FastAPI API、持久化格式、H3 提交/恢复规则或后处理协议。
- 业务组件只能组合组件库，不新增自制按钮、输入框、卡片、弹窗、标签、进度条、图标或媒体控件。
- 原生 `<video controls>` 仅用于浏览器播放；认证媒体仍须通过 Bearer fetch 转为 Object URL。

## 取舍

- 服务端状态由 TanStack Query 管理；不引入 Redux、Zustand、React Router、Storybook 或 Ant Design X SDK。
- 3213 是独立 origin，因此独立登录；同源 `/api` 代理避免修改后端 CORS。
