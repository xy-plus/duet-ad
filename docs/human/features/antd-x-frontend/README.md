---
name: antd-x-frontend
type: feature
status: building
owner: human
updated: 2026-08-25
tdd: N/A
links: [conversation-task]
---

# Ant Design X 前端重构

## 要什么

用 Ant Design 与 Ant Design X 重新实现完整会话式 H3 工作台。现行 3211/本机 3212 保留旧 UI；候选 HTTPS 3213 提供新 UI，并通过同源 `/api` 共享同一个 3212 后端和会话数据。

## 为什么

把视觉语言、交互状态和组件边界固化到主题、组件入口和自动检查中，避免本次及后续功能继续手写视觉组件或偏离既定设计。

## 验收

- [ ] 现有登录、创建、分析、提示词、短长视频生成、恢复/重试、媒体和后处理功能完整保留
- [ ] 所有可见控件和表面由 AntD/AntD X 提供，仅原生视频控件例外
- [ ] 主题 Token、组件导入、CSS 和视觉回归门禁可阻止后续显示风格漂移
- [ ] 旧 `web/` 和后端不修改，3211/3212 与 3213 两套前端共享同一个 3212 后端
- [ ] 3213 使用现有 Caddy 与 HTTPS，不使用 Vite preview 作为生产服务器
- [ ] 生产只读 smoke 通过后再把本 feature 和对应 Agent 文档改为 `done`

## 用户行为

- [登录、登出与会话导航](behaviors/authentication-navigation.md)
- [创建会话与分析进度](behaviors/create-analysis.md)
- [提示词、生成参数与付费安全](behaviors/generation-safety.md)
- [认证媒体与关键帧后处理](behaviors/media-postprocess.md)

这些文件描述前端可观察行为；H3 receipt、状态机与供应商重试真相仍以 [会话任务](../conversation-task/README.md) 为准，不在这里复制。

## 边界

- 不修改 FastAPI API、持久化格式、H3 提交/恢复规则或后处理协议。
- 业务组件只能组合组件库，不新增自制按钮、输入框、卡片、弹窗、标签、进度条、图标或媒体控件。
- 原生 `<video controls>` 仅用于浏览器播放；认证媒体仍须通过 Bearer fetch 转为 Object URL。
- 旧 schema 或能力字段不完整的会话保持可查看但不可操作；前端不会从其他字段猜测可提交状态。
- 仓库中已有 3213 候选配置不代表线上已发布；上线前本 feature 保持 `building`。

## 取舍

- 服务端状态由 TanStack Query 管理；不引入 Redux、Zustand、React Router、Storybook 或 Ant Design X SDK。
- 3213 是独立 origin，因此独立登录；同源 `/api` 代理避免修改后端 CORS。
