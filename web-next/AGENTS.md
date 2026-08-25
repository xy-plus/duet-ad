# web-next 前端约束

本目录只使用 Ant Design 与 Ant Design X 建立可见界面。目标不是复刻库组件，而是让所有功能自然服从同一套 Token、状态语义和交互习惯。

## 组件边界

- 业务代码只能从 `src/ui/antd` 导入可见组件和图标，禁止直接导入 `antd`、`@ant-design/x` 或 `@ant-design/icons`。
- 可以创建有业务含义的组合组件，但不能创建自己的 Button、Input、Card、Modal、Tag、Progress、Upload、Image、播放器控件或通用视觉包装器。
- 禁止原生交互控件、手写 SVG 和内联样式。唯一例外是 `src/ui/video.tsx` 中由 AntD Card 承载的原生 `<video controls>`。
- 如果门面缺少组件，先把确实需要的官方组件加入 `src/ui/antd.ts`；不得通过直引或自制组件绕过门禁。

## 视觉与样式

- 颜色、字体、圆角、阴影和控件尺寸只在 `src/ui/theme.tsx` 定义。功能代码不得声明新的视觉常量。
- 功能 CSS 只能处理页面几何、滚动、响应式断点和视频宽高比，并使用 Ant Token CSS 变量。
- 禁止 `.ant-*` 内部选择器、原始颜色、`!important`、渐变、玻璃拟态和组件式 CSS 类。

## 页面骨架与密度

- 这是会话式 AI 工作台，不是后台管理页。桌面固定为 272px 会话侧栏、固定顶栏和不超过 900px 的居中内容流；页面高等于 `100dvh`，只允许内容区和会话列表滚动。
- 分析完成态的首屏必须容纳结论、真实指标、紧凑源视频和关键帧入口。源视频桌面宽度不得超过 320px；移动端才改为单列全宽。
- 连续阅读内容不要逐段包 Card。Card 只用于媒体、参数或独立状态边界；摘要使用 Descriptions，提示词使用 Collapse，反馈使用 Alert/Result，进度使用 Progress/ThoughtChain。
- 提示词默认折叠；生成设置使用一块轻量参数表面；创建/更新时间属于末尾次要信息，不得单独占据大卡片。
- 修改骨架、密度或组件表面前，先更新 `docs/human/features/antd-x-frontend/behaviors/visual-language.md` 和 `docs/agent/reference/web-next.md`，再更新对应 Playwright 截图；不得只覆盖截图掩盖无规则的漂移。

## 数据与交互

- 服务端状态由 TanStack Query 管理；API 地址始终使用相对 `/api`，不得引入 CORS fallback 或写死端口。
- 详情、生成参数和恢复动作只信服务端冻结值。`submission_unknown` 禁止提交；resume 和 stitch retry 复用旧请求 id，确定失败重试使用新 id。
- 认证媒体必须通过 Bearer fetch 生成 Object URL，并在切换资源或卸载时回收。
- 不得加入后端不存在的聊天入口、统计值、媒体元数据、假计时器或生产 mock。

## 验证

- 功能先写失败测试，再做最小实现。
- 提交前运行 `npm run check`；真实浏览器契约与截图运行 `npm run test:e2e`。
- 视觉验收固定覆盖 1440×1000 的首屏/生成区和 390×844 的 Drawer/正文；文档滚动、内容宽度、源视频宽度和横向溢出另有几何断言。
- 不得通过禁用 ESLint、Stylelint、TypeScript 或截图断言来换取通过。
