# Duet AI Ant Design X 原型

独立的前端交互原型，使用 React 19、Ant Design 6 与 Ant Design X 2。它不读取现有 `web/`，不调用后端或供应商接口，刷新页面后所有本地状态复位。

```bash
npm install
npm run dev -- --host 127.0.0.1
```

浏览器打开 `http://127.0.0.1:5173/`。验证命令：

```bash
npm test -- --run
npm run lint
npm run build
```
