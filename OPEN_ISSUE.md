# OPEN_ISSUE

<!-- 活文档，只许这三节；每条 ≤3 行：一句现状 + 一句出路 + 指针（细节放 docs）；完成即删、不留 ✅ 编年史；全文超 ~60 行＝有条目该清 -->

## ⚠️ 待拍板

<!-- 需人决策才能推进；自愈终审失败件放这里，标 ⚠️需人确认 -->

## 🚧 进行中

<!-- 开工登记、集成即删；重开任务记重开链：继承未解 blockers 与已用轮次 -->

- A 后端核心：FastAPI 骨架(config/auth/health/静态托管/run.sh)+会话存储+上传校验，pytest 绿、401/非法文件拦截可 curl 复现；待派发。（→ 分支 task/backend-core）
- F 前端：Apple 风格 ChatGPT 式 UI（登录口令/会话列表/上传/轮询/关键帧 Bento/prompt 卡片/占位视频/提交按钮禁用态），按既定 API 契约；待派发。（→ 分支 task/frontend）
- B 流水线：ffprobe 校验→抽 40 帧→codex exec 沙箱选帧写 prompt(dry-run)→产物白名单校验→ffmpeg 占位视频；依赖 A。（→ 分支 task/pipeline）
- C 预留提交接口：POST /submit 默认 501，开关+confirm+dry-run 复核门控齐全，不触 ARK_API_KEY；依赖 A。（→ 分支 task/submit-gate）
- D e2e+文档：样例视频全链路冒烟、README+docs/human+docs/agent；依赖 A/B/C/F 集成后。（→ 主线串行）

## 📌 待办与限制

<!-- 已知但当前不做：TODO、技术债、接受的边界，含审查 nits；升级即挪进「进行中」，做完/失效即删 -->

- Seedance 真实提交：接口仅预留(501)，ENABLE_SEEDANCE_SUBMIT=true+confirm 才启用；本阶段不触密钥。（→ app/seedance.py）
- 状态推送用轮询而非 SSE/WebSocket：当前阶段够用；量起来再升级。
