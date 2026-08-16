# OPEN_ISSUE

<!-- 活文档，只许这三节；每条 ≤3 行：一句现状 + 一句出路 + 指针（细节放 docs）；完成即删、不留 ✅ 编年史；全文超 ~60 行＝有条目该清 -->

## ⚠️ 待拍板

<!-- 需人决策才能推进；自愈终审失败件放这里，标 ⚠️需人确认 -->

## 🚧 进行中

<!-- 开工登记、集成即删；重开任务记重开链：继承未解 blockers 与已用轮次 -->

（空）

## 📌 待办与限制

<!-- 已知但当前不做：TODO、技术债、接受的边界，含审查 nits；升级即挪进「进行中」，做完/失效即删 -->

- Seedance 真实提交：接口仅预留(501)，ENABLE_SEEDANCE_SUBMIT=true+confirm 才启用；本阶段不触密钥。（→ app/seedance.py）
- 状态推送用轮询而非 SSE/WebSocket：当前阶段够用；量起来再升级。
- codex 认证只支持 CODEX_HOME 文件认证：env 清洗会杀 OPENAI_API_KEY/CODEX_API_KEY 等 env 认证路径（有意设计）。（→ app/codex_runner.py）
- TikTok 链接依赖第三方 TikWM API + TIKTOK_PROXY（境外），TikWM 故障时 TikTok 分支不可用、直链不受影响；商标/Logo 会被 skill 硬规则拒停（e2e 实测"Jeetee"锅被拒）。（→ app/downloader.py, skills/）
- 审查 nits 汇总（R1/R2，不阻塞）：/api/login 无限流；_RateLimiter IP 条目只增不删、反代后共享一桶；save_upload async 内同步写盘；note 无长度上限；submit_locks 字典常驻内存；503 排在 dry-run 复核后；_SECRET_KEY_MARKERS 与 env 清洗口径不齐；_render_preview glob 不过滤前缀；has_video 磁盘探测与 meta 标记双源；URL 上传且无 note 时标题取整串 URL。
