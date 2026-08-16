# OPEN_ISSUE

<!-- 活文档，只许这三节；每条 ≤3 行：一句现状 + 一句出路 + 指针（细节放 docs）；完成即删、不留 ✅ 编年史；全文超 ~60 行＝有条目该清 -->

## ⚠️ 待拍板

<!-- 需人决策才能推进；自愈终审失败件放这里，标 ⚠️需人确认 -->

## 🚧 进行中

<!-- 开工登记、集成即删；重开任务记重开链：继承未解 blockers 与已用轮次 -->

- video-maker skill 对齐：DoD=项目自用与 web/video-maker.zip 逐字节一致、pipeline 跑通新契约（4fps 抽帧/基本校验/生成位留空）、目录改名 video-maker、测试全绿、公网服务重启验证；当前阶段=实现。（→ 分支 video-maker-skill）

## 📌 待办与限制

<!-- 已知但当前不做：TODO、技术债、接受的边界，含审查 nits；升级即挪进「进行中」，做完/失效即删 -->

- Seedance 真实提交：接口仅预留(501)，ENABLE_SEEDANCE_SUBMIT=true+confirm 才启用；本阶段不触密钥。（→ app/seedance.py）
- 状态推送用轮询而非 SSE/WebSocket：当前阶段够用；量起来再升级。
- codex 认证只支持 CODEX_HOME 文件认证：env 清洗会杀 OPENAI_API_KEY/CODEX_API_KEY 等 env 认证路径（有意设计）。（→ app/codex_runner.py）
- TikTok 链接依赖第三方 TikWM API + TIKTOK_PROXY（境外），TikWM 故障时 TikTok 分支不可用、直链不受影响；新版 video-maker skill 已移除商标/版权硬规则（内容后处理归将来的第二个 skill）。（→ app/downloader.py, skills/video-maker/）
- data 目录无磁盘清理策略（LRU/TTL）：会话只进不出，量起来后需清理机制。
- 失败任务无重试按钮：TrendScout 有 retry 模式可参考；当前只能新建会话重传；pipeline copytree 无重入保护，加重试前需处理。
- "payload changed since review" 文案在新契约下语义已变（无评审 payload 可比对，实际=产物缺失/不可构建）：改文案连带冻结测试，留到下轮。（→ app/seedance.py）
- web/video-maker.zip 与 skills/video-maker/ 的字节一致性无 CI 防漂移：改 skill 后需手动重打包。（→ skills/video-maker/）
- 闸等待占线程池线程：管道闸阻塞的是 anyio 线程（默认 40），极端排队场景会拖慢 URL 下载/静态文件；正确性不受影响，量大再改异步闸。（→ app/main.py run_pipeline_gated）
- 审查 nits 汇总（R1/R2，不阻塞）：/api/login 无限流；_RateLimiter IP 条目只增不删、反代后共享一桶；save_upload async 内同步写盘；note 无长度上限；submit_locks 字典常驻内存；503 排在 dry-run 复核后；_SECRET_KEY_MARKERS 与 env 清洗口径不齐；_render_preview glob 不过滤前缀；has_video 磁盘探测与 meta 标记双源；URL 上传且无 note 时标题取整串 URL；限流先于幂等查重（重试消耗限流额度，有意外先撞 429）；幂等命中时首请求未落盘完成则随后 422 回滚会让重试方短暂 404。
