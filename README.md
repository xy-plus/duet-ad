# 视频工作室（duet-ad1）

FastAPI + 双前端的参考视频复刻工具。`web/` 是 3211 的原生旧前端，`web-next/` 是 3213 的 React 19 + Ant Design/Ant Design X 新前端；两者只消费同一个 FastAPI 和会话目录。生产生成链路是：本地输入准备 → AutoDL Art 的 MiniMax-H3 workflow → `generated.mp4`。H3 是视频模型名称，不是 HTTP/3；公网入口由 Caddy 以 HTTP/1.1、HTTP/2 提供。

## Quickstart

依赖 Python 3.12+、`ffmpeg`/`ffprobe`、发行版 `bwrap`；真实输入准备还需要已完成文件认证的 `codex` CLI。

```bash
python3 -m venv --without-pip .venv
pip3 --python .venv/bin/python install -r requirements.txt

# 本地开发；生产监听和密钥配置见 .deploy/runbook.md
ACCESS_TOKEN='<local-only-token>' HOST=127.0.0.1 PORT=3211 ./run.sh
```

```bash
.venv/bin/python -m pytest tests -q
```

### Ant Design X 前端

```bash
cd web-next
npm ci
npm run dev       # 本地组件开发；不代理真实 /api
npm run check     # TypeScript + ESLint + Stylelint + Vitest + build
npm run test:e2e  # Chromium 真实浏览器契约和截图基线
```

真实 API 始终使用同源相对路径 `/api`；本地 Vite 不提供后端代理，生产联调以 Caddy `:3213` 拓扑为准。`npm run preview` 只用于本地检查构建产物，不是生产服务器。

开发前先读 [web-next 约束](web-next/AGENTS.md)。组件只能从 [`src/ui/antd.ts`](web-next/src/ui/antd.ts) 导入，Token 与 Provider 只在 [`src/ui/theme.tsx`](web-next/src/ui/theme.tsx) 定义；ESLint、Stylelint 和 [`governance.test.ts`](web-next/src/governance.test.ts) 是不可绕过的治理入口。

## 生产契约摘要

- 新会话是 `schema_version=2`；`duration_s` 固定为 ffprobe `v:0` 视觉时长（缺失时依次用 `duration_ts*time_base`、视频包 PTS 范围，绝不使用 OpenCV `frame_count/fps` 元数据），音频或容器尾巴不参与 15/300 秒门禁。`≤15s` 保持单请求；`>15s` 冻结为单段不超过 14 秒的首尾帧生成子任务。
- 原文保持使用本地 `whisper.cpp` multilingual small，只读取规范化音频并自动识别语言；改写/翻译仍使用音频专用隔离区。两条路径都看不到源视频、帧、OCR 或视觉 prompt。
- 输入准备只从结构化台词生成发声块。`auto` 同时保留 `spoken` 与 `sung`，无音轨是合法的空台词输入；MP3 编码尾部先按音频分析，再把最终台词裁到视频时间轴。OCR、字幕、画面文字和备注永远不能被提升为台词。
- 自动生成的 H3 源提示词可在首次 H3 attempt 创建前二次修改；保存后重写绑定 receipt，attempt 创建后即锁定。
- 生成前可选 `16:9/9:16` 画幅和 `480p/768p` 清晰度；服务端按真正 H3 输入帧推荐画幅、按源视频短边推荐清晰度。目标画幅不完全匹配时默认居中 `crop`，用户仍可改为黑边 `pad`；生成后继续展示服务端冻结的参数摘要。
- 短视频用 `prepared_input.json` 绑定输入；长视频用 `long_video_plan.json` 绑定完整源文件、分段、首尾锚点、提示词和台词摘要。客户端提交当前 `plan_receipt` 做 CAS，任何漂移都在付费前拒绝。
- 新网页入口固定以“快速模式”新建长视频，确认页不显示开关或说明，生成结果参数摘要也不展示该模式：所有分段先完成本地 receipt/输入冻结，再有界并发 POST；`continue` 直接复用上一段已绑定且已适配的源末帧 bytes，因此不依赖上游成片。后端仍兼容 `fast_mode=false` 及已冻结的历史任务：关闭时 `hard_cut` 新建生成链，`continue` 以上一生成段真实尾帧为首帧，同链串行、不同链最多两条并发。两种模式都不承诺逐帧无缝，也不是供应商原生 extend。
- 分段视觉提示词只声明“与源片段时长一致”，不携带浮点秒数；冻结 plan 和服务端整秒换算是时长唯一真源。历史 plan v1 的 11–15 秒 attempt 仅允许 GET 恢复，不能创建新付费 POST。
- 4fps 关键帧由 ffmpeg 按 `v:0` PTS 顺序批量解码，VFR 视频不依赖 OpenCV 毫秒随机 seek。所有 FL2VA 子片段去除自身音轨后由 ffmpeg 归一化、拼接。`auto` 以视频 presentation start 为零点，让解码器处理 AAC/Opus priming，并由音频时间戳确定前置静音或视频零点前裁剪，最后在画面终点裁剪或补静音；`none` 输出静音。两者都不改变画面时长，拼接失败只重跑本地拼接。
- Web 只调用 `POST /submit`。已知 task 的查询、超时、下载或输出故障只继续同一 attempt；唯一自动新 POST 例外是供应商明确返回 `FAILED/ERROR/FAIL`，且上一 attempt 的 task id、receipt、诊断和同一 input receipt 已完整落盘。此时沿用原 `client_request_id`，按 `AUTO_RETRY_INTERVAL_S` 等待后新建顺序 attempt，累计不超过 `1 + AUTO_RETRY_COUNT`（默认总计 3 次 POST）。快速模式成功兄弟不重提；串行模式失败段成功后才推进下游。拼接重试只做本地工作，新增供应商 POST 为 0。FL2VA 原始输出允许不短于源段目标超过一帧、且不长于整秒请求 1 秒；最终拼接仍按源段帧预算精确裁补并校验全片时长。
- `submission_unknown` 完全锁死，必须先到供应商侧核对。快速模式把 unpaid `prepare`、单次 POST `submit` 与恢复推进分开：全部 child receipt 落盘后才允许第一笔 POST，结果未知的 child 不会二次 POST，已提交兄弟仍可完成 GET 收敛。重启恢复默认 GET-only；只对上述完整确认且有额度的 `h3_provider_failed` 创建下一 attempt，或提交该失败后已经落盘的 `ready_to_submit/h3.ready` 自动 attempt。旧 schema 会话仍可查看，但提交和后处理均为只读。
- H3 成片只接受无 userinfo、全部预解析地址和实际 socket peer 均为公网的 HTTPS URL；拒绝重定向，流式下载最多 200 MiB，并在原子替换前用 ffprobe 验证正时长视频流。
- 关键帧冻结后，隔离 Codex 按段执行 `skills/image-postprocess`，只读取本段关键帧与后端编辑模式，产出供用户审阅的真实 Seedream 提示词；不读取或复制 H3 提示词。关键帧后处理有三个独立选项，按每段 `MediaKit 文字/字幕擦除 → MediaKit Logo/图标擦除 → Seedream 图片优化` 执行阶段屏障；段之间并行，帧级请求受各供应商的全局信号量限制。短视频是逻辑段 `0`，长视频为 `1..N`。图片二次编辑是用户可选的生成式操作，生成提示词不代表一定执行、成功或被 H3 采用。完成后 H3 必须读取完整同名 `postprocessed/`，缺帧或任一段失败时拒绝提交而不回退原图。
- MediaKit 仅在完整 HTTP 429 + `RequestLimitExceeded` 明确未受理时按通用预算重试。Seedream 每帧最多 3 次 POST，只有完整 HTTP 429 + 精确 `QuotaExceeded` 且响应不含 `data` 时才重试；网络/超时、取消或不明响应记为 `submission_unknown`，启动恢复绝不自动重发。人工分段重试必须带 revision CAS，并保留旧 attempt 以提示潜在重复计费。
- `face_hold`、Seedance 生产提交路径和 MiniMax Context IR 接入均已删除；上线失败只沿直接 H3 链路修复。

## 生产拓扑

```text
public :3211 (Caddy, h1/h2) ── all paths ──> 127.0.0.1:3212
                                                   uvicorn, legacy web + /api

public :3213 (Caddy, h1/h2)
  ├─ /api/* ───────────────────────────────> 127.0.0.1:3212, path preserved
  └─ other paths ──────────────────────────> web-next/dist, SPA fallback

127.0.0.1:3212 ───────────────────────────> AutoDL Art H3
```

`:3211` 与 `:3213` 是不同 origin，浏览器登录状态互不共享；3213 复用同一个后端、数据和 Caddy unit，不新增 CORS、前端 service 或第二个 uvicorn。新前端已发布在 `https://8.166.140.227:3213`；预检、只读发布 smoke、回滚和付费保护见 [部署 runbook](.deploy/runbook.md)。

## 目录与文档

- `app/`：API、输入准备、长视频计划/编排/拼接、可恢复 H3 状态机、文件存储，以及可选 MediaKit → Seedream 关键帧后处理。
- `web/`：由 uvicorn 提供的现行原生单页 UI。
- `web-next/`：3213 的 React/Ant Design X 生产 UI；TanStack Query 只在运行态详情每 2 秒轮询，业务组件受组件门面与 Token/CSS 门禁约束。
- `skills/video-maker/`：输入准备阶段的关键帧选择/视觉 prompt skill。
- `data/<cid>/`：会话文件、短链 receipt 或长链 plan receipt、每段 `.h3/` attempt 状态和最终视频。
- [功能与验收](docs/human/features/conversation-task/README.md)
- [Ant Design X 用户行为](docs/human/features/antd-x-frontend/README.md)
- [运行架构](docs/agent/architecture/architecture.md)
- [Ant Design X 前端架构](docs/agent/architecture/antd-x-frontend.md)
- [API、schema 与配置参考](docs/agent/reference/reference.md)
- [web-next 模块、API 与治理参考](docs/agent/reference/web-next.md)
- [部署 runbook](.deploy/runbook.md)
- [H3 正式 API smoke](.deploy/smoke-h3.sh)
- `OPEN_ISSUE.md`：仍未解决的限制；不是现行契约来源。

## 安全边界

`ACCESS_TOKEN` 必填，所有 `/api/conversations*` 和文件读取都需要 Bearer 鉴权。3213 的 token 只属于该 origin；任一请求返回 401 时新前端会清除 token 和 Query cache。`AUTODL_ART_TOKEN`、`VOLC_MEDIAKIT_API_KEY` 与 `ARK_API_KEY` 只从服务环境读取，不进入公开 API、meta 或日志；供应商回执与中间产物只存于私有目录。生产使用单进程 uvicorn；进程内锁、信号量和文件锁不是多 worker 协调机制。
