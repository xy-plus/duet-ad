# 视频工作室（duet-ad1）

FastAPI + 原生 Web 的参考视频复刻工具。生产生成链路是：本地输入准备 → AutoDL Art 的 MiniMax-H3 workflow → `generated.mp4`。H3 是视频模型名称，不是 HTTP/3；公网入口仍由 Caddy 以 HTTP/1.1、HTTP/2 提供。

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

## 生产契约摘要

- 新会话是 `schema_version=2`；源视频最长 300 秒，上传后以 ffprobe 实测时长做服务端门禁，超限会话不入队。`≤10s` 保持原 Ref2VA 单请求；`>10s` 冻结为 1–15 秒的 FL2VA 首尾帧子任务。
- 原文保持使用本地 `whisper.cpp` multilingual small，只读取规范化音频并自动识别语言；改写/翻译仍使用音频专用隔离区。两条路径都看不到源视频、帧、OCR 或视觉 prompt。
- 输入准备只从结构化台词生成发声块。`auto` 同时保留 `spoken` 与 `sung`，无音轨是合法的空台词输入；MP3 编码尾部先按音频分析，再把最终台词裁到视频时间轴。OCR、字幕、画面文字和备注永远不能被提升为台词。
- 自动生成的 H3 源提示词可在首次 H3 attempt 创建前二次修改；保存后重写绑定 receipt，attempt 创建后即锁定。
- 短视频提交可选 `auto/edit/custom/none`；长视频一期只接受源音频 `auto` 或静音 `none`。非 9:16 必须选择居中 `crop` 或黑边 `pad`。
- 短视频用 `prepared_input.json` 绑定输入；长视频用 `long_video_plan.json` 绑定完整源文件、分段、首尾锚点、提示词和台词摘要。客户端提交当前 `plan_receipt` 做 CAS，任何漂移都在付费前拒绝。
- 长视频在 `hard_cut` 处新建生成链；同链 `continue` 段以上一生成段尾帧为首帧、当前源片段末帧为目标尾帧。同链串行，不同链最多两条并发；这是连续性约束，不承诺逐帧无缝，也不是供应商原生 extend。
- 所有 FL2VA 子片段去除自身音轨后由 ffmpeg 归一化、拼接。`auto` 复用完整源音轨，`none` 输出静音；拼接失败只重跑本地拼接。
- Web 只调用 `POST /submit`。已知 task 的查询、超时、下载或输出故障只继续同一 attempt，不自动重复付费提交；长链只重做确定失败及其未完成下游段，成功段复用。
- `submission_unknown` 完全锁死，必须先到供应商侧核对。重启只对 `queued/running` generation 做 GET-only `resume`，不会补发供应商 POST。旧 schema 会话仍可查看，但提交和后处理均为只读。
- H3 成片只接受无 userinfo、全部预解析地址和实际 socket peer 均为公网的 HTTPS URL；拒绝重定向，流式下载最多 200 MiB，并在原子替换前用 ffprobe 验证正时长视频流。
- Seedream 的去字幕水印/去品牌后处理仍可选，但 H3 只读取原始关键帧或显式 `crop/pad` 派生帧，绝不读取 `postprocessed/`。
- `face_hold`、Seedance 生产提交路径和 MiniMax Context IR 接入均已删除；上线失败只沿直接 H3 链路修复。

## 生产拓扑

```text
public :3211 (Caddy, h1/h2) -> 127.0.0.1:3212 (uvicorn, single process)
                                     |
                                     +-> AutoDL Art H3
```

systemd 示例、预检、上线步骤和付费 smoke 保护见 [部署 runbook](.deploy/runbook.md)。

## 目录与文档

- `app/`：API、输入准备、长视频计划/编排/拼接、可恢复 H3 状态机、文件存储和可选 Seedream 后处理。
- `web/`：同源单页 UI；详情和 generation 均以 2 秒轮询刷新。
- `skills/video-maker/`：输入准备阶段的关键帧选择/视觉 prompt skill。
- `data/<cid>/`：会话文件、短链 receipt 或长链 plan receipt、每段 `.h3/` attempt 状态和最终视频。
- [功能与验收](docs/human/features/conversation-task/README.md)
- [运行架构](docs/agent/architecture/architecture.md)
- [API、schema 与配置参考](docs/agent/reference/reference.md)
- [部署 runbook](.deploy/runbook.md)
- [H3 正式 API smoke](.deploy/smoke-h3.sh)
- `OPEN_ISSUE.md`：仍未解决的限制；不是现行契约来源。

## 安全边界

`ACCESS_TOKEN` 必填，所有 `/api/conversations*` 和文件读取都需要 Bearer 鉴权。`AUTODL_ART_TOKEN` 以及可选 Seedream 凭据只从服务环境读取，不进入 API、meta、receipt 或日志。生产使用单进程 uvicorn；进程内锁、信号量和文件锁不是多 worker 协调机制。
