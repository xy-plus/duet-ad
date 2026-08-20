# 视频工作室（duet-ad1）

FastAPI + 原生 Web 的参考视频复刻工具。生产生成链路是：本地输入准备 → MiniMax Context IR → AutoDL Art 的 MiniMax-H3 workflow → `generated.mp4`。H3 是视频模型名称，不是 HTTP/3；公网入口仍由 Caddy 以 HTTP/1.1、HTTP/2 提供。

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

- 新会话是 `schema_version=2`；源视频最长 15 秒。`10 < duration_s <= 15` 合法，但界面提示稳定性 warning。
- 原文保持使用本地 `whisper.cpp` multilingual small，只读取规范化音频并自动识别语言；改写/翻译仍使用音频专用隔离区。两条路径都看不到源视频、帧、OCR 或视觉 prompt。
- 输入准备只从结构化台词生成发声块。`auto` 同时保留 `spoken` 与 `sung`，无音轨是合法的空台词输入；MP3 编码尾部先按音频分析，再把最终台词裁到视频时间轴。OCR、字幕、画面文字和备注永远不能被提升为台词。
- Context IR 返回的全部 `<d>...</d>` 台词必须与冻结台词在数量、顺序和文本上精确一致；少、多、改写或 OCR 衍生发声都会在 H3 提交前被拒绝。
- 页面先显式提交 Context IR，再展示完整优化提示词；用户可带 SHA-256 保存二次修改，校验通过后才开放 H3 生成。成片存在后仍可查看 IR。
- 提交时显式选择 `auto/edit/custom/none` 和画幅策略。非 9:16 必须选择居中 `crop` 或黑边 `pad`；实际时长保留在 receipt，供应商请求时长为 `ceil(duration_s)` 秒。
- `prepared_input.json` 绑定源文件、可选音频、关键帧、视觉 prompt、最终 prompt、台词与引擎参数的哈希。文件或台词漂移即拒绝提交/恢复。
- Web 先调用 `POST /context-ir`，确认/编辑 IR 后才调用 `POST /submit`。已知 task 的查询、超时、下载或输出故障只继续同一 attempt，不自动重复付费提交。
- `submission_unknown` 完全锁死，必须先到供应商侧核对。重启只对 `queued/running` generation 做 GET-only `resume`，不会补发供应商 POST。旧 schema 会话仍可查看，但提交和后处理均为只读。
- H3 成片只接受无 userinfo、全部预解析地址和实际 socket peer 均为公网的 HTTPS URL；拒绝重定向，流式下载最多 200 MiB，并在原子替换前用 ffprobe 验证正时长视频流。
- Seedream 的去字幕水印/去品牌后处理仍可选，但 H3 只读取原始关键帧或显式 `crop/pad` 派生帧，绝不读取 `postprocessed/`。
- `face_hold` 与 Seedance 生产提交路径已删除；上线失败只沿 Context IR → H3 修复，不回退 Seedance。

## 生产拓扑

```text
public :3211 (Caddy, h1/h2) -> 127.0.0.1:3212 (uvicorn, single process)
                                     |
                                     +-> MiniMax Context IR
                                     +-> AutoDL Art H3
```

systemd 示例、预检、上线步骤和付费 smoke 保护见 [部署 runbook](.deploy/runbook.md)。

## 目录与文档

- `app/`：API、输入准备、可恢复 H3 状态机、文件存储和可选 Seedream 后处理。
- `web/`：同源单页 UI；详情和 generation 均以 2 秒轮询刷新。
- `skills/video-maker/`：输入准备阶段的关键帧选择/视觉 prompt skill。
- `data/<cid>/`：会话文件、冻结 receipt、`.h3/` attempt 状态和最终视频。
- [功能与验收](docs/human/features/conversation-task/README.md)
- [运行架构](docs/agent/architecture/architecture.md)
- [API、schema 与配置参考](docs/agent/reference/reference.md)
- [部署 runbook](.deploy/runbook.md)
- [H3 正式 API smoke](.deploy/smoke-h3.sh)
- `OPEN_ISSUE.md`：仍未解决的限制；不是现行契约来源。

## 安全边界

`ACCESS_TOKEN` 必填，所有 `/api/conversations*` 和文件读取都需要 Bearer 鉴权。`MINIMAX_API_KEY`、`AUTODL_ART_TOKEN` 以及可选 Seedream 凭据只从服务环境读取，不进入 API、meta、receipt 或日志。生产使用单进程 uvicorn；进程内锁、信号量和文件锁不是多 worker 协调机制。
