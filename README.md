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

### Ant Design X 交互原型

`web-next/` 是不连接后端的本地原型，用于评审 React + Ant Design X 重建后的会话式布局与主要操作。它不替换生产 `web/`，不发起 API 或 provider 请求。

```bash
cd web-next
npm install
npm run dev -- --host 127.0.0.1
```

## 生产契约摘要

- 新会话是 `schema_version=2`；`duration_s` 固定为 ffprobe `v:0` 视觉时长（缺失时依次用 `duration_ts*time_base`、视频包 PTS 范围，绝不使用 OpenCV `frame_count/fps` 元数据），音频或容器尾巴不参与 10/300 秒门禁。`≤10s` 保持原 Ref2VA 单请求；`>10s` 冻结为 provider 整秒时长不超过 10 秒的 FL2VA 首尾帧子任务。
- 原文保持使用本地 `whisper.cpp` multilingual small，只读取规范化音频并自动识别语言；改写/翻译仍使用音频专用隔离区。两条路径都看不到源视频、帧、OCR 或视觉 prompt。
- 输入准备只从结构化台词生成发声块。`auto` 同时保留 `spoken` 与 `sung`，无音轨是合法的空台词输入；MP3 编码尾部先按音频分析，再把最终台词裁到视频时间轴。OCR、字幕、画面文字和备注永远不能被提升为台词。
- 自动生成的 H3 源提示词可在首次 H3 attempt 创建前二次修改；保存后重写绑定 receipt，attempt 创建后即锁定。
- 生成前可选 `16:9/9:16` 画幅和 `480p/768p` 清晰度；服务端按真正 H3 输入帧推荐画幅、按源视频短边推荐清晰度。目标画幅不完全匹配时默认居中 `crop`，用户仍可改为黑边 `pad`；生成后继续展示服务端冻结的参数摘要。
- 短视频用 `prepared_input.json` 绑定输入；长视频用 `long_video_plan.json` 绑定完整源文件、分段、首尾锚点、提示词和台词摘要。客户端提交当前 `plan_receipt` 做 CAS，任何漂移都在付费前拒绝。
- 长视频在 `hard_cut` 处新建生成链；同链 `continue` 段以上一生成段尾帧为首帧、当前源片段末帧为目标尾帧。同链串行，不同链最多两条并发；这是连续性约束，不承诺逐帧无缝，也不是供应商原生 extend。
- 分段视觉提示词只声明“与源片段时长一致”，不携带浮点秒数；冻结 plan 和服务端整秒换算是时长唯一真源。历史 plan v1 的 11–15 秒 attempt 仅允许 GET 恢复，不能创建新付费 POST。
- 4fps 关键帧由 ffmpeg 按 `v:0` PTS 顺序批量解码，VFR 视频不依赖 OpenCV 毫秒随机 seek。所有 FL2VA 子片段去除自身音轨后由 ffmpeg 归一化、拼接。`auto` 以视频 presentation start 为零点，让解码器处理 AAC/Opus priming，并由音频时间戳确定前置静音或视频零点前裁剪，最后在画面终点裁剪或补静音；`none` 输出静音。两者都不改变画面时长，拼接失败只重跑本地拼接。
- Web 只调用 `POST /submit`。已知 task 的查询、超时、下载或输出故障只继续同一 attempt，不自动重复付费提交；长链只重做确定失败及其未完成下游段，成功段复用。FL2VA 原始输出允许不短于源段目标超过一帧、且不长于整秒请求 1 秒；最终拼接仍按源段帧预算精确裁补并校验全片时长。
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
- `web-next/`：Ant Design + Ant Design X 无后端交互原型；所有任务状态只存在浏览器内存。
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
