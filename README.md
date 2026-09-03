# Duet H3 Max R2V Research

这是从 `github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf` 的极简前端出发建立的独立研究仓库。它不依赖 `/home/xy/duet-ad1` 的运行目录，也不修改源仓库。

后端使用 Python 3.12、FastAPI 和纯文件系统存储。上游保持固定 Master 的视频分析语义；`video-maker` 之后改走逐段 DeepSeek 提示词 Skill、Picsart `minimax-h3-max-r2v`，最后裁回源片段时长并合并。

## 研究版边界

- 不增加媒体质量、替换成功率、人物一致性、台词、口型、分辨率或成片可播放性门禁。
- 不增加模型输出质量重试、CAS、幂等冲突、恢复状态机或旧生产回执框架。
- 只保留鉴权、解析、项目与路径归属、SSRF、命令参数隔离、请求大小、供应商必填组装和超时等技术边界。
- 每个执行节点首次失败后最多重试一次。Picsart submit 在未取得任务 ID 的未知结果下也会重提；研究方案明确接受可能重复付费的风险。
- 日志故意保存未脱敏的请求、响应、鉴权信息、签名 URL 和媒体副本。日志仅存服务器本地，不通过 API 提供，目录权限为 `0700`、文件权限为 `0600`。
- 当前实现只支持单进程实例。后台任务、签名素材 token 与文件状态不支持多 worker 共享。

## 链路

```text
原视频 4 fps / 音频 / YAMNet / Whisper
  -> scene-first 4-10 秒分段
  -> 每段 scene-aware exact-9
  -> MediaKit 可选去字幕、去标志
  -> video-maker 逐段视觉描述
  -> 项目级台词翻译
  -> video-maker 项目级对象索引
  -> 各段并行：新提示词 Skill -> H3 Max R2V -> 裁回源时长
  -> 顺序合并 generated.mp4
```

`video-maker` 的两个历史阶段保留固定 JSON Schema 的 `submit_result` 调用，因为这是 Skill 自身格式合同。新 H3 提示词 Skill 输出纯文本；后端只机械附加翻译台词及冻结时间窗。

## 配置

参见 `/home/xy/duet-ad1-h3-max-r2v/.env.example`。至少需要显式提供：

- `ACCESS_TOKEN`：前端登录 Bearer token；禁止使用缺省或空值。
- `DEEPSEEK_API_KEY`、`PICSART_API_KEY`。
- 启用 MediaKit 时提供 `VOLC_MEDIAKIT_API_KEY`。
- `PUBLIC_BASE_URL`：Picsart 能从公网访问的 HTTPS 服务根地址，不能使用 loopback 地址。
- 可执行的 `ASR_CLI`、存在的 `ASR_MODEL` 和 `YAMNET_MODEL_PATH`。

系统还需要 `ffmpeg` 与 `ffprobe`。Python 依赖安装后，可单进程启动：

```bash
python3 /home/xy/duet-ad1-h3-max-r2v/run.py
```

不要用 `uvicorn --workers` 或并行启动第二个实例指向同一 `DATA_DIR`。

## API 与存储

前端使用登录、能力声明、项目列表/创建/详情、项目媒体和替换图七组 API。成功项目在同一状态快照中返回 `project_progress.percent=100` 与 `has_video=true`；前端每两秒轮询，无 SSE/WebSocket。

```text
/home/xy/duet-ad1-h3-max-r2v/data/projects/{project_id}/
  state.json
  inputs/
  work/segments/{segment_id}/
  output/generated.mp4
  logs/
```

重启时遗留的 `queued/running` 项目会失败为 `process_restarted`，不做断点恢复。通用文件接口只暴露源视频和最终视频，不能读取状态、日志或中间文件。

## 验证

离线测试使用假 DeepSeek、MediaKit 和 Picsart，不产生付费任务：

```bash
python3 -m pytest /home/xy/duet-ad1-h3-max-r2v/tests -q
```

真实供应商冒烟测试需要另行明确授权，并且必须先配置可公网回源的 `PUBLIC_BASE_URL`。
