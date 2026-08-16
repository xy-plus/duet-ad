# 视频工作室（duet-ad1）

ChatGPT 式单页应用（Apple 风格）+ FastAPI 后端：用户上传参考视频（任意类别），后台由沙箱化的本地 codex agent 按 `skills/video-maker` 处理，产出 ≤9 张关键帧 + Seedance prompt；预留 Seedance 真实提交接口（默认 501 未开放）。

## Quickstart

依赖：Python 3.12+、`ffmpeg`/`ffprobe` 在 PATH 上；跑真实流水线还需 `codex` CLI（0.147.0 实证基线，CODEX_HOME 文件认证，见下文）。

```bash
# 本机 python3 缺 ensurepip，必须 --without-pip 建 venv，再用系统 pip3 往里装
python3 -m venv --without-pip .venv
pip3 --python .venv/bin/python install -r requirements.txt

# 起服务（单进程 uvicorn，默认 0.0.0.0:3211；HOST/PORT 可覆盖）
ACCESS_TOKEN=<访问口令> ./run.sh
# 浏览器打开 http://localhost:3211，输入口令进入
```

跑测试（不需要 codex；需要 ffmpeg 生成样例视频）：

```bash
.venv/bin/python -m pytest tests -q
```

## 结构

```
app/            FastAPI 后端：main(路由) config auth storage downloader pipeline codex_runner seedance seedance_task(提交脚本)
web/            单页前端（原生 JS，同源 /api/*，由后端 StaticFiles 挂载在 /）
skills/video-maker/   codex agent 使用的技能（SKILL.md + extract_keyframes/crop_image 两脚本）
tests/          pytest（conftest 直建 Settings，默认 enable_pipeline=False 不跑流水线）
data/           运行期产物，每会话一目录（见 docs/agent/architecture）
run.sh          uvicorn 启动脚本（优先 .venv/bin/uvicorn）
OPEN_ISSUE.md   待拍板/进行中/待办与限制（活文档）
```

## 文档地图

- 人要什么、验收、行为规则：`docs/human/features/conversation-task/`
- 怎么实现（模块/数据流/状态机/数据布局/安全模型/配置表）：`docs/agent/architecture/architecture.md`
- 接口契约（API 全字段/状态码/门控矩阵/校验链）：`docs/agent/reference/reference.md`
- 当前未决与已知限制：`OPEN_ISSUE.md`

## 开发须知（最易踩的坑）

- `ACCESS_TOKEN` 必填，缺了 `get_settings()` 直接 RuntimeError；所有 `/api/conversations*` 走 `Authorization: Bearer <token>`。
- `ENABLE_PIPELINE`：`get_settings()`（生产）默认开；直建 `Settings`（测试）默认关——测试不会真跑流水线。
- codex 认证只支持 CODEX_HOME 文件认证：调起 codex 前进程级 env 清洗会剔除名字含 KEY/TOKEN/SECRET/PASSWORD 的变量，`OPENAI_API_KEY` 类 env 认证路径不可用（有意设计，原因见 architecture 安全模型）。
- 改 API 响应字段先读 `docs/agent/reference/reference.md`：detail 是冻结的 13 字段契约，meta.json 内部字段（submitted_at/task_id 等）不外泄。
