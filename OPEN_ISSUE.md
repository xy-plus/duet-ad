# OPEN_ISSUE

<!-- 活文档，只许这三节；每条 ≤3 行。完成即删，不保留编年史。 -->

## ⚠️ 待拍板

- 旧人物策略资料包仍在根目录，但不进入现行 Context IR → H3 运行链；需决定删除归档，还是只保留独立的授权/商标版权规则。（→ `seedance-cleaning-video-maker-中文完整资料包.zip`）

## 🚧 进行中

<!-- 当前无跨轮未完成实施项；新工作开工时登记，完成后删除。 -->

## 📌 待办与限制

- Context IR + H3 只能做高相似复刻，不能保证逐像素、逐帧、文字、原音或节奏一致；若目标升级为“一模一样”，需重新定义输入契约、模型与验收指标。（→ `temp/09-restore-h3-no-face/`, `docs/human/features/conversation-task/`）
- 从仓库根直接运行 `pytest -q` 会收集 `temp/09-*`、`temp/10-*` 的同名测试模块并冲突；规范命令暂为 `pytest tests -q`，后续应增加 pytest 收集隔离或重命名模块。（→ `temp/`）
- 全量测试仍有 FastAPI `on_event` 与 Starlette TestClient/httpx 弃用 warning；迁移 lifespan 和兼容测试客户端后再升级依赖。（→ `app/main.py`, `tests/`）
- 状态更新仍用 2 秒轮询，不用 SSE/WebSocket；规模和实时性要求上升后再升级。（→ `web/app.js`）
- Codex 认证只支持 `CODEX_HOME` 文件认证；服务会清除名称含 KEY/TOKEN/SECRET/PASSWORD 的 agent 子进程环境，环境变量 API key 路径有意不可用。（→ `app/codex_runner.py`）
- 当前宿主禁止 bubblewrap 所需的非特权 user namespace，故 CodexRunner 固定使用 0.147 的 legacy Landlock；该开关已弃用，升级 Codex 前必须验证替代沙箱。（→ `app/codex_runner.py`）
- TikTok 链接依赖 TikWM 与可选 `TIKTOK_PROXY`，第三方故障会影响该分支；普通受支持直链不依赖 TikWM。（→ `app/downloader.py`）
- `data/` 尚无 TTL/LRU 或磁盘水位清理，长期运行会持续增长。（→ `app/storage.py`）
- pipeline 闸等待占用 anyio worker thread；极端排队会拖慢同池下载/文件工作，正确性不受影响。（→ `app/main.py:run_pipeline_gated`）
- `/api/login` 未限流、note/target language 无长度上限、进程内锁字典不驱逐；量级和攻击面扩大时统一处理。（→ `app/main.py`）
