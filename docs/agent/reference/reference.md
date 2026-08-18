---
name: backend-api
type: reference
status: done
owner: agent
updated: 2026-08-18
links: [conversation-task, app/main.py, app/storage.py, app/downloader.py, app/pipeline.py, app/codex_runner.py, app/seedance.py, app/seedance_task.py, app/seedream.py, app/seedream_task.py, app/postprocess.py, app/sanitize.py, app/vocal.py]
---

# backend-api · 接口（How/Now）

## 公开接口（HTTP）

除标注外均需 `Authorization: Bearer <ACCESS_TOKEN>`，失败一律 401 `{"detail":"unauthorized"}`（口令比对用 `hmac.compare_digest`）。

### `GET /api/health`（无鉴权）

- 200 → `{"ok": true}`

### `POST /api/login`（无鉴权）

- 请求：`{"token": "<口令>"}`（JSON）
- 200 → `{"ok": true}`；口令错 → 401 `{"detail":"invalid token"}`

### `GET /api/conversations`

- 200 → 数组，按 `created_at` 倒序，每项恰 6 字段：`id, title, note, status, created_at, has_video`
- `has_video` 为磁盘实况探测（`generated.mp4` 是否存在），不读 meta

### `POST /api/conversations`

- 请求：multipart，`file` 与 `reference_url`（http(s) 直链 / TikTok 视频页）恰好一个 + `note`（可选，默认 `""`）+ `client_request_id`（可选幂等键，格式 `^[0-9A-Za-z-]{8,64}$`，空 = 不参与幂等）+ `voice_mode`（`none|keep|rewrite|translate`，默认 `none`）+ `target_language`（仅 `translate` 时必填非空，其余模式忽略）。Web 端恒传 `keep/rewrite/translate` 之一（默认 `keep`），恒触发音轨探测；API 直接调用默认 `none`（不探测音轨，兼容 curl 等存量用法）
- 201 → `{"id": "<32位hex>", "status": "queued"}`；`enable_pipeline` 开时登记后台流水线（经管道闸，见架构）
- 200 → `{"id", "status"}`：`client_request_id` 命中既有会话 meta（扫 meta.json 查重），不建目录、不重复入队
- 400 `provide exactly one of file or reference_url`：`file`/`reference_url` 都不给或都给
- 400 `invalid client_request_id`：幂等键非空但不合格式
- 422 `invalid voice_mode: <v>`：`voice_mode` 不在枚举
- 422 `target_language required for translate`：`translate` 模式缺目标语言（strip 后为空）
- 429 `too many uploads`：同 IP 1 分钟超 10 次（内存滑动窗口，进程重启清零）
- 429 `too many queued tasks`：`queued` 状态会话数达 `MAX_QUEUED`（processing/done/failed 不计；查重+计数+建目录在同一把锁内，无竞态）
- 422：上传/下载校验链失败（`UploadError`/`DownloadError`），detail 为具体原因；失败即回滚删除整个会话目录

上传校验链（顺序执行，`storage.save_upload` → `storage.probe_video` → [口播模式] `storage.probe_audio`）：

1. 扩展名 ∈ `{.mp4, .mov, .webm}`（小写化），否则 `unsupported extension: <ext>`
2. 流式落盘（1MB 块），累计超 `MAX_UPLOAD_MB*1024*1024` → `file exceeds <n> bytes`，已写部分删除
3. ffprobe 实探（30s 超时）：打不开 → `ffprobe failed: ...` / `unreadable video file`；时长解析失败 → `cannot parse video duration`；时长超 `MAX_DURATION_S` → `duration <d>s exceeds <n>s`
4. `voice_mode != none` 时音轨探测（`ffprobe -select_streams a`，30s 超时）：无音轨 → 422 `no audio track in video` + 回滚；探测故障 → `ffprobe failed: ...` / `unreadable video file` / `cannot parse audio probe result` + 回滚

注：`voice_mode`/`target_language`（仅 translate）在建目录时（`new_conversation`，早于本校验链）即落 meta 内部字段，不进 detail 响应；校验失败回滚删除整个会话目录

URL 分支（`downloader.fetch_reference`，线程池执行不堵事件循环）：TikTok 视频页先经 TikWM API 解析出 play 直链；下载带 SSRF 防护（见架构安全模型），任一步失败（含解析到私网、HTTP 非 2xx、超限/超时/空文件、连接/读取异常归一）抛 `DownloadError` → 422 + 回滚；落盘 `source.<ext>`（后缀取 URL path，白名单外默认 `.mp4`）后与文件分支汇合同一 `probe_video`

### `GET /api/conversations/{cid}`

- 200 → **冻结的 16 字段契约**（显式键，meta 内部字段不外泄）：

| 字段 | 来源 |
| --- | --- |
| `id, title, note, status, error, created_at, updated_at` | meta.json |
| `keyframes` | meta.json（字符串数组，缺省 `[]`；多段模式保持 `[]`） |
| `prompt` | meta.json（缺省 `null`；多段模式保持 `null`） |
| `segments` | meta.json（缺省 `[]`；多段模式为各段产物数组，见关键数据形） |
| `voice_lines` | meta.json（缺省 `[]`；口播模式为全片台词数组） |
| `has_source` | `source.*` 磁盘探测 |
| `has_video` | `generated.mp4` 磁盘探测 |
| `submit_enabled` | `settings.enable_seedance_submit` |
| `postprocess` | meta.json `postprocess` 字段（缺省 `null`；后处理启动后回传 `{status, options, frames, error}`，running 状态也回传——前端靠它轮询） |
| `postprocess_enabled` | `settings.enable_seedream_edit` |

- 404 `not found`：cid 非法（非 `^[0-9a-f]{32}$`）或目录/meta 不存在

### `GET /api/conversations/{cid}/files/{name:path}`

- 200 → `FileResponse`
- files 白名单（`storage.resolve_file`，此外一律 404）：`source.mp4`（映射唯一的 `source.*`，扩展名不定）、`preview.mp4`、`generated.mp4`、`contact_sheet.jpg`（映射 `work/contact_sheet.jpg`）、`keyframes/<fn>`（映射 `work/keyframes/<fn>`）、`postprocessed/<fn>`（映射 `work/postprocessed/<fn>`）、`segments/<N>/work/keyframes/<fn>` 与 `segments/<N>/work/postprocessed/<fn>`（映射 `work/segments/<N>/work/` 下同名目录；`<N>` 为正整数，`<fn>` 必须是不含路径的纯文件名）
- 防御：cid 正则校验 + `resolve()` 后必须 `is_relative_to` 会话目录且是文件；symlink 越界/穿越一律 404

### `POST /api/conversations/{cid}/postprocess`（T5b，默认 501）

- 请求：JSON `{"options": {"face_hold": bool, "remove_subtitle": bool, "remove_brand": bool}, "confirm": true}`；三选项至少一个为真，否则 422 `at least one option required`；选项值非 bool → 422 `options must be booleans`；未知键忽略（旧客户端残留的 `change_bg` 静默丢弃）
- 成功 200 → `{"status": "running", "frames": []}`（受理即返回；后台并行逐帧编辑，结果经 detail 的 `postprocess` 字段轮询，复用 2s 轮询；终态 `{status: "done"|"failed", options, frames, error}`）
- 门控矩阵（`postprocess.start`，**固定顺序**）：

| 序 | 条件 | 状态码 / detail |
| --- | --- | --- |
| 1 | `enable_seedream_edit` 关 | 501 `Seedream edit is disabled.` |
| 2 | 会话不存在 | 404 `not found` |
| 3 | `confirm` 不为 `true` | 409 `confirmation required` |
| 4 | `options` 非对象/三选项全假 | 422 `at least one option required`；选项值非 bool → 422 `options must be booleans` |
| 5 | `status != "done"` | 409 `artifacts not ready` |
| 6 | `meta.postprocess.status` 已 `running` | 409 `already running` |
| 7 | 上次 `done/failed` 的 `options` 与本次不同 | 409 `options changed since last run`（防旧产物贴新标签；同选项重跑照常跳过已有图；锁定比对只认当前三选项与上次 options 的共有键——旧会话四键 options 里的废弃 `change_bg` 键忽略；纯废弃形态（上次在当前键上无任何 True，如旧版只勾 `change_bg`）不 409，放行重跑并清除旧 postprocessed 产物强制全帧重编辑） |
| 8 | 每会话锁内复查：meta 消失或已在 running | 409 `already running` |
| 9 | 锁内复查：目标帧目录缺失/为空 | 409 `artifacts not ready` |
| 10 | 后台逐帧执行中任一帧失败 | 受理后 meta 落 `postprocess.status=failed`，`error` 列失败帧名（其余帧照常跑完；已成功帧保留） |

- 后台任务（`postprocess.run_task`，BackgroundTasks，独立路径不吃管道闸；每会话一把锁管门控，逐帧编辑经进程级信号量 `SEEDREAM_CONCURRENCY` 限并发——单个 uvicorn 进程内跨会话共享，多 worker 部署时每进程独立限额）：
  - 收集目标帧：单段 = `work/keyframes/*.png`；多段 = `work/segments/N/work/keyframes/*.png`（N 来自 `meta.segments`）
  - 每帧按勾选选项构造中文编辑指令（多选项用 `；` 连接）：含人脸遮挡「如果图片中含有人脸：将图片中的人物改为用手捂住脸的造型。如果图片中不含人脸：跳过捂脸处理，仅执行其余修改。」（条件式指令，由 Seedream 自己判断有无脸；多选项时条件句放最前）；去字幕水印「移除图片中的所有字幕、水印和贴纸元素，其余（尺寸、内容等）保持不变」；去版权物品「图片中的所有品牌标志、logo、商标等版权元素改为不侵权的类似视觉效果的等效物，其余（尺寸、内容等）保持不变」
  - 所有目标帧都发编辑请求（不做人脸预判/过滤）：无人脸帧由 Seedream 按条件指令输出近似原图，直接存 postprocessed 展示；将来可加输出-输入变化判定过滤（见 OPEN_ISSUE）
  - 未跳过帧经 `asyncio.gather(return_exceptions=True)` **并行**调用 `seedream.edit_image(..., confirm=True, size=...)`（路由层已校验 confirm；`size` 为按帧实际像素等比放大到 ≥3,686,400 像素的 `"WxH"`——如 720×1280 → `1440x2560`——不传时模型默认 2048 方形、方向可能失真；cv2 读不出的帧不传 size）；产出 `work/postprocessed/<帧名>.png`（单段）或 `work/segments/N/work/postprocessed/<帧名>.png`（多段）；已存在的输出跳过（重跑不重复扣费）
  - 任一帧失败 → 整体 failed（`error` 列失败帧名，脱敏 ≤300 字）；其余帧照常跑完，已成功帧保留；`meta.postprocess.frames` 记有优化版的帧名列表（单段 = 帧名；多段 = `segments/N/work/postprocessed/帧名` 全形路径，与 files 白名单路径同形，前端按段前缀过滤展示；终态列表为字典序 `sorted(frames)`——段 ≥10 时 `segments/10` 排 `segments/2` 前，前端按段前缀过滤不受影响——与并发完成顺序无关）
  - running 期间每成功一帧即写回 `frames`（单协程 gather 回调串行执行，无写竞态），前端 2s 轮询据此显示 n/m 实时进度
- 幂等：`meta.postprocess.status == "running"` 时再提交一律 409；锁常驻内存，进程重启即失效；done/failed 后可重跑（已有优化图跳过）

### `POST /api/conversations/{cid}/submit`（预留，默认 501）

- 请求：JSON，仅接受 `{"confirm": true}`；不接受任何 prompt/参数覆盖（提交时由 `work/prompt.txt` + 关键帧现构建请求）
- 成功 200 → `{"status": "succeeded", "video": "generated.mp4"}`，并回写 meta：`has_video/submitted_at/task_id`（内部字段，不进响应）
- 门控矩阵（`seedance.submit`，**固定顺序**）：

| 序 | 条件 | 状态码 / detail |
| --- | --- | --- |
| 1 | `enable_seedance_submit` 关 | 501 `Seedance submission is disabled.` |
| 2 | 会话不存在 | 404 `not found` |
| 3 | `confirm` 不为 `true` | 409 `confirmation required` |
| 4 | `status != "done"` | 409 `artifacts not ready` |
| 5 | `has_video` 已真 | 409 `already submitted` |
| 6 | 每会话锁内复查：meta 消失或 `has_video` 已真 | 409 `already submitted` |
| 7 | `work/prompt.txt` 缺失或为空 | 409 `payload changed since review` |
| 8 | `work/keyframes/` 下无 PNG | 409 `payload changed since review` |
| 9 | dry-run 预检（120s 超时）构建 payload 失败或未落 payload-out | 409 `payload changed since review` |
| 10 | 服务进程无 `ARK_API_KEY` | 503 `ARK_API_KEY not configured` |
| 11 | 真实提交超时（1800s）/ 执行器不可用 / 非零退出 | 502 `seedance task timed out` / `seedance runner unavailable` / 脱敏后输出（≤300 字） |

- 提交 argv：提交时由当前 `work/prompt.txt` + `work/keyframes/*.png` 现构建 `app/seedance_task.py create --confirm-submit --wait --state-file work/task.json --download generated.mp4`，建模固定 `doubao-seedance-2-0-260128 / 9:16 / 15s / 720p / --generate-audio / --no-watermark`；argv 列表、无 shell、env 缺省继承服务进程（`ARK_API_KEY` 由此进入脚本）；T5b：每张帧若存在 `work/postprocessed/<同名>` 优化图则 `--ref-images` 优先用优化图（dry-run 预检同一 argv；多段暂不涉及——seedance 提交仅支持单段 work/ 契约）
- 幂等：成功后 `has_video=true` 落盘，重复/并发提交一律 409；锁常驻内存，进程重启即失效

## 公开接口（Python 模块）

- `app.config.get_settings() -> Settings` — 读环境变量建配置；缺 `ACCESS_TOKEN` 抛 RuntimeError
- `app.config.Settings` — frozen dataclass，14 字段（见架构配置表）；直建默认 `enable_pipeline=False`
- `app.auth.require_auth(request, cred)` — FastAPI 依赖，Bearer 校验
- `app.main.create_app(settings) -> FastAPI` — 应用工厂（测试注入用）；模块级 `app = create_app(get_settings())`
- `app.storage.new_conversation(data_dir, note, orig_name, client_request_id="", voice_mode="none", target_language="") -> dict` — 建目录 + 初始 meta（status=queued）；幂等键/目标语言非空才落 meta，`voice_mode` 恒落
- `app.storage.probe_audio(path) -> bool` — `ffprobe -select_streams a` 探测音轨；探测失败抛 `UploadError`
- `app.voice.extract_audio(cdir) -> Path | None` — ffmpeg 抽音轨为 work/voice.mp3；无音轨 → None；失败 → PipelineError
- `app.vocal.analyze(audio) -> VocalAnalysis` — ffmpeg 解码 16kHz f32le → YAMNet 逐窗推理 → `VocalAnalysis(windows, has_bgm)`；模型 sha256 校验不符/解码失败 → `VocalError`
- `app.vocal.classify_segment(start_ms, end_ms, windows) -> "sung" | "spoken" | None` — 台词片段与声学窗口重叠加权判定；None = 无人声证据（假转录，调用方丢弃）
- `app.voice.validate_voice_lines(raw, duration_s) -> list[dict]` — 台词 JSON 白名单校验（raw ≤ 32KB、条目 ≤ 200、每行 text ≤ 500 字、text/start_s/end_s 三字段、时间单调且落在时长内）；返回净化列表
- `app.storage.update_meta(data_dir, cid, **changes) -> dict | None` — 合并写字段并刷新 `updated_at`
- `app.storage.load_meta(data_dir, cid) -> dict | None` — cid 正则不过/文件缺 → None
- `app.storage.list_conversations(data_dir) -> list[dict]` — 扫描合法目录，按 `created_at` 倒序
- `app.storage.remove_conversation(data_dir, cid)` — 回滚删目录（cid 校验）
- `app.storage.mark_submitted(data_dir, cid, task_id) -> dict` — 提交标记回写（供幂等门控）
- `app.storage.save_upload(cdir, upload, max_bytes) -> Path` — 流式落盘 `source.<ext>`；`UploadError` 由 HTTP 层转 422
- `app.downloader.fetch_reference(url, cdir, settings) -> Path` — URL 分流（TikTok 经 TikWM / http(s) 直链）下载落盘 `source.<ext>`；`DownloadError` 由 HTTP 层转 422
- `app.storage.probe_video(path, max_duration_s) -> float` — ffprobe 探测，返回时长
- `app.storage.resolve_file(data_dir, cid, name) -> Path | None` — files 白名单解析
- `app.pipeline.run(settings, cid, runner)` — 后台任务入口；任何失败 → `failed`+`error`，不抛
- `app.pipeline.validate_work_dir(work) -> (list[str], str)` — agent 产物白名单校验，返回 (关键帧名, prompt)
- `app.pipeline.attribute_lines(lines, segments) -> dict[int, list[dict]]` — 台词按 start_s 落入段 [start_s, end_s) 归段（恰在边界归后段；超出末段终点 ≤0.01s 浮点误差归末段，更远不归段），返回 {index: [台词]}
- `app.seedream.edit_image(settings, cdir, image, prompt, out, confirm, size="") -> Path` — 编辑门控纯函数：开关 + confirm 门控（confirm 须严格 True）+ 入口 out 已存在 409 + dry-run 预检 + 真实提交；不自带并发锁（同 out 仅一次提交由 postprocess 编排保证）；`size`（`"WxH"`，可选）非空时 dry-run 与真实提交 argv 均带 `--size`；失败抛 `SeedreamError(status, detail)`
- `app.postprocess.start(settings, cid, payload, locks) -> dict` — 后处理门控（换选项重跑 409）+ 置 `meta.postprocess=running`；返回勾选选项（路由层据此调度后台任务）；失败抛 `PostprocessError(status, detail)`
- `app.postprocess.run_task(settings, cid, options, sem)` — 后处理后台任务：收集目标帧、条件式指令构造、未跳过帧 gather 并行 `seedream.edit_image(confirm=True, size=按帧像素等比放大的 "WxH")`（`sem` 为进程级信号量）、写 `meta.postprocess` 终态（含父任务取消时写 failed(error=cancelled) 后继续传播取消）；不抛
- `app.postprocess._fit_size(w, h) -> str` — 输出尺寸落在 Ark 合法面积区间 [3,686,400, 4,624,220]：原图已在区间 → 原尺寸不动；否则等比缩放（浮点 scale 非整数倍，`scale = sqrt(3686400/(w*h))`）到下限附近，round 亏空加宽一维补足，返回 `"WxH"`（如 720×1280 → `1440x2560`、1080×1920 → `1440x2560`；整数 ceil 倍会超上限，实测 400）
- `app.pipeline.FACE_HOLD_CONDITION_LINE` — 捂脸配套条件动作行，后端机械加进 prompt（所有模式；多段模式排在「不要生成背景音乐」行之后）
- `app.sanitize.sanitize(text, limit=300) -> str` — 公共脱敏（seedance/seedream/postprocess 共用）：删含 key|authorization 的行 + 抹 `ARK_API_KEY` 字面值 + 截断
- `app.codex_runner.CodexRunner(timeout_s, concurrency)` — `.build_argv(workdir, prompt)` / `.run(workdir, prompt)`；`CodexError` 包装超时/非零/找不到二进制
- `app.codex_runner.clean_stderr(text, limit=500)` — 剔环境变量行 + 截断（pipeline 的 `_run_cmd` 复用）
- `app.seedance.submit(settings, cid, payload, locks) -> dict` — 门控 + 执行；`SubmitError(status, detail)` 由 HTTP 层转响应

## 关键数据形

meta.json（`data/<cid>/meta.json`）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | str | uuid4 hex，32 位小写 |
| `title` | str | `note` 或净化文件名（去路径/控制字符，≤80 字，空则 `untitled`） |
| `note` | str | 用户备注（可无长度上限，见 OPEN_ISSUE nits） |
| `status` | str | `queued/processing/done/failed` |
| `error` | str \| null | 失败原因，≤500 字 |
| `created_at` / `updated_at` | str | ISO8601 UTC |
| `keyframes` | list[str] | 关键帧文件名（done 后写入） |
| `prompt` | str \| null | Seedance prompt（done 后写入） |
| `client_request_id` | str | 前端幂等键（仅提交时带才存在；内部字段，查重依据；查重不比对任何提交参数，靠前端换键保证同键同参数） |
| `voice_mode` | str | `none/keep/rewrite/translate`（恒落，默认 `none`；内部字段） |
| `target_language` | str | 翻译目标语言（仅 `translate` 且非空时落；内部字段） |
| `voice_lines` | list[dict] | 口播台词（`voice_mode≠none` 时 ASR 校验 + YAMNet 声学过滤后写入；进 detail 响应 `voice_lines` 字段） |
| `has_bgm` | bool | 口播模式 YAMNet 判定的整片背景音乐有无（`voice_mode≠none` 时恒写；内部字段，不进 detail 响应） |
| `segments` | list[dict] | 多段模式逐段产物：`index`（1 起）/`start_s`/`end_s`/`keyframes`/`prompt`/`lines`（该段台词 text 列表）；单段模式不写（缺省） |
| `scenes_note` | str | 场景检测失败或 scenes.json 非法回退单段的留痕（内部字段，仅回退时写） |
| `voice_lines_dropped` | int | 多段模式下未归段的越界台词数（内部字段，仅 >0 时写） |
| `voice_lines_vocal_dropped` | int | 口播模式被声学验证丢弃的假转录句数（唱歌/无人声；内部字段，仅 >0 时写） |
| `has_video` | bool | 提交标记（仅提交后存在；内部字段） |
| `submitted_at` / `task_id` | str | 提交时间 / Ark 任务 id（内部字段，读不到 task.json 则为 null） |
| `postprocess` | dict \| null | 后处理状态（仅后处理启动后存在）：`status`（`running/done/failed`）/`options`（勾选选项）/`frames`（有优化版的帧名列表，单段=帧名、多段=`segments/N/work/postprocessed/帧名` 全形路径）/`error`（失败原因，指明帧名）；进 detail 响应 `postprocess` 字段 |

scenes.json（`work/scenes.json`，`app/scenes.py` 产物）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `duration_s` | float | 视频时长（= manifest.duration_seconds，round 3 位） |
| `scenes` | list | 场景边界：`index`（1 起）/`start_s`/`end_s`/`frames`（帧按 time_seconds 落入 [start_s, end_s) 分组的文件名） |
| `segments` | list | 拆段边界建议：`index`/`start_s`/`end_s`，每段 4~15s、覆盖全程无缝隙；`duration_s` ≤ 20 时为空数组（时长 ∈ (15, 20] 超 Seedance 单段上限，流水线按单段处理） |

产物白名单校验（`pipeline.validate_work_dir`，任一不过即 PipelineError → `failed`）：

- `work/keyframes/*.png`：数量 ∈ 1..9（新契约该目录只有选定帧 `01.png…N.png`）
- `work/prompt.txt`：存在、非空、≤ 32KB（`MAX_PROMPT_BYTES`）

多段模式每段目录 `work/segments/N/` 的嵌套 `work/` 按同规则校验；校验通过后由后端在 `prompt.txt` 开头机械加两行「不要生成背景音乐」+ 捂脸条件动作行（`FACE_HOLD_CONDITION_LINE`，不依赖 codex 写），meta.segments 存的 prompt 含这两行（两行恒为中文，不随 translate 目标语言翻译）。单段模式同样机械加条件动作行（无 BGM 行），meta.prompt 含该行；前缀后总长复核 ≤32KB（超限 → 会话 failed）。scenes 检测失败（无场景切点/缺 PySceneDetect）或 scenes.json 的 segments 违反结构不变量（4~15s/相邻无缝/覆盖全程）→ 回退单段模式（meta.scenes_note 留痕），不判失败。段 codex 的 cwd 即段目录（物理隔离，看不到段外内容）：`scripts/` 与 `source.mp4` 留在段根，抽帧/联系表/manifest/台词/关键帧/prompt/后处理均在段 `work/`；scenes.json 不拷入段目录；codex `-o` 落盘的 codex_last_message.txt 随段目录各自独立。

## 依赖

- Python 包（`requirements.txt`）：fastapi、uvicorn[standard]、python-multipart、opencv-python-headless `>=4.8,<5`（skill 脚本与 pipeline 关键帧解码校验用；`<5` 因 skill 脚本的 cv2 API 兼容性仅在 4.x 验证）、scenedetect（场景检测，`app/scenes.py` 用；`>=0.7`——0.6.x 无 FrameTimecode.seconds 属性）、pytest、httpx（TestClient）、ai-edge-litert `==2.1.6`（YAMNet 推理，口播声学验证用）
- 外部可执行：ffmpeg/ffprobe（探测+抽帧+测试造样例）、codex CLI（0.147.0 实证基线，仅流水线用）
- 模型：`models/yamnet.tflite`（4.1MB，进仓库即部署自带；AudioSet 521 类；加载前 sha256 校验；环境变量 `YAMNET_MODEL_PATH` 可覆盖默认路径）

口播声学验证环节（`voice_mode≠none`，codex 听写校验之后）：对 work/voice.mp3 跑 `app.vocal.analyze`（YAMNet 521 类逐窗推理，窗长 15600 样本），每句 `classify_segment` 只保留 `"spoken"`，`"sung"` 与 None（无人声证据的假转录）丢弃并计 `voice_lines_vocal_dropped`；同时判整片 BGM 落 `has_bgm`。验证失败 → 整体 `failed`（`vocal classification unavailable: <原因>`），不静默降级。类索引与判定阈值照搬 TrendScout 盘上实测校准（2026-07-28），不自行调参。
- 技能脚本：`skills/video-maker/scripts/extract_keyframes.py`（`--fps`/`--times`/`--sample-count`/`--prefix`/`--columns`/`--out-dir`）、`skills/video-maker/scripts/crop_image.py`（裁字幕/水印）；提交脚本 `app/seedance_task.py`（`create --dry-run|--confirm-submit --wait`，模型默认 `doubao-seedance-2-0-260128`，Ark `https://ark.cn-beijing.volces.com/api/v3`）；编辑脚本 `app/seedream_task.py`（`edit --dry-run|--confirm-submit [--size WxH]`，模型默认 `doubao-seedream-5-0-pro-260628`，实测契约：JSON 图生图 POST `https://ark.cn-beijing.volces.com/api/v3/images/generations`，`image` 为 data URI 字符串数组，`size` 可选（不传不加入请求体，模型默认 2048 方形），同步 200 返回 `data[0].b64_json`（缺失/为空/非法即失败退出））；场景脚本 `app/scenes.py`（`<video> --work-dir <work>`，PySceneDetect 场景检测 + 拆段建议，写 scenes.json）
- 流水线固定参数：抽帧 `--fps 4`（分页联系表落 `work/`；段内落 `work/segments/N/work/`）；scenes 检测超时 300s；拆段切分 `ffmpeg -ss <start> -i <src> -to <len>` 重编码落 `work/segments/N/source.mp4`（切出时长与边界误差 <0.1s）；提交建模 `9:16 / 15s / 720p / --generate-audio / --no-watermark`（提交时现构建，无评审 payload）
