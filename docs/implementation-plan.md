# MiniMax H3 Max R2V 研究版后端实施计划

## 总结

- 在 `/home/xy/duet-ad1-h3-max-r2v` 内实现全新的独立 Python/FastAPI 后端，源仓库 `/home/xy/duet-ad1` 保持不动。
- 前端继续使用已从 `github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf` 导出的极简版。
- 从 Master 复制 video-maker 及其之前的功能和 Skill，保留算法、执行顺序和数据语义；不复制旧生产系统的回执、CAS、恢复、质量门禁和严格验证框架。
- video-maker 之后改为：项目级翻译和对象索引完成 → 各片段并行执行新提示词 Skill → Picsart MiniMax H3 Max R2V → 裁回原片段长度 → 合并视频。

## 实施内容

### 1. 研究版原则

明确写入 README、架构文档和代码注释：

- 除固定 Master `video-maker` 两阶段自身的 strict `submit_result` 合同外，不增加任何质量门禁、业务参数校验、模型输出校验或输入输出 schema 校验。
- 不检查媒体质量、画面替换成功率、人物一致性、台词准确率、口型、语言、分辨率、码率或成片可播放性。
- 不因缺少索引项、对象映射不唯一、时长偏差或提示词质量问题阻止任务；仅当固定 `video-maker` 的既有结构化响应无法解析时按自然执行错误处理。
- 不为新节点实现 JSON Schema，也不实现哈希回执、CAS、幂等冲突、恢复状态机或质量型重试；`video-maker` 只保留原有两个 Schema。
- 只保留使程序能够执行和保护主机所必需的技术边界：登录鉴权、HTTP/JSON/multipart 解析、项目归属、路径穿越、SSRF、命令注入、基础请求体上限、供应商必填字段组装以及进程超时。
- 文件不存在、命令退出非零、HTTP 请求失败或供应商拒绝属于自然执行错误：记录日志并让任务失败，不包装成质量门禁。
- 现有严格版后端合同文档全部改写为研究版合同；前端要求的固定响应形状仅用于兼容 UI，不视为模型质量校验。

### 2. 前端兼容 API 与存储

实现前端实际可达的七组接口：

- `POST /api/login`
- `GET /api/capabilities`
- `GET /api/conversations`
- `POST /api/conversations`
- `GET /api/conversations/{project_id}`
- `GET /api/conversations/{project_id}/files/{path}`
- `GET /api/conversations/{project_id}/creation-input/replacement-image`

接口约定：

- `POST /api/conversations` 接收 `file` 或 `reference_url`、`generation_request`、`client_request_id` 以及可选 `replacement_image`。
- 上传文件和 URL 同时存在时优先使用上传文件；未知字段直接忽略。
- `client_request_id` 只写日志和项目元数据，不做幂等判断；每次创建都生成新项目并返回 `201`。
- `GET /api/capabilities` 返回现有前端期待的 `minimal_creation` v1 固定结构。
- 项目状态只使用 `queued`、`running`、`succeeded`、`failed`；成功时同一快照包含 `percent: 100` 和 `has_video: true`。
- 无 SSE/WebSocket，继续支持前端每两秒轮询。
- 重启时把遗留的 `queued/running` 项目标记为 `failed/process_restarted`，不做断点恢复。

使用纯文件系统存储：

```text
data/projects/{project_id}/
  state.json
  inputs/
  work/
    segments/{segment_id}/
  output/generated.mp4
  logs/
```

不引入数据库、迁移系统或对象存储。

### 3. 视频流水线

保留并移植 Master 的以下行为：

1. 保存原视频和可选用户替换图。
2. 原视频按 4 fps 解帧。
3. ffmpeg 提取音频，YAMNet 判断语音，本地 Whisper 转写。
4. PySceneDetect `ContentDetector(threshold=27)` 检测场景。
5. 按场景优先切成约 4–10 秒片段。
6. 每段确定性选取 9 张关键帧。
7. 按前端请求调用 MediaKit 去字幕/品牌处理。
8. 每段并行运行 video-maker，生成 `visual_prompt`。
9. 使用全部片段描述进行一次项目级 DeepSeek 翻译，并按原时间语义回挂到各片段。
10. 显式运行 video-maker 的项目索引阶段，生成 `element_index.json`；它仍属于 video-maker，不新增同类业务节点。
11. 在所有公共输入准备完成后，启动每段一个并行 R2V 分支。

每段 R2V 分支：

1. 新增一个独立 DeepSeek Skill，输入：
   - 本段 `visual_prompt`
   - 完整全局 `element_index`
   - 本段本地化后的翻译台词及相对时间
   - 可选用户替换图片
   - 可选用户替换说明
2. Skill 输出纯文本 R2V 提示词，规则固定为：
   - 优先描述用户指定替换的对象、目标和替换方式。
   - 对本段出现且用户未点名的人物：替换为同性别、种族和整体风格，脸略有不同；服装保持颜色和风格但款式稍有变化的新人物。
   - 对场景：保持场景类别和整体风格，但改成不重复的同类地点或布局。
   - 保持原动作、运动、机位、构图、镜头、焦段和光线。
   - 使用全局稳定 ID 帮助跨片段保持同一替换语义，但映射不完整时继续按最佳努力生成。
3. 后端机械附加台词及时间窗，Skill 不得改写台词：
   - 使用目标语言自然说话。
   - 不出现原语言台词。
   - 不生成字幕或屏幕文字。
   - 台词在原片段时长内结束，补齐时段不新增台词。
4. 将本段原视频静音，生成内置临时签名 URL；用户图片存在时，先按 MiniMax 的 `256–5760` 双边范围与 `1:2.5–2.5:1` 比例范围生成确定性 PNG，再生成签名 URL。归一化只做等比缩放和必要黑边补齐，不拉伸或裁剪内容。
5. 调用 Picsart：
   - Advanced workflow：`minimax/h3-max/reference-to-video`
   - `prompt_expansion_mode: "quality"`
   - `reference_video_urls`: 当前静音片段
   - `reference_image_urls`: 可选的一张用户图片
   - `prompt`: Skill 输出与机械附加台词
   - `duration`: `max(5, ceil(原片段秒数))`
6. 轮询任务、下载结果，用 ffmpeg 将音视频裁回原片段准确时长。
7. 全部片段完成后按原顺序 concat，生成 `output/generated.mp4`。

4.x 秒片段因此提交为 5 秒，供应商结果再裁回原片段长度；不因 4 秒不可提交而阻止整个流程。H3 原生音频直接使用，不增加 TTS、配音混合或生成后 ASR。

### 4. 供应商传输、重试与日志

新增无 Bearer 鉴权的内部素材读取端点：

`GET|HEAD /internal/provider-assets/{project_id}/{asset_id}?token=...`

- URL 基于 `PUBLIC_BASE_URL` 生成，供 Picsart 从公网读取；`HEAD` 与 `GET` 使用完全相同的 token、归属和路径校验。
- 每个素材使用独立高熵 token，默认有效期为素材注册后 24 小时。
- token、项目归属和路径安全属于技术访问控制，不属于质量校验。
- 不接入 Picsart Drive 或外部对象存储。

所有节点统一执行一次初始尝试和最多一次自动重试：

- 包括 ffmpeg、Whisper、YAMNet、场景切分、MediaKit、video-maker、翻译、项目索引、新 Skill、Picsart 提交/轮询/下载、裁剪和合并。
- Picsart 提交结果未知且没有 `taskId` 时也重新提交一次；接受可能产生重复付费任务。
- 除固定 Master 在 YAMNet 检出人声而第一次 Whisper 为空时进行一次 ASR 复核外，不根据质量、语义或媒体属性触发额外重试；总 attempt 仍不超过两次。

每个项目写 `trace.jsonl`，每个节点尝试写独立目录：

```text
logs/{node}/{segment_or_project}/attempt-{1|2}/
  input.json
  input-media/
  output.json
  output-media/
  error.json
  stdout.log
  stderr.log
```

日志要求：

- 原样记录每个节点的输入、输出、完整请求/响应、HTTP headers、鉴权信息、签名 URL、提示词、台词、对象索引、DeepSeek 原始响应、Picsart 提交与轮询响应、异常堆栈及命令输出。
- 图片、音频和视频复制进对应尝试的日志目录，不只记录路径。
- 不脱敏、不清理签名 URL query。
- 日志目录权限 `0700`，文件权限 `0600`；不新增日志下载 API，默认永久保留、人工删除。
- 日志写入失败时只输出到进程 stderr 并继续执行，不能成为任务门禁。

### 5. 测试与验收

- 保持现有 47 个极简前端契约测试全部通过。
- 增加 API 测试：登录、能力声明、创建、轮询、详情、源视频、结果视频和替换图片读取。
- 使用假 DeepSeek、MediaKit 和 Picsart 服务完成端到端测试，测试中不发起真实付费生成。
- 验证 Master 上游链仍产生每段 9 帧、视频描述、本地化台词和全局对象索引，但不对其内容质量做断言。
- 验证所有片段在公共输入准备完成后并行进入新 Skill 和 H3 R2V。
- 验证 4.x 秒片段使用供应商 5 秒参数并裁回原时长。
- 验证 `prompt_expansion_mode` 固定为 `"quality"`，静音视频和可选图片同时传入。
- 验证用户未提供替换图时仍能执行；图片和说明不配对时也不由后端门禁拦截。
- 验证每个节点的两次尝试、未知提交重提及原始输入输出日志均被记录。
- 验证签名素材 URL 可供无 Bearer 的供应商读取，但错误 token 不能跨项目访问。
- 验证签名素材的 `GET/HEAD` 均可用，并验证参考图满足 MiniMax 双边和比例约束。
- 测试只断言流程、API 兼容和日志存在，不增加模型输出、媒体质量或业务语义断言。

## 默认边界

- 使用 Python 3.12、FastAPI、Uvicorn、httpx、OpenCV、NumPy、PySceneDetect、LiteRT、ffmpeg/ffprobe、本地 whisper-cli 和 `ggml-small`。
- DeepSeek 默认使用当前链路的 `deepseek-v4-flash-vision-exp`，新仓库自行读取配置，不依赖源仓库运行目录。
- 不迁移 image-postprocess、video-prompt-fusion、Context IR、旧 H3 后端和旧恢复框架。
- 不实现旧前端中不可达的 `/submit`、提示词编辑、后处理和手工重试接口。
- 默认同时启动全部片段任务，不增加业务并发门禁；HTTP 连接池和操作系统资源限制保持库默认行为。
- `PUBLIC_BASE_URL` 必须是 Picsart 可以访问的公网地址。
- Picsart 契约以其当前 [H3 Max R2V Advanced 文档](https://picsart.com/api-platform/models/minimax-h3-max-r2v/advanced)为准。
