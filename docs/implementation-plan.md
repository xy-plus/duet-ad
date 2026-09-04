# MiniMax H3 Max R2V 研究版后端实施计划

> 2026-09-04 最新需求覆盖：video-maker 后新增项目级 DeepSeek Object Replace，逐段 Fusion 由后端内部选择 DeepSeek 或 Codex，前端和公开 API 不提供模型切换。所有 DeepSeek 请求开启 thinking 且不发送供应商级输出 token 上限；结构化节点使用 Responses API 原生 `text.format=json_schema`。Schema 只保证结构与输入绑定，后端不得把 Skill 语义要求变成二次质量门禁。

## 总结

- 在 `/home/xy/duet-ad1-h3-max-r2v` 内实现全新的独立 Python/FastAPI 后端，源仓库 `/home/xy/duet-ad1` 保持不动。
- 前端继续使用已从 `github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf` 导出的极简版。
- 从 Master 复制 video-maker 及其之前的功能和 Skill，保留算法、执行顺序和数据语义；不引入模型语义复审或质量型重试。
- video-maker 之后改为：项目级翻译和 raw 对象索引 → 项目级 Object Replace 与 enhanced 索引/音色 → 各片段 Fusion → Picsart MiniMax H3 Max R2V → 裁回原片段长度 → 合并视频。

## 实施内容

### 1. 研究版原则

明确写入 README、架构文档和代码注释：

- closed JSON Schema 仅用于 `video-maker` 两阶段、项目级翻译、Object Replace 和 H3 提示词 Fusion 的结构化传输及输入绑定；DeepSeek 走 Responses API 原生 Structured Outputs，不把 Schema 扩展成业务效果门禁。
- 不检查媒体质量、画面替换成功率、人物一致性、台词准确率、口型、语言、分辨率、码率或成片可播放性。
- 不因 replacement/voice 为 `null`、对象没有按 Skill 要求替换、普通实体不足两个、关系未闭包、提示词未出现 ID/Image/Audio、空 ASR 或翻译缺行而阻止任务；翻译缺失或空文本回退源文。
- Object Replace 和 Fusion 都使用动态哈希绑定 Schema；本地只检查 JSON 形状、字段类型和绑定。Fusion 的约束块由后端按实际可用字段 best-effort 编译。
- 只保留使程序能够执行和保护主机所必需的技术边界：登录鉴权、HTTP/JSON/multipart 解析、项目归属、路径穿越、SSRF、命令注入、基础请求体上限、必要媒体可用性、Picsart 硬限制、付费幂等以及进程超时。
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
- 非空 `client_request_id` 持久绑定规范化请求、实际源视频摘要和可选替换图摘要。首次创建返回 `201`；同键同输入返回原项目 `200` 且不重复启动流水线；同键异输入返回 `409`；索引损坏或悬空时返回 `503`，不得冒险创建第二项目。空 ID 仍按每次独立创建处理。
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
    element_index.json
    enhanced_element_index.json
    voice-references/audio-01.wav ... audio-03.wav
    segments/{segment_id}/
  output/generated.mp4
  logs/
```

不引入数据库、迁移系统或对象存储。

### 3. 视频流水线

保留并移植 Master 的以下行为：

1. 保存原视频和可选用户替换图。
2. 原视频按 4 fps 解帧。
3. ffmpeg 提取音频，YAMNet 判断语音，本地 Whisper 转写；空 ASR 或 YAMNet/ASR 无重叠 spoken 区间时以空台词继续，不做内容型复核。
4. PySceneDetect `ContentDetector(threshold=27)` 检测场景。
5. 按场景优先切成约 4–10 秒片段。
6. 每段确定性选取 9 张关键帧。
7. 按前端请求调用 MediaKit 去字幕/品牌处理。
8. 每段并行运行 video-maker，生成 `visual_prompt`。
9. 使用全部片段描述进行一次项目级 DeepSeek 翻译，通过 Responses API 原生 JSON Schema 请求 `lines[{text}]`，按原时间语义回挂到各片段；缺行或空文本回退对应源文。
10. 显式运行 video-maker 的项目索引阶段，生成 raw `element_index.json`；people/entities/scenes/relations 允许为空，不设置集合大小、描述长度或 scene 非空门禁。
11. 运行一次项目级 DeepSeek Object Replace：输入 raw 索引、全部视觉提示、翻译台词、可选 Image 1/说明和当前音色目录，输出 nullable replacement/voice patch。
12. 后端机械合并 enhanced 索引；按稳定人物顺序保留最先出现的最多 3 个可用不同音色，更多选择降级为 `null`，生成 0–3 条 `Audio N`。
13. 所有公共输入可传递后，启动逐段 R2V 准备与执行。

Object Replace Skill 本身仍明确要求：人物和场景都应替换；用户点名的物品应使用 Image 1；其他实体至少替换两个，不足两个则全部替换；`replace_together` 关系应协同；人物应从目录中选择符合性别、口音和特征的音色。后端不对这些生成职责逐项打分，nullable 输出可直接流向下游。

每段 R2V 分支：

1. 新增一个由后端内部选择执行器的 Fusion Skill，输入：
   - 本段 `visual_prompt`
   - 完整项目级 `enhanced_element_index`
   - 本段本地化后的翻译台词及相对时间
   - Fusion 调用不接收用户图片、用户原始说明或独立音色目录/选择。
2. Skill 通过动态绑定本段输入哈希和段号的 closed Schema 输出字符串 `prompt`，负责读取 enhanced index 并表达其中的替换、`Image 1`、`Audio N`、preserve 和 relations。后端只验证 closed 结构与绑定，再按实际存在的字段 best-effort 附加约束块；不检查正文是否覆盖 stable ID、`Image 1`、`Audio N` 或任何替换规则。
3. 后端机械附加可用台词及时间窗：
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
   - `reference_audio_urls`: 项目级同序的 0–3 条参考音频
   - `prompt`: Skill 输出与机械附加台词
   - `duration`: `max(5, ceil(原片段秒数))`
6. 轮询任务、下载结果，用 ffmpeg 将音视频裁回原片段准确时长。
7. 全部片段完成后按原顺序 concat，生成 `output/generated.mp4`。

4.x 秒片段因此提交为 5 秒，供应商结果再裁回原片段长度；这是 Picsart 的硬参数适配，不是质量复审。H3 原生音频直接使用，不增加 TTS、配音混合或生成后 ASR。音色目录当前随仓库提供 30 条不超过 5 秒的素材，但运行时不要求数量恰好为 30，也不主观复审年龄、性格等标签。

### 4. 供应商传输、重试与日志

新增无 Bearer 鉴权的内部素材读取端点：

`GET|HEAD /internal/provider-assets/{project_id}/{asset_id}?token=...`

- URL 基于 `PUBLIC_BASE_URL` 生成，供 Picsart 从公网读取；`HEAD` 与 `GET` 使用完全相同的 token、归属和路径校验。
- 每个素材使用独立高熵 token，默认有效期为素材注册后 24 小时。
- token、项目归属和路径安全属于技术访问控制，不属于质量校验。
- 不接入 Picsart Drive 或外部对象存储。

普通节点执行一次初始尝试和最多一次自然错误重试：

- ffmpeg、Whisper、YAMNet、场景切分、MediaKit、video-maker、翻译、项目索引、Object Replace、Fusion、轮询/下载、裁剪和合并只在命令、传输或结构无法继续时进入第二 attempt。
- 空 ASR、YAMNet 与台词无重叠、翻译缺行/空行、nullable 替换/音色及语义覆盖不足都直接继续，不触发重试。
- Picsart submit 是付费边界，不使用通用重试。POST 前先持久化 `submission_started`；未知结果不得自动重投，只有供应商明确拒绝或已有任务业务状态明确 `FAILED` 时才允许后续新提交。

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

所有 segment 的免费准备完成后发布 `state.h3_prepared_artifacts` v2。它逐段绑定 source/muted/prompt 文件回执、Fusion 输入和包含 index/start/end/duration/join mode 的 segment metadata，同时绑定 enhanced index、voice references、Object Replace 输入、可选 replacement image 与 output request。该 manifest 和提交回执只服务于安全恢复与付费幂等，不评判模型输出质量。

### 5. 测试与验收

- 保持现有 47 个极简前端契约测试全部通过。
- 增加 API 测试：登录、能力声明、创建、轮询、详情、源视频、结果视频和替换图片读取。
- 使用假 DeepSeek、MediaKit 和 Picsart 服务完成端到端测试，测试中不发起真实付费生成。
- 验证 Master 上游链仍产生每段 9 帧、视频描述、本地化台词和全局对象索引，但不对其内容质量做断言。
- 验证所有片段在公共输入准备完成后并行进入新 Skill 和 H3 R2V。
- 验证 4.x 秒片段使用供应商 5 秒参数并裁回原时长。
- 验证 `prompt_expansion_mode` 固定为 `"quality"`，静音视频和可选图片同时传入。
- 验证用户未提供替换图时仍能执行；nullable replacement/voice 和超过 3 个不同音色都按确定性降级继续。
- 验证 Object Replace/Fusion 只受 closed 结构和输入绑定约束，不因替换数量、关系闭包或提示词中缺少 ID/Image/Audio 而失败。
- 验证空 ASR 与 YAMNet 无重叠直接继续，翻译缺行或空行回退源文。
- 验证普通节点的自然错误重试、Picsart 未知提交禁止重投及原始输入输出日志均被记录。
- 验证签名素材 URL 可供无 Bearer 的供应商读取，但错误 token 不能跨项目访问。
- 验证签名素材的 `GET/HEAD` 均可用，并验证参考图满足 MiniMax 双边和比例约束。
- 测试只断言流程、API 兼容和日志存在，不增加模型输出、媒体质量或业务语义断言。

当前完整离线结果：`251 passed, 1 warning`。

## 默认边界

- 使用 Python 3.12、FastAPI、Uvicorn、httpx、OpenCV、NumPy、PySceneDetect、LiteRT、ffmpeg/ffprobe、本地 whisper-cli 和 `ggml-small`。
- DeepSeek 默认使用当前链路的 `deepseek-v4-flash-vision-exp`，新仓库自行读取配置，不依赖源仓库运行目录。
- 不迁移 image-postprocess、Context IR、旧 H3 后端和旧恢复框架；仅保留满足当前内部 Fusion 与历史项目兼容所需的 Codex 执行器。
- 不实现旧前端中不可达的 `/submit`、提示词编辑、后处理和手工重试接口。
- 默认同时启动全部片段任务，不增加业务并发门禁；HTTP 连接池和操作系统资源限制保持库默认行为。
- `PUBLIC_BASE_URL` 必须是 Picsart 可以访问的公网地址。
- Picsart 契约以其当前 [H3 Max R2V Advanced 文档](https://picsart.com/api-platform/models/minimax-h3-max-r2v/advanced)为准。
