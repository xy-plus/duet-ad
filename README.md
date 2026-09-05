# Duet H3 Max R2V Research

当前完整链路图：[docs/current-pipeline.svg](docs/current-pipeline.svg)

这是从 `github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf` 的极简前端出发建立的独立研究仓库。它不依赖 `/home/xy/duet-ad1` 的运行目录，也不修改源仓库。

后端使用 Python 3.12、FastAPI 和纯文件系统存储。上游沿用固定 Master 的视频分析语义；`video-maker` 先逐段生成视觉描述，再以九帧像素为事实源生成项目级原始索引，逐段描述只作为防漏清单。DeepSeek Object Replace v2 随后一次性冻结全项目的替换、人物音色和单张新形象图版提示词，后端机械合并为增强索引，并由 Seedream 每项目生成一张包含所有实际替换元素的新形象图版。此后的逐段 Fusion 根据同一冻结索引及图版映射编译 H3 提示词，再调用 Picsart `minimax-h3-max-r2v`，最后裁回源片段时长并合并。Fusion 执行器由后端选定，前端和公开 API 均不暴露模型选择。

## 研究版边界

- 不增加媒体质量、替换成功率、人物一致性、台词、口型、模型语义覆盖或成片观感门禁。只要节点结果能按结构传给下一节点，就不因“做得不够好”拦截、失败或重试。
- closed Schema 只保证 JSON 形状、字段类型和本次输入绑定，不承担业务语义复审。Skill 里的替换与音色规则是生成职责，不是后端再次评分的通过条件。
- 只保留鉴权、解析、项目与路径归属、SSRF、命令参数隔离、请求大小、必要媒体可用性、Picsart 硬参数限制、付费提交幂等和超时等技术边界。
- 普通执行节点首次失败后最多重试一次；Seedream 生成 POST 和 Picsart submit 都不做通用重试。发出 Seedream POST 前持久化与输入绑定的提交回执；结果未知时禁止自动重提。Picsart 同样在 POST 前持久化不可自动重投的提交回执，只有供应商明确拒绝时才允许后续重提；未知结果必须人工对账。
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
  -> video-maker 项目级原始对象索引：九帧为事实源，逐段 visual_prompt 仅作防漏清单
  -> 后端从 raw 索引生成全 stable-key replacement_set 和去除条目级 preserve/replaceable 的规划视图
  -> 一次项目级 DeepSeek Object Replace v2：规划视图 + replacement_set + 全部视觉提示 + 用户图/说明 + 当前音色目录；输出 patch、voice 和 reference_sheet_prompt
  -> 后端机械合并增强索引；按可用 voice_id 生成全项目 Audio 1..N（N <= 3）
  -> 后端按冻结 stable key 确定性生成 TILE_XX 图版 spec；Seedream 每项目生成一张包含全部实际替换元素的新形象图版
  -> 第一屏障：所有段并行完成 Fusion、best-effort 约束编译、最终提示词和静音 Video 1
  -> 第二屏障：所有段完成有序图片、音频和 Video 1 的素材注册及 H3 参数预检
  -> 各段并行 H3：Video 1 + 同一组有序 Images + 全项目同序 Audio 1..N -> 裁回源时长
  -> 顺序合并 generated.mp4
```

`video-maker` 的两个历史阶段、项目级翻译、Object Replace 和 DeepSeek Fusion 都通过 DeepSeek Responses API 的原生 `text.format=json_schema` 输出 closed JSON；不使用工具调用、`tool_choice` 或虚构的提交函数。Schema 只约束可传递的结构和输入绑定：Object Replace 的 patch 中 `replacement` 与人物 `voice_id` 都允许为 `null`，后端不复审人物/场景是否替换正确、用户对象是否命中、普通实体是否达到数量或关系是否闭包；Fusion 只校验 closed 结构、`input_sha256` 和 `segment_index`，不检查正文是否覆盖 stable ID、`Image 1` 或 `Audio N`。DeepSeek thinking 以 `reasoning.effort=high` 显式开启，请求不设置供应商级 `max_output_tokens`。

项目级索引在运行时根据本次请求的正整数段号构造动态 closed Schema；人物、实体、场景和关系的所有 occurrence 只能引用这组精确段号，不能输出 `0`、未知段号或自行重排为另一套编号。`segments[].visual_prompt` 只提示模型回看对应九帧，不是新的视觉事实，也不新增本地语义门禁。

Object Replace 调用前，后端从 raw stable index 确定性生成 `replacement_set`：`people/entities/scenes` 分别包含本类全部 stable key 并排序。送模型的 `global_element_index` 是独立规划视图，仅移除这三类记录顶层的 `preserve/replaceable`；relations 和其他事实不变。完整 raw index 继续冻结，并作为机械合并 enhanced index 的基底。Object Replace v2 除 replacement/voice patch 外还输出项目级 `reference_sheet_prompt`；后端而非模型把增强索引中的实际非空替换稳定映射为 `stable key -> TILE_XX -> row/column`，因此全局索引是图版、Fusion 与 H3 共用的唯一身份锚点。

Object Replace Skill 要求 `replacement_set` 中每个人物、实体和场景都生成非空替换。同一 stable key 跨图片、分段和 occurrence 使用同一套新设计；Object Replace 调用中的 Image 1 只约束用户文字唯一语义对应的一个既有 stable key，其他 key 仍使用 `generated_variant`。人物保持明确的类型、性别、种族、年龄段、整体风格和服装主色，但更换身份，使人脸或可见身体产生差异；只露出身体局部时，统一改变局部形态、比例、肤色、纹理或标记，不为具体部位写特例；服装保持颜色和风格但改变款式。场景保持类型、用途、布局、色调、构图、机位、镜头运动、焦段和光线，但改成同类不同实例。实体保持功能、动作接口、尺度、接触和相对关系，但改变材质、颜色、品种、轮廓、结构或造型。增强索引仍保存 raw `preserve`；对 `REPLACE`，Fusion 不带入与新设计冲突的源外观。上述是模型生成职责：Schema/normalizer 仍允许 nullable，后端不以是否完成替换作为失败、拦截或重试条件。

新形象图版为项目级单张 2048×2048 PNG。Seedream 同时接收后端生成的空白格、同布局源帧证据和可选用户图，并严格沿用冻结 TILE spec；用户图只可影响唯一有效的 `source=image_1` stable key，不约束其他 tile。存在该绑定时，图版实际送入 H3 的顺序为 `Image 1=用户图`、`Image 2=新形象图版`；没有绑定时不扩散原始用户图，图版单独作为 `Image 1`。每个 segment 都使用同一张图版、同一顺序和同一 `stable key -> TILE_XX -> Image N` 映射；Fusion 只消费这套后端映射，不自行识别或重编号图版对象。

音色目录位于 `/home/xy/duet-ad1-h3-max-r2v/assets/voice-library`。当前随仓库提供 30 条 2–5 秒单声道 PCM WAV；运行时只要求目录中存在可用条目，不把“恰好 30 条”设为门禁，也不复审年龄、性格等标签是否与声音主观相符。Object Replace 每条只接收一行 `voice_id - file - language - gender - 短特征`。Skill 仍负责为人物选择合适音色；后端只消费可用选择，并按稳定人物顺序保留最先出现的最多 3 个不同 `voice_id`，后续新增音色稳定降级为 `null`，再机械映射为 `Audio 1..N`。Picsart 的每个 segment 接收同一份、同一顺序的 0–3 条参考音频。

## 配置

参见 `/home/xy/duet-ad1-h3-max-r2v/.env.example`。至少需要显式提供：

- `ACCESS_TOKEN`：前端登录 Bearer token；禁止使用缺省或空值。
- `DEEPSEEK_API_KEY`、`PICSART_API_KEY`、`ARK_API_KEY`。DeepSeek 仍负责 video-maker、翻译、Object Replace 和可选 DeepSeek Fusion；Ark Seedream 负责项目级单张新形象图版。
- `SEEDREAM_API_BASE`、`SEEDREAM_MODEL` 和 `SEEDREAM_TIMEOUT_S`：Seedream 同步生成端点、模型和请求超时；默认值见 `.env.example`。
- `FUSION_AGENT`：内部逐段 Fusion 执行器，只能为 `codex` 或 `deepseek`，缺省值为 `deepseek`。客户端传入的同名字段会被忽略，服务端选择在创建时冻结；Object Replace 则固定为项目级 DeepSeek 节点，不受该配置影响。
- Codex Fusion 固定通过 `/home/xy/.local/bin/codex-acct acct2 exec` 使用 `gpt-5.6-sol`、reasoning `low`；`CODEX_TIMEOUT_S` 和 `CODEX_CONCURRENCY` 只控制执行边界，不改变账号、模型或 reasoning。
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
  work/element_index.json
  work/enhanced_element_index.json
  work/global-reference-sheet/spec.json
  work/global-reference-sheet/reference-sheet.png
  work/segments/{segment_id}/
  work/voice-references/audio-01.wav ... audio-03.wav
  output/generated.mp4
  logs/
```

`work/element_index.json` 与 `state.element_index` 保存 video-maker 原始结果；`work/enhanced_element_index.json` 与 `state.enhanced_element_index` 保存后端机械合并的 replacement/voice patch。`state.object_replace` 保存版本、输入哈希、目录哈希和图版提示词；`state.replacement_sheet` 保存冻结 TILE spec、Seedream 提交回执及最终单图回执；`state.voice_references` 按 `Audio 1..N` 顺序冻结实际可用的文件路径与哈希。`state.h3_prepared_artifacts` v3 把各段 prompt/源片段/静音片段哈希绑定到增强索引、图版状态、音色列表、Object Replace 输入、可选用户图、输出配置，以及每段的起止时间、时长和 join mode；旧 v2 只用于兼容既有恢复数据。它用于保证恢复时复用的是同一批技术输入，不判断替换质量。`state.picsart_submission_receipts` 是付费提交恢复的权威账本；签名 URL 不持久化。

重启时遗留的 `queued/running` 项目会失败为 `process_restarted`，不做自动断点恢复。`resume-pre-picsart` 只接受没有任何 Picsart submit 痕迹的项目：失败发生在图版 checkpoint 之前时，可复用前置媒体后重新进入免费上游；已冻结 `prepared/succeeded/not_needed` checkpoint 时，先在任何写入前验证其全局索引、segments、文件和请求绑定，再只恢复或复用原图版，不重跑翻译、原始索引或 Object Replace。已有 Seedream 付费证据只允许本地 replay，结果未知时停止，不能盲目再 POST。`resume-picsart` 对新项目只复用 v3 manifest 冻结的 `h3_prompt.txt`、静音 Video 1、有序图片（包括同一张冻结图版）和参考音频，不重跑 Object Replace、Seedream 或 Fusion；恢复时只校验文件、哈希、路径和请求绑定等技术一致性，不复审 enhanced index 或图版的替换质量。旧 v2 项目继续使用兼容路径。通用文件接口只暴露源视频和最终视频，不能读取状态、日志或中间文件。

## 验证

离线测试使用假 Codex、DeepSeek、Seedream、MediaKit 和 Picsart，不产生付费任务：

```bash
python3 -m pytest /home/xy/duet-ad1-h3-max-r2v/tests -q
```

真实供应商冒烟测试需要另行明确授权，并且必须先配置可公网回源的 `PUBLIC_BASE_URL`。

非空 `client_request_id` 会持久绑定规范化请求、源视频内容和可选替换图内容。首次创建返回 `201`；相同 ID 与相同输入的重放返回原项目 `200`，不会重写输入或再次启动流水线；相同 ID 绑定不同输入时返回 `409`。索引损坏或无法证明绑定关系时以 `503` 停止创建，避免产生第二个可能付费的项目。

Seedream 图版改造已通过当前离线全量测试 `383 passed, 1 warning`，并于 2026-09-05 部署到独立 3301/3302 实例；部署后完成了服务、鉴权只读 API、静态文件哈希和 443 供应商素材回源验证，但尚未创建新的付费 E2E。最近一次真实付费 E2E 项目 `265935eb58f0441bbb4bcb2621039688` 验证的是更早的 `8c1515d` 与 `285733b`，不能替代本改造的真实生成验收。
