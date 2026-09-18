# MiniMax H3 Max R2V 新链路交接

## 当前状态

- 工作区：`/home/xy/duet-ad1-h3-max-r2v`
- 分支：`main`
- 前端固定基线：`github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf`
- 本工作树使用项目级 DeepSeek Object Replace v3 和项目级 Seedream 单张新形象图版；原始索引、增强索引、`stable key -> TILE_XX -> 实际 Image N` 映射、人物音色和 H3 有序媒体分层冻结。全局 stable index 是这些层共同且唯一的身份锚点。
- 逐段 Fusion 仍由后端 `FUSION_AGENT` 选择 `codex` 或 `deepseek`，缺省为 `deepseek`；前端和公开 capability/request 不暴露该选择，客户端夹带的 `fusion` 会被服务端配置覆盖并在创建时冻结。Object Replace 始终使用 DeepSeek，不受 `FUSION_AGENT` 影响。
- DeepSeek 代码路径显式开启 thinking，不发送供应商级输出 token 上限；video-maker 两阶段、项目级翻译、Object Replace 和 DeepSeek Fusion 均使用 Responses API 原生 `text.format=json_schema`。本地只处理 closed 结构与输入绑定，不做模型语义二次校验。
- Object Replace v3、sheet bundle v4/spec v2 与 H3 manifest v4 的最终离线结果以本工作树合并完成后的实际测试为准；当前改造尚未部署，也尚未创建真实付费 E2E。

## 已实现链路

```text
保存原视频和可选替换图
  -> 项目级 MediaKit 视频字幕擦除（remove_subtitle 开关；pro 固定 v5 / standard 由 MEDIAKIT_VIDEO_ERASE_MODE 选择），产出 work/source-nosubs.mp4，下游一律改用擦除后文件
  -> 原视频 4 fps 解帧
  -> 视频视觉轴音频 / YAMNet / Whisper / spoken 台词
  -> ContentDetector(threshold=27)
  -> scene-first 4-10 秒分段
  -> 每段 Hamilton scene-aware exact-9
  -> 可选 MediaKit：先去字幕、再去标志
  -> 每段并行 video-maker 视觉分析
  -> 一次项目级台词翻译并按时间交集回挂
  -> 一次 video-maker 项目级 raw 全局对象索引：九帧为事实源，逐段 visual_prompt 仅作防漏清单
  -> 后端从 raw 索引生成全 stable-key replacement_set 和去除条目级 preserve/replaceable 的规划视图
  -> 一次项目级 DeepSeek Object Replace v4：规划视图 + replacement_set + 全部视觉提示 + 用户图/说明 + 当前音色目录；输出分离外观/交互语义的 patch、voice 和逐行 dialogue_voices 台词归属
  -> 后端机械合并 enhanced 索引，按可用 voice_id 生成同序 Audio 1..N（N <= 3）；有台词但无人分配到音色时从音色库兜底一条默认音色（优先目标语言，否则目录第一条）作 Audio 1，记 voice_reference_default_fallback 事件和 state.object_replace.voice_reference_fallback
  -> 逐行豆包语音合成 2.0 台词 TTS（按目标语言选 voice_type；归属人物音色库条目的性别选男/女声，无归属用该语言默认音色），拼整条音轨并切每段 dialogue-track.wav；失败即 fail-closed
  -> 后端从 frozen enhanced index 生成 sheet bundle v4/spec v2：stable key -> TILE_XX
  -> Seedream 每项目一次，以 Image 1 空白布局 + 可选 Image 2 用户图生成一张隔离的新形象图版
  -> 公共输入屏障
  -> 第一屏障：所有段并行完成 Fusion、最终提示词和静音 Video 1（有替换人物或物品的段由 fal.ai SAM 一次多目标检测统一深灰块打码；未配置 FAL_KEY 不触发）
  -> 第二屏障：所有段完成有序 Images、素材注册和 H3 参数预检
  -> 每段并行 Picsart H3：Video 1 + 所有段完全同序的 Images + 本段台词语音切片 Audio 1（无台词段不传音频）
  -> 供应商片段裁回原时长并统一为 24 fps
  -> 按原顺序合并 output/generated.mp4；有台词时整条 TTS 音轨覆盖成片音轨
```

无音轨视频会显式跳过音频、YAMNet、Whisper 和 spoken 筛选节点，以空台词继续。跨段台词按各段时间交集分割文本；台词不参与切点选择。

## video-maker 合同

`skills/video-maker/SKILL.md` 以固定 Master 为基线。逐段视觉阶段保持原合同；项目索引阶段增加逐段防漏清单、跨段覆盖账本和精确段身份绑定。两个阶段都沿用 closed JSON Schema 传输；DeepSeek 使用 Responses API 原生 Structured Outputs，不扩展为新链路的通用质量门禁。

### 逐段视觉阶段

每个 segment 并行执行一次。输入包括：

- 原 segment 计划和 4 fps manifest。
- `generation_config`：`remove_subtitle`、`remove_watermark`。
- MediaKit 后的九张权威关键帧所生成的半分辨率分析副本。
- 每张分析帧的 `frame_order`、项目内 `path`、`sha256`，以及九张图片本身。
- 固定 `video-maker` Skill。

这个阶段不接收台词、全局对象索引、用户替换图或用户替换说明，也无权选帧或改变九帧的数量、顺序和字节。

模型通过 `VISUAL_PROMPT_SCHEMA` 返回：

```json
{"prompt": "本段可见主体、场景、动作因果、镜头、构图、运动与节奏的视觉描述"}
```

后端只解包 `prompt` 字符串，写入本段 `visual_prompt.txt`；不设置任意最短长度或内容覆盖要求。完整 DeepSeek 请求和响应只进入该节点的 attempt 日志。

### 项目级索引阶段

所有逐段视觉阶段完成后执行一次。请求包括：

```json
{
  "phase": "project_index",
  "generation_config": {
    "remove_subtitle": true,
    "remove_watermark": true
  },
  "segments": [
    {
      "segment_index": 1,
      "visual_prompt": "本段已发现的人物、手持物、场景与动作描述",
      "frames": [
        {
          "frame_order": 1,
          "path": "work/segments/0001/analysis-keyframes/01.png",
          "sha256": "..."
        }
      ]
    }
  ]
}
```

图片仍是所有 segment 的九张半分辨率分析副本，按 segment/frame 排序。`segments[].visual_prompt` 只作为防漏清单：它提示模型逐项回看对应九帧，不能替代或推翻像素证据。这个阶段不接收台词、用户图片或用户说明。

模型通过本次请求动态生成的 project-index closed Schema 返回四个数组：`people`、`entities`、`scenes`、`relations`。人物、实体、场景和关系的所有 occurrence 只能引用请求中的精确正整数段号，不能输出 `0`、未知段号或自行重新编号。四类集合允许为空，不要求场景非空或达到任意数量。后端把已有项的调用内 `key` 机械映射为 `person-01`、`entity-01`、`scene-01`、`relation-01` 等稳定 key，保存为 `work/element_index.json` 和 `state.element_index`。这两处是不可覆盖的 raw index；供应商 record 不进入业务索引。

## video-maker 之后

项目级 DeepSeek 翻译读取 spoken 台词和所有逐段视觉描述，通过 Responses API 原生 closed JSON Schema 请求 `lines[{text}]`；后端按原时间语义把可用译文切回每个 segment。返回缺行或某行为空时直接回退对应源文，不因翻译完整度失败或重试。

新增 `skills/object-replace/SKILL.md` 在 raw index 完成后设置一个 DeepSeek 项目级公共节点，不按 segment 重复；如首次自然失败，通用节点边界仍允许第二个 attempt。调用前，后端从 raw stable index 确定性生成 `replacement_set={people,entities,scenes}`，每类包含全部 stable key 并排序。输入包括：

- 只供模型规划的 `global_element_index` 深拷贝：仅移除人物、实体和场景记录顶层的 `preserve/replaceable`，relations 和其他字段不变。
- 后端确定性生成的全 stable-key `replacement_set`。
- 按 segment 顺序的全部 `segment_visual_prompts`。
- 目标语言和项目级翻译台词。
- 可选用户图（Object Replace 调用中的 Image 1）与用户替换说明；有图时同时传图片本身和 SHA-256。
- 目录 SHA-256 及当前可用的 `voice_id - file - language - gender - 短特征` 行。仓库当前提供 30 条，但运行时不要求恰好 30 条。

Object Replace v4 不回吐完整索引，只通过动态 closed JSON Schema 输出与 raw stable key 对齐的 patch。`replacement` 和人物 `voice_id` 都允许为 `null`；每个非空 replacement 包含 `source`、`appearance_description` 和 `target_visual_description`。顶层另有 `dialogue_voices`：与 `translated_dialogue` 等长，每行归属一个 `people` stable key 或 `null`（旁白/画外音），合并时写入增强索引的独立块。`appearance_description` 只供 sheet，限定为单个 stable key 自身的内在可见外观，不得包含动作、接触、相对位置、所在场景或其他 stable key；`target_visual_description` 只供 Fusion/H3 承载动作、接触和关系。Skill 自身要求 `replacement_set` 中所有人物、实体和场景输出非空替换，并为人物选择合适音色。同一 stable key 跨图片、分段和 occurrence 使用同一套新设计。用户图只约束用户文字唯一语义对应的一个既有 stable key；其他 key 仍使用 `generated_variant`，不因参考图只指向一个元素而停止替换。

人物保持索引明确的类型、性别、种族、年龄段、整体风格和服装主色，但更换身份，使人脸或可见身体产生差异；只露出身体局部时，统一改变局部形态、比例、肤色、纹理或标记，不为具体部位写特例；服装保持颜色和风格但改变款式。场景保持类型、用途、布局、色调、构图、机位、镜头运动、焦段和光线，但改成同类不同实例。实体保持功能、动作接口和尺度，但改变材质、颜色、品种、轮廓、结构或造型；动作、接触和相对关系只能写入 `target_visual_description`，不能回流到图版外观。完整 raw index 仍冻结，并作为 patch 的机械合并基底；Fusion 读取的 enhanced index 因而保留完整 raw 事实。上述是模型生成职责：Schema/normalizer 仍允许 nullable，后端不逐项复审，也不因未完成而失败或重试。

patch 结构可读取后，后端深拷贝 raw index，机械添加 nullable replacement 和人物 voice，原样保留 stable ID、原始描述、occurrences、preserve 与 relations，并写入 `work/enhanced_element_index.json` 与 `state.enhanced_element_index`。后端按稳定人物顺序保留最先出现的最多 3 个不同可用 `voice_id`；第四个及后续新音色稳定降级为 `null`，不会阻断项目。保留项依次分配 `Audio 1..N` 并复制为 `work/voice-references/audio-01.wav` 至 `audio-03.wav`。运行时只校验音频文件与 Picsart 输入可用性，不复审年龄、性格等标签。

增强索引是新形象图版唯一身份锚点。后端只为实际非空 replacement 确定性生成一个项目级 sheet bundle v4/spec v2，冻结每个 tile 的 stable key、类别、行列和 `appearance_description`，并机械编译完整图版提示词；Seedream 不得重新识别、合并、重命名或排序对象。每个人物格画该人物的头肩面部特写（索引只见到身体局部时只画该局部），每个实体格只画该实体，每个场景格只画无人物、动物和可移动实体的空置环境，不能重演源视频关系。Seedream 每项目调用一次，输入固定为 `Image 1=2048×2048 空白布局`，存在绑定时追加 `Image 2=规范化用户图`；不再生成或发送源帧证据。用户图只可影响 `source=image_1` 对应的 stable key，后端把这个 Object Replace/H3 用户图角色确定性映射到 Seedream Image 2，不扩散到其他 tile。输出仍只做解码、尺寸归一、标签复写及文件/哈希绑定；不增加替换覆盖或画面效果的本地质量门禁。

新增 `skills/h3-max-r2v-prompt/SKILL.md` 每段执行一次。Fusion 的业务事实源包括本段 `visual_prompt`、完整 `enhanced_element_index`、本段本地化后的翻译台词，以及后端冻结的 `h3_reference_images`/`replacement_sheet` 元数据；不再单独传入用户说明或独立音色选择。Skill 负责读取索引中的 replacement/voice，并指示 H3 从权威 `stable key -> TILE_XX -> 实际 Image N` 对应格取得新形象、替换 Video 1 中的源形象；后端仅根据实际存在的字段 best-effort 生成 `authoritative_constraints`，不把它当成质量门禁。增强索引继续保存 raw `preserve`，但 `REPLACE` 项只编译动作、功能、尺度、接触、相对位置、运动连续性、人物明确属性及场景色调/构图/机位/光线等类别安全不变量；可能冲突的源脸、服装款式、物体外观和场景陈设不会带入，raw `preserve` 只直接用于 `KEEP`。

内部 Codex 和 DeepSeek Fusion 共用动态 closed JSON Schema：固定 `schema/version`，并把 `input_sha256/segment_index` 绑定到本次输入，模型填写字符串 `prompt`。本地只校验这套结构和绑定，不扫描正文里的 stable key、TILE、`Image N`、`Audio N` 或替换覆盖率。后端把能从增强索引和图版 spec 得到的对象/图片映射、目标语言台词和时间窗机械附加，并按请求的 remove_subtitle/remove_logo 默认注入 Overlay removal 块（要求成片不保留源视频的字幕、文字、水印和 Logo；注意 MediaKit 擦除只作用于分析关键帧，不改变送给 H3 的静音 Video 1）；缺少可选语义时继续出片。关系任一端被替换时，权威约束只保留通用交互类型、接触、相对位置和运动连续性，不带入指向源对象外观的 predicate/preserve；两端都未替换时保留原关系描述。Codex 路径仍保留独立 `/tmp` stage、bwrap 遮蔽 checkout、只读输入、剔除秘密环境变量、agent shell 断网、进程组硬超时和唯一有界输出的隔离边界。

随后通过 Picsart Advanced workflow `minimax/h3-max/reference-to-video` 提交最终提示词及 H3 参考素材：

- `prompt_expansion_mode`: `quality`
- `reference_video_urls`: 当前静音片段 Video 1 的临时签名公网 URL
- `reference_image_urls`: 每段相同的有序图片 URL；新形象图版固定为 `Image 1`（身份锚点占主参考位），存在唯一有效用户图绑定时追加用户图，对应 `Image 2`；没有该绑定时不扩散原始用户图，仅发送 `[新形象图版]`
- `reference_audio_urls`: 有台词段为该段 TTS 台词语音切片（H3 的 Audio 1，只用于口型同步）；无台词段省略字段。音色库 wav 只作 TTS 克隆参考，不再送 H3
- `duration`: `max(5, ceil(原片段秒数))`
- `resolution`: 前端 `480p`/`768p` 映射为供应商 Advanced API 的 `480P`/`768P`
- `aspect_ratio`: 前端请求值

每个 segment 都必须提交完全相同且顺序相同的整组图片，不允许按段过滤、重排、截断或重新去重，否则提示词中的 `Image N` 会漂移；音频只携带本段台词切片，段间天然不同。用户图片沿用 `duet-ad1` 的 MiniMax 图片代理机制：内容保持原比例等比缩放；宽、高均限制在 `256–5760`，宽高比限制在 `1:2.5–2.5:1`；极端比例通过黑边补齐，不拉伸、不裁掉内容。签名素材端点同时支持 `GET` 和 `HEAD`。

## API、存储与技术边界

- 七组极简前端 API 已实现，`creation_input` 为前端要求的 `{version, source, replacement_image}`。
- 项目只使用 `queued/running/succeeded/failed` 四态；成功快照同时包含 `percent=100` 和 `has_video=true`。
- 上传使用服务端固定磁盘名并流式执行字节上限；外链视频逐跳做 URL、DNS、全局 IP、重定向和累计字节限制。
- 通用文件 API 只允许源视频与成功态最终视频；状态、日志和中间文件不能下载。
- 供应商素材 token 与项目、asset 和真实项目内路径绑定；归一化替换图以 `.png` 和 `image/png` 提供，并支持 `GET/HEAD`。
- `work/element_index.json`/`state.element_index` 是 raw index；`work/enhanced_element_index.json`/`state.enhanced_element_index` 是机械合并后的下游权威，两层不互相覆盖。
- `state.object_replace.version=3` 冻结输入哈希和目录哈希；`state.replacement_sheet.version=4` 冻结 `spec.version=2`、Seedream 输入/提交/输出回执、最终图版文件回执及 H3 图片角色；`state.voice_references` 是与 `work/voice-references/audio-XX.wav` 一一对应的有序列表，包括 `audio_reference`、`voice_id`、项目内路径、文件哈希和目录特征行。
- `state.h3_prepared_artifacts.version=4` 逐段冻结 source、muted、prompt 的路径/大小/哈希、Fusion 输入哈希与 segment metadata 哈希；metadata 覆盖 index、start/end、duration 和 join mode。bindings 同时绑定 enhanced index、replacement sheet、voice references、Object Replace 输入、可选用户图及 output request。Object Replace、sheet bundle、spec 和 H3 manifest 的版本字段都必须严格等于当前合同值；缺失或不等立即失败，不能转换、推断或混用。
- `state.picsart_submission_receipts` 是付费恢复权威：按 segment/run 保存 request 哈希、`submission_started/accepted/unknown/rejected/provider_failed` 和 task ID。研究日志可以丢失，但不能改变是否允许再次 POST 的判定。
- 状态文件原子写入；项目与日志目录为 `0700`，文件为 `0600`。
- 原始请求、响应、headers、密钥、签名 URL、媒体副本、异常和命令输出按节点/attempt 本地保存，不脱敏；日志失败不阻断业务。
- 只支持单进程实例；同一数据目录由生命周期文件锁阻止重叠 worker。

## 重试与付费边界

普通节点初始执行一次，失败后最多再尝试一次。Seedream 生成 POST 和 Picsart submit 都不使用通用重试：

- Seedream 在 POST 前原子持久化与模型、prompt 和输入图片哈希绑定的 receipt/claim；已有成功响应只复用同一结果，`submission_unknown`、明确拒绝和协议错误都不会自动再次 POST。未知结果必须先人工对账。
- 每次允许的 submit 在 POST 前先原子冻结 `submission_started` 回执；拿到 `taskId` 后立即写 accepted。无 task ID 的未知结果禁止 resume 重提；只有 submit 自身明确被拒绝，或已有 task 的业务状态明确为 `FAILED`，才允许后续再次 POST。poll 的网络错误或 HTTP 4xx/5xx 都不证明 task 已失败。
- 一旦取得 `taskId`，poll 和 download 的失败只重试自身，不能返回 submit。
- 磁盘写满（ENOSPC）是暂时性故障：节点内按通用退避策略不限次重试；失败终态落盘（`_persist_failure`）和 `pipeline_entry` 兜底 `mark_failed` 在 ENOSPC 下同样重试直到写成功，避免 worker 死了 status 仍停在 `running` 的假死。研究日志写入本身 best-effort，不会在重试路径上再炸一次。按文件存在性复用的断点产物（MediaKit stage checkpoint、optimized/proxy/关键帧、节点 artifacts）复用前必须非空且图片类可解码（`media.is_decodable_image`），0 字节/损坏文件视为不存在并重新生成；记录里带确定性 4xx（除 408/429）的 ProviderFailure 不再无限重试，直接失败，只有网络/5xx/超时类无限退避。
- `POST /api/conversations/{id}/resume-pre-picsart` 只接受从未出现 Picsart submit 痕迹的失败项目。死在分段前（无 segments 状态）的项目走 early resume：按已落盘节点产物逐节点复用或重跑，补齐分段后照常继续。没有图版 checkpoint 时，它复用原切段、4 fps manifest、exact-9、MediaKit 帧、分析副本和逐段 video-maker，再重建免费上游；存在 checkpoint 时先在零写入边界验证冻结上游与图版绑定，`prepared` 只首次提交或本地 replay 同一 Seedream receipt，`succeeded/not_needed` 直接复用，均不重跑翻译、raw index 或 Object Replace。未知/损坏的付费证据停止恢复，不能盲目再 POST。兜底逻辑上线前冻结的 voice_references 为空但有待合成台词的旧项目，续跑装载后自动从音色库物化默认音色为 Audio 1 并重算图版 upstream_bindings（voice_reference_default_fallback 事件，reason 带 resume: 前缀）。
- `POST /api/conversations/{id}/resume-picsart` 只复用 H3 manifest v4 冻结的静音 Video 1、R2V 提示词、同一张图版、相同顺序的有序图片和有序音频。它不重跑 raw/enhanced 索引、Object Replace、Seedream 或 Fusion；严格检查版本、路径、哈希、媒体和请求绑定，不把图版画面或 voice 与 enhanced index 做语义交叉复审。已有且未明确失败的 task ID 只继续 poll；只有供应商业务结果明确为 `FAILED`，或 submit 自身明确被拒绝且未产生 task ID，才允许重新 submit。
- 不因生成质量、替换效果或媒体观感增加重试。
- 明确决策（2026-09-07）：不做段级产出验收 + 自动重跑。H3 单段产出存在随机波动（场景替换、字幕复现都有随机性，已由受控实验证明与 prompt/素材无关），但我们的目标是通过输入侧控制让一次就跑过——凡是不想让 H3 保留的信息，在参考视频进 H3 之前从像素层面抹掉（fal.ai SAM 统一深灰块打码、MediaKit 视频字幕擦除），而不是生成后再验收补救。
- Whisper 返回空 ASR，或 YAMNet 与 ASR 没有重叠 spoken 区间时，直接以空台词继续，不因内容为空进行复核、失败或重试。

## 验证状态

本工作树新增/调整的离线用例覆盖以下合同；最终全量结果需以本工作树合并完成后的实际测试为准：

- Object Replace v3 每项目只设一个公共节点，不随 segment 重复；后端从 raw index 确定性生成包含全部排序 stable key 的 `replacement_set`，模型只收到移除三类记录顶层 `preserve/replaceable` 的规划视图，relations 不变；nullable replacement/voice 可直接流向增强索引，非空 replacement 的 `appearance_description` 与 `target_visual_description` 分别只流向图版和 Fusion/H3。
- project-index 收到逐段 `visual_prompt` 防漏清单，但仍以九帧像素为唯一视觉事实；动态 Schema 只接受本次请求中的精确正整数段号，拒绝 `0`、未知段号和重新编号。
- raw index 不变，enhanced index 由后端机械合并；人物共用可用 `voice_id` 时共用同一 `Audio N`，第四个不同音色稳定降级而不阻断。
- sheet bundle v4/spec v2 只从 frozen enhanced index 生成，确定性绑定全部实际非空 replacement 的 stable key/TILE/行列；Seedream 每项目只生成一张图版，仅接收 canvas Image 1 和可选用户 Image 2，不生成/发送源帧证据，人物、实体与空置场景严格分格描述。H3 图片顺序固定为图版 Image 1、可选用户图 Image 2，保持冻结且跨段一致。
- 每段 Fusion 收到视觉提示、enhanced index、本段翻译台词和后端冻结的图片角色/TILE 映射；原始用户说明、图片二进制和独立音色字段不能泄漏进 Fusion。
- `REPLACE` 权威约束不带入冲突的源脸、服装款式、物体外观或场景陈设；关系任一端被替换时也不带入指向源对象外观的 predicate/preserve，只保留通用交互、接触、相对位置和运动连续性。`KEEP` 项及两端均未替换的关系仍保留各自 raw 描述。
- Seedream 的 pre-POST receipt 会阻止 `submission_unknown` 自动重投；图版后处理不以视觉效果阻断项目。
- 每个 Picsart segment 收到同一顺序的图片：有唯一有效用户图绑定时为用户图、图版；否则只有图版。图片与 Fusion 中的实际 `Image N` 映射一致。
- 每个 Picsart segment 收到同序的 0/1/2/3 条音频；超限、重复、空 URL 在网络请求前被拒绝。
- `resume-pre-picsart` 只按当前版本合同重建公共屏障；`resume-picsart` 只使用 H3 manifest v4 冻结的 prompt、同一图版、有序图片和音频，不重跑 Seedream/Fusion。版本字段缺失或不等时立即停止。

非空 `client_request_id` 已实现持久化创建幂等：绑定规范化请求与实际输入文件摘要；同键同输入返回原项目且不重启流水线，同键异输入返回 `409`，索引无法可信读取时以 `503` 停止创建。当前 v3/v4 合同的最终测试结果待本工作树合并完成后实测。
