# MiniMax H3 Max R2V 新链路交接

## 当前状态

- 工作区：`/home/xy/duet-ad1-h3-max-r2v`
- 分支：`main`
- 前端固定基线：`github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf`
- 本轮工作树把旧的项目级单音色选择改为一次项目级 DeepSeek Object Replace；原始索引、增强索引、人物音色映射和有序 Audio 列表分层冻结。
- 逐段 Fusion 仍由后端 `FUSION_AGENT` 选择 `codex` 或 `deepseek`，缺省为 `deepseek`；前端和公开 capability/request 不暴露该选择，客户端夹带的 `fusion` 会被服务端配置覆盖并在创建时冻结。Object Replace 始终使用 DeepSeek，不受 `FUSION_AGENT` 影响。
- DeepSeek 代码路径显式开启 thinking，不发送供应商级输出 token 上限；video-maker 两阶段、项目级翻译、Object Replace 和 DeepSeek Fusion 均使用 Responses API 原生 `text.format=json_schema`。本地只处理 closed 结构与输入绑定，不做模型语义二次校验。
- 当前功能代码基线为 `3184ea7`，已部署到 3301；完整离线结果为 `259 passed, 1 warning`。
- 最近一次真实付费 E2E 项目 `265935eb58f0441bbb4bcb2621039688` 验证的是 `8c1515d` 与 `285733b`，不是 `3184ea7`。`3184ea7` 的关系约束修复目前只有离线测试与部署证据，没有新的付费 E2E。

## 已实现链路

```text
保存原视频和可选替换图
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
  -> 一次项目级 DeepSeek Object Replace：规划视图 + replacement_set + 全部视觉提示 + Image 1/说明 + 当前音色目录
  -> 后端机械合并 enhanced 索引，按可用 voice_id 生成同序 Audio 1..N（N <= 3）
  -> 公共输入屏障
  -> 第一屏障：所有段并行完成 Fusion、最终提示词和静音 Video 1
  -> 第二屏障：所有段完成 Image 1 归一化、素材注册和 H3 参数预检
  -> 每段并行 Picsart H3：Video 1 + 可选 Image 1 + 所有段完全同序的 Audio 1..N
  -> 供应商片段裁回原时长并统一为 24 fps
  -> 按原顺序合并 output/generated.mp4
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
- 可选 Image 1 与用户替换说明；有图时同时传图片本身和 SHA-256。
- 目录 SHA-256 及当前可用的 `voice_id - file - language - gender - 短特征` 行。仓库当前提供 30 条，但运行时不要求恰好 30 条。

Object Replace 不回吐完整索引，只通过动态 closed JSON Schema 输出与 raw stable ID 对齐的 patch；`replacement` 和人物 `voice_id` 都允许为 `null`。Skill 自身要求 `replacement_set` 中所有人物、实体和场景输出非空替换，并为人物选择合适音色。同一 stable key 跨图片、分段和 occurrence 使用同一套新设计。Image 1 只约束用户文字唯一语义对应的一个既有 stable key；其他 key 仍使用 `generated_variant`，不因参考图只指向一个元素而停止替换。

人物保持索引明确的类型、性别、种族、年龄段、整体风格和服装主色，但更换身份，使人脸或可见身体产生差异；只露出身体局部时，统一改变局部形态、比例、肤色、纹理或标记，不为具体部位写特例；服装保持颜色和风格但改变款式。场景保持类型、用途、布局、色调、构图、机位、镜头运动、焦段和光线，但改成同类不同实例。实体保持功能、动作接口、尺度、接触和相对关系，但改变材质、颜色、品种、轮廓、结构或造型。完整 raw index 仍冻结，并作为 patch 的机械合并基底；Fusion 读取的 enhanced index 因而保留完整 raw 事实。上述是模型生成职责：Schema/normalizer 仍允许 nullable，后端不逐项复审，也不因未完成而失败或重试。

patch 结构可读取后，后端深拷贝 raw index，机械添加 nullable replacement 和人物 voice，原样保留 stable ID、原始描述、occurrences、preserve 与 relations，并写入 `work/enhanced_element_index.json` 与 `state.enhanced_element_index`。后端按稳定人物顺序保留最先出现的最多 3 个不同可用 `voice_id`；第四个及后续新音色稳定降级为 `null`，不会阻断项目。保留项依次分配 `Audio 1..N` 并复制为 `work/voice-references/audio-01.wav` 至 `audio-03.wav`。运行时只校验音频文件与 Picsart 输入可用性，不复审年龄、性格等标签。

新增 `skills/h3-max-r2v-prompt/SKILL.md` 每段执行一次。Fusion 的业务事实源只有本段 `visual_prompt`、完整 `enhanced_element_index` 和本段本地化后的翻译台词；调用不传任何图片，也不再单独传入用户说明、Image 1 元数据或音色选择。Skill 负责读取索引中的 replacement/voice 并将其编译进提示词；后端仅根据实际存在的字段 best-effort 生成 `authoritative_constraints`，不把它当成质量门禁。增强索引继续保存 raw `preserve`，但 `REPLACE` 项只编译动作、功能、尺度、接触、相对位置、运动连续性、人物明确属性及场景色调/构图/机位/光线等类别安全不变量；可能冲突的源脸、服装款式、物体外观和场景陈设不会带入，raw `preserve` 只直接用于 `KEEP`。

内部 Codex 和 DeepSeek Fusion 共用动态 closed JSON Schema：固定 `schema/version`，并把 `input_sha256/segment_index` 绑定到本次输入，模型填写字符串 `prompt`。本地只校验这套结构和绑定，不扫描正文里的 stable ID、`Image 1`、`Audio N` 或替换覆盖率。后端把能从增强索引得到的约束、目标语言台词和时间窗机械附加；缺少可选语义时继续出片。关系任一端被替换时，权威约束只保留通用交互类型、接触、相对位置和运动连续性，不带入指向源对象外观的 predicate/preserve；两端都未替换时保留原关系描述。Codex 路径仍保留独立 `/tmp` stage、bwrap 遮蔽 checkout、只读输入、剔除秘密环境变量、agent shell 断网、进程组硬超时和唯一有界输出的隔离边界。

随后通过 Picsart Advanced compatibility workflow `minimax/h3-max/reference-to-video` 提交最终提示词及 H3 参考素材：

- `prompt_expansion_mode`: `quality`
- `reference_video_urls`: 当前静音片段 Video 1 的临时签名公网 URL
- `reference_image_urls`: 可选 Image 1 经过 MiniMax 约束归一化后的 PNG 临时签名公网 URL
- `reference_audio_urls`: 冻结的 0–3 条音频，按 `Audio 1..N` 原序写入；空列表时省略字段
- `duration`: `max(5, ceil(原片段秒数))`
- `resolution`: 前端 `480p`/`768p` 映射为兼容 API 的 `480P`/`768P`
- `aspect_ratio`: 前端请求值

每个 segment 都必须提交完全相同且顺序相同的整个音频列表，不允许按段过滤、重排、截断或重新去重，否则提示词中的 `Audio N` 会漂移。用户图片沿用 `duet-ad1` 的 MiniMax 图片代理机制：内容保持原比例等比缩放；宽、高均限制在 `256–5760`，宽高比限制在 `1:2.5–2.5:1`；极端比例通过黑边补齐，不拉伸、不裁掉内容。签名素材端点同时支持 `GET` 和 `HEAD`。

## API、存储与技术边界

- 七组极简前端 API 已实现，`creation_input` 为前端要求的 `{version, source, replacement_image}`。
- 项目只使用 `queued/running/succeeded/failed` 四态；成功快照同时包含 `percent=100` 和 `has_video=true`。
- 上传使用服务端固定磁盘名并流式执行字节上限；外链视频逐跳做 URL、DNS、全局 IP、重定向和累计字节限制。
- 通用文件 API 只允许源视频与成功态最终视频；状态、日志和中间文件不能下载。
- 供应商素材 token 与项目、asset 和真实项目内路径绑定；归一化替换图以 `.png` 和 `image/png` 提供，并支持 `GET/HEAD`。
- `work/element_index.json`/`state.element_index` 是 raw index；`work/enhanced_element_index.json`/`state.enhanced_element_index` 是机械合并后的下游权威，两层不互相覆盖。
- `state.object_replace` 冻结 schema 版本、输入哈希和目录哈希；`state.voice_references` 是与 `work/voice-references/audio-XX.wav` 一一对应的有序列表，包括 `audio_reference`、`voice_id`、项目内路径、文件哈希和目录特征行。
- `state.h3_prepared_artifacts` v2 逐段冻结 source、muted、prompt 的路径/大小/哈希、Fusion 输入哈希与 segment metadata 哈希；metadata 覆盖 index、start/end、duration 和 join mode。bindings 同时绑定 enhanced index、voice references、Object Replace 输入、可选 replacement image 及 output request；它用于恢复时确认技术输入未漂移，不复审替换语义。
- `state.picsart_submission_receipts` 是付费恢复权威：按 segment/run 保存 request 哈希、`submission_started/accepted/unknown/rejected/provider_failed` 和 task ID。研究日志可以丢失，但不能改变是否允许再次 POST 的判定。
- 状态文件原子写入；项目与日志目录为 `0700`，文件为 `0600`。
- 原始请求、响应、headers、密钥、签名 URL、媒体副本、异常和命令输出按节点/attempt 本地保存，不脱敏；日志失败不阻断业务。
- 只支持单进程实例；同一数据目录由生命周期文件锁阻止重叠 worker。

## 重试与付费边界

普通节点初始执行一次，失败后最多再尝试一次。Picsart submit、poll、download 是三个独立节点，submit 不使用通用重试：

- 每次允许的 submit 在 POST 前先原子冻结 `submission_started` 回执；拿到 `taskId` 后立即写 accepted。无 task ID 的未知结果禁止 resume 重提；只有 submit 自身明确被拒绝，或已有 task 的业务状态明确为 `FAILED`，才允许后续再次 POST。poll 的网络错误或 HTTP 4xx/5xx 都不证明 task 已失败。
- 一旦取得 `taskId`，poll 和 download 的失败只重试自身，不能返回 submit。
- MediaKit 调用使用进程级 `MEDIAKIT_CONCURRENCY` 信号量；可配置为 `1` 以序列化多项目/多分段请求。
- `POST /api/conversations/{id}/resume-pre-picsart` 只接受从未出现 Picsart submit 痕迹的失败项目。它复用原切段、4 fps manifest、exact-9、已完成 MediaKit 帧、分析副本和已落盘的逐段 video-maker；随后重新生成翻译、raw index、Object Replace、enhanced index 和有序音频，通过新公共屏障后才能 submit。失败时的部分 raw/enhanced/state 组合不是已完成回执。
- `POST /api/conversations/{id}/resume-picsart` 只复用 v2 manifest 冻结的静音 Video 1、R2V 提示词、可选 Image 1 和有序音频。它不重跑 raw/enhanced 索引、Object Replace 或 Fusion；只检查路径、哈希、媒体和请求绑定，不把 voice 与 enhanced 人物索引做语义交叉复审。只有新字段缺失的历史项目才使用旧单音频兼容回退。已有且未明确失败的 task ID 只继续 poll；只有供应商业务结果明确为 `FAILED`，或 submit 自身明确被拒绝且未产生 task ID，才允许重新 submit。
- 不因生成质量、替换效果或媒体观感增加重试。
- Whisper 返回空 ASR，或 YAMNet 与 ASR 没有重叠 spoken 区间时，直接以空台词继续，不因内容为空进行复核、失败或重试。

## 验证状态

本次合同变更的离线测试已覆盖：

- Object Replace 每项目只设一个公共节点，不随 segment 重复；后端从 raw index 确定性生成包含全部排序 stable key 的 `replacement_set`，模型只收到移除三类记录顶层 `preserve/replaceable` 的规划视图，relations 不变；nullable replacement/voice 可直接流向增强索引。
- project-index 收到逐段 `visual_prompt` 防漏清单，但仍以九帧像素为唯一视觉事实；动态 Schema 只接受本次请求中的精确正整数段号，拒绝 `0`、未知段号和重新编号。
- raw index 不变，enhanced index 由后端机械合并；人物共用可用 `voice_id` 时共用同一 `Audio N`，第四个不同音色稳定降级而不阻断。
- 每段 Fusion 只收到视觉提示、enhanced index 和本段翻译台词，其输入媒体为空；原始用户说明、图片和独立音色字段不能泄漏进 Fusion。
- `REPLACE` 权威约束不带入冲突的源脸、服装款式、物体外观或场景陈设；关系任一端被替换时也不带入指向源对象外观的 predicate/preserve，只保留通用交互、接触、相对位置和运动连续性。`KEEP` 项及两端均未替换的关系仍保留各自 raw 描述。
- 每个 Picsart segment 收到同序的 0/1/2/3 条音频；超限、重复、空 URL 在网络请求前被拒绝。
- `resume-pre-picsart` 可为缺少 enhanced 字段的历史失败项目重建新公共屏障；`resume-picsart` 只使用冻结 prompt/媒体，新的空 `voice_references` 不得回退到旧音频，而字段缺失的旧单音频项目仍可恢复。

非空 `client_request_id` 已实现持久化创建幂等：绑定规范化请求与实际输入文件摘要；同键同输入返回原项目且不重启流水线，同键异输入返回 `409`，索引无法可信读取时以 `503` 停止创建。功能代码基线 `3184ea7` 的完整离线测试为 `259 passed, 1 warning`，相关 Skill 均通过 `quick_validate.py`。

## 2026-09-04 部署与真实 E2E

功能代码基线 `3184ea7` 已部署到 3301 实例；3301 的 Caddy 反向代理到本机 3302 后端。3213 仍是独立的原 `duet-ad1` 实例，未被本次部署改动。`3184ea7` 使关系任一端被替换时不再把源对象外观描述带进权威约束；该提交目前只有离线测试与部署证据，没有新的付费 E2E。

真实 A→B 测试从 3213 成功项目 `5607f61caeac41bf8168354187aa2599` 取得源视频、参考图和冻结请求配置，并以 multipart 上传到 3301。源视频是 16.95 秒、720×1280、30 fps、带 AAC 音轨的真实狗狗梳理画面，SHA-256 为 `6c55df6dcbcac2314f19b9c7d0cc4adb6bf73b7d63da5fa7ac15580cc1188cd4`；用户说明为“把视频中的狗替换成图片里的狗”。请求使用 9:16、480p、fit auto、中文自动改写，并开启图片优化、去标识和去字幕。

3301 新项目 `beebc22c68dd4b5fb934552a1dee944b` 从 queued 一次运行至 succeeded，没有调用任何 resume。两段均只有 `attempt-1`，每段各有且仅有一个 accepted Picsart submit receipt，没有第二次付费 POST。Object Replace、两段 Fusion、音色物化、Picsart poll/download/trim 和最终 stitch 均完成。

最终 `output/generated.mp4` 为 H.264 480×832、24 fps，带 AAC 48 kHz 双声道音频，时长 16.957683 秒，大小 1,743,355 字节，SHA-256 为 `409e8a71928681a8f3e77fd812063baaa7d6b18a365883527c598c1385a919fb`。全量 ffmpeg 解码成功；3301 公网下载返回 `video/mp4`，Range 请求为 206，完整下载字节数与 SHA-256 均和本地成品一致。两段抽帧均为连贯的真实狗狗梳理场景，白色犬只替换贯穿两段，不是色卡、空白或测试占位视频。

### 多候选验证矩阵

下表把“传输成功”和“视觉语义结论”分开；较早项目可以证明多种真实输入、付费回执与恢复链路，但不能自动证明后续提交的替换效果。

| 3213 来源项目 | 3301 项目 | 用户替换要求 | 分段与付费回执 | 结论边界 |
| --- | --- | --- | --- | --- |
| `5607f61caeac41bf8168354187aa2599` | `beebc22c68dd4b5fb934552a1dee944b` | 狗替换为参考图中的狗 | 2 段，各 1 条 accepted | 首个完整真实 E2E；狗替换贯穿两段。只露出手，不能据此评价换脸。 |
| `88da16ebf6604fa19d94f5c8346735d2` | `8b12473d76b643f2bbc10fe6ede359eb` | 逗猫棒替换为参考图中的逗猫棒 | 3 段；第 1、2 段各 1 条 accepted，第 3 段为 provider_failed 后 receipt-safe 重提并 accepted | 真实链路成功；当时项目索引漏掉视觉描述中的第三只猫，猫和场景差异偏弱，不能作为后续索引修复的通过证据。 |
| `03ed892b63bd44ba85d43553f8e7a40e` | `5e5931219c2844dea50bfd6ac2a4e1b3` | 项链替换为参考图中的项链 | 4 段；第 1、4 段各 1 条 accepted，第 2 段两次 provider_failed 后 accepted，第 3 段一次 provider_failed 后 accepted | 验证只对供应商明确失败段进行 receipt-safe 恢复；不扩展为最新替换语义证据。 |
| `7ef9e18fe47d4080a2f90f4c4ffd5044` | `04c089b55ecc4fe28db40ade504514bb` | 牙刷、杯子、药片等手持物替换为参考图小熊 | 5 段，各 1 条 accepted，无 unknown | 验证 `934dc2a` 强差异 Skill；小熊和场景替换明显，但人物变化偏弱且第 5 段仍有瓶子，暴露 raw preserve 冲突和跨段索引漏项。 |
| 同上 | `265935eb58f0441bbb4bcb2621039688` | 与上一行完全相同的 A/B 输入 | 首次在付费前因零起始段号停止，无 Picsart 回执；部署 `285733b` 后 `resume-pre-picsart`，5 段各 1 条 accepted，无 unknown | 项目 succeeded，验证 `8c1515d` 与 `285733b` 的提示词和精确段号修复，不验证后来的 `3184ea7`。逐时点 A/B 确认 7 类用户指定手持物和场景均明显替换，第 5 段不再残留绿、粉瓶；人物仍接近源人物，项链、手表、手链 3 个自动替换不明显。增强索引和 H3 提示词已包含这些目标，剩余缺口在 H3 的最终视觉遵循。 |

`265935eb58f0441bbb4bcb2621039688` 的成品为 H.264 768×1344、24 fps，带 AAC 48 kHz 双声道音频，时长 34.749349 秒，大小 6,720,643 字节，SHA-256 为 `05186a746da78aebfb25419c43e9bf138b7db7736db84fa5ed2be6bfb2ef70f7`。全量 ffmpeg 解码无错误；经 3301 鉴权完整下载返回 `video/mp4`，字节数、SHA-256 和逐字节比较均与本地文件一致。当前 `3184ea7` 已用该项目的真实增强索引离线重编译第 1 段权威块，确认不再含 `toothbrush` 或 `brush-head`，受影响关系只保留通用交互、接触、相对位置和运动连续性；这仍是确定性离线验证，不冒充新的付费 E2E。
