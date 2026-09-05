# MiniMax H3 Max R2V 研究版架构

## 目标与基线

本仓库的本质目标不是改造生产后端，而是让固定 Master 极简前端拥有一个独立、可研究的新后端。前端基线与 `video-maker` 之前的算法基线固定为 `github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf`；源仓库的后续变化不会隐式改变本链路。

技术边界与质量判断严格分开：服务端负责让请求安全、可归属、可执行和可追踪，但不判断生成内容“够不够好”。供应商拒绝、网络错误、命令失败和文件缺失是自然执行错误；结构化模型节点只校验 closed JSON 形状、字段类型及输入绑定，不二次验证索引、替换、音色、台词或提示词语义，也不因此失败或重试。

## 数据流与屏障

项目级公共输入依次产生：

1. 原视频精确解码帧轴、4 fps 分析帧与规范音频。
2. YAMNet 声学分类和 Whisper 台词；自动台词只投影 spoken 行。
3. `ContentDetector(threshold=27)` 的精确镜头边界。
4. 4–10 秒 scene-first 片段计划及逐段 exact-9。
5. 可选 MediaKit 处理，先文本擦除、再图标擦除。
6. 每段 `video-maker` 视觉提示词。
7. 一次项目级台词翻译和一次 `video-maker` 全局原始对象索引；索引同时接收各段 `visual_prompt` 作为回看九帧的防漏清单，像素仍是唯一视觉事实。
8. 后端从 raw stable index 确定性生成 `replacement_set`：`people/entities/scenes` 分别包含本类全部 stable key 并排序；同时生成仅移除这三类记录顶层 `preserve/replaceable` 的模型规划视图，relations 和其他事实不变。
9. 一次项目级 DeepSeek Object Replace v3：读取规划视图、`replacement_set`、全部逐段视觉提示、目标语言与项目级翻译台词、可选用户图及说明，以及当前音色目录；输出绑定 stable key 的 replacement/voice patch，每个非空 replacement 分离 `appearance_description` 与 `target_visual_description`。
10. 后端深拷贝完整 raw index 并机械合并 patch；对可用的 `voice_id` 按稳定顺序映射 `Audio N`，生成项目级增强索引和同序的 0–3 条冻结音频。
11. 后端以冻结 enhanced index 为唯一身份锚点，把所有实际非空 replacement 确定性映射为 `stable key -> TILE_XX -> row/column`，生成 sheet bundle v3/spec v2 并从 `appearance_description` 编译隔离式提示词；Seedream 每项目调用一次，以空白布局为 Image 1、可选规范化用户图为 Image 2，生成一张 2048×2048 新形象图版，不生成或发送源帧证据。

只有上述公共输入完成并能传给下游，才启动逐段 R2V 准备。所有片段先并行完成 Fusion、最终提示词和静音 Video 1；随后完成有序参考图、素材 URL 注册和 Picsart 硬参数预检，才允许发出 Picsart POST。这些屏障只防止缺文件、供应商参数不合法和半批次付费，不判断模型内容质量。

每段分支按以下顺序执行：

```text
h3-max-r2v-prompt Skill
  -> 后端机械附加台词与相对时间窗
  -> H3 接收静音源片段 Video 1、项目级同序 Images 和项目级同序 Audio 1..N
  -> Picsart submit
  -> 使用同一 task ID poll
  -> 立即下载供应商结果
  -> 24 fps 帧预算裁回源片段时长
```

最终按原 segment 顺序统一媒体规格并 concat。

## 上游算法语义

- 分段在每个起点之后的 4–10 秒窗口内优先选择最晚真实镜头边界；没有候选才使用时长切点；不足 4 秒的尾段会回撤前一切点。
- 台词区间只做技术解析，不参与切点选择。跨段台词按各段时间交集切分文本，再转换为段内相对时间。
- exact-9 从原视频完整 PTS/解码序号清单中选取。场景不超过九个时先给每个场景一帧，再用 Hamilton 最大余数法分配剩余名额；稀疏输入确定性重复；超过九个场景时压成九个连续分析槽，但保留完整 source cut timeline。
- MediaKit 的文本擦除先于图标擦除。处理后的全尺寸九帧是两个 `video-maker` 阶段的共同权威输入；送模型时另建半尺寸分析副本。

## 模型合同

`video-maker` 以固定 Master 的既有 Skill 为基线；逐段视觉合同保持不变，项目索引阶段增加防漏清单和精确段身份绑定：

- 逐段阶段用 `VISUAL_PROMPT_SCHEMA` 返回 `{prompt}`，业务层只解包字符串；不设置任意最短长度或内容覆盖门禁。
- 项目索引阶段接收 `phase`、生效的 `generation_config`，以及按 segment/frame 排序的 `segment_index`、`visual_prompt`、`frame_order`、项目内路径、SHA 和全部分析图片；不接收台词或用户素材。`visual_prompt` 只用于逐项提示模型回看对应九帧，不能替代或推翻像素证据。运行时根据本次请求中的正整数段号构造动态 closed Schema，人物、实体、场景和关系的 occurrence 只能引用这组精确段号，不能输出 `0`、未知段号或自行重新编号。输出 closed 结构后由后端机械分配已有项的稳定 ID，原样写入 `work/element_index.json` 和 `state.element_index`。people/entities/scenes/relations 允许为空，不要求必须识别场景或达到任意集合大小；Object Replace 不回写 raw index。

项目级 `object-replace` Skill 固定使用 DeepSeek，每项目只设一个公共逻辑节点，不按 segment 重复；首次自然执行错误仍受下文“每节点最多两个 attempt”边界约束。调用前，后端从 raw stable index 确定性生成 `replacement_set={people,entities,scenes}`，每类包含全部 stable key 并排序；送模型的 `global_element_index` 是深拷贝的规划视图，仅移除这三类记录顶层的 `preserve/replaceable`，relations 和其他字段不变。完整 raw index 始终冻结，并在返回 patch 后作为机械合并基底；Fusion 接收的 enhanced index 因而仍包含完整 raw 事实。

closed JSON Schema v3 由 raw index 中已有 stable key、音色 ID 集合和输入哈希动态生成；patch 字段中的 `replacement` 与人物 `voice_id` 均允许 `null`。每个非空 replacement 包含 `source`、`appearance_description` 和 `target_visual_description`：`appearance_description` 只供 sheet，限定为单个 stable key 的内在可见外观，不能写动作、接触、相对位置、所在场景或其他 stable key；`target_visual_description` 只供 Fusion/H3 承载动作、接触与关系。Skill 自身要求 `replacement_set` 中所有人物、实体和场景输出非空替换，并为人物选择符合特征的音色。同一 stable key 跨图片、分段和 occurrence 使用同一套新设计。用户图只约束用户文字唯一语义对应的一个既有 stable key；其他 key 仍各自使用 `generated_variant`，不得因参考图只指向一个元素而停止替换。人物保持索引明确的类型、性别、种族、年龄段、整体风格和服装主色，但更换身份，使人脸或可见身体产生差异；只露出身体局部时，统一改变局部形态、比例、肤色、纹理或标记，不为具体部位写特例；服装保持颜色和风格但改变款式。场景保持类型、用途、布局、色调、构图、机位、镜头运动、焦段和光线，但改成同类不同实例。实体保持功能、动作接口和尺度，但改变材质、颜色、品种、轮廓、结构或造型；动作、接触与相对关系不得回流到图版外观。上述都是模型生成职责；后端只校验结构和输入绑定，不把 replacement 非空、用户目标命中、替换覆盖、关系协同或声音标签语义变成质量门禁。

后端在 patch 结构可读取后深拷贝 raw index 并机械合并：`occurrences`、`preserve`、relations 和原始描述不变，对应记录增加模型给出的 nullable `replacement`，人物增加 nullable `voice`。人物可共用 `voice_id`；后端按稳定人物顺序保留最先出现的最多 3 个不同可用 ID，后续新 ID 稳定降级为 `null`，而不是让项目失败。保留下来的文件依次复制为 `work/voice-references/audio-01.wav` 至 `audio-03.wav` 并写入增强索引。当前仓库附带 30 条 2–5 秒单声道 PCM WAV，但运行时不要求目录恰好 30 条，也不复审年龄、性格等描述是否主观准确；只保留文件可读、媒体可用和供应商最多 3 条参考音频等技术限制。

图版节点不让 Seedream 重新建立身份：sheet bundle v3/spec v2 的 TILE 顺序、位置、stable key、类别和 `appearance_description` 都由后端从 frozen enhanced index 生成并冻结，Seedream 只在每个固定格内画出对应新设计。人物格只包含该人物或可见身体局部，实体格只包含该实体，场景格必须是无人物、动物和可移动实体的空置环境；图版不是源视频画面的重演。Seedream 输入固定为 `Image 1=空白布局`，存在绑定时才追加 `Image 2=规范化用户图`，不生成或发送源帧证据。用户图只可影响 `source=image_1` 的 tile，不得扩散到其他 stable key；这里的 `source=image_1` 是 Object Replace/H3 的用户图角色，后端确定性映射到 Seedream Image 2。图版生成结果只做图片解码、统一尺寸、标签复写和文件/哈希绑定等技术处理；不新增“是否画得像”“是否每格替换成功”之类的本地视觉质量门禁。

新 `h3-max-r2v-prompt` Skill 每段执行一次并通过 closed JSON Schema 返回提示词。它的业务事实源是本段视觉提示、完整增强索引、本段翻译台词，以及后端冻结的 `h3_reference_images`/`replacement_sheet` 映射；不再传用户原始说明，也不接收独立音色输入。Skill 负责读取增强索引，并让 H3 从权威 `stable key -> TILE_XX -> 实际 Image N` 指向的图版格取得新形象、替换 Video 1 中对应源形象。后端只校验返回对象的 closed 结构、`input_sha256` 与 `segment_index`，随后根据增强索引和图版 spec 中实际可用字段 best-effort 编译并附加约束块及台词时间窗。增强索引保存 raw `preserve`，但 `REPLACE` 项编译时只保留动作、功能、尺度、接触、相对位置、运动连续性、人物明确属性、场景色调/构图/机位/光线等类别安全不变量，不带入冲突的源脸、服装款式、物体外观或场景陈设；raw `preserve` 只直接用于 `KEEP`。关系任一端被替换时，同样只保留通用交互类型、接触、相对位置和运动连续性，不带入指向源对象外观的 predicate/preserve；两端都未替换时保留原关系描述。后端不扫描 Fusion 正文是否出现或遗漏 stable key、TILE、`Image N`、`Audio N`，也不因增强索引中 nullable/缺失的替换或音色而失败。最终语义覆盖与画面执行都交给模型和供应商。

Picsart 使用 Advanced workflow `minimax/h3-max/reference-to-video`。提交地址为 `/workflows/minimax/h3-max/reference-to-video/submit`，请求体只有一层 `params`，字段使用供应商原生的 `reference_video_urls`、`reference_image_urls`、`reference_audio_urls`、`aspect_ratio` 和 `prompt_expansion_mode: "quality"`；分辨率映射为 `480P`/`768P`。对 H3 而言，当前静音片段是 Video 1；存在唯一有效的用户图 stable-key 绑定时，`reference_image_urls=[用户图, 新形象图版]`，即 `Image 1=用户图`、`Image 2=图版`；没有该绑定时不发送原始用户图，`reference_image_urls=[新形象图版]`，图版为 `Image 1`。冻结音频列表按原序对应 `Audio 1..N`。所有 segment 都提交同一份、同一顺序的完整图片与音频列表，不按段删除、去重、排序或重新编号。用户参考图先复用 `duet-ad1` 的 MiniMax 代理规则，等比落入 `256–5760` 双边和 `1:2.5–2.5:1` 比例范围，必要时补黑边，再编码为 PNG。轮询同一 workflow 下的 `/{taskId}/result`，从 `response.result.video.url` 读取结果。4.x 秒片段提交为 5 秒，结果再裁回源时长。

## 原始/增强索引与续跑边界

- `work/element_index.json` 和 `state.element_index` 始终保留 `video-maker` raw index；`work/enhanced_element_index.json` 和 `state.enhanced_element_index` 只保存后端机械合并结果。两者不覆盖、不混用。
- `state.object_replace.version=3` 保存 `input_sha256` 与音色目录哈希；`state.replacement_sheet.version=3` 保存 frozen `spec.version=2`、Seedream 输入/提交/输出回执、H3 图片角色和最终图版文件回执；`state.voice_references` 保存按 `Audio 1..N` 排列的 `voice_id`、项目内路径、文件哈希与特征行。增强索引先原子落盘，再与这些项目级元数据一起更新 state；签名 URL 不是冻结状态。
- `state.h3_prepared_artifacts.version=4` 在所有段免费准备完成后一次发布，逐段冻结 source、muted、prompt 的路径/大小/哈希、Fusion 输入哈希和 segment metadata 哈希；metadata 覆盖 index、start/end、duration 与 join mode。bindings 同时覆盖增强索引、冻结图版状态、音色列表、Object Replace 输入、可选用户图的文件回执和 output request。Object Replace、sheet bundle、spec 和 H3 manifest 的版本字段都必须严格等于当前合同值；任何缺失或不等都立即失败，不能转换、推断或跨版本拼装。`state.picsart_submission_receipts` 记录每段每次允许提交的 request 哈希、run identity、状态和 task ID；研究日志只是证据副本，不承担付费恢复真相。
- `resume-pre-picsart` 仅允许没有任何 Picsart submit 痕迹的失败项目。没有图版 checkpoint 时可复用合格的源片段、exact-9、MediaKit/分析副本和逐段视觉提示，再按当前合同重建免费上游；一旦存在图版 checkpoint，就必须先在零写入边界校验冻结的版本字段、segments、source、enhanced index、Object Replace、音色、请求、prompt 和有序输入。`prepared` 只允许首次提交或本地回放同一 Seedream receipt，`succeeded/not_needed` 直接复用，均不得重跑上游或另建 stable-key 映射。
- `resume-picsart` 已越过付费屏障，只复用 H3 manifest v4 冻结的每段 `source-muted.mp4`、`h3_prompt.txt`、同一张图版和相同顺序的有序图片，以及按哈希验证的有序音频文件；它不重跑 raw index、Object Replace、Seedream 或 Fusion。恢复严格校验版本、路径、哈希、媒体和请求绑定，不把图版画面或 `voice_references` 与 enhanced index 做二次语义复审。已有且未明确失败的 task ID 只继续 poll/download。

## 重试、证据与付费边界

普通节点有独立的 `attempt-1`/`attempt-2`，首次抛出自然执行错误后最多重试一次，不根据生成质量触发重试。Seedream 生成 POST 与 Picsart submit 都是付费边界，不使用这条通用重试规则。Whisper 返回空 ASR，或 YAMNet 与 ASR 没有可重叠的 spoken 区间时，直接以空台词继续，不因内容为空再跑一次；翻译结果缺行或某行为空时回退对应源文，也不失败或重试。

Seedream 在发出 POST 前原子持久化与 prompt、模型和输入图片哈希绑定的 receipt/claim。已存在成功响应时只物化同一结果；明确拒绝、协议错误或提交结果未知都不会自动再 POST，其中 `submission_unknown` 必须先人工对账。

Picsart submit、poll、download 是三个节点。每次被允许的 submit 在 POST 前先原子冻结 `submission_started` 回执；取得 `taskId` 后立即写成 accepted 回执，再进入 poll。没有 task ID 的未知结果默认禁止 resume 再 POST，必须先对账；只有 submit 自身被供应商明确拒绝，或已有 task 的业务状态明确为 `FAILED`，才可标记为允许重提。poll 的传输错误和 HTTP 状态都不能证明 task 失败。取得 ID 后，poll 和 download 的失败只重试自身，不能回到 submit。

供应商适配器把业务值与 provider record 分离。业务对象、raw/enhanced 索引和下游 prompt 不携带原始供应商请求；原始记录只进入对应 attempt 日志。二进制响应作为媒体证据保存，JSON 只记录状态、headers、长度和摘要，不把二进制误解码成文本。

日志失败只能写进进程 stderr，不能改变业务结果。日志目录逐层为 `0700`，文件为 `0600`，默认永久保留并由人工删除。

## 主机与访问边界

- 登录和前端 API 使用 Bearer 鉴权；供应商素材 URL 只使用独立高熵、项目绑定且限时的 token。
- 上传磁盘名由服务端决定，客户端 filename 只作显示元数据；视频和替换图分别执行字节上限。
- 外链视频每次跳转都重新做 URL、DNS 与全局 IP 校验，实际连接钉扎到已批准 IP，同时保留 Host/SNI；下载流同时受 Content-Length 和实际累计字节限制。
- 项目文件路由是逻辑名 allowlist，不接受任意项目相对路径。签名素材在注册和读取时都重新验证真实路径属于对应项目，`GET` 与供应商预检使用的 `HEAD` 共用同一验证路径。
- 状态只使用 `queued/running/succeeded/failed`。纯文件存储、后台任务和内存 token 表限定单实例；同一数据目录不允许重叠进程。
