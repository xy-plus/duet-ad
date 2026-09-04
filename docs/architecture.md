# MiniMax H3 Max R2V 研究版架构

## 目标与基线

本仓库的本质目标不是改造旧生产后端，而是让固定 Master 极简前端拥有一个独立、可研究的新后端。前端基线与 `video-maker` 之前的算法基线固定为 `github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf`；源仓库的后续变化不会隐式改变本链路。

技术边界与质量判断严格分开：服务端负责让请求安全、可归属、可执行和可追踪，但不判断生成内容“够不够好”。供应商拒绝、网络错误、命令失败和文件缺失是自然执行错误；除固定 `video-maker` 两阶段既有结构化响应无法解析外，模型内容不完整或替换效果不理想不是后端门禁。

## 数据流与屏障

项目级公共输入依次产生：

1. 原视频精确解码帧轴、4 fps 分析帧与规范音频。
2. YAMNet 声学分类和 Whisper 台词；自动台词只投影 spoken 行。
3. `ContentDetector(threshold=27)` 的精确镜头边界。
4. 4–10 秒 scene-first 片段计划及逐段 exact-9。
5. 可选 MediaKit 处理，先文本擦除、再图标擦除。
6. 每段 `video-maker` 视觉提示词。
7. 一次项目级台词翻译和一次 `video-maker` 全局对象索引。
8. 若存在翻译后台词，项目级 Fusion 从固定 30 条音色目录中选择一个音色并冻结；无台词时显式跳过。

只有上述公共输入全部完成，才同时启动所有片段的 R2V 分支。这样每个分支读取同一份冻结对象索引、翻译台词和音色，不会因并发完成顺序产生不同语义。

每段分支按以下顺序执行：

```text
h3-max-r2v-prompt Skill
  -> 后端机械附加台词与相对时间窗
  -> 静音源片段、可选用户图片和统一 Audio 1 签名 URL
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

`video-maker` 是固定 Master 的既有 Skill：

- 逐段阶段用 `VISUAL_PROMPT_SCHEMA` 返回 `{prompt}`，业务层解包为纯字符串。
- 项目索引阶段接收 `phase`、生效的 `generation_config`，以及按 segment/frame 排序的 `segment_index`、`frame_order`、项目内路径、SHA 和全部分析图片；不接收逐段视觉提示词、台词或用户素材。输出四类数组后由后端机械分配稳定 ID。

项目级 `voice-selection` Skill 只返回一个由输入哈希和 30 个允许 ID 共同约束的 strict Schema 结果。模型所见目录不是 JSON，而是每条一行的 `voice_id - file - language - gender - 短特征` 文本；未知字段、来源、许可和哈希不占用模型上下文。后端把选中文件复制为项目内只读事实 `work/voice-reference.wav`，冻结 ID、目录哈希、文件哈希和完整来源标签。目录校验要求恰好 30 个单声道 PCM WAV，时长为 2–5 秒且文件哈希完全匹配；标签只采用数据集明确元数据，录音风格不扩写为人格。

新 `h3-max-r2v-prompt` Skill 每段执行一次并通过 closed JSON Schema 返回提示词。它读取本段视觉提示词、完整且无供应商记录污染的对象索引、本段翻译台词、可选图片、所选 Audio 1 的一行短描述和用户说明。后端随后机械附加“不说原语言、不生成字幕、在冻结时间窗内结束、所有片段沿用同一 Audio 1 音色且忽略参考音频字词”的要求。

Picsart 使用 Advanced compatibility workflow `minimax/h3-max/reference-to-video`。提交地址为 `/workflows/minimax/h3-max/reference-to-video/submit`，请求体只有一层 `params`，字段使用供应商原生的 `reference_video_urls`、`reference_image_urls`、`reference_audio_urls`、`aspect_ratio` 和 `prompt_expansion_mode: "quality"`；分辨率映射为 `480P`/`768P`。每段都提交同一个项目音色 URL；用户参考图先复用 `duet-ad1` 的 MiniMax 代理规则，等比落入 `256–5760` 双边和 `1:2.5–2.5:1` 比例范围，必要时补黑边，再编码为 PNG。轮询同一 workflow 下的 `/{taskId}/result`，从 `response.result.video.url` 读取结果。4.x 秒片段提交为 5 秒，结果再裁回源时长。

## 重试、证据与付费边界

每个节点有独立的 `attempt-1`/`attempt-2`。首次抛出自然执行错误后重试一次，不根据生成质量触发重试。唯一保留的固定上游语义是：YAMNet 检出人声而第一次 Whisper 为空时，在同一两次 attempt 边界内复核一次；第二次仍空则继续。

Picsart submit、poll、download 是三个节点。submit 第一次失败且未取得 `taskId` 时可进行第二次 POST，其中结果未知时明确接受重复付费风险；取得 ID 后，poll 和 download 的失败只重试自身，不能回到 submit。

供应商适配器把业务值与 provider record 分离。业务对象、`element_index.json` 和下游 prompt 不携带原始供应商请求；原始记录只进入对应 attempt 日志。二进制响应作为媒体证据保存，JSON 只记录状态、headers、长度和摘要，不把二进制误解码成文本。

日志失败只能写进进程 stderr，不能改变业务结果。日志目录逐层为 `0700`，文件为 `0600`，默认永久保留并由人工删除。

## 主机与访问边界

- 登录和前端 API 使用 Bearer 鉴权；供应商素材 URL 只使用独立高熵、项目绑定且限时的 token。
- 上传磁盘名由服务端决定，客户端 filename 只作显示元数据；视频和替换图分别执行字节上限。
- 外链视频每次跳转都重新做 URL、DNS 与全局 IP 校验，实际连接钉扎到已批准 IP，同时保留 Host/SNI；下载流同时受 Content-Length 和实际累计字节限制。
- 项目文件路由是逻辑名 allowlist，不接受任意项目相对路径。签名素材在注册和读取时都重新验证真实路径属于对应项目，`GET` 与供应商预检使用的 `HEAD` 共用同一验证路径。
- 状态只使用 `queued/running/succeeded/failed`。纯文件存储、后台任务和内存 token 表限定单实例；同一数据目录不允许重叠进程。
