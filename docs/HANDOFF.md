# MiniMax H3 Max R2V 新链路交接

## 当前状态

- 工作区：`/home/xy/duet-ad1-h3-max-r2v`
- 分支：`main`
- 前端固定基线：`github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf`
- 已有仓库提交：
  - `2141263 chore: import minimal frontend from duet-ad Master`
  - `4f68eb6 test: preserve minimal frontend API contract`
- 独立 FastAPI 后端、媒体链路、供应商适配、Skills、测试和文档均已实现。
- 本轮实现已写入本仓库工作树；源仓库 `/home/xy/duet-ad1` 没有被修改。
- 3301 前端入口和 3302 内部后端已由 user systemd 单元运行，Caddy 使用现有公网证书。
- DeepSeek、MediaKit 和 Picsart 已接通运行配置；真实冒烟已走通 Picsart 前的全部节点。
- 真实 Picsart 请求在提交阶段返回 `unknown_model`，未取得 taskId；因此当前没有成功生成视频的证据，也没有可核对的付费任务回执。

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
  -> 一次 video-maker 项目级全局对象索引
  -> 公共输入屏障
  -> 每段并行：新 R2V 提示词 Skill -> 静音原片段 -> Picsart H3 Max R2V
  -> 供应商片段裁回原时长并统一为 24 fps
  -> 按原顺序合并 output/generated.mp4
```

无音轨视频会显式跳过音频、YAMNet、Whisper 和 spoken 筛选节点，以空台词继续。跨段台词按各段时间交集分割文本；台词不参与切点选择。

## video-maker 合同

`skills/video-maker/SKILL.md` 保持固定 Master 原文。它有两个阶段，均沿用 Skill 自身的 strict `submit_result` 结构化传输；这是旧 Skill 的既有合同，不扩展为新链路的通用质量门禁。

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

后端只解包 `prompt` 字符串，写入本段 `visual_prompt.txt`。完整 DeepSeek 请求和响应只进入该节点的 attempt 日志。

### 项目级索引阶段

所有逐段视觉阶段完成后执行一次。请求只包括：

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

图片仍是所有 segment 的九张半分辨率分析副本，按 segment/frame 排序。这个阶段明确不接收逐段 `visual_prompt`、台词、用户图片或用户说明。

模型通过 `PROJECT_INDEX_SCHEMA` 返回四个数组：`people`、`entities`、`scenes`、`relations`。后端把调用内 `key` 机械映射为 `person-01`、`entity-01`、`scene-01`、`relation-01` 等稳定 key，保存为 `work/element_index.json`。供应商 record 不进入业务索引。

## video-maker 之后

项目级 DeepSeek 翻译读取 spoken 台词和所有逐段视觉描述，保持原行顺序；后端再按原时间语义把翻译台词切回每个 segment。

新增 `skills/h3-max-r2v-prompt/SKILL.md` 每段执行一次，输入：

- 本段 `visual_prompt`。
- 完整全局 `element_index`。
- 本段本地化后的翻译台词和相对时间。
- 可选用户替换图片。
- 可选用户替换说明。

它输出纯文本 R2V 提示词，要求优先执行用户替换；其余人物换为同性别、种族和整体风格但脸略有不同、服装同色同风格但款式略变的新人物；场景换为同类别同风格但不同布局和细节的同类场景；动作、机位、构图、运动、焦段、光线和转场保持不变。

后端不让 Skill 改写台词，而是在其输出后机械附加目标语言台词、段内时间窗、禁止原语言和禁止字幕的要求。随后把静音原片段、可选用户图及最终提示词提交 Picsart：

- `model`: `minimax-h3-max-r2v`
- `promptExpansionMode`: `quality`
- `videoUrls`: 当前静音片段的临时签名公网 URL
- `imageUrls`: 可选用户图片的临时签名公网 URL
- `duration`: `max(5, ceil(原片段秒数))`
- `resolution`、`aspectRatio`: 前端请求值

## API、存储与技术边界

- 七组极简前端 API 已实现，`creation_input` 为前端要求的 `{version, source, replacement_image}`。
- 项目只使用 `queued/running/succeeded/failed` 四态；成功快照同时包含 `percent=100` 和 `has_video=true`。
- 上传使用服务端固定磁盘名并流式执行字节上限；外链视频逐跳做 URL、DNS、全局 IP、重定向和累计字节限制。
- 通用文件 API 只允许源视频与成功态最终视频；状态、日志和中间文件不能下载。
- 供应商素材 token 与项目、asset 和真实项目内路径绑定；替换图签名响应保留冻结的图片媒体类型。
- 状态文件原子写入；项目与日志目录为 `0700`，文件为 `0600`。
- 原始请求、响应、headers、密钥、签名 URL、媒体副本、异常和命令输出按节点/attempt 本地保存，不脱敏；日志失败不阻断业务。
- 只支持单进程实例；同一数据目录由生命周期文件锁阻止重叠 worker。

## 重试与付费边界

每个节点初始执行一次，失败后最多再尝试一次。Picsart submit、poll、download 是三个独立节点：

- submit 第一次失败且没有 `taskId` 时可再次 POST；结果未知时明确接受重复付费风险。
- 一旦取得 `taskId`，poll 和 download 的失败只重试自身，不能返回 submit。
- 不因生成质量、替换效果或媒体观感增加重试。
- 固定 Master 的 YAMNet 检出人声但第一次 Whisper 为空时，仍在同一个最多两次的 ASR attempt 边界内复核一次；第二次仍空则以空台词继续。

## 验证与剩余事项

已完成离线测试，覆盖前端合同、API、SSRF/路径归属、供应商 MockTransport、研究日志、固定 Master 媒体算法、YAMNet、video-maker 两阶段、公共屏障、逐段并行、Picsart 重试和 4.x 秒裁回。当前结果：`139 passed, 1 warning`；唯一 warning 是 FastAPI TestClient 引入的 Starlette/httpx 弃用提示。

YAMNet 模型：

- 路径：`/home/xy/duet-ad1-h3-max-r2v/models/yamnet.tflite`
- 字节数：`4,126,810`
- SHA-256：`4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf`

部署与剩余事项：

1. 运行入口为 `https://8.166.140.227:3301`；登录使用部署环境的 `ACCESS_TOKEN`，凭据保存在受限配置文件，不写入仓库。
2. Picsart 官方文档使用的模型 ID 仍为 `minimax-h3-max-r2v`，但当前 token 对该模型返回 `unknown_model`。需要在 Picsart 账户/目录侧开通该模型后，使用同一输入再次运行最后一步；不自动替换为其他模型。
3. 当前 `.env` 仅作为本机部署覆盖配置，权限为 `0600` 且被 `.gitignore` 忽略；其他密钥由受限源配置文件加载。
