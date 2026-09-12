# 物品灰块 + 多参考图实验分支 设计文档

日期：2026-09-12
分支：`experiment/object-gray-mask-multiref`（基于生产 main `e61b99d`）
工作树：`/home/xy/duet-ad1-h3-max-r2v-object-mask`
状态：设计已经用户确认，spec 审查问题已修正

## 背景与目标

生产 main 已有"人脸灰块"：`app/face_mask.py` 用 YuNet（CPU）对分段视频逐帧检测人脸，bbox 外扩后填纯灰矩形 `(128,128,128)`，产出 `source-faces-masked.mp4` 作为 H3 参考视频；触发条件是该段有换脸人物，持续失败可降级不打灰。

本实验在同一位置增加**物品灰块**：把 OR（object-replace）判定要替换的物品也从参考视频里遮掉，让 H3 拿不到原物品信息，只能使用我们提供的用户图/图版做替换。

同时移植实验线的**多参考图**能力（合图 A + 逐图直传 B 两套），用后端开关控制。

三条铁律约束：增量改动，不新增门禁；人脸灰块原样不动；多参考图只有 composed/direct 两条链路，单图是两种模式各自的 1 图情形，不保留 main 的旧单图链路（本分支是独立实验线，不承担存量项目兼容）。

## 用户已拍板的决策

| 决策点 | 结论 |
|---|---|
| 检测方式 | 整段视频直接交给视频理解模型 API（不做"抽帧+本地传播"重方案；拿到稀疏时间点框后仍需本地线性插值续帧，这是纯 CPU 几行代码，不算重方案） |
| 遮盖范围 | 只遮 OR 判定要替换的物品；检测提示词用 OR 输出的目标名称/描述 |
| 接入范围 | 所有满足条件的视频模型 API（见下表 7 家）；研究阶段后端配置选择，逐家测完效果再定主力 |
| 多参考图 | cherry-pick 移植 multiref-common 公共层 + A（composed）+ B（direct），后端开关 `multiref_mode` 只有 `composed | direct` 两个值，不保留旧单图链路（单图 = 所选模式的 1 图情形），不暴露前端 |
| 掩码→灰块 | 不需要模型，OpenCV 确定性填色；物品块用深灰 `(64,64,64)` 与人脸块 `(128,128,128)` 区分 |
| 提示词联动 | 生成提示词节点（fusion / H3 为主，object-replace 补一句色块语义）同步修改，明确介绍两种色块的含义 |
| 对外 API | v1 公开 API 同步支持多参考图上传（随公共层移植）；**不暴露任何内部模型切换**（`multiref_mode`、`object_detect_providers`、provider 名称一律不出现在公开 API 参数与响应里） |
| 验收方式 | 通过对外 API 做 E2E 验收；最后一轮用 TikTok 链接验收（TikTok 解析修复已在 main 的 `app/safe_download.py` / `app/public_api.py`，本分支已继承，移植过程中不得丢失） |

## 检测模型 API 调研结论

满足条件（视频输入 + 开放词汇按文字定位 + 可输出 bbox + 公开自助 API）共 7 家：

| # | 服务 | 梯队 | Endpoint 形态 | 视频输入 | bbox 输出 | 10s 视频成本（约） | 国内可达 |
|---|---|---|---|---|---|---|---|
| 1 | 豆包 / 火山方舟 | 主力 | `https://ark.cn-beijing.volces.com/api/v3`，OpenAI 兼容，Bearer key | Files API / base64 ≤50MB / URL ≤50MB；fps 0.2–5 可调 | `<bbox>x1 y1 x2 y2</bbox>`，0–1000 归一化（图像官方支持，视频为 prompt 用法需实测） | ≈ ¥0.02 | 直连 |
| 2 | Qwen-VL / 阿里百炼 | 主力 | `https://dashscope.aliyuncs.com/compatible-mode/v1`，OpenAI 兼容 | URL / 本地文件，fps 可控；视频 2s–10min | `bbox_2d` 绝对像素坐标 | ≈ ¥0.02（新用户 100 万 tokens 免费 90 天） | 直连 |
| 3 | GLM-4.5V / 智谱 BigModel | 主力 | `https://open.bigmodel.cn/api/paas/v4/chat/completions`，OpenAI 兼容 | 官方模型页明确支持视频输入 | Grounding 是官方卖点（按文本描述定位人物/物体）；视频 bbox 仍需实测 | 输入 ¥2/百万 tokens；新用户 2000 万 tokens 免费 | 直连 |
| 4 | Google Gemini API | 上限参照 | `https://generativelanguage.googleapis.com`，File API 上传 + generateContent，API key | File API（≤2GB）/ inline ≤20MB；1 FPS 采样 | `box_2d`（0–1000 归一化），唯一可能同时出 mask | < $0.003 | 需代理（本机 127.0.0.1:7897） |
| 5 | 腾讯混元视觉 TokenHub | 互验/兜底 | `https://api.hunyuan.cloud.tencent.com/v1`，OpenAI 兼容 | URL / Base64，官方支持 | bbox 未文档化，prompt 用法 | 首次开通 100 万 tokens 免费 | 直连 |
| 6 | Moonshot Kimi k2.6 | 互验/兜底 | `https://api.moonshot.cn/v1`，OpenAI 兼容 | `video_url`（base64 / URL / 文件上传），k2.5/k3 不支持视频 | bbox 未文档化，prompt 用法 | 按 token，确切单价未公开 | 直连 |
| 7 | 阶跃星辰 Step | 互验/兜底 | `https://api.stepfun.com/v1`，OpenAI 兼容 | Step-1.5V 起支持视频理解 | bbox 未文档化，prompt 用法 | ≈ ¥0.005–0.015/千 tokens | 直连 |

排除及原因：OpenAI/Claude（API 不支持视频文件输入）、Twelve Labs（只出时间区间不出 bbox）、DeepSeek（无视觉 API）、讯飞星火（仅图片理解）、百度千帆 ERNIE-VL（托管 API 视频输入查不到支持证据）、商汤（视频能力偏 to-B 无自助文档）、MiniMax（无自助视频理解 VL API）、Amazon Nova（无 grounding 文档、境外账号）。闭集检测 API（Google/AWS/Azure、腾讯商品识别、阿里分割、百度 EasyDL）不满足"按文字指定物品"，全部不接。

已知风险与工程补偿：

1. 时间粒度粗：VLM 视频理解本质是 1 FPS（豆包可到 5）抽帧 + 时间戳拼接，bbox 最多每秒一个 → 本地线性插值补全每帧。
2. bbox 漂移/幻觉：学术 benchmark（VUE-STG, arXiv 2511.19529）显示通用 VLM 时空定位不强，小物品（项链）风险最大 → 坐标合法性严格校验 + 多家对比日志。
3. 输出格式漂移：JSON 结构、坐标系（归一化/绝对像素）随模型版本变 → 每家独立解析器 + 解析失败重试。

## 架构设计

### 1. 多参考图移植

- cherry-pick `feature/multiref-common`（提交范围 `0c5b843..ed080d3`，含 `c316c60` 共 5 个提交）公共层：`app/main.py` + `app/public_api.py` 多文件上传（表单字段 `replacement_image` 重复 1–4 次，`REPLACEMENT_MAX_IMAGES = 4`）、`app/store.py` 回执 version 2（`replacement_images` 有序 sha256）、`app/public_quotes.py` 暂存、`web/app.js` 多图上传 UI。
- 整合方案 A（`experiment/multiref-product-views`）：`app/product_views.py` 合图（2–4 张 → 768px 格 2 列拼图，`compose_product_views` + `frozen_product_views` 冻结清单 + `view_instruction` 提示词注入）。**纳入提交**：`f1a98e4`（冻结有序合图为单一 provider reference）、`8860a78`（合图适配 H3 宽高比）、`8877739`（请求体校验失败不付费重提）、`9440830`（合图保守去白边）、`d8d357e` + `8044783`（共享用户图去白边预处理，两模式共用，只移植这一份）、replay 基建 `ccf255a` + `581722c` + `8decac6`（A/B 内容相同只取一份；属测试工具链，main 的 `node_replay_support.py` 缺这 +78 增量，阶段 7 重放门禁需要；其"boundary fixtures 缺失 fail-closed"语义只作用于测试，不是产品门禁）。**排除**：deploy `f8d04cd`/`8220f31`、web `f4409fe`/`b3ce734`（与 common 层 `1ac22e6`/`f167e18` 等价）、`66e2c0c`/`814edd6`（common 层等价已覆盖）。
- 整合方案 B（`experiment/multiref-prompt-directed`）：逐图直传，`reference_image_ids` 贯穿 `app/video_maker_schemas.py`、`app/object_replace_schema.py`、`app/replacement_sheet.py`；pipeline 的 `_replacement_image_paths` / `_canonical_user_reference_path` / `h3_reference_images` 构造 / bindings `replacement_images` 校验链。**纳入提交**：`2dc40e5`（B pipeline 有序用户图贯穿与 H3 冻结输入）、`90e9482`（图版 tile 映射有序用户图）、`e6fd179`（identity 归并保留绑定）、`f6cf067` + `b0e092c`（允许未绑定用户图、保留全部上传图）、`bb54034`（归一化冗余参考标注，不加门禁）。**排除**：docs `2c34681`、replay 基建 `0c8cad6`/`6653a8c`/`2b2fb44`（A 侧等价已取）、deploy `a71b522`、web `771c1c3`/`a37aa0e`（与 common 层等价）、`5bcda0a`/`9a3d60c`（common 层等价已覆盖）、`38e55aa`/`fe02171`（去白边用 A 的共享版本）。B 树 `app/media.py` +71 与 A 逐字节相同，随 A 的去白边提交带入，不重复移植。
- 去重策略：common 层 5 个提交先完整 cherry-pick；A/B 按上述清单逐个 cherry-pick，与已应用提交等价的改动跳过，其余冲突手工整合。
- A/B 在同一分支共存。**分叉点在"参考图构造与冻结"环节**：composed 模式在参考图准备时把多图合成单一 provider reference（下游与单图完全同构）；direct 模式的 `reference_image_ids` 贯穿 video-maker / OR / 图版 / bindings；H3 提交处两模式共用"按 `h3_reference_images` 顺序传图"。同一模式内 1 图与多图共享链路，仅按上传数量分叉（沿用 B 树的单图兼容语义：单图 checkpoint 字节稳定、bindings 不加 `replacement_images` 键），保证同模式断点续跑一致。

开关：`Settings.multiref_mode`，取值**只有 `composed | direct` 两个**，仅从后端环境变量 `MULTIREF_MODE` 读（变量名固定），必须显式配置、不设缺省，缺失或非法值启动报错，不进前端、不进公开 API 参数。**不存在第三条"旧单图链路"**：上传 1 张图就是所选模式的 1 图情形（composed 模式 1 图直接规范化装入图版位，direct 模式 1 图即 Image 序列里的一张）。

### 2. 物品检测 provider 层

新模块 `app/object_detect.py`，**7 家 provider 平行可插拔，对外暴露同一套接口**：

- 统一接口契约：所有 provider 实现相同的 `detect(video_path: Path, targets: list[str]) -> list[ObjectDetection]`，输入输出完全一致。`ObjectDetection` 为 dataclass：`{label: str, timestamp_s: float, bbox_xyxy_normalized: tuple[float, float, float, float]}`；bbox 为 **0–1 浮点、相对帧宽/高**，`__post_init__` 校验坐标 ∈ [0,1]、x0<x1、y0<y1、timestamp ≥ 0。`detect` 直接返回 list；失败抛 `ProviderFailure`；`ProviderResult` 只用于底层 `_send` 的证据记录，不作为 `detect` 返回形态。内部实现因模型而异（上传方式、提示词模板、响应解析），对调用方完全透明；pipeline 只面向统一接口，不知道背后是哪家模型。
- 实现结构：`OpenAICompatibleGroundingClient` 基类覆盖 6 家（豆包、Qwen、GLM、混元、Kimi、阶跃），endpoint/model/prompt 参数化；`GeminiGroundingClient` 单独实现（File API 上传 + generateContent，走代理配置）。两家实现都满足同一接口契约。
- 遵循现有 `app/providers.py` 模式：构造注入 `Settings` + 可注入 `transport`（测试 seam）；同步 httpx；统一 `_send` 重试（transport 错误 + 502/503/504，backoff 2s/5s）。注意重试成本：每次重试是重新上传 + 重新推理，单家单次 detect 费用上限 = 3 次推理；Gemini 已上传文件句柄在一次 detect 的重试间复用，不重复上传。
- 统一输出契约已在上方接口契约中定义；每家独立解析器适配格式差异；坐标越界/结构非法 → 判失败重试；**单目标粒度**：某目标在重试上限内仍无合法框 → 跳过该目标（记日志，不挡任务），其余目标正常遮盖。
- 配置：`config.py` 每家加 `<name>_api_key`（`field(repr=False)`，无默认）+ `<name>_api_base` + `<name>_model`；没配 key = 该 provider 不启用。研究期选择：`object_detect_providers`（有序名称列表，环境变量 `OBJECT_DETECT_PROVIDERS` 逗号分隔，无缺省，名称必须是已知 7 家之一），pipeline 内按序回退。**多家效果对比不在 pipeline 内并发**，由研究期独立对比脚本逐家跑（见实现计划阶段 8），各家结果落日志供对比。
- 缺 key 语义（与人脸节点 "model is missing 才致命" 对齐）：`object_detect_providers` 未配置或全部未启用 → 物品灰块节点不触发，直接跳过（等同该段无 OR 目标）；**显式启用但缺 key / 配置错误 → 配置错误致命**。

### 3. 物品灰块节点

新模块 `app/object_mask.py`，镜像 `app/face_mask.py` 结构：

- `mask_objects(source, destination, targets, *, detector=None) -> dict`：逐帧读视频，`detector` 抽象（生产实现调 object_detect provider；测试注入假检测器，同 `_FixedDetector` seam）。
- 稀疏时间点 bbox → 每帧线性插值（相邻时间点间插值；IoU 匹配续帧；物品出画时灰块在最后已知位置续 `OBJECT_MASK_HOLD_FRAMES` 帧后消失——`object_mask.py` 模块常量，显式取值 12，不进 config）。
- 物品块填**深灰 `(64,64,64)`**，与人脸块 `(128,128,128)` 明确区分；先写 mp4v 临时文件再 ffmpeg 重编码 H.264/yuv420p。错误抛 `ObjectMaskError`。

pipeline 集成（`app/pipeline.py`）：

- 新增独立节点 `mask-replaceable-objects`，串联在 `mask-reference-faces` 之后，作用于同一条参考视频流。
- 触发条件：该段 OR 判定有对象替换（类比 `_segment_has_replaced_person` 的判定）且物品检测可用（见上节缺 key 语义）；目标描述列表来自 OR 输出。
- 容错照抄人脸模式：重试上限内持续失败 → **降级到未遮物品的原视频继续渲染** + `object_mask_degraded` 日志；显式启用的 provider 缺 key / 配置错误 → 致命。**不新增任何门禁**。人脸灰块节点本身（YuNet、0.73 阈值、外扩比例、降级逻辑）原样不动。

提示词联动：fusion / H3 提示词节点同步修改，明确介绍两种色块——人脸灰块 `(128,128,128)` 遮源人脸、物品深灰块 `(64,64,64)` 遮被替换物品，H3 不得绘制/复原色块内容、须用提供的参考图补全；物品部分仅在物品灰块节点实际产出遮盖时出现（按段冻结，保证续跑一致）。object-replace 在遮盖之前运行、看不到遮盖后视频且输出 schema 固定，只在其 skill 补一句色块语义说明，保持 OR 描述与最终遮盖一致，不改其输出结构。

### 4. 测试

- `tests/test_object_mask.py`：注入假检测器，合成视频断言物品块像素 = `(64,64,64)`、人脸块仍为 `(128,128,128)`、插值连续性、出画续帧、缺模型/降级分支（镜像 `tests/test_face_mask.py`）。
- `tests/test_object_detect.py`：`httpx.MockTransport` 测 7 家解析器（正常/格式漂移/坐标越界/重试）；参数化断言 7 家同一输入输出同构（统一契约）；单目标失败跳过；缺 key 跳过与致命两路。
- `tests/test_pipeline.py` 增补：monkeypatch `object_mask.mask_objects`，验证节点日志目录、降级到原视频路径、`object_mask_degraded` 事件、无可用 provider 时节点不触发；`multiref_mode` 两条路径（composed、direct，各含 1 图与多图情形）覆盖。
- 节点重放：采集 fixtures 后扩展 `test_node_replay_*`；E2E 前按仓库门禁 `pytest tests/test_node_replay_{local,external,boundary}.py -n auto` 全绿。
- 真实 API 对比测试等用户配 key 后进行（独立对比脚本逐家跑），是本实验核心产出。

### 5. 明确不做

- 不接闭集检测/分割云 API。
- 不做本地 GPU 模型部署；CPU 备选（YOLO-World + MobileSAM）本轮不接。
- 不做"掩码转灰块"模型（确定性 OpenCV 填色）。
- 不动 seedream 图版链路；不改人脸灰块现有算法（YuNet、0.73 阈值、外扩比例）。
- 开关不进前端、不进公开 API（含 provider 名称、`multiref_mode`、检测模型选择，一律不出现在 v1 参数与响应里）；不新增任何质量/语义/产品门禁（replay 测试基建的 fail-closed 属测试工具链，不在此列）。
- 不丢 main 已有的 TikTok 链接解析与下载修复（`app/safe_download.py` / `app/public_api.py` 的 TikWM 解析、代理、重试），移植冲突时保留这些修复。

## 部署与端口

实验端口 3241（前端）→ 3242（后端，127.0.0.1），服务命名沿用 `duet-ad1-h3-max-r2v-object-mask{,-caddy}.service` 模式。**部署只在全部测试通过且用户确认后进行**；部署前确认无在飞项目。

## 账号与密钥

7 家 key 由用户后续配置。代码侧只加 config 字段，没配 key 的 provider 自动不启用，不影响其它链路和测试。
