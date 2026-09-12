# 物品灰块 + 多参考图实验分支 设计文档

日期：2026-09-12
分支：`experiment/object-gray-mask-multiref`（基于生产 main `e61b99d`）
工作树：`/home/xy/duet-ad1-h3-max-r2v-object-mask`
状态：设计已经用户确认，待写实现计划

## 背景与目标

生产 main 已有"人脸灰块"：`app/face_mask.py` 用 YuNet（CPU）对分段视频逐帧检测人脸，bbox 外扩后填纯灰矩形 `(128,128,128)`，产出 `source-faces-masked.mp4` 作为 H3 参考视频；触发条件是该段有换脸人物，持续失败可降级不打灰。

本实验在同一位置增加**物品灰块**：把 OR（object-replace）判定要替换的物品也从参考视频里遮掉，让 H3 拿不到原物品信息，只能使用我们提供的用户图/图版做替换。

同时移植实验线的**多参考图**能力（合图 A + 逐图直传 B 两套），用后端开关控制。

三条铁律约束：增量改动，不影响旧链路；不加新门禁；单图链路行为与 main 字节级一致。

## 用户已拍板的决策

| 决策点 | 结论 |
|---|---|
| 检测方式 | 整段视频直接交给视频理解模型 API（不做"抽帧+本地传播"重方案；拿到稀疏时间点框后仍需本地线性插值续帧，这是纯 CPU 几行代码，不算重方案） |
| 遮盖范围 | 只遮 OR 判定要替换的物品；检测提示词用 OR 输出的目标名称/描述 |
| 接入范围 | 所有满足条件的视频模型 API（见下表 7 家）；研究阶段后端配置选择，逐家测完效果再定主力 |
| 多参考图 | cherry-pick 移植 multiref-common 公共层 + A（composed）+ B（direct），后端开关 `multiref_mode`，不暴露前端 |
| 掩码→灰块 | 不需要模型，OpenCV 确定性填灰（与人脸同一画法） |

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
2. bbox 漂移/幻觉：学术 benchmark（VUE-STG, arXiv 2511.19529）显示通用 VLM 时空定位不强，小物品（项链）风险最大 → 坐标合法性严格校验 + 多家互验日志。
3. 输出格式漂移：JSON 结构、坐标系（归一化/绝对像素）随模型版本变 → 每家独立解析器 + 解析失败重试。

## 架构设计

### 1. 多参考图移植

- cherry-pick `feature/multiref-common`（`c316c60..ed080d3`，5 个提交）公共层：`app/main.py` + `app/public_api.py` 多文件上传（表单字段 `replacement_image` 重复 1–4 次，`REPLACEMENT_MAX_IMAGES = 4`）、`app/store.py` 回执 version 2（`replacement_images` 有序 sha256）、`app/public_quotes.py` 暂存、`web/app.js` 多图上传 UI。
- 整合方案 A（`experiment/multiref-product-views`）：`app/product_views.py` 合图（2–4 张 → 768px 格 2 列拼图，`compose_product_views` + `frozen_product_views` 冻结清单 + `view_instruction` 提示词注入）。
- 整合方案 B（`experiment/multiref-prompt-directed`）：逐图直传，`reference_image_ids` 贯穿 `app/video_maker_schemas.py`、`app/object_replace_schema.py`、`app/replacement_sheet.py`；pipeline 的 `_replacement_image_paths` / `_canonical_user_reference_path` / `h3_reference_images` 构造 / bindings `replacement_images` 校验链。
- A/B 在同一分支共存，pipeline 分叉处按开关选路。单图时保持 main 行为字节级不变（receipt v1、checkpoint 稳定、bindings 不加 `replacement_images` 键——沿用 B 树 `pipeline.py:1946` 的兼容注释语义）。

开关：`Settings.multiref_mode`，取值 `off | composed | direct`，仅从后端环境变量读（如 `MULTIREF_MODE`），不进前端、不进公开 API 参数。`off` 或未配 = 现状单图链路；上传只有 1 张时无论开关都走单图链路（沿用数据驱动分叉）。

### 2. 物品检测 provider 层

新模块 `app/object_detect.py`：

- `OpenAICompatibleGroundingClient` 基类覆盖 6 家（豆包、Qwen、GLM、混元、Kimi、阶跃），endpoint/model/prompt 参数化；`GeminiGroundingClient` 单独实现（File API 上传 + generateContent，走代理配置）。
- 遵循现有 `app/providers.py` 模式：构造注入 `Settings` + 可注入 `transport`（测试 seam）；同步 httpx；统一 `_send` 重试（transport 错误 + 502/503/504，backoff 2s/5s）；返回 `ProviderResult` / 抛 `ProviderFailure`（全 HTTP 证据记录）。
- 统一输出契约 `list[ObjectDetection]`：`{label, timestamp_s, bbox_xyxy_normalized}`。每家独立解析器适配格式差异；坐标越界/结构非法 → 判失败重试；仍失败则该目标跳过（记日志，不挡任务）。
- 配置：`config.py` 每家加 `<name>_api_key`（`field(repr=False)`，无默认）+ `<name>_api_base` + `<name>_model`；没配 key = 该 provider 不启用。研究期选择：`object_detect_providers`（有序名称列表），按序回退；支持配多家同时跑互验（各家结果都落日志供对比）。

### 3. 物品灰块节点

新模块 `app/object_mask.py`，镜像 `app/face_mask.py` 结构：

- `mask_objects(source, destination, targets, *, detector=None) -> dict`：逐帧读视频，`detector` 抽象（生产实现调 object_detect provider；测试注入假检测器，同 `_FixedDetector` seam）。
- 稀疏时间点 bbox → 每帧线性插值（相邻时间点间插值；IoU 匹配续帧；物品出画时灰块在最后已知位置续 `hold_frames` 帧后消失）。
- 填灰与人脸完全一致：`frame[y0:y1, x0:x1] = (128,128,128)`，先写 mp4v 临时文件再 ffmpeg 重编码 H.264/yuv420p。错误抛 `ObjectMaskError`。

pipeline 集成（`app/pipeline.py`）：

- 新增独立节点 `mask-replaceable-objects`，串联在 `mask-reference-faces` 之后，作用于同一条参考视频流。
- 触发条件：该段 OR 判定有对象替换（类比 `_segment_has_replaced_person` 的判定）；目标描述列表来自 OR 输出。
- 容错照抄人脸模式：重试上限内持续失败 → 降级为不打灰继续渲染 + `object_mask_degraded` 日志；缺 key / 配置错误 → 致命。**不新增任何门禁**。
- H3 prompt 的 `mask_block` 文案从 faces 泛化为 "faces and replaced objects"，仅当物品灰块节点实际产出遮盖时变化。

### 4. 测试

- `tests/test_object_mask.py`：注入假检测器，合成视频断言灰块像素 = `(128,128,128)`、时序保持、缺模型/降级分支（镜像 `tests/test_face_mask.py`）。
- `tests/test_object_detect.py`：`httpx.MockTransport` 测 7 家解析器（正常/格式漂移/坐标越界/重试）。
- `tests/test_pipeline.py` 增补：monkeypatch `object_mask.mask_objects`，验证节点日志目录、降级路径、`object_mask_degraded` 事件；`multiref_mode` 三条路径（off 字节不变、composed、direct）覆盖。
- 节点重放：采集 fixtures 后扩展 `test_node_replay_*`；E2E 前按仓库门禁 `pytest tests/test_node_replay_{local,external,boundary}.py -n auto` 全绿。
- 真实 API 对比测试等用户配 key 后进行，是本实验核心产出。

### 5. 明确不做

- 不接闭集检测/分割云 API。
- 不做本地 GPU 模型部署；CPU 备选（YOLO-World + MobileSAM）本轮不接。
- 不做"掩码转灰块"模型（确定性 OpenCV 填灰）。
- 不动 seedream 图版链路；不改人脸灰块现有算法（YuNet、0.73 阈值、外扩比例）。
- 开关不进前端、不进公开 API；不新增任何质量/语义门禁。

## 部署与端口

实验端口 3241（前端）→ 3242（后端，127.0.0.1），服务命名沿用 `duet-ad1-h3-max-r2v-object-mask{,-caddy}.service` 模式。实现完成并测试通过后再部署；部署前确认无在飞项目。

## 账号与密钥

7 家 key 由用户后续配置。代码侧只加 config 字段，没配 key 的 provider 自动不启用，不影响其它链路和测试。
