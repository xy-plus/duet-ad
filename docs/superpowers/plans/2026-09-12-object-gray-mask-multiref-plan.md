# 物品灰块 + 多参考图 实现计划

日期：2026-09-12（经 spec/计划双审查后修订）
对应 spec：`docs/superpowers/specs/2026-09-12-object-gray-mask-multiref-design.md`（同目录 specs/，审查修正版）
工作树：`/home/xy/duet-ad1-h3-max-r2v-object-mask`（分支 `experiment/object-gray-mask-multiref`，基于 main `e61b99d`）
执行方式：子代理按 TDD 实施（先写失败测试再实现），每阶段完成后 git 提交。

## 移植源与提交清单（审查核实版）

分叉基线：`0c5b843`（在 main 上）。cherry-pick 区间端点注意：`A..B` 不含左端点。

| 来源 | 范围 | 纳入 | 排除 |
|---|---|---|---|
| 公共多图层 `feature/multiref-common` | `0c5b843..ed080d3`（含 `c316c60` 共 5 提交） | 全部 5 个：`app/main.py`、`app/public_api.py`、`app/store.py`（receipt v2）、`app/public_quotes.py`、`web/app.js` 多图 UI、公开 API 契约测试 | 无 |
| A `experiment/multiref-product-views`（树 `/home/xy/duet-ad1-h3-max-r2v-multiref-a`） | `0c5b843..8044783` 共 15 提交中 | `f1a98e4`、`8860a78`、`8877739`（含 `tests/test_picsart_validation_retry.py`）、`9440830`、`d8d357e`、`8044783`（去白边共享预处理，两模式共用只取这份，含 `app/media.py` +71）、replay 基建 `ccf255a`/`581722c`/`8decac6`（测试工具链，main 缺此 +78 增量，阶段 7 需要） | deploy `f8d04cd`/`8220f31`；web `f4409fe`/`b3ce734`（common 层 `1ac22e6`/`f167e18` 等价）；`66e2c0c`/`814edd6`（common 等价） |
| B `experiment/multiref-prompt-directed`（树 `/home/xy/duet-ad1-h3-max-r2v-multiref-b`） | `0c5b843..2c34681` 共 17 提交中 | `2dc40e5`、`90e9482`、`e6fd179`、`f6cf067`、`b0e092c`、`bb54034` | docs `2c34681`；replay 基建 `0c8cad6`/`6653a8c`/`2b2fb44`（A 侧已取）；deploy `a71b522`；web `771c1c3`/`a37aa0e`；`5bcda0a`/`9a3d60c`（common 等价）；`38e55aa`/`fe02171`（去白边取 A 版）；B 的 `app/media.py` +71 与 A 逐字节相同不重复 |

去重策略：common 层先完整 cherry-pick；A/B 按清单逐个 cherry-pick，等价改动跳过，冲突手工整合。A/B 分叉点在"参考图构造与冻结"环节，H3 提交处两模式共用。

## 参照的现有模式（main 树上的锚点）

- 人脸灰块：`app/face_mask.py`（`mask_faces(source, destination, *, detector=None)`、`FaceMaskError`、`GRAY_FILL=(128,128,128)`、YuNet CPU）；pipeline 节点 `mask-reference-faces`（`app/pipeline.py` 约 5290–5350），触发 `_segment_has_replaced_person`（约 4942），`FACE_MASK_MAX_ATTEMPTS = 3`（约 158），失败降级 + `face_mask_degraded` 日志。
- provider 模式：`app/providers.py` —— `Settings` 注入 + 可注入 `transport`、同步 httpx、模块级 `_send`（约 170，首试+2 重试、backoff 2s/5s、只重试 transport 错误和 502/503/504）、`ProviderResult`（约 59）/ `ProviderFailure`（约 65）全证据。
- 配置模式：`app/config.py` `Settings` frozen dataclass（约 19），密钥字段 `field(repr=False)` 无默认，`__post_init__`（约 55）严格校验，`get_settings()`（约 185）从环境变量读。
- 测试模式：`tests/test_face_mask.py`（注入 `_FixedDetector`）、`tests/test_providers.py`（MockTransport + autouse 消 backoff）、`tests/test_pipeline.py`（monkeypatch `face_mask.mask_faces`）、节点重放 `tests/test_node_replay_*.py` + `scripts/harvest_node_fixtures.py` + `docs/node-replay-tests.md`。

## 阶段 1：multiref 公共层 + `multiref_mode` 开关

1. cherry-pick `0c5b843..ed080d3` 共 5 个提交；解决与 main（`0c5b843` 之后）的冲突，预期主要在 `app/main.py` / `app/public_api.py` / `app/store.py`。
2. cherry-pick replay 基建三提交（A 侧 `ccf255a`/`581722c`/`8decac6`）。
3. `app/config.py` 新增 `multiref_mode: str`，`__post_init__` 校验必须 ∈ {`composed`, `direct`}，**无缺省**；`get_settings()` 从环境变量 `MULTIREF_MODE`（名称固定）读，缺失/非法即启动报错。
4. 前端 `web/app.js` 多图上传 UI 随公共层落地，UI 上不出现模式概念。
5. 测试：config 校验（合法两值/非法值/缺失报错）；公共层上传/回执/公开 API 契约测试随 cherry-pick 带入并跑通。
6. 提交。

## 阶段 2：composed（A 方案）移植

1. 按清单 cherry-pick A 纳入提交；`app/product_views.py`、`app/media.py` 去白边共享预处理落地。
2. `app/pipeline.py`：`_stored_replacement_paths(project, state)`；`_materialize_canonical_user_reference(..., replacement_paths=)` 在 `multiref_mode == "composed"` 且 `len(paths) > 1` 时走 `compose_product_views`，1 图走 `user_reference_png`。
3. 提示词注入三处：object-replace payload（`views/instruction`）、object-replace skill 尾部 `view_instruction`、fusion 权威块 `view_instruction(..., "Image 2")`；seedream 图版 `_build_seedream_prompt(user_view_count=)`。
4. `8877739` 的 `_poll_failure_is_request_validation` 修复随阶段带入。
5. 测试：合图确定性（同输入同 sha256）、2/3/4 图布局、3 图留空格、1 图路径、冻结清单篡改 fail-closed、提示词注入断言。
6. 提交。

## 阶段 3：direct（B 方案）移植

1. 按清单 cherry-pick B 纳入提交；`app/pipeline.py` 的 `_replacement_image_paths` / `_canonical_user_reference_path(order)` / `_user_reference_metadata` / `h3_reference_images` 构造 / bindings `replacement_images`（仅 >1 图新增，单图 checkpoint 字节稳定）及校验链落地。
2. schema 层：`app/object_replace_schema.py`、`app/video_maker_schemas.py`、`app/replacement_sheet.py` 的 `reference_image_ids`。
3. skills 更新：object-replace（按 `reference_id` 绑定）、video-maker（用户图排在分析帧前、非源证据）、h3-max-r2v-prompt（图片编号以 `h3_reference_images` 为准）。
4. 与阶段 2 整合：分叉点（参考图构造与冻结环节）按 `settings.multiref_mode` 选路；H3 提交处两模式共用按 `h3_reference_images` 顺序传图。
5. 测试：1 图与 2–4 图 Image 编号、bindings 兼容（单图字节稳定）、三 schema 的 `reference_image_ids` 校验、seedream 输入图序。
6. 提交。

## 阶段 4：物品检测 provider 层（并行子代理：1 脚手架 + 每模型 1 代理）

统一接口契约（所有代理必须严格遵守，写进脚手架）：
- `detect(video_path: Path, targets: list[str]) -> list[ObjectDetection]`，直接返回 list，失败抛 `ProviderFailure`（`ProviderResult` 仅底层 `_send` 用）。
- `ObjectDetection` dataclass：`{label: str, timestamp_s: float, bbox_xyxy_normalized: tuple[float,float,float,float]}`；bbox 为 0–1 浮点相对帧宽/高；`__post_init__` 校验坐标 ∈[0,1]、x0<x1、y0<y1、timestamp≥0。
- 解析失败/坐标越界 → `_send` 重试（首试+2，上限 3 次推理费用）；单目标重试上限内仍无合法框 → 跳过该目标记日志，不挡其它目标。

1. **脚手架代理**（先行，其余 7 家代理等其提交后并行）：新建 `app/object_detect.py` 骨架——`ObjectDetection`、provider 注册表（名称→client 类）、`OpenAICompatibleGroundingClient` 基类（视频 base64 inline、fps 参数默认 1、统一提示词模板要求"每秒时间戳+bbox"结构化输出、`_send` 重试复用 `app/providers.py` 模式）、选择函数（按 `settings.object_detect_providers` 顺序回退，未配 key 跳过并记日志）；`app/config.py` 增加 `object_detect_providers: tuple[str, ...]`（环境变量 `OBJECT_DETECT_PROVIDERS` 逗号分隔，无缺省，名称校验）；并以**豆包**为参考实现完整落地（endpoint `https://ark.cn-beijing.volces.com/api/v3`，`<bbox>` 0–1000 归一化解析）。
2. **6 个并行代理**（各自独立 worktree，从脚手架提交拉分支，完成后合并）：Qwen（`https://dashscope.aliyuncs.com/compatible-mode/v1`，`bbox_2d` 绝对像素÷宽高归一化）、GLM（`https://open.bigmodel.cn/api/paas/v4/chat/completions`）、混元（`https://api.hunyuan.cloud.tencent.com/v1`，保守 JSON/bbox 文本解析）、Kimi（`https://api.moonshot.cn/v1`，仅 k2.6 支持视频）、阶跃（`https://api.stepfun.com/v1`）、Gemini（`https://generativelanguage.googleapis.com`，File API 上传 + generateContent，`box_2d` 0–1000 归一化，代理字段从 config 读；单次 detect 内重试复用已上传文件句柄）。
3. 每家 config 字段：`<name>_api_key`（`field(repr=False)`、无默认）、`<name>_api_base`、`<name>_model`；每家改动集中在独立段落，便于 7 分支合并。
4. 测试：脚手架带豆包全套测试（MockTransport：正常/格式漂移/坐标越界/重试/单目标跳过/缺 key 跳过与致命两路）；每家代理带自己解析器的同款测试 + 统一契约参数化用例注册。合并后 `tests/test_object_detect.py` 全集绿。
5. 提交（脚手架 1 个 + 每家 1 个 + 合并 1 个）。

## 阶段 5：物品灰块模块 + pipeline 节点

1. 新模块 `app/object_mask.py`，镜像 `face_mask.py`：`mask_objects(source, destination, targets, *, detector=None) -> dict`；`OBJECT_BLOCK_FILL = (64,64,64)`；`OBJECT_MASK_HOLD_FRAMES = 12`（模块常量，不进 config）；`ObjectMaskError`；detector 抽象注入（生产实现包装阶段 4 选择逻辑）。
2. 插值：稀疏时间点 bbox → 每帧线性插值；IoU 匹配续帧；出画续 12 帧后消失。
3. 输出管线复用人脸做法：mp4v 临时文件 → ffmpeg H.264/yuv420p。
4. `app/pipeline.py`：新节点 `mask-replaceable-objects` 串联在 `mask-reference-faces` 之后；触发 = 该段 OR 判定有对象替换（参照 `_segment_has_replaced_person` 写 `_segment_has_replaced_object`）且检测可用；目标描述取自 OR 输出。
5. 容错：`OBJECT_MASK_MAX_ATTEMPTS = 3`；持续失败 → 降级到未遮物品的原视频继续 + `object_mask_degraded` 日志；`object_detect_providers` 未配置/全部未启用 → 节点不触发直接跳过；显式启用但缺 key/配置错误 → 致命。不加新门禁。
6. 测试 `tests/test_object_mask.py` + `tests/test_pipeline.py` 增补：假检测器合成视频断言 `(64,64,64)` 像素、插值连续性、出画续帧、降级路径、节点不触发路径、节点日志目录。
7. 提交。

## 阶段 6：提示词联动

1. fusion / H3 提示词的 `mask_block` 文案改为介绍两种色块：中灰 `(128,128,128)` 遮源人脸、深灰 `(64,64,64)` 遮被替换物品；H3 不得绘制/复原色块内容，须用参考图补全。
2. 物品部分仅在物品灰块节点实际产出遮盖时出现（按段冻结，保证续跑一致）。
3. object-replace skill 只补一句色块语义说明（OR 在遮盖前运行、输出 schema 不变）。
4. 测试：提示词快照断言（有/无物品遮盖两种情形）。
5. 提交。

## 阶段 7：回归与节点重放门禁

1. `pytest -n auto` 全量绿。
2. `scripts/harvest_node_fixtures.py` 采集新节点 fixtures（依赖阶段 1 带入的 replay 基建增量）；`pytest tests/test_node_replay_{local,external,boundary}.py -n auto` 全绿。
3. 提交。

## 阶段 8（等用户配 key，不在本轮实施）

1. 独立对比脚本对同一分段用已配置的各家逐家跑，bbox 结果与视频截图对比落日志；pipeline 内不并发多跑。
2. 用户根据效果定主力与回退顺序（调整 `OBJECT_DETECT_PROVIDERS` 顺序即可）。
3. 部署（3241→3242，`duet-ad1-h3-max-r2v-object-mask{,-caddy}.service`）只在全部测试通过且用户确认后进行；部署前确认无在飞项目。

## 执行约束（给工作子代理）

- TDD：每个功能先写失败测试再实现。
- 所有路径写完整绝对路径；shell 命令经 `rtk`。
- 每阶段结束 git 提交，提交信息说明阶段内容。
- 不改动 `app/face_mask.py`；不加产品门禁；开关不进前端与公开 API。
- 遇到与 spec/计划冲突的现实约束，先记录并报告，不擅自改设计。
