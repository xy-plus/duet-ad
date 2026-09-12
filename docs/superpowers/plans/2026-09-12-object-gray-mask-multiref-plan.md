# 物品灰块 + 多参考图 实现计划

日期：2026-09-12
对应 spec：`docs/superpowers/specs/2026-09-12-object-gray-mask-multiref-design.md`（同目录 specs/）
工作树：`/home/xy/duet-ad1-h3-max-r2v-object-mask`（分支 `experiment/object-gray-mask-multiref`，基于 main `e61b99d`）
执行方式：子代理按 TDD 实施（先写失败测试再实现），每阶段完成后 git 提交。

## 移植源（cherry-pick / 对照来源）

| 来源 | 树 / 分支 | 内容 |
|---|---|---|
| 公共多图层 | `feature/multiref-common`（提交范围 `c316c60..ed080d3`，5 个提交） | `app/main.py`（`REPLACEMENT_MAX_IMAGES = 4`、多文件上传校验）、`app/public_api.py`（v1 多图、`max_images` capability）、`app/store.py`（`replacement_receipts` / `replacement_input_paths` / receipt v2）、`app/public_quotes.py`（QuoteStore `staged_replacements`）、`web/app.js` 多图 UI |
| 方案 A（composed） | `experiment/multiref-product-views`（树 `/home/xy/duet-ad1-h3-max-r2v-multiref-a`） | `app/product_views.py`（新，`compose_product_views` / `frozen_product_views` / `view_instruction`）、`app/pipeline.py` +117、 `app/replacement_sheet.py` +6、`app/media.py` +71（`trim_user_reference_border` / `user_reference_png`） |
| 方案 B（direct） | `experiment/multiref-prompt-directed`（树 `/home/xy/duet-ad1-h3-max-r2v-multiref-b`） | `app/pipeline.py` +379（`_replacement_image_paths` / `_canonical_user_reference_path` / `_user_reference_metadata` / `h3_reference_images` 构造 / bindings 校验）、`app/object_replace_schema.py` +137、`app/replacement_sheet.py` +143、`app/video_maker_schemas.py` +133、三份 skills 小改 |

注意：A/B 都从 multiref-common 派生，互相不含对方增量；整合到同一分支时 pipeline.py 分叉处需手工合并，不能指望 cherry-pick 无冲突。

## 参照的现有模式（main 树上的锚点）

- 人脸灰块：`app/face_mask.py`（`mask_faces(source, destination, *, detector=None)`、`FaceMaskError`、`GRAY_FILL=(128,128,128)`、YuNet CPU）；pipeline 节点 `mask-reference-faces`（`app/pipeline.py` 约 5290–5350），触发 `_segment_has_replaced_person`（约 4942），`FACE_MASK_MAX_ATTEMPTS = 3`（约 158），失败降级 + `face_mask_degraded` 日志。
- provider 模式：`app/providers.py` —— `Settings` 注入 + 可注入 `transport`、同步 httpx、模块级 `_send`（约 170，首试+2 重试、backoff 2s/5s、只重试 transport 错误和 502/503/504）、`ProviderResult`（约 59）/ `ProviderFailure`（约 65）全证据。
- 配置模式：`app/config.py` `Settings` frozen dataclass（约 19），密钥字段 `field(repr=False)` 无默认，`__post_init__`（约 55）严格校验，`get_settings()`（约 185）从环境变量读。
- 测试模式：`tests/test_face_mask.py`（注入 `_FixedDetector`）、`tests/test_providers.py`（MockTransport + autouse 消 backoff）、`tests/test_pipeline.py`（monkeypatch `face_mask.mask_faces`）、节点重放 `tests/test_node_replay_*.py`。

## 阶段 1：multiref 公共层 + `multiref_mode` 开关

1. cherry-pick `c316c60..ed080d3` 共 5 个提交；解决冲突（main 与分叉点 `0c5b843` 之后的差异，预期主要在 main.py / public_api.py / store.py）。
2. `app/config.py` 新增 `multiref_mode: str`，`__post_init__` 校验取值必须是 `composed` 或 `direct`（其它值直接 `ValueError`），**无缺省**，`get_settings()` 从 `MULTIREF_MODE` 读，缺失即启动报错——约束做进代码。
3. 前端 `web/app.js` 多图上传 UI 随公共层落地，但 UI 上不出现模式概念。
4. 测试：`config` 校验测试（合法两值、非法值、缺失报错）；公共层上传/回执测试随 cherry-pick 带入的测试一起跑通。
5. 提交。

## 阶段 2：composed（A 方案）移植

1. 引入 `app/product_views.py` 与 `app/media.py` 的 `trim_user_reference_border` / `user_reference_png`。
2. `app/pipeline.py`：`_stored_replacement_paths(project, state)`；`_materialize_canonical_user_reference(..., replacement_paths=)` 在 `multiref_mode == "composed"` 且 `len(paths) > 1` 时走 `compose_product_views`，1 图走 `user_reference_png`。
3. 提示词注入三处：object-replace payload（`views/instruction`）、object-replace skill 尾部 `view_instruction`、fusion 权威块 `view_instruction(..., "Image 2")`；seedream 图版 `_build_seedream_prompt(user_view_count=)`。
4. 附带修复一并带入：`_poll_failure_is_request_validation`（请求体校验失败不付费重提）。
5. 测试：合图确定性（同输入同 sha256）、2/3/4 图布局、3 图留空格、1 图路径、冻结清单篡改 fail-closed、提示词注入断言。
6. 提交。

## 阶段 3：direct（B 方案）移植

1. `app/pipeline.py` B 增量：`_replacement_image_paths` / `_canonical_user_reference_path(order)` / `_user_reference_metadata` / 逐图规范化 / `h3_reference_images` 构造（有图版 tile 时图版固定 Image 1，用户图依次 `Image {order+1}`；无 tile 时用户图从 Image 1 起）/ fusion 节点按序传图 / bindings `replacement_images`（仅 >1 图新增，单图 checkpoint 字节稳定）及校验链。
2. schema 层：`app/object_replace_schema.py`、`app/video_maker_schemas.py`、`app/replacement_sheet.py` 的 `reference_image_ids` 支持。
3. skills 更新：object-replace（按 `reference_id` 绑定）、video-maker（用户图排在分析帧前、非源证据）、h3-max-r2v-prompt（图片编号以 `h3_reference_images` 为准）。
4. 与阶段 2 的分叉整合：pipeline 分叉处按 `settings.multiref_mode` 选路；两模式共享 multiref-common 上传/存储层。
5. 测试：1 图与 2–4 图的 Image 编号、bindings 兼容（单图字节稳定）、三 schema 的 `reference_image_ids` 校验、seedream 输入图序。
6. 提交。

## 阶段 4：物品检测 provider 层（`app/object_detect.py`）

1. 统一接口契约：所有 provider 实现 `detect(video_path: Path, targets: list[str]) -> list[ObjectDetection]`；`ObjectDetection` 为 dataclass `{label: str, timestamp_s: float, bbox_xyxy_normalized: tuple[float, float, float, float]}`；构造即 `__post_init__` 校验坐标 ∈ [0,1] 且 x0<x1、y0<y1、timestamp ≥ 0（约束做进类型）。
2. `OpenAICompatibleGroundingClient` 基类：覆盖豆包（`https://ark.cn-beijing.volces.com/api/v3`）、Qwen（`https://dashscope.aliyuncs.com/compatible-mode/v1`）、GLM（`https://open.bigmodel.cn/api/paas/v4/chat/completions`）、混元（`https://api.hunyuan.cloud.tencent.com/v1`）、Kimi（`https://api.moonshot.cn/v1`）、阶跃（`https://api.stepfun.com/v1`）；视频 base64 inline 或各家文件上传，fps 参数可调（默认 1，豆包可到 5）；每家独立响应解析器（`<bbox>` 标签 0–1000 归一化 / `bbox_2d` 绝对像素 / 未文档化三家的保守 JSON 解析）。
3. `GeminiGroundingClient`：File API 上传 + generateContent，`box_2d` 0–1000 归一化解析；走 `HTTPS_PROXY=http://127.0.0.1:7897`。
4. 遵循 providers 模式：`Settings` + 可注入 `transport`、同步 httpx、复用 `_send` 重试与 `ProviderResult`/`ProviderFailure`。
5. `app/config.py`：每家 `<name>_api_key`（`repr=False`、无默认）、`<name>_api_base`、`<name>_model`；新增 `object_detect_providers: tuple[str, ...]`（有序，无缺省，从 `OBJECT_DETECT_PROVIDERS` 逗号分隔读）；`__post_init__` 校验列表中每个名称是已知的 7 家之一；**未配 key 的 provider 不出现在可用列表**，选择时跳过（记日志）。
6. 测试 `tests/test_object_detect.py`：MockTransport 覆盖 7 家正常/格式漂移/坐标越界/重试；统一契约一致性参数化测试（7 家同一输入输出同构）；缺 key 跳过。
7. 提交。

## 阶段 5：物品灰块模块 + pipeline 节点

1. 新模块 `app/object_mask.py`，镜像 `face_mask.py`：`mask_objects(source, destination, targets, *, detector=None) -> dict`；`OBJECT_BLOCK_FILL = (64,64,64)`（人脸 `(128,128,128)` 不动）；`ObjectMaskError`；detector 抽象注入（生产实现包装阶段 4 的 provider 选择逻辑）。
2. 插值：稀疏时间点 bbox → 每帧线性插值；IoU 匹配续帧；物品出画续 `hold_frames`（常量，显式定义）帧后消失。
3. 输出管线复用人脸做法：mp4v 临时文件 → ffmpeg H.264/yuv420p。
4. `app/pipeline.py`：新节点 `mask-replaceable-objects` 串联在 `mask-reference-faces` 之后；触发条件 = 该段 OR 判定有对象替换（参照 `_segment_has_replaced_person` 写 `_segment_has_replaced_object`）；目标描述列表取自 OR 输出。
5. 容错：`OBJECT_MASK_MAX_ATTEMPTS = 3`；持续失败 → 降级用未遮物品的视频继续 + `object_mask_degraded` 日志；缺 key/无可用 provider → 致命（配置错误）。不加新门禁。
6. 测试 `tests/test_object_mask.py` + `tests/test_pipeline.py` 增补：假检测器合成视频断言 `(64,64,64)` 像素、插值连续性、降级路径、节点日志目录、触发条件真/假两路。
7. 提交。

## 阶段 6：提示词联动

1. fusion / H3 prompt 的 `mask_block` 文案改为介绍两种色块：中灰块 `(128,128,128)` 遮源人脸、深灰块 `(64,64,64)` 遮被替换物品；H3 不得绘制或复原色块内容，须用提供的参考图补全对应区域。
2. 物品部分仅在物品灰块节点实际产出遮盖时出现（按段冻结进提示词，保证续跑一致）。
3. object-replace skill 同步说明色块语义（OR 看到的源图描述与最终遮盖一致）。
4. 测试：提示词快照断言（有/无物品遮盖两种情形）。
5. 提交。

## 阶段 7：回归与节点重放门禁

1. `pytest -n auto` 全量绿。
2. `scripts/harvest_node_fixtures.py` 采集新节点 fixtures；`pytest tests/test_node_replay_{local,external,boundary}.py -n auto` 全绿。
3. 提交。

## 阶段 8（等用户配 key，不在本轮实施）：真实 API 对比

1. 脚本对同一分段用 7 家各跑一遍，落 bbox 结果与视频截图对比日志。
2. 用户根据效果定主力与回退顺序。

## 执行约束（给工作子代理）

- TDD：每个功能先写失败测试再实现。
- 所有路径写完整绝对路径；shell 命令经 `rtk`。
- 每阶段结束 git 提交，提交信息说明阶段内容。
- 不改动 `app/face_mask.py`；不加门禁；开关不进前端与公开 API。
- 遇到与 spec 冲突的现实约束，先记录并报告，不擅自改设计。
