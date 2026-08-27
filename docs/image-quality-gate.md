# 图片替换质量门

`app.image_quality` 是纯验收层，不调用图片供应商，也不负责自动重试。只有确定性指标和语义验收全部为 `pass` 时，`QualityReceipt.publishable` 才为 `true`。

## 调用顺序

1. 用冻结 v2 plan 生成并持久化 `_image_edit_masks`。
2. 调用 `mask_manifest_receipt(...)` 验证 plan、帧清单和 producer receipt 绑定。
3. 调用 `load_frame_masks(...)` 验证项目内路径、PNG、SHA、尺寸、alpha、目标域重叠。
4. 先调用 `evaluate_reference_packs(...)`；只有 durable receipt 重新读取后仍可发布，才允许逐帧付费 POST。
5. 图片编辑结束后调用 `evaluate_image_quality(...)`。
6. 发布前调用 `quality_receipt(...)` 从磁盘重新绑定 plan、mask manifest 及源/结果图片。

## Mask manifest v1

顶层 exact keys：

```text
schema, version, plan_sha256, frames, sha256
```

每帧 exact keys：

```text
segment_index, frame_index, source, persons, scene, protected_non_target
```

`source` 和 `frame_inventory[].source` 都是：

```text
path, sha256, width, height
```

`persons[]` 是 `person_id + path/sha256/width/height + producer_receipt`。receipt 必须为 `duet.image-mask-producer` v1 且 purpose 为 `person`。

`protected_non_target` 使用相同 artifact 结构，purpose 必须为 `protected_non_target_people`。它证明受保护人物没有进入 target；确定性门实际使用的完整非目标域是 `NOT(person targets UNION scene target)`，因此核心道具和互动实体也不会漏出保护范围。

`scene` 直接使用 scene-mask gateway 的完整 `SceneMaskItem`：

```text
purpose, channel, component_id, shot_id, frame_id, path, sha256,
byte_size, width, height, producer_receipt
```

scene receipt 必须为 `duet.scene-mask.producer` v1，绑定同一个 plan、源帧、component、shot 和 frame；membership 只能是 SAM2，BiRefNet 只能精修 SAM2 不确定边缘，禁止 fallback。

所有路径必须是项目内相对 POSIX 路径。person/protected 接受 provider 原始 BGRA PNG 的 alpha；scene 接受单通道 grayscale-alpha PNG。空 mask、整图 mask、逃逸、symlink、SHA/尺寸/计数不匹配以及 target/protected 重叠都拒绝。

## Profile 与门

调用方必须显式传 `QualityProfile`。`POC_PROFILE_V1` 的 calibration 固定为 `uncalibrated_poc`，仅用于离线 POC，不能视为生产校准。

默认确定性门包括：

- 尺寸；
- 全局及 protected 区域曝光、白点、对比度、色温代理；
- 人物和场景局部 Delta-E 76；
- 场景目标域边缘变化；
- protected 区域边缘 IoU；
- protected 区域显著性质心偏移。

语义 verify 必须逐人物证明身份变化、源身份无残留、局部色变化；逐场景分别证明语义、几何、纵深、布局和局部颜色变化；并验证全局光照、互动/遮挡、逐帧及跨段连续性。任何 `unknown` 都不能发布。

这些确定性指标是 POC 代理，不等价于人物识别、深度或真实 3D 几何。远端身份、depth、flow 等能力应作为额外 `QualityGate` 注入；能力缺失时必须返回 `unknown`，不能用 bbox 或全图 mask 代替。

## Reference pack 前置门

`evaluate_reference_packs(...)` 要求 source/generated mappings 的 keys 精确覆盖全部人物和场景 ID。每个人物至少两张不同生成视图；source slot 只作为验收证据，禁止直接作为最终 target reference。

语义 verifier 必须返回每个人物的：

```text
identity_changed, source_identity_absent, multi_view_consistency,
local_color_change
```

每个场景的：

```text
semantic_change, geometry_change, depth_change, layout_change,
local_color_change
```

项目级还必须分别验证全局光向、曝光、WB/CCT 和 tone curve。

## 重试与发布

所有 receipt 的 `provider_retry_allowed` 固定为 `false`。质量失败、语义失败、unknown、Codex 失败或 receipt 损坏都只会阻止发布；本模块没有 provider POST，也不产生自动重试信号。
