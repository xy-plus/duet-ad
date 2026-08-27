---
name: image-postprocess
description: 分析或验收视频项目冻结关键帧，为全部叙事主人物与真实新场景的同时替换生成严格结构化计划，并以连续性、物理关系和光色守恒为门禁。用于图片后处理的项目级 plan 与 verify 阶段。
---

# image-postprocess

## 原则

最终视频的连续性、现实合理性和叙事关系优先。人物与真实新场景必须同时替换，形成明显表象差异；画面内核、动作目的、相机和全局光色保持不变。任一必要目标无法安全完成时判项目不合格，不得输出成功 no-op。

本 Skill 只生成结构化设计或结构化验收结论，不编辑图片、不调用供应商。`plan` 只生成结构化设计，不直接编写 Seedream 提示词；后端确定性编译器负责加入不可删除的执行约束。用户自由文本不得删除或覆盖硬约束。

执行器逐帧并行编辑：图1始终是唯一编辑画布；其他输入图只提供冻结人物身份、场景设计和本段布局，不传递构图、机位、动作、光线或实体关系。不得使用首帧内容锚定其他帧。

不得出现素材特调：不写只适用于某个测试视频的物体、房间、颜色、构图或位置规则；所有结论只来自本次允许读取的冻结图片。

## 输入边界

`work/request.json` 的 `phase` 只能是 `plan` 或 `verify`。

### plan

只读取：

- `work/request.json`：含 `phase=plan`、`edit_mode`、按段号升序的 `index/chain_id/join_mode`。
- `work/segments/<段号>/keyframes/01.png、02.png …`：每段全部冻结关键帧。

只写 `work/image_optimization.json`。

### verify

只读取：

- `work/request.json`：含 `phase=verify` 与 `segment_indices`。
- `work/frozen_plan.json`：后端冻结的 v2 plan receipt。
- `work/metrics.json`：确定性质量指标。
- `work/segments/<段号>/source/NN.png` 与 `output/NN.png`：一一对应的源帧和结果帧。

只写 `work/image_verification.json`。

两个阶段均不得读取视频、音频、台词、视频生成提示词、项目目录、环境变量、其他路径或未列出的文件；不得联网，不得写其他文件。

## plan 工作流

1. 查看全项目全部冻结帧，先建立镜头、场景组件、人物轨道、遮挡和互动关系。`continue` 优先视为连续画面；`hard_cut` 不自动表示人物或场景改变。
2. 找出全部叙事主人物。叙事主人物是持续承担动作、对白、演示或剧情作用的人；背景路人不是替换目标，写入 `protected_non_target_people`。不能可靠区分多位主人物或可能串人时，保守判不合格，不得错误合并。
3. 为每个稳定叙事主人物建立一个 `person_plans` 项。替换人物保持源人物可见的性别呈现、肤色与族裔外观风格范围、年龄范围和整体气质，但不推断或命名敏感身份；使用长相略有不同的新脸。服装保持用途、颜色关系和整体风格，但款式与局部主色必须产生可见差异。
4. 每段 `persons` 必须按 ID 完整枚举所有主人物。人物在本段任一帧可观察且边界可定位时写 `replace`，并列出所有可观察帧；确实不可观察时写 `not_observable`，不得新增人物。所有可观察主人物都必须替换，不能漏人或二选一。
5. 按同一物理环境建立 `scene_plans`；每段恰好属于一个场景组件。每个场景方案必须同时包含：环境语义真实更换、形状/几何变化、纵深变化、空间布局变化、局部材质或颜色变化。不得只改色相、材质或全局调色，不得把原场景换皮后冒充新场景。
6. 人物与新场景都保持画幅、裁切、机位、镜头、透视、构图、焦点和景深；固定全局光源方向、曝光、白平衡/CCT、tone curve 和整体色彩风格。新几何可以产生物理正确的局部阴影和反射变化，但不能改变全局照明。
7. 冻结人物与核心实体之间的接触、持握、装配、支撑、遮挡、前后顺序、数量、姿态、动作目的和叙事关系。新人物和新场景不得扭曲、融合、穿模或违反现实规则。
8. 每个人物和场景组件选一张目标完整、清晰、遮挡少的冻结帧作为 `reference`。每段选一张最能表达本段源构图与空间布局的帧作为 `layout_reference_frame_index`。引用只能指向对应人物可观察或场景所属的真实输入帧。
9. 禁止恢复或新增字幕、文字、Logo、水印、贴纸、界面、品牌标识或乱码。

### 不合格原因

按下列顺序选择首个成立的稳定 `reason`：

1. `no_observable_narrative_person`
2. `narrative_person_tracks_ambiguous`
3. `person_replacement_unsafe`
4. `scene_components_ambiguous`
5. `scene_structure_replacement_unsafe`

不合格时必须输出 `eligible=false`、对应 reason，且 `person_plans/scene_plans/segments` 全为空。不得输出提示词或不改动方案。

### plan 唯一输出

结构必须严格如下；短视频使用 `segment_indices=[0]`，也必须完整表达人物与场景双目标。所有数组按段号或 ID 升序，ID 从 `01` 连续编号，除规定字段外不得增加字段：

```json
{
  "version": 2,
  "phase": "plan",
  "segment_indices": [1, 2],
  "eligible": true,
  "reason": null,
  "person_plans": [
    {
      "id": "PERSON_01",
      "source_identity": "源主人物的稳定可见特征",
      "replacement_identity": "统一的新人物身份与不同脸部设计",
      "wardrobe_change": "同用途同风格但不同款式的服装设计",
      "local_color_change": "人物局部固有色的明确变化",
      "reference": {"segment_index": 1, "frame_index": 1},
      "observable_segments": [1, 2]
    }
  ],
  "scene_plans": [
    {
      "id": "SCENE_01",
      "source_scene": "源环境的稳定可见语义",
      "replacement_scene": "同叙事用途但不同设计的真实新环境",
      "semantic_change": "环境语义替换说明",
      "geometry_changes": ["可见形状和几何变化"],
      "depth_changes": ["纵深和前后层级变化"],
      "layout_changes": ["空间布局变化"],
      "local_color_change": "场景局部材质或固有色变化",
      "reference": {"segment_index": 1, "frame_index": 1},
      "segments": [1, 2]
    }
  ],
  "segments": [
    {
      "segment_index": 1,
      "persons": [
        {
          "id": "PERSON_01",
          "state": "replace",
          "observable_frames": [1, 2],
          "target_region": "本段人物完整目标域",
          "boundary": "本段人物真实可见边界"
        }
      ],
      "scene": {
        "scene_id": "SCENE_01",
        "target_region": "本段完整场景目标域",
        "boundary": "场景与人物、前景物体的停止边界",
        "layout_reference_frame_index": 1
      },
      "protected_non_target_people": [],
      "protected_relations": ["必须保持的可见空间或物理关系"]
    }
  ]
}
```

`not_observable` 项的 `observable_frames` 必须为 `[]`，`target_region` 与 `boundary` 必须为 `null`。`replace` 项必须至少列一个真实可观察帧。同一 `scene_plans.segments` 集合必须无重叠覆盖全部段；每个 `person_plans.observable_segments` 必须等于段内该人物为 `replace` 的段集合。

## verify 工作流

逐帧对照 source/output、冻结 plan 和确定性指标。不能确认即 `unknown`，不得把缺失证据写成 pass；不要修图或建议重试。

逐人物验证：目标身份确实改变、源身份不再可识别、冻结新身份与局部颜色在可观察帧保持一致；`not_observable` 只能为 `not_applicable`，并确认没有新增该人物。

逐场景验证：环境语义、形状、纵深和空间布局都真实改变，局部材质或固有色不同，不能只有调色或纹理变化。验证全局光源方向、曝光、WB/CCT、tone curve 保持；允许新几何产生合理局部阴影。验证接触、遮挡、动作和现实关系，检查段内连续性、跨段人物身份与场景组件连续性、漏人物、串人和计划外人物。

每项检查严格输出 `{status,evidence}`；status 只能是 `pass/not_applicable/fail/unknown`，evidence 必须简短且来自可见证据或 metrics。任一 `fail` 或 `unknown` 均不得通过。

### verify 唯一输出

```json
{
  "version": 2,
  "phase": "verify",
  "plan_sha256": "逐字复制 frozen_plan.json 的 sha256",
  "segment_indices": [1, 2],
  "passed": false,
  "reason": "scene_replacement_failed",
  "segments": [
    {
      "segment_index": 1,
      "passed": false,
      "person_checks": [
        {
          "person_id": "PERSON_01",
          "identity_changed": {"status": "pass", "evidence": "可见证据"},
          "source_identity_absent": {"status": "pass", "evidence": "可见证据"},
          "local_color_change": {"status": "pass", "evidence": "可见证据"}
        }
      ],
      "scene_checks": {
        "semantic_change": {"status": "fail", "evidence": "可见证据"},
        "geometry_change": {"status": "pass", "evidence": "可见证据"},
        "depth_change": {"status": "pass", "evidence": "可见证据"},
        "layout_change": {"status": "pass", "evidence": "可见证据"},
        "local_color_change": {"status": "pass", "evidence": "可见证据"}
      },
      "invariants": {
        "lighting_preservation": {"status": "pass", "evidence": "可见证据"},
        "interaction_preservation": {"status": "pass", "evidence": "可见证据"},
        "cross_frame_continuity": {"status": "pass", "evidence": "可见证据"}
      }
    }
  ],
  "project_checks": {
    "narrative_person_completeness": {"status": "pass", "evidence": "可见证据"},
    "no_identity_swap": {"status": "pass", "evidence": "可见证据"},
    "no_unplanned_person": {"status": "pass", "evidence": "可见证据"},
    "person_identity_continuity": {"status": "pass", "evidence": "可见证据"},
    "scene_continuity": {"status": "pass", "evidence": "可见证据"}
  }
}
```

reason 取首个成立项：存在 unknown 为 `verification_unknown`；然后依次为 `narrative_person_incomplete`、`identity_swap_detected`、`unplanned_person_detected`、`person_replacement_failed`、`scene_semantic_change_failed`、`scene_geometry_change_failed`、`scene_depth_change_failed`、`scene_layout_change_failed`、`local_color_change_failed`、`lighting_preservation_failed`、`interaction_preservation_failed`、`cross_frame_continuity_failed`、`person_identity_continuity_failed`、`scene_continuity_failed`。全部通过时 reason 为 null。

两个输出均写 UTF-8 裸 JSON，不得使用 Markdown 代码围栏，不得输出解释或其他字段。
