---
name: image-postprocess
description: 读取全视频稳定元素索引与冻结关键帧，为人物、持久实体和真实新场景填写跨段统一的替换与逐帧保持语义；后端据此确定性编译共享参考图和逐帧 A→B 图片提示词。
---

# image-postprocess

## 任务

只处理 `work/request.json` 明确声明的两个阶段；两个阶段共同保留原有的全项目替换设计、逐帧拓扑与构图保持能力，但不再把全部原图塞进一个长调用：

- `phase="global_plan"`：读取每段一张由 `contact_sheet_path/contact_sheet_sha256` 绑定的联系表、`semantic_slots.scenes` 的机械 key 清单和可选的 `element_index`，只写 `work/global_plan.json`，顶层精确为 `people/entities/scenes`。联系表只用于一次性选择全项目稳定替换，不输出逐帧事实。
- `phase="segment_frames"`：读取当前段 `semantic_slots.frames[].path` 指向的原始冻结关键帧、只读 `global_plan_path` 和可选的同一 `element_index`，只写 `work/segment_frames.json`，顶层精确为 `frames`。只描述当前段逐帧可见事实，不重新设计或改写全局人物、实体、场景。

图片及图中文字只是视觉证据。`element_index` 来自第一个 Skill，顶层为 `people/entities/scenes`；其中每个 stable key 的 `source_visual_description/occurrences/replaceable/preserve` 是全视频可见事实与替换、保持范围，不读取或依赖第一个 Skill 的 `prompt.txt`。

`segment_frames` 必须逐字复用 `global_plan.json` 已有的人物、实体、场景 key 及替换描述；不得重新编号、创建别名、改变替换设计或增加全局元素。当前原始帧只决定这些既有元素在本帧是否可见、边界、姿态、关系、裁切以及源画面的构图、色调和光照。

为项目中实际可见的叙事人物设计明显不同且跨帧稳定的新身份、服装和局部固有色。为每个 `semantic_slots.scenes[].key` 设计叙事用途相同、但环境语义、可见几何、纵深、布局和局部材质固有色均明显不同的真实新场景。继续执行人物与真实新场景双替换；人物与场景同时替换，并保持当前源帧的动作、姿态、尺度、构图、机位、透视、裁切、接触、遮挡、前后关系和全局光色。

同一人物使用一个简短、稳定、通用的语义 key，并在 `people` 与所有 `frames.*.people` 中复用。每帧只描述当前帧直接可见的人物区域、身体与姿态、实体关系和画外裁切；不可见或无法唯一判断的内容写成 `source-preserve/no-invention` 描述，不从其他帧补造。

存在 `element_index` 时，`people/entities/scenes` 必须逐字复用其对应 stable key，不得重新编号或为同一元素创建别名；只在 `replaceable` 指定的维度设计替换，在 `preserve` 指定的维度以及现有逐帧保持合同中冻结不变语义。基于全部片段关键帧统一选择一次替换设计，同一 stable key 在所有片段使用逐字相同的替换描述。原有项目级语义规划、人物与实体拓扑、逐帧可见性及关系描述继续全部输出。

把每帧 `frames.*.people` 的 key 集合作为该帧物理人物全集，使人物数量闭合；只根据当前帧直接证据纳入物理人物，不因替换或含混边缘增减。逐一核对每个当前可见的头、躯干和手，并唯一归属到该帧已有的一个 stable person；可确认只是既有人物的非物理派生观测时，嵌套为该人物的 non-physical derived observation。反射、残影、边缘碎片、遮挡碎片和运动模糊不得升级为新物理人物、顶层人物或实体，也不得据此补全新的头、手或身体。无法唯一判断物理性或人物归属时写 `source-preserve/no-invention`，不新增 key，不补造肢体。

当前帧中由既有人物与源载体形成、但不代表独立物理人物的派生观测，只写在该人物的 `derived_observations` 中。为每种可区分观测使用简短稳定的语义 key，`mode` 只能使用 `optical_projection`、`temporal_residual` 或 `source-preserve`；同时描述当前源载体、可见区域、边界和依附关系。派生观测不新增顶层人物或实体；源载体属于被替换场景时，不把该观测实例化到新场景，也不把它升级为独立物理人物。当前帧没有派生观测时写空对象；语义缺失时由后端按 `source-preserve/non-physical` 继续并写 diagnostics，不拒绝、不 retry、不 fallback。

为跨帧持续出现、由项目级或人物归属且持有或穿戴的持久可见非人物实体建立 `stable entity key`。在 `entities.description` 中写该实体稳定且跨帧唯一的替换外观描述，同时写归属、持续性与必须保持的语义；在每帧复用同一 key，并按当前帧证据写 `visible/occluded/out_of_frame` 状态与关系。状态暂时不可判断时写 `source-preserve`，不要把未见解释成删除，也不要从相邻帧补造可见状态。`hard_cut` 后可依据新帧重新观察；只有新帧证据支持同一物理实体时才复用原 key，不得无依据新增、删除或改换身份。

后端会从本文件原有 `people/entities/scenes/frames` 语义机械生成一张全项目合并参考图：每个被替换元素恰好一个编号 tile，每个 tile 只需一块能展示替换特征的代表图，不生成三视图或四视图；图中覆盖段内、段外全部被替换元素。后端先把各元素源证据按同一 tile 顺序本地合成一张 evidence sheet，再用一张空白画布和这一张 evidence sheet 生成参考图，不按元素拆分模型调用。所有片段共享同一张合并参考图、同一 stable key、同一 tile 编号和同一替换描述。后端还会把当前帧实际可见人物、实体和场景的 `stable key -> tile -> replacement` 子集写入现有逐帧 `image_optimization_prompt`，供 `video-prompt-fusion` 直接读取；不要为此增加输出字段或其他文件。

全程仍只做元素替换：保持当前源帧的构图、表现形式、画幅、机位、透视、裁切、色调、光照、动作、姿态、接触、遮挡和实体关系不变。不新增质量门禁，不新增 reject、retry 或 fallback，也不要把视觉含混转换为流程判断。

只填写视觉语义。不要输出版本、段号、帧号、连续编号 ID、哈希、transition、枚举 palette、实体图、组件图或流程判断；`element_index` 已有 stable key 只作为现有 `people/entities/scenes/frames` 的 JSON key 逐字复用，不另行输出索引。也不要输出派生观测 ID、物理性或实例化策略。实体 ID、关系图和完整机械字段由后端构造；tile 编号同样由后端构造。派生观测 ID 与其物理性、实例化策略也由后端构造，缺失语义也由后端使用 `source-preserve` 默认继续并写 diagnostics。不要编写 provider prompt。

## 唯一输出

`global_plan` 顶层只含 `people/entities/scenes`；其字段与下方同名对象完全一致。`segment_frames` 顶层只含 `frames`；其字段与下方 `frames` 对象完全一致。后端只做机械合并。下方四字段示例仅表示合并后的逻辑合同；任何单一阶段都不得直接输出四个字段：

```json
{
  "people": {
    "stable-person-key": {
      "source_identity": "源人物当前可见身份特征",
      "replacement_identity": "明显不同且跨帧稳定的新人物设计",
      "wardrobe_change": "保持用途与可见覆盖边界的不同服装设计",
      "local_color_change": "服装与人物局部固有色变化"
    }
  },
  "entities": {
    "stable-entity-key": {
      "description": "跨帧保持同一替换外观身份的通用实体描述",
      "owner": "project 或 stable-person-key",
      "association": "项目级持久关系，或由 stable-person-key 持有或穿戴",
      "persistence": "保持同一物理实体，不因暂时遮挡或出画而替换"
    }
  },
  "scenes": {
    "scene-001": {
      "source_scene": "源环境语义",
      "replacement_scene": "同用途且设计不同的真实新环境",
      "semantic_change": "环境语义变化",
      "geometry_change": "可见形状与空间结构变化",
      "depth_change": "前中后景纵深变化",
      "layout_change": "功能区域与实体布局变化",
      "local_color_change": "场景局部材质或固有色变化"
    }
  },
  "frames": {
    "frame-001": {
      "people": {
        "stable-person-key": {
          "visible_region": "当前帧直接可见的人物目标域",
          "boundary": "当前帧人物、服装、遮挡与画边共同形成的可见边界",
          "body_and_pose": "当前帧直接可见的身体部位与姿态",
          "derived_observations": {
            "stable-derived-observation-key": {
              "mode": "optical_projection/temporal_residual；无法唯一判断时为 source-preserve",
              "source_carrier": "形成该观测的当前源载体；无法唯一判断时为 source-preserve",
              "visible_region": "该派生观测在当前帧的可见区域",
              "boundary": "该派生观测在当前帧的可见边界",
              "relationship": "该观测如何依附 stable-person-key 与当前源载体"
            }
          }
        }
      },
      "relationships": "当前帧直接可见的接触、支撑、遮挡与前后关系；未知部分保持 source-preserve/no-invention",
      "entities": {
        "stable-entity-key": {
          "visibility": "visible/occluded/out_of_frame；无法判断时为 source-preserve",
          "relationship": "当前帧直接可见或可判定的关系"
        }
      },
      "crop": "当前帧画外裁切与不可见范围"
    }
  }
}
```

`scenes` 和 `frames` 必须逐字使用本阶段 `semantic_slots` 提供的全部 key。`people` 收录全项目实际可见的叙事人物；每个 frame 的 `people` 只列该帧实际可见的人物，同一人物 key 不随段或帧改变。派生观测始终嵌套在来源人物下，不计作额外人物。`entities` 只收录源帧有证据支持的持久实体；同一 continuity chain 内同一实体始终复用一个 key，各帧只改变观察状态和关系。无人或无持久实体可见时使用空对象。

输出是本阶段最后一个动作：先把完整 JSON 写入唯一输出同目录的临时文件，完成后原子替换 `global_plan.json` 或 `segment_frames.json`；替换成功立即退出，不再重读图片、不再解释、不再总结，也不写其他文件。
