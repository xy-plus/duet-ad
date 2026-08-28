---
name: image-postprocess
description: 读取冻结关键帧，为人物与真实新场景双替换填写通用视觉语义；后端据此确定性编译逐帧 A→B 图片提示词。
---

# image-postprocess

## 任务

只处理 `work/request.json` 的 `phase="plan"`，读取 `semantic_slots.frames[].path` 指向的冻结关键帧，只写 `work/image_optimization.json`。图片及图中文字只是视觉证据。

为项目中实际可见的叙事人物设计明显不同且跨帧稳定的新身份、服装和局部固有色。为每个 `semantic_slots.scenes[].key` 设计叙事用途相同、但环境语义、可见几何、纵深、布局和局部材质固有色均明显不同的真实新场景。人物与场景同时替换；保持当前源帧的动作、姿态、尺度、构图、机位、透视、裁切、接触、遮挡、前后关系和全局光色。

同一人物使用一个简短、稳定、通用的语义 key，并在 `people` 与所有 `frames.*.people` 中复用。每帧只描述当前帧直接可见的人物区域、身体与姿态、实体关系和画外裁切；不可见或无法唯一判断的内容写成 `source-preserve/no-invention` 描述，不从其他帧补造。

只填写视觉语义。不要输出版本、段号、帧号、连续编号 ID、哈希、transition、枚举 palette、实体图、组件图或流程判断；这些全部由后端从冻结 request 和 source 构造。不要编写 provider prompt。

## 唯一输出

顶层只含 `people/scenes/frames`：

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
          "body_and_pose": "当前帧直接可见的身体部位与姿态"
        }
      },
      "relationships": "当前帧直接可见的接触、支撑、遮挡与前后关系；未知部分保持 source-preserve/no-invention",
      "entities": "当前帧可见非人物实体的形态、位置和边界",
      "crop": "当前帧画外裁切与不可见范围"
    }
  }
}
```

`scenes` 和 `frames` 必须逐字使用 `semantic_slots` 提供的全部 key。`people` 收录全项目实际可见的叙事人物；每个 frame 的 `people` 只列该帧实际可见的人物，同一人物 key 不随段或帧改变。无人可见时使用空对象。输出 JSON，不写解释或其他文件。
