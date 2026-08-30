---
name: video-maker
description: 分析预抽取视频帧，为每个 segment 选择恰好 9 张未修改原始关键帧并生成旧视频提示词；随后以一次项目级调用建立跨段人物、实体、场景及其物理或功能关系的稳定索引。用于原视频视觉分析和下游替换对齐，不做图片替换或最终提示词融合。
---

# video-maker

## 逐段分析

项目统一为 `segments[N>=1]`。每次只处理当前 segment 的 `work/NN_frame_*.png`、联系表、`manifest.json` 和可选 `scenes.json`；像素证据优先于辅助信息。

1. 查看全部联系表，必要时查看原始帧；提取主体、场景、对象、动作、结果、镜头、构图、转场、空间关系和因果。
2. 选择恰好 9 张不同原始帧，覆盖初始、推进、必要转场和结果，按源时间升序逐字节复制为 `work/keyframes/01.png` 至 `09.png`。不得重复、裁剪、调色、修图或生成替代帧；不足 9 张则说明输入不足。
3. 只据已查看帧写 `work/prompt.txt`：使用用户语言，描述可见主体与对象、镜头与构图、按图片顺序的动作因果、相对节奏和 segment 时间轴；与源片段时长一致但不写具体秒数，不引入优化后内容、声音或不可见事实。

逐段输出只有 9 张关键帧和旧视频提示词；`prompt.txt` 最后完成，校验和与发布由后端负责。

## 项目级索引

仅当 `work/project_index_request.json` 声明 `phase="project_index"` 时执行一次。输入是所有 segment 的半尺寸分析副本及 SHA；按 segment、frame 顺序查看全部帧，不读取 `prompt.txt`，只输出元素索引。

先逐帧列出直接可见的人物、可独立移动或被动作作用的持久实体、场景以及元素间关系，再跨帧回查并合并。使用不可变中性 ID：`person-01`、`entity-01`、`scene-01`、`relation-01`。stable key 只是绑定 ID，不得含源属性。同一实例逐字复用；证据不足、特征冲突或同类实例可分别移动/接触时分开。硬切后重新依据当前帧确认；位置、相邻帧、服装颜色或叙事角色不能单独证明同一性。碎片、倒影、残影、模糊、过渡扫到的背景杂物不升格；occurrence 只记录当前帧可证实内容。

顶层精确为 `people/entities/scenes/relations`：

```json
{
  "people": {"person-01": {"source_visual_description": "string", "occurrences": [{"segment_index": 1, "frame_orders": [1]}], "replaceable": ["string"], "preserve": ["string"]}},
  "entities": {"entity-01": {"source_visual_description": "string", "occurrences": [{"segment_index": 1, "frame_orders": [1]}], "replaceable": ["string"], "preserve": ["string"]}},
  "scenes": {"scene-01": {"source_visual_description": "string", "occurrences": [{"segment_index": 1, "frame_orders": [1]}], "replaceable": ["string"], "preserve": ["string"]}},
  "relations": {"relation-01": {"subject_key": "entity-01", "predicate": "string", "object_key": "entity-02", "occurrences": [{"segment_index": 1, "frames": [{"frame_order": 1, "state": "string", "geometry": "string"}]}], "preserve": ["string"], "replace_together": true}}
}
```

关系必须是像素可证实的物理、空间或功能关系，例如连接、容纳、持有、驱动、释放、接触、支撑或组成；用 `subject_key/predicate/object_key` 固定角色，不能把主客体互换。逐帧记录当前状态和相对几何，使装配、作用、释放、分离等变化保持同一关系 ID。仅当两个元素需要保持接口、尺度或功能配合时设 `replace_together=true`。不因常识补造功能，不把动作先后误写为关系。

`replaceable` 只写可替换属性，`preserve` 只写必须保持的身份、形态、关系或连续性。空类别写 `{}`。最终回答只返回上述 JSON object；后端捕获并发布。
