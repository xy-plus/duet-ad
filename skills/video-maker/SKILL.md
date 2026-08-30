---
name: video-maker
description: 分析后端为每个 segment 冻结的 9 张原始关键帧并生成旧视频提示词；随后以一次项目级调用建立跨段人物、实体、场景及其物理或功能关系的稳定索引。用于原视频视觉分析和下游替换对齐，不做选帧、图片替换或最终提示词融合。
---

# video-maker

## 逐段分析

项目统一为 `segments[N>=1]`。每次只处理当前 segment 的 `work/keyframes/01.png` 至 `09.png`、联系表和 `manifest.json`；像素证据优先于辅助信息。九帧由后端按源时间冻结，模型无权选帧或修改其数量、顺序和字节。

1. 按 01 至 09 顺序查看全部冻结帧；提取主体、场景、对象、动作、结果、镜头、构图、转场、空间关系和因果。
2. 只据已查看帧写 `work/prompt.txt`：使用用户语言，描述可见主体与对象、镜头与构图、按图片顺序的动作因果、相对节奏和 segment 时间轴；与源片段时长一致但不写具体秒数，不引入优化后内容、声音或不可见事实。

逐段模型只输出旧视频提示词；九帧、校验和与正式发布均由后端负责。

## 项目级索引

仅当 `work/project_index_request.json` 声明 `phase="project_index"` 时执行一次。输入是所有 segment 的半尺寸分析副本及 SHA；按 segment、frame 顺序查看全部帧，不读取 `prompt.txt`，只输出元素索引。

先逐帧列出直接可见的人物、可独立移动或被动作作用的持久实体、场景以及元素间关系，再跨帧回查并合并。使用不可变中性 ID：`person-01`、`entity-01`、`scene-01`、`relation-01`。stable key 只是绑定 ID，不得含源属性。同一实例逐字复用；证据不足、特征冲突或同类实例可分别移动/接触时分开。硬切后重新依据当前帧确认；位置、相邻帧、服装颜色或叙事角色不能单独证明同一性。碎片、倒影、残影、模糊、过渡扫到的背景杂物不升格；occurrence 只记录当前帧可证实内容。

输出格式以本次调用注入的 JSON Schema 为唯一权威。`people/entities/scenes/relations` 都是数组，发现的稳定 ID 写入各记录的 `key`；人物、实体和场景记录包含 `source_visual_description/occurrences/replaceable/preserve`，关系记录包含 `subject_key/predicate/object_key/occurrences/preserve/replace_together`。后端按 `key` 建索引，不接受重复 key、额外字段或缺字段。

`scenes` 必须非空，且每个输入帧必须且只能归属一个 scene occurrence；所有 scene occurrences 合并后逐项等于输入中的全部 segment/frame，不得引用未知帧、重复归属或漏帧。人物、实体和关系只记录有直接证据的内容，可以为空。

关系必须是像素可证实的物理、空间或功能关系，例如连接、容纳、持有、驱动、释放、接触、支撑或组成；用 `subject_key/predicate/object_key` 固定角色，不能把主客体互换。逐帧记录当前状态和相对几何，使装配、作用、释放、分离等变化保持同一关系 ID；这是下游逐帧关系状态和几何的唯一 producer，后续 phase 不重新生成。仅当两个元素需要保持接口、尺度或功能配合时设 `replace_together=true`。不因常识补造功能，不把动作先后误写为关系。

`replaceable` 只写可替换属性，`preserve` 只写必须保持的身份、形态、关系或连续性。空类别写 `[]`。只填写 Schema 字段；后端校验冻结输入并原子发布。
