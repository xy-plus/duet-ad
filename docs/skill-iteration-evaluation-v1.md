# Image Skill 离线迭代评分合同 v1

本合同只用于离线比较 `image-postprocess` 与 `video-prompt-fusion` 的 Skill 版本。它不被生产代码导入，不改变项目状态，也不构成生成门禁。

## 共同实验规则

本评估的目标是降低视觉重复和查重相似性，同时保持原有叙事、动作、关系与镜头语义不变。

- 每个数据集样例、每个 Skill 版本独立运行 3 次；冻结输入、Skill 与输出 SHA-256。
- 语义评分采用 0 至 4 分，并必须附带冻结 artifact SHA、JSON Pointer、stable key、分段和帧序号；不能只写自然语言理由。
- 确定性事实和主观评分分离。事实违反对应规则时，该评分维度强制计 0 分，任一核心规则失败时总分封顶 59；Schema、输入/输出 SHA、数量或顺序无效时整轮记 0。
- Image 必须同时冻结原始 `global_plan`、每段原始 `segment_frames`、后端编译后的 plan 与 prompts。评分以原始输出为主，防止后端规范化掩盖 source/target 无实质差异。
- 数据集 manifest、oracle、runner、evaluator、盲评 prompt、模型及模型配置都冻结 SHA；候选代理只能修改对应 `SKILL.md`。
- 比较版本时，数据集 SHA、样例集合、重复序号与每次输入 SHA 必须相同。
- 第一轮只精简 Skill：先删除重复描述、后端 Schema/校验器已经保证的细节和不影响模型决策的微约束，以最少文字保留任务目标、证据权威、核心不变量与输出边界。10% 只是防止删除空白冒充精简的最低门槛，不是优化目标；应逐级压缩并选择所有非退化候选中最短的一版。结构有效率、策略满足率、中位分、最低分和各维度中位分、最低分均不得下降。
- 后续语义轮必须提升策略满足率、中位分或最低分之一，同时不得降低上述任何指标。
- 评分结果只给出是否建议进入下一轮离线实验；不允许接入线上创建、恢复、后处理或 H3 流程。

## image-postprocess 预期

1. 用户提供参考图和说明时，必须把它绑定到唯一正确的 stable key，并按其指定目标替换。
2. 所有出现的人物都必须替换：保持可见性别表达、种族呈现和整体风格；脸部身份要有轻微但可辨识的变化；服装保持颜色与风格，但剪裁或款式有轻微变化。
3. 非人物类别不预先指定；除人物外替换数量必须达到 `min(2, non_person_candidate_count)`。用户指定的非人物目标计入这个数量，但当候选数大于 1 时，`replaced_non_person_keys` 仍必须包含至少一个非用户目标；候选不足 2 个时替换全部候选。
4. 选中场景时，保持场景类别和叙事功能，但必须换成可辨识的不同实例，例如厨房仍是厨房，但不是原厨房。
5. 替换目标不能只是原描述改写；`source` 与 `target` 实质相同视为未替换。
6. 机位、构图、光线、镜头、动作、姿态、遮挡和接触关系保持不变。
7. 同一人物、道具或场景跨图片、跨分段只允许有一个转换后的设计。

| 维度 | 权重 |
| --- | ---: |
| Schema 与 stable-key 绑定 | 10 |
| 用户参考图绑定 | 10 |
| 所有人物完成替换 | 15 |
| 人脸身份轻微变化 | 10 |
| 人物风格与服装相似性 | 10 |
| 至少两个非人物替换 | 15 |
| source/target 实质差异 | 10 |
| 场景同类不同实例 | 5 |
| 机位、光线、镜头保持 | 5 |
| 跨帧 stable-key 一致 | 10 |

## video-prompt-fusion 预期

1. 新关键帧是静态视觉事实的唯一权威；旧视频提示词只能提供动作、机位运动、时间节奏与前后关系。
2. 不得从旧提示词恢复已经被替换掉的人脸、服装、物体或场景静态特征。
3. image-postprocess 的所有替换 stable key 都必须在对应区间得到传播，且跨段保持一致。
4. 保留原视频的动作阶段、方向、机位、光线、镜头节奏、关系主客体和功能角色。
5. 硬切前后的静态事实不得跨边界投射。
6. 对话与音频文字不得泄漏到纯视觉描述中。

| 维度 | 权重 |
| --- | ---: |
| Schema、数量与顺序 | 10 |
| 新关键帧静态事实权威 | 20 |
| 排除旧静态事实 | 15 |
| 替换目标完整传播 | 15 |
| 动作、机位、节奏保持 | 15 |
| 硬切不跨界 | 10 |
| 关系保持 | 5 |
| 音频与视觉分离 | 5 |
| 跨段 stable-key 一致 | 5 |

## 使用方式

评分输入遵循 `tests/skill_iteration_score.py` 的严格 JSON 合同。示例命令：

```bash
/home/xy/duet-ad1/.venv/bin/python /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/tests/skill_iteration_score.py score --report /absolute/path/report.json --dataset-manifest /absolute/path/manifest.json --oracle /absolute/path/oracle.json --output /absolute/path/summary.json
/home/xy/duet-ad1/.venv/bin/python /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next/tests/skill_iteration_score.py compare --baseline /absolute/path/baseline.json --candidate /absolute/path/candidate.json --dataset-manifest /absolute/path/manifest.json --oracle /absolute/path/oracle.json --phase simplify --output /absolute/path/comparison.json
```
