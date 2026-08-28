---
name: submit-gate
type: behavior
status: done
owner: human
updated: 2026-08-28
tdd: N/A
links: [conversation-task, processing-state, long-video, postprocess]
---

# H3 统一提交门控

当前 v4 项目只有一个提交合同。单段项目是 `segments.length=1`，不得选择独立 short 提交器；多段项目也不得复制另一套参数、音频或恢复逻辑。

```json
{
  "confirm": true,
  "client_request_id": "request-123456",
  "dialogue_mode": "auto",
  "dialogue_delivery": "off_screen",
  "fit_mode": "none",
  "aspect_ratio": "9:16",
  "resolution": "768p",
  "expected_plan_receipt": "64 lowercase hex characters",
  "fast_mode": true
}
```

- `edit/custom` 还必须带非空 `lines`；`auto/none` 禁止带 `lines`。
- `dialogue_mode != none` 时必须由用户显式选择 `on_screen` 或 `off_screen`；`none` 不接受声音呈现和音频 reference。
- `fast_mode` 使用统一执行器；`N=1` 时没有并行兄弟，字段不产生另一条业务路径。
- `expected_plan_receipt` 对所有当前 v4 项目生效，用来防止旧页面、图片确认或生成参数覆盖新状态。

## 唯一编译规则

提交门内只允许普通后端纯函数做下列确定性投影，不调用 Skill：

| 输入 | H3 冻结结果 |
| --- | --- |
| `dialogue_mode=none` | 零台词、零声音 reference |
| `dialogue_mode=auto` | 仅保留 `classification=spoken` 的 ASR 行；既有 `sung`（包含 YAMNet 归并的 singing/chant/rap/humming）全部排除。没有真实口播时冻结零台词、零声音 reference |
| `dialogue_delivery=off_screen` | 逐行保留冻结 text/start/end，`delivery=off_screen`、无画内 subject，只能绑定已证明为 clean voice 的唯一 reference；完整源混音必须由同一既有 YAMNet 收据证明 `spoken && has_bgm=false` |
| `dialogue_delivery=on_screen` 且主分析已有同一行的画内人物/时间证据 | 逐行绑定既有证据与唯一 voice reference |
| `dialogue_delivery=on_screen` 但证据缺失 | 409 `on_screen_authority_unavailable`，Context/H3 POST 为 0；不得调用额外 Skill 补证 |

语言字段只复用上游已经冻结的权威值；没有权威值时使用格式占位 `und`，不得由后端按样本文字猜测语言。

如果确定性投影更新了 plan receipt，服务可以返回既有的 409 refresh/CAS 提示；Web 只刷新详情并保留用户选择，不自动再次 POST。用户再次确认后继续同一统一链路。这是输入冻结，不是新的产品阶段或 Skill。

## 门控规则

| 条件 | 结果 |
| --- | --- |
| `ENABLE_H3_SUBMIT` 未开 | 501 `H3 submission is disabled.` |
| 会话不存在 | 404 `not found` |
| schema 不是 v2 | 409 `read_only` |
| 未完成图片后处理或每段不是恰好 9 张图 | 409 `postprocess_artifacts_invalid`；不回退原图 |
| v4 图片尚未由用户确认，或确认绑定的图、顺序、计划已漂移 | 409 image acceptance 错误；不创建付费任务 |
| 自动模式只检测到歌词或歌唱 | 按无台词冻结，完整源混音不进入 Fusion/H3；无音频 H3 输出在拼接时静音 |
| 既有 YAMNet 检测到 BGM，或无法给当前完整混音证明无 BGM | 该混音禁止成为 voice reference；不调用替代分类器或新 Skill，无法走无台词路径时付费前拒绝 |
| `spoken/edit/custom` 台词非空但 YAMNet 检测到或无法排除 BGM | 409 clean voice reference 错误；Context/H3 POST 为 0，不得回退整轨混音。只有用户显式 `none` 才能丢弃真实口播 |
| 输入准备未 `done` | 409 `artifacts not ready` |
| 请求缺字段、有未知字段、`confirm` 非 true、id 或枚举不合法 | 422；在状态写入和供应商调用前拒绝 |
| plan receipt 缺失、格式非法或已变化 | 422/409；刷新后由用户再次确认，Web 不自动重发 |
| 图片确认后缺少 `video-prompt-fusion` 最终提示词，或四类输入 SHA/顺序漂移 | 409；重新融合提示词，Context/H3 POST 为 0；禁止旧视觉 prompt 直达 Context/H3 |
| 画内声音缺少主分析权威 | 409 `on_screen_authority_unavailable`；不调用 Skill，不创建 H3 attempt |
| Context IR 试图改变帧序、源硬切时点、台词、声音呈现、music policy 或 voice reference | fail closed；不创建 H3 attempt |
| generation active/succeeded，或参数与冻结输入不一致 | 409；不创建新供应商任务 |
| generation 为 `resume_required` | 只接受原 id 和全部原冻结参数，继续原 attempt |
| generation 为 `submission_unknown` | 任意提交均 409；只允许查询供应商已有任务 |
| 全部门控通过 | 202 queued；统一 segment coordinator 异步执行 Context、H3 和拼接 |

## 图片与 Skill 边界

- 全链仅解析并调用 `video-maker`、`image-postprocess` 与 `video-prompt-fusion` 三个 Skill；出现第四个 Skill 名、音频 Skill、Binding Skill 或 speaker-visibility phase，测试和启动门必须失败。
- `video-maker` 第一次调用冻结 segments、每段 9 张原始关键帧、原始视频提示词、动作/镜头/时间和它已经能证明的结构化事实。
- `image-postprocess` 已冻结，不再迭代；它只把每段 9 张原始关键帧变成 9 张优化关键帧，不做素材准入。
- 用户确认后的 9 张优化图按 segment 和帧序进入统一 frozen receipt；随后 `video-prompt-fusion` 以有序新关键帧、旧视频提示词、图片优化提示词和音频内容为唯一四类输入生成最终视频提示词。
- `new_keyframes` 仍属于第一类输入，但每项必须同时绑定源时间、source scene 与 transition；扩充字段不是第五类输入，也不是新阶段。
- 音频内容必须冻结既有 ASR/YAMNet 的真实口播筛选结果、BGM 判定与 `music_policy=forbid`；`work/voice.mp3` 默认只用于分析，只有同一收据证明 `spoken && has_bgm=false` 时才可成为 reference。
- 第一次生成确认只冻结四类输入并项目级调用一次 `video-prompt-fusion`；完成后返回可刷新状态，Web 不自动重提。相同设置由用户再次明确确认后，才允许进入 Context IR 和 H3。
- Context IR 与 H3 只能消费 `video-prompt-fusion` 输出及其绑定的四类输入；禁止旧视觉 prompt 直接覆盖新人物、新场景或新对象。

## 付费与恢复边界

- 每个供应商 POST 前必须先落 attempt receipt；输入 SHA、顺序、参数或上游 receipt 漂移时拒绝。
- `submission_unknown` 永远 GET-only。
- 已知供应商明确失败的自动 attempt 必须沿用同一冻结输入并受统一次数预算约束；成功 segment 不重提。
- 本地拼接失败只重做拼接，不重新提交 H3。
- H3 输出下载、ffprobe、PTS、时长、原生音轨和最终拼接验证通过后才原子发布 `generated.mp4`。
- 有合法口播的段以 H3 原生输出音轨为成片音频；源混音和 conditioning voice 不得 overlay 或回挂。无真实口播的段丢弃 H3 音轨并输出静音。
