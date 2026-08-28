---
name: conversation-task
type: feature
status: done
owner: human
updated: 2026-08-28
tdd: N/A
links: []
---

# 会话式 H3 视频复刻

## 唯一权威链路

本节是当前产品链路的唯一权威说明。修改生产代码、API、Web、测试或文档前，必须逐项对照本节；任何实现若增加阶段、增加 Skill、复制单段/多段逻辑，或改变已冻结图片 Skill，均视为偏离需求，不得合入或部署。

```text
原视频
  -> video-maker Skill
  -> segments[N>=1]，每段 9 张原始关键帧 + 视频提示词
  -> 既有 ASR + YAMNet 音频语义分析，区分真实口播、歌唱和背景音乐（非 Skill）
  -> 可选 MediaKit 去字幕和/或去 Logo；保持帧数、顺序和 segment 归属不变
  -> image-postprocess Skill
  -> 每段 9 张优化关键帧
  -> Web 用户确认图片并设置台词、声音呈现、画幅、清晰度和适配方式
  -> 后端只保留真实口播并拒绝把源混音冒充人声参考
  -> video-prompt-fusion Skill 融合优化图、旧视频提示词、图片优化提示词和音频内容
  -> Context IR 只优化最终提示词
  -> H3 按 segment 生成原生音视频
  -> 按统一时间轴拼接，N=1 仍走同一拼接实现
  -> Web 同时展示原视频与最终新视频
```

硬约束：

- 原有去字幕、去 Logo 能力必须保留。它们是 `image-postprocess` 前的可选关键帧预处理，不是新 Skill；未选择时原始关键帧直接进入图片优化。选择两项时固定按去字幕后去 Logo 执行，处理结果仍须每段恰好 9 张并保持原顺序。

- 全链只允许调用三个 Skill：`video-maker` 负责关键词/片段分析、每段 9 张关键帧和旧视频提示词；`image-postprocess` 只负责关键帧图片优化；`video-prompt-fusion` 只负责最终提示词融合。不得派生 Audio、Binding、Speaker 或其他 Skill/phase。
- 收口后的 `video-maker` Skill 权威 SHA-256 为 `0bbb22baeb8f14fef737b279e2ab2e8f70bf8965d41b182f1987537e1e3e4785`；不得恢复已删除的音画绑定、说话人可见性或图片后融合 phase。
- `video-prompt-fusion` Skill 权威 SHA-256 为 `e366785045805547e984926d6d6ce4ff6ce6589a9fb0b0ed6b2816da14eb8249`；只允许一次项目级调用并只消费四类冻结输入。音频块必须采用标签与 `lines_json` 字节紧邻的单行包络，不得扩展为新的分析、评测或供应商阶段。
- 已验收的 `image-postprocess` 生成效果、目标视觉和 A→B 提示词已经冻结，不再迭代。职责清理已经完成：只删除素材拒绝、流程控制、验收门和发布逻辑，完整保留原验收版 A→B 生成合同。最终权威 Skill SHA-256 为 `4839f0f0673a9e05cc44f2938c1068a63a142c8c31a9f6b2b2f138842b68cd03`。
- 除上述三个 Skill 外，不存在 Binding Skill、Audio Skill、Speaker Skill 或第四个 Skill。音频呈现由普通后端代码根据冻结台词、Web 的画内/画外选择和现有 voice reference 做确定性投影，再作为只读输入交给融合 Skill。
- 单段和多段不是两条链。所有当前项目统一为 `segments[N>=1]`；所谓短视频只是 `N=1`，使用同一冻结、Context、H3、attempt、恢复、拼接和验收实现。
- 图片 Skill 收到合法可解码关键帧就执行，不做素材资格审查，不因人物、场景、遮挡或内容复杂而拒绝开始。
- `video-prompt-fusion` 的四类输入固定为：有序新关键帧、旧视频提示词、图片优化提示词、音频相关内容。新人物、新场景、新对象、服装、材质、空间结构、锚点景别和裁切以新关键帧为准；旧视频提示词只负责同一硬切区间内的动作顺序、因果、镜头运动类型和相对节奏；源时间、source scene 与硬切以既有分析产物为准；图片优化提示词只解释替换目标和保持约束；音频文本、时间、画内/画外和 voice reference 逐字保持。
- 音频分类只复用已经存在并经过验证的 ASR + YAMNet 节点，不引入替代分类器、声源分离器或新 Skill。当前冻结枚举仍只有 `spoken/sung/null`；YAMNet 检出的 singing、chant、rap、humming 等歌唱类统一归入既有 `sung`，不新增分类值。自动台词只接受 `spoken`；仅有 `sung` 时必须冻结 `lines=[]`、`voice_references=[]`，不得把歌词改名为画外台词。
- 从源视频抽取的 `work/voice.mp3` 是完整混音，只能用于既有 ASR/YAMNet 分析。只有 YAMNet 冻结结果确认存在真实口播且不存在 BGM 的当前音轨才可继续作为 voice reference。`spoken/edit/custom` 需要声音但检测到或无法排除 BGM 时必须在付费前明确拒绝；只有用户显式选择 `dialogue_mode=none` 才能丢弃真实口播，不得由 `auto` 静默降级。
- 默认不生成背景音乐是贯穿 Fusion、Context IR 和最终发布的合同，不是一个可丢失的旧提示词前缀。无真实口播的 segment 使用现有无音频 H3 路径并在拼接时输出静音；不得仅依靠 `no music` 自然语言承诺消除模型生成的音乐。
- 每张冻结关键帧必须同时绑定源时间、源 scene 和与前一帧的转场类型。源视频的硬切是时间轴权威：不得被后端误标为 `same_camera`，不得被 Fusion 或 Context IR 改到其他时刻，也不得让单个 H3 请求跨该硬切连续变形。
- 首轮先修复现有 transition、anchor、Fusion 与 Context 合同，在同一请求中明确冻结硬切而非连续 morph；若真实 A/B 仍出现跨切点变形，再复用现有 `segments[N>=1]` 与 stitch 做物理隔离。物理拆分上线前必须证明短逻辑段仍有 9 个唯一源帧、供应商合法时长不会丢动作、台词不跨边界，且逻辑/供应商时长由版本化 receipt 分别绑定；证明不足时付费前拒绝。
- 用户第一次确认生成时，后端冻结包括声音呈现在内的四类输入，并以一次项目级 `video-prompt-fusion` 调用覆盖全部 `segments[N>=1]`；完成后 Web 只刷新展示，不自动再次提交。用户使用相同设置再次明确确认后，才进入 Context IR 和 H3。
- Context IR 只优化 `video-prompt-fusion` 输出的文字表达，不得恢复旧视觉元素，也不得改变关键帧顺序、台词、时间、画内/画外选择或 voice reference；IR 完成后直接进入 H3，中间不得增加新阶段。
- 有真实口播且存在合法 clean voice reference 时，H3 最终音频以供应商原生输出为准；源混音或 conditioning audio 不得在拼接时覆盖、回挂或混入成片。无真实口播时丢弃 H3 音轨并输出静音，确保模型新生背景音乐不会进入成片。
- `submission_unknown` 只允许查询已有任务，不得自动或人工再次 POST。
- 完整可用链最终只部署到 Web 端口 `3211`，以同端口可回滚替换旧版本；不得新建其他对外预览端口代替交付。

### 变更前对照门

每个变更必须在评审或测试中明确回答并证明：

1. 是否仍只调用上述三个 Skill，且职责与输入输出没有扩张？
2. 是否同时适用于 `N=1` 与 `N>1`，且没有复制两套业务逻辑？
3. 是否保持每段原始/优化关键帧各 9 张及其冻结顺序？
4. 是否保持图片 Skill 字节不变？
5. 图片确认后是否只由 `video-prompt-fusion` 使用四类冻结输入生成最终提示词，并禁止旧视觉 prompt 直接进入 H3？
6. 是否由 Web 明确控制图片确认、台词、画内/画外、画幅、清晰度和适配方式？
7. 是否让 Context IR 后的冻结输入直接进入 H3？
8. 是否对有真实口播的 segment 保留 H3 原生音频、对无台词 segment 强制静音，并始终阻止源音回挂？
9. 是否没有新增产品阶段、状态机、Skill 或用户概念？
10. 是否只把真实口播作为 dialogue，并在付费前阻止完整源混音成为 voice reference？
11. 是否冻结并逐值保持每张图的源时间、scene 和硬切，且 H3 请求不跨源硬切？

任一回答为否，变更即停止，不以补丁、兼容分支或额外收据绕过。

## 要什么

运营/创作者上传一支最长 300 秒的参考视频，系统用唯一分段链准备每段 9 张关键帧、旧视频提示词与自动台词；用户确认优化图并显式选择台词、声音呈现和画面参数后，`video-prompt-fusion` 融合四类冻结输入，Context IR 优化后直接交给 H3 生成各段原生音视频，最终由同一拼接流程输出新视频。

## 为什么

把看片、选帧、台词来源、提示词组合和付费生成固化成可审计流程；尤其要保证画面 OCR 不会变成角色台词、重复点击和服务重启不会静默重复扣费。

## 验收

- [x] 新会话使用 schema v2；无音轨视频合法，自动台词为空也能继续
- [ ] 自动台词只来自 ASR 且声学分类必须为 `spoken`；`sung/chant/rap/humming`、OCR、字幕、画面文字和备注绝不成为台词
- [x] 每段 `video-prompt-fusion` 生成的最终提示词对用户可见，并与四类输入及对应 9 张优化图一起由统一 plan receipt 冻结
- [x] 生成前可选 `auto/edit/custom/none`、画内/画外声音呈现、`16:9/9:16` 画幅和 `480p/768p` 清晰度；服务端先按真实 H3 输入与源视频推荐，目标不匹配时默认裁切但允许改为留边
- [x] 生成完成后成片继续可播放，并只用服务端冻结值展示画幅、清晰度、台词、声音呈现、适配、实际总时长和 segment 数
- [x] 上传后以首个视频流的视觉时长校验；音频/容器尾巴不影响 300 秒门禁，超过 300 秒时拒绝，其余统一规划为 `segments[N>=1]`，每段 provider 整秒时长不超过 14 秒
- [ ] 每个片段固定使用自身 9 张有序关键帧；关键帧源时间与 scene 逐项冻结，源硬切两侧不得进入同一 H3 请求；`N=1` 使用完全相同的实现
- [ ] Fusion 与 Context IR 逐值保持源硬切时点、图片顺序和镜头角色；任一漂移在 H3 POST 前拒绝
- [ ] 完整源混音不得成为 voice reference；仅有歌曲歌词的 segment 冻结为空台词、零音频参考，并发布静音成片
- [x] 用户提交前看到本次准确新增的付费子任务数；确定失败时复用成功段，只重做失败段及同链下游，拼接失败则零付费本地重拼
- [x] VFR/CFR 都按视频时间戳批量抽帧；台词、声音呈现与 voice reference 由单一确定性编译器冻结，画内才要求说话人时序，画外不要求嘴型，`none` 不发送台词或声音参考
- [x] 提交冻结版本化 receipt，随后异步执行 H3，并持续显示 `queued/running/resume_required/succeeded/failed/submission_unknown`
- [x] 已知 task 故障只允许原参数继续同一 attempt；确定失败才用新请求 id 重试；`submission_unknown` 锁死提交并要求先核对供应商
- [x] 旧会话可查看但不能提交、重试或后处理
- [x] 可选 MediaKit 文字/字幕、Logo/图标擦除与 Seedream 图片优化；按段有序执行、失败段可定向重试，全部完成后结果进入 H3 冻结输入
- [x] `face_hold` 与 Seedance 生产提交/回退路径已删除
- [x] H3 成片下载只接受预解析地址与实际连接 peer 均为公网的 HTTPS、拒绝重定向、限制 200 MiB，并经 ffprobe 视频门禁后原子落盘

## 边界

- H3 是模型名，不代表服务启用了 HTTP/3。
- 新计划中多图参考工作流的单次请求最多 14 秒，总输入最多 300 秒；历史 plan v1 的 11–15 秒 attempt 只允许 GET 恢复，不允许新 POST。
- 同镜头内的普通动作不臆造秒数；源硬切必须写入冻结的绝对 segment 内时点，Fusion 与 Context IR 不得改写。最终拼接严格服从源逻辑片段帧预算。
- 跨段连续性由冻结的关键帧顺序、统一提示词和拼接时间轴共同约束；验收不通过时保留产物供用户查看，但不得伪装为通过。
- 分段链路不是供应商原生 extend；服务端为每段建立独立、可审计的多图参考 task，再本地拼接。
- 单共享口令，无用户级权限；状态用 2 秒轮询，不用 SSE/WebSocket。
- 供应商明确终态 `FAILED/ERROR/FAIL` 且 task id、receipt、诊断与同一 input receipt 完整落盘时，沿用原 client id 自动补交，默认首次加 2 次、总计最多 3 次 POST；其他确定失败仍需人工新 id。已知 task 的查询/超时/下载故障等待人工继续但不创建 attempt；`submission_unknown` 连人工继续也被拒绝，须先核对是否已创建任务。

## 取舍

- receipt 和 attempt 状态都落文件系统，以输入哈希、先落状态再 POST 和会话文件锁换取可恢复性；部署保持单 uvicorn 进程。
- 图片优化结果先供用户查看并确认；确认后 H3 必须绑定对应 9 张优化图及其画幅派生帧，禁止回退原图。

> 表现规格见本目录 `behaviors/`；纯文档变更不适用 TDD。
