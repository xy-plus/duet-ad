---
name: submit-gate
type: behavior
status: done
owner: human
updated: 2026-08-25
tdd: N/A
links: [conversation-task, processing-state]
---

# H3 提交门控

## 规则

短视频的 `POST /api/conversations/{id}/submit` 只接受：

```json
{
  "confirm": true,
  "client_request_id": "request-123456",
  "dialogue_mode": "auto",
  "fit_mode": "none",
  "aspect_ratio": "9:16",
  "resolution": "768p"
}
```

`edit/custom` 还必须带非空 `lines`；`auto/none` 禁止带 `lines`。请求存在未知字段、无严格 `confirm:true`、id 不合规、台词时间越界或画幅选择不匹配时均拒绝。

长视频严格接受：

```json
{
  "confirm": true,
  "client_request_id": "request-123456",
  "dialogue_mode": "auto",
  "fit_mode": "none",
  "aspect_ratio": "9:16",
  "resolution": "768p",
  "expected_plan_receipt": "64 lowercase hex characters",
  "fast_mode": true
}
```

长链只允许 `dialogue_mode=auto|none`，不允许 `lines/edit/custom`；`expected_plan_receipt` 必须与当前详情一致。当前页面新建长视频固定显式发送 `fast_mode=true`，确认页不显示快速模式开关或说明，结果参数摘要也不展示该模式；后端继续兼容显式 `false`，历史调用字段缺失等价于 `false`。

如果旧标签页缺少 `expected_plan_receipt`，服务会提示“页面版本已更新，请刷新页面后重试”。此请求不会自动补 receipt、不会提交 H3，也不会产生付费任务；刷新页面后再确认即可。页面入口和 `app.js` 禁止缓存，确保刷新取得当前契约。

| 条件 | 结果 |
| --- | --- |
| `ENABLE_H3_SUBMIT` 未开 | 501 `H3 submission is disabled.` |
| 会话不存在 | 404 `not found` |
| schema 不是 v2 | 409 `read_only` |
| 精确旧版四键长视频提交 | 409 `client_refresh_required` + 中文刷新提示；不创建付费任务 |
| 输入准备未 `done` | 409 `artifacts not ready` |
| AutoDL 凭据缺失 | 503 `h3_credentials_missing` |
| `aspect_ratio` / `resolution` 缺失或不在闭集 | 422 `invalid_aspect_ratio` / `invalid_resolution`；在 claim 和供应商 POST 前拒绝 |
| 冻结输入、receipt 或画幅派生失败 | 409 `prepared_input_invalid` / `frame_fit_failed` |
| 历史未冻结长会话的 anchors 缺失、损坏或路径越界 | 409 `fit_requirement_unknown`；不静默按 `none`，不创建付费任务 |
| 长视频 plan receipt 缺失/格式非法或已变化 | 422 `invalid_plan_receipt` / 409 `long_video_plan_changed`；付费前拒绝 |
| 同一 generation active/succeeded，或 H3 阶段确定失败后复用旧 id | 409；不创建新供应商任务。长链 `stage=stitch` 失败必须复用旧 id，只本地重拼 |
| generation 为 `resume_required` | 只接受原 id、原台词、原画幅、原清晰度和原 fit；合法时返回 202 + 原 attempt，新 id/参数漂移分别 409 |
| generation 为 `submission_unknown` | 任意 id 均 409 `submission_outcome_unknown`；先核对供应商 |
| 全部门控通过 | 202 `{"status":"queued","attempt":N}`；后台异步执行 |

## 边界

- 首次 start 和 H3 阶段确定失败后的 retry 会在锁内冻结画幅、清晰度、台词、适配方式和长链快速模式；长链所有 segment 共用同一选择。长链另冻结当前 plan receipt，新 id retry、原 id 拼接重试与 `resume_required` 均复用服务端冻结的 `fast_mode`，历史 generation 缺失时按 `false`；不会被页面草稿或缓存改变。长链拼接失败复用原 id 且不创建供应商任务。`resume_required` 只加载既有 receipt，不重写它、不递增 attempt。
- 提交门控后的唯一自动新 POST 是完整确认的供应商终态 `h3_provider_failed`：沿用原 `client_request_id` 和同一 input receipt，新建顺序 attempt，累计不超过 `1 + AUTO_RETRY_COUNT`。已落盘的 `ready_to_submit/h3.ready` 自动 attempt 已占额度；`submission_unknown`、提交拒绝、结果缺失和输入/安全错误均不进入该例外。
- “生成最终视频”点击后必须立即进入提交态或显示错误；前端异常不得表现为无响应。
- 自动 H3 源提示词只允许在 H3 attempt 创建前通过 CAS 保存；attempt 创建后锁定，防止页面内容与实际生成输入不一致。
- 未做素材优化时，短链和长链每段使用原关键帧；优化完成时必须使用对应 `postprocessed/`，需要 crop/pad 时再从所选帧产生画幅派生图。所有最终 bytes 在付费 POST 前冻结并进入 receipt，禁止缺帧时静默回退。
- 新长链每个不超过 14 秒的 segment 固定使用多图参考 workflow，传入该段 1–9 张冻结关键帧。已有首尾帧付费 attempt 仅按原 receipt GET-only 恢复，不切换接口。
- 已有 generation + frozen plan 的历史长会话沿用冻结 `fit_mode`；即使旧 meta 的 `fit_required` 为 null，也不重写 active/failed/resume 的 receipt、输入或重试参数。
- 成片下载先验证全部 DNS 解析地址，再在读取 status/body 前验证实际 socket peer 为公网；拒绝 userinfo 和重定向，限制 200 MiB，并在原子落盘前通过 ffprobe 正时长视频流验证。
- 旧 Seedance 提交实现和 `face_hold` 参数/提示词注入已删除，不是失败回退选项。

## 图片优化与 H3 门控

- schema v2 的关键帧冻结后，同一 `image-postprocess` Skill 先执行 `phase=plan`，结构化冻结全部叙事主人物和全部场景组件的双目标替换。后端确定性编译提示词；任一主人物不可安全替换，或任一场景不能同时完成语义、形状、纵深、布局和局部颜色变化，均在付费前判 `eligible=false`。
- v2 编译提示词不能自由 PATCH，避免删除人物、场景、光色或关系硬约束；旧 v1 receipt 只读兼容且不迁移。图片编辑后必须以同一 frozen plan 执行 `phase=verify`，人物身份、源身份残留、真实换景、局部颜色、全局光色、关系和跨帧/跨段连续性任一 `fail/unknown` 均阻止 H3 提交。
- 后处理选项 canonical 为 `remove_subtitle/remove_brand/optimize_image` 三个布尔值；精确旧两字段请求兼容为 `optimize_image:false`，其余缺字段、未知字段或非布尔值拒绝。实际执行对每段保持 `文字擦除 -> 品牌擦除 -> 图片优化` 阶段屏障，段之间并行；任一失败段不会阻止其他段完成，整体保持 failed，H3 不得静默回退原图。
- `postprocess.segments[]` 仅公开 `index/status/stage/completed_frames/total_frames/revision/error`。失败段通过 `POST /api/conversations/{id}/postprocess/segments/{index}/retry` 重试，请求严格为 `{"confirm":true,"expected_revision":N}`；只复用该段已成功的阶段/帧与服务端冻结选项、模型、模式、提示词，不接受页面重新指定。
- Seedream 付费 POST 前先持久化 attempt 输入摘要；只有明确的 HTTP 429 `QuotaExceeded` 且响应无 `data` 才按统一预算重试。网络或读写超时记为 `submission_unknown`，自动恢复不得再次 POST；人工分段重试也必须保留旧 attempt 记录。
- Seedream 默认模型为 `doubao-seedream-5-0-pro-260628`。Pro 请求不发送 `sequential_image_generation`；Lite、4.5、4.0 请求固定发送 `sequential_image_generation:"disabled"`。失败项目不改冻结模型，仍通过上述分段重试产生新 revision，旧 attempt 回执保留。
- 图片优化默认采用逐帧独立并行模式；显式配置仍可选择锚帧一致性模式。模式在项目开始时私有冻结，历史项目与失败重试不会随服务缺省变化。
