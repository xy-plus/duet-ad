---
name: processing-state
type: behavior
status: done
owner: human
updated: 2026-08-28
tdd: N/A
links: [conversation-task, submit-gate]
---

# Current v4 operation 状态

current v4 对外只有一个从 accepted A 到 commit B 的 operation，不把内部阶段错误投影成要求用户刷新、消费 409 或另起请求的产品状态。

| 内部阶段 | current operation 投影 | 行为 |
| --- | --- | --- |
| analysis / postprocess | `202 running` | 生成每段 exact-9 图片；技术验收 A 后自动继续 |
| prompt_fusion | `202 running` | 同一 accepted claim 调用一次 Fusion v2；输出 shape 暂不可用时保留 durable work item，不创建备用 prompt |
| Ref2VA compile | `202 running` | 后端从 visual prose + frozen mechanics 编译唯一 provider prompt |
| Context | `202 running` | local identity 同字节 receipt；HTTP 0 |
| H3 / stitch | `202 running` | exact-9 Picture、零 source audio reference；按 receipt 推进 task 与 EDL |
| 最终媒体验收通过 | `200 succeeded / commit_b` | 原子发布 `generated.mp4` |

相同 CID 的重放只确保既有 operation 继续；即使请求 id 或兼容 image-acceptance payload 不同，也不能创建竞争 operation。进程重启认领同一 input owner、Fusion continuation 或已冻结 generation。

## 内部付费安全状态

H3 attempt 仍持久化 `queued/running/resume_required/succeeded/failed/submission_unknown` 供恢复和审计，但 current operation 不把这些状态变成另一条产品 workflow：

- 已知 task 的查询、下载或探测故障继续同一 attempt；
- receipt 完整的 provider terminal failure 可在统一预算内创建下一顺序 attempt；
- `submission_unknown` 只允许 GET，不猜测、不二次 POST；
- stitch 失败只重做本地 EDL；
- quality score/diagnostics 不改变任何 attempt 状态，不触发 retry 或 fallback。

## Current / history 边界

- current Context 始终是 Ref2VA 同字节 local identity。
- 旧 Context HTTP、multimodal audio、short/long、speaker visibility 与 quality verdict receipt 只读；已有 task 只按原 receipt GET 恢复。
- 历史成片可查看；历史 receipt 不得迁移、改写为 current 或用新 id 创建 fallback POST。

## 例子

- Fusion 内部暂未产出合法 shape：operation 仍为同一 `202 running / prompt_fusion`，后台继续同一 accepted claim；客户端不刷新重提。
- 某个已知 H3 task 查询超时：operation 保持 `running`，后台只对该 task GET；不会发送第二个 provider POST。
- 某段 H3 无音轨：operation 继续到 stitch，在同一 EDL 插入该段静音，最终仍只产生一个 commit B。
