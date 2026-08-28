---
name: submit-gate
type: behavior
status: done
owner: human
updated: 2026-08-28
tdd: N/A
links: [conversation-task, processing-state, long-video, postprocess]
---

# H3 统一提交合同

当前 v4 只有一个 A→B operation。请求在技术验收 A 前校验 shape、receipt 和付费参数；一旦 A 被接受，同一 CID 自动推进到 receipt 绑定的成片 B，不再要求刷新、二次确认或第二个 submit。

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

当前 v4 接受后公开结果固定为：

```text
202 {operation_id:<cid>,status:"running",stage:<current-stage>}
  -> 同一 durable operation 自动继续
200 {operation_id:<cid>,status:"succeeded",stage:"commit_b"}
```

重放或兼容 image-acceptance 调用只读取/确保同一个 operation，不接受另一组参数创建竞争生成。内部 `prompt_fusion_refresh_required`、图片 acceptance CAS 或质量诊断不构成 current 的公开 refresh/409 步骤。

## 唯一编译规则

| 冻结输入 | 后端结果 |
| --- | --- |
| 每段 source timeline | exact 9 个有序 Picture 槽位；scene 改变机械编译为 hard cut，scene 不变为 continuous |
| Fusion v2 `visual[]` | 只作为每个冻结 hard-cut 区间的视觉 prose；无 provider 标签解释权 |
| `dialogue_mode=none` | 零台词、零 source audio reference |
| `dialogue_mode=auto` | 只投影冻结的 `spoken` 文本和时间；逐行 `voice_ref=null`，项目 `voice_references=[]` |
| `dialogue_delivery` | 当前 Ref2VA compiler 固定投影为 off-screen voiceover；不新增 speaker/binding Skill |
| `music_policy=forbid` | 后端机械编译 `non_diegetic_music:` / `N/A` |
| Fusion visual + timeline + dialogue + music | 后端确定性编译唯一 Ref2VA prompt |
| 后端 Ref2VA prompt | Context local identity 原样绑定，同字节 effective prompt，HTTP 0 |
| H3 输出 | 成片使用 H3 native audio；缺音轨的 segment 在同一 EDL 补静音 |

## A 前技术校验

- schema、字段闭集、`confirm`、client id、枚举和 receipt 必须合法。
- 每个 segment 必须具备 exact 9 张确认图片及其有序 SHA、source scene/time/transition；极短 scene 的受 receipt 证明重复帧合法。
- Fusion 四类输入、Skill SHA、input/output SHA 和 segment 顺序必须一致；Fusion v2 输出只能是 `{index,visual[]}`。
- Ref2VA compiler 必须从冻结机械字段生成 Picture 1…9、Shot/cut 时间、台词和 music policy；Skill 文本不能覆盖这些字段。
- H3 request 必须为零 source audio reference；源音频路径和 bytes 不能出现在 Fusion、Context、H3 或 stitch 输入中。
- 付费 attempt 必须先落 exact input receipt；`submission_unknown` 仍是 GET-only。

这些是技术完整性校验。人物、背景、旧静态残留、动作表现、画面质量、semantic score 和 diagnostics 只进入测试与 Skill 迭代，不得阻断 A→B、不触发 retry、不生成备用 prompt，也不得选择另一 workflow。

## A 后自动延续

- A 与 B 共用同一个 `operation_id=cid`、冻结请求 id 和输入 owner。
- Fusion 是未付费的内部阶段；输出暂不可用时保留同一 accepted claim 并继续调度，不公开失败终态或要求重提。
- Fusion done 后后端自动 finalize plan、创建唯一 generation，并进入 Context local identity、H3 和 EDL。
- 进程重启会认领同一 stale continuation；已进入 provider 边界的 attempt 仍按 receipt 安全恢复。
- 已知 provider task 的恢复和确定 provider failure 的预算内 retry 沿用既有 attempt 规则；不会派生备用 current path。

## Current / history 边界

- current create contract 只有 v4 + Fusion v2 + backend Ref2VA + Context local identity。
- Fusion v1、旧 Context HTTP、多模态 source-audio reference、旧 short/long、speaker visibility 与 quality-verdict receipt 均只读。
- 历史已知 task 只按原 receipt GET；历史成片可查看，但不得迁移为 current、覆盖输出或作为 fallback 新 POST。
