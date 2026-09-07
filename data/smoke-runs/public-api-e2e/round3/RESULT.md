# 对外 v1 API 正式验收轮 round3（2026-09-08）

目标：https://8.166.140.227:3211（release HEAD 55af160，含音色兜底 55319ea + 节点重放套件）。
四单参数与 round2 完全一致（原始长度，幂等键 `pub-e2e-r3-*` / `pub-generate-necklace-v3-<ts>`），提交于 04:55（UTC+8）。

## 判定：验收轮未达标

判定标准为四单一次性全成。实际 **3/4 一次性成功，mouthwash 首跑失败且断点续跑也确定性失败**，验收轮未达标。

| 单 | job id | 终态 | 耗时 | 一次性成功 | 结算 |
|---|---|---|---|---|---|
| dog（换木质气垫梳，中文改编） | vg_b7d990abe2f449e9889bacf1aa915822 | succeeded | 19.7 min | 是 | 1000 |
| necklace（换项链，保持原台词） | vg_b918c0089e164995ac34461efe0a29e6 | succeeded | 44.9 min | 是 | 1000 |
| mouthwash（换 Listerine，英文改编） | vg_019b9d61fc694d368eed3661aeb5bb60 | **failed** | 49.6 min（含续跑 4 min） | **否** | 0 |
| generate-necklace（generate + brief，英文） | vg_c391e60450174c43905fbca8e1695715 | succeeded | 45.2 min | 是 | 1000 |

三个成功单成片已下载并 sha256 校验通过（`dog.mp4`/`necklace.mp4`/`generate-necklace.mp4`）。

## mouthwash 失败与续跑详情

- 首跑失败点：5 个段的 `mask-reference-faces` 全部成功之后、Picsart submit 之前，pipeline 级报 `prepared H3 prompt is not canonical`（pipeline.py:1809：冻结校验要求 final_prompt 无首尾空白）。
- 根因定位：`work/segments/0001/h3_prompt.txt` 尾部多一个空格（结尾 `...0s] You `␣），其余 4 段均 canonical。英文台词行尾空格被带进冻结的 H3 prompt，属确定性数据问题。
- 续跑：05:40 走 `resume-pre-picsart`（无 submit 痕迹，门禁接受），前置节点重跑（attempt-2）后 05:44 在同一校验再次失败（`pre_picsart_resume_failed`，同一错误）。续跑复用冻结的 h3_prompt.txt，不会重新生成 prompt，因此无法自愈。**需要代码侧在冻结前 strip 或在校验失败时重生成该段 prompt。**

## 任务 2：round2 狗单（dfe6479538184759ba6cc953a0bda138）续跑验证

- 04:56 走 `resume-pre-picsart`（新代码已部署），约 0.4 秒后再败，错误仍是 `dialogue TTS requires a voice reference`（`pre_picsart_resume_failed` ×2）。
- 根因：音色兜底（55319ea）在 `object-replace` 节点的 `_materialize_voice_references` 内生效；续跑复用旧代码冻结的 object-replace 产物、跳过该节点重放，兜底不会补建，voice-references 仍为空。**音色兜底只对 fresh run 生效，续跑路径不生效。**
- 对照：round3 新狗单 fresh run 一次性成功（本轮模型自行分配了音色，`voice-references/audio-01.wav` 存在，未触发兜底事件）——上轮"无人脸视频必败"在本轮未复现，但该确定性缺口在续跑路径依然存在。

## 新链路事件统计（trace.jsonl + 节点 output.json）

| 事件 | dog | necklace | mouthwash | generate |
|---|---|---|---|---|
| json_repairs（修正循环） | 0 | 0 | 0 | 0 |
| face_mask_degraded | 0 | 0 | 0 | 0 |
| voice_reference_default_fallback | 0 | 0 | 0 | 0 |
| picsart_terminal_failure_resubmit | 0 | 0 | 0 | 0 |
| auto_resumed / seedream_receipt_reset | 0 | 0 | 0 | 0 |
| pre_picsart_resume_started/failed | – | – | 1/1 | – |
| dialogue-tts-line 重试（永不失败吸收 TTS 抖动） | 3 | 15 | – | 3 |

另：necklace 与 generate 在 API 层各出现一次瞬时 `submission_unknown`，约 30 秒内自行恢复 running 并成功；内部回执全部 accepted、无重提，对账链路与上轮一致工作正常。本轮四单模型节点全部 attempt-1 通过，无 JSON 修正需求。

## 产物（data/smoke-runs/public-api-e2e/round3/）

- `dog.mp4` / `necklace.mp4` / `generate-necklace.mp4`（sha256 均校验通过）
- `frames/`：4 张 `*-reference-sheet.png`（含失败的 mouthwash）；成功单每段首中尾各 1 帧（dog 2 段、necklace 4 段、generate-necklace 4 段，共 30 帧）
- `jobs.json`、`generate-necklace.json`、`generate-necklace-submit.json`、`submit-time.txt`、`public_api_e2e_round3.py`

## generate-necklace 台词原文（translated_voice_lines）

```text
[0.0-10.0s]   You know how annoying it is when a necklace tarnishes after just two wears? It makes the whole outfit look cheap.
[10.0-18.6s]  But this is titanium steel, so it won't fade. You can shower and swim in it. And the minimalist design matches everything.
[18.6-28.4s]  Plus, it brings out your collarbone and looks amazing whether you're at the office or on a date. It's the perfect touch.
[28.4-36.8s]  And the best part? It's only a fraction of what big brands charge. Stock is limited, so tap the link below!
```

## 结论与建议

1. **验收轮未达标**：mouthwash 首跑失败 + 续跑确定性失败（H3 prompt 尾部空格未 canonicalize）。
2. 续跑路径两个已知缺口均在本轮实证：冻结的 object-replace 不重放 → 音色兜底不补建（round2 狗单）；冻结的 h3_prompt 不重生成 → 非 canonical prompt 无法自愈（round3 mouthwash）。建议：续跑门禁检测到这两类失败时允许重放对应节点，或在冻结时做 canonicalize/strip。
