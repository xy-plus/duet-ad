# 对外 v1 API 质量回归 round2（2026-09-08）

目标：https://8.166.140.227:3211（release 新部署：灰块打码、DeepSeek 同会话 JSON 修正循环、解析容错、永不失败链路）。
四单参数与 round1 完全一致（原始长度视频，新 job，幂等键 `pub-e2e-r2-*` / `pub-generate-necklace-v3-<ts>`），提交于 03:02（UTC+8）。

## 终态与耗时

| 单 | job id | 终态 | 耗时 | 结算 |
|---|---|---|---|---|
| dog（换木质气垫梳，中文 auto_rewrite） | vg_dfe6479538184759ba6cc953a0bda138 | **failed** | 27.1 min（含一次断点续跑） | 0（失败退返） |
| necklace（换项链，保持原台词） | vg_410a240354ba475cb3d60c35978b064c | **succeeded** | 48.6 min | 1000 |
| mouthwash（换 Listerine，English auto_rewrite） | vg_be663f3ce6534deeb75b7d916f3bb168 | **succeeded** | 52.8 min | 1000 |
| generate-necklace（dialogue_mode=generate，English） | vg_7f85e3040aea44aab204d9e3132c96fb | **succeeded** | 50.8 min | 1000 |

三个成功单成片均已下载并 sha256 校验通过：`necklace.mp4`、`mouthwash.mp4`、`generate-necklace.mp4`。

对比 round1：上轮 necklace/mouthwash/generate 三单全部死于模型输出 JSON 解析/形状校验（attempt-1 即终）；本轮同类错误被同会话 JSON 修正循环兜住，三单全成。

## 新链路事件（logs/trace.jsonl + 节点记录）

- **json_repairs（同会话修正循环）**：触发 2 次，均修复成功——
  - necklace `object-replace` attempt-1：`JSONDecodeError: Expecting ',' delimiter: line 1 column 6013`（与 round1 致命错误同类）→ 修正后通过；
  - generate-necklace `video-maker-project-index` attempt-1：`Expecting value: line 1 column 1` → 修正后通过。
  （该循环记录在节点 output.json 的 `json_repairs` 字段，不产生 trace 事件。）
- **picsart_terminal_failure_resubmit**：mouthwash 触发 1 次——segment 2 供应商任务终态 FAILED（reason=asset_fetch，failed_task_id=fcd35e73-…），自动重提后成功，全程无需人工对账。
- **face_mask_degraded**：四单均未触发（灰块打码路径无降级）。
- **auto_resumed**：未触发（跑批期间无服务重启）。
- **永不失败重试**：dialogue-tts-line 节点在 necklace/mouthwash/generate 分别重试 15/13/3 次后全部成功（TTS 供应端抖动被退避重试吸收，未传导为项目失败）。

## dog 单失败（疑似回归，已按规则续跑验证）

- 失败点：`replacement-sheet-finalize` 之后、`dialogue-tts-line` 之前，pipeline 级报错 `dialogue TTS requires a voice reference`；**Picsart submit 之前**，无任何 submit 痕迹。
- 按 release AGENTS.md 走 `POST /api/conversations/dfe6479538184759ba6cc953a0bda138/resume-pre-picsart`（confirm），续跑被接受但 **1 秒内以同一错误失败**（trace: `pre_picsart_resume_started` → `pre_picsart_resume_failed`），属确定性失败。
- 证据：`work/voice-references/` 目录为空——本轮未物化任何音色参考；round1 同参数同素材的狗单是成功的（`voice-references` 正常产出）。该视频无人脸（只有手和狗），疑似新打码/人脸链路变更后，无人脸视频在 auto_rewrite 模式下拿不到音色参考（含 catalog 兜底音色，`pipeline.py:1116` 注释声称 fresh run 总会物化兜底）。**建议主代理上报为回归缺陷。**

## 产物清单（data/smoke-runs/public-api-e2e/round2/）

- `necklace.mp4` / `mouthwash.mp4` / `generate-necklace.mp4`（sha256 校验通过）
- `frames/`：四单 `*-reference-sheet.png` 各 1 张；三个成功单每段首/中/尾各 1 帧（necklace 4 段、mouthwash 5 段、generate-necklace 4 段，共 39 帧）
- `jobs.json`、`generate-necklace.json`、`generate-necklace-submit.json`、`submit-time.txt`、`public_api_e2e_round2.py`（round2 脚本：主工作树参数 + release 版 poll 修正）
- 抽帧初查：necklace/generate-necklace 帧中替换项链清晰可见；mouthwash seg3 中帧人物手持玻璃杯（该段是否应出现漱口水瓶待主代理逐帧验收）。

## generate-necklace 台词原文（state.json translated_voice_lines）

```text
[0.0-10.0s]   Okay, I have to be honest—most necklaces I've bought turn black and look cheap after wearing them just a couple of times. It's so frustrating!
[10.0-18.6s]  But this one? It's made of titanium steel. No tarnishing, ever. I can wear it in the shower, in the pool—I literally never take it off.
[18.6-28.4s]  And the minimalist design goes with everything. It highlights your collarbones perfectly—whether you're heading to work or going on a date.
[28.4-36.8s]  And the price? It's a fraction of what you'd pay for those designer brands. But stock is limited—tap the link below and grab yours now!
```
