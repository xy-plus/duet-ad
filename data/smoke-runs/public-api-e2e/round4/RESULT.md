# 对外 v1 API 正式验收轮 round4（2026-09-08）——达标

目标：https://8.166.140.227:3211（release 最终版 a5e103b：删中间层校验 + 续跑音色兜底 + 健壮性改造）。
四单参数与 round2/round3 完全一致（原始长度，幂等键 `pub-e2e-r4-*` / `pub-generate-necklace-v3-<ts>`），提交于 09:05:48（UTC+8）。

## 判定：达标 —— 四单全部一次性成功（无 failed、无人工续跑）

| 单 | job id | 终态 | 耗时 | 一次性成功 | 结算 |
|---|---|---|---|---|---|
| dog（换木质气垫梳，中文改编） | vg_9e282d490c7f429893b01ac7bbc7b163 | succeeded | 20.3 min | 是 | 1000 |
| necklace（换项链，保持原台词） | vg_3bf64b383fa94744bcd98473b50fb61d | succeeded | 46.7 min | 是* | 1000 |
| mouthwash（换 Listerine，英文改编） | vg_c2f75102b76f40bd932f5e9567705b3f | succeeded | 54.4 min | 是* | 1000 |
| generate-necklace（generate + brief，英文） | vg_850125267f5c40138c35b46bd55b6689 | succeeded | 51.9 min | 是* | 1000 |

四单成片均已下载并 sha256 校验通过（`dog.mp4`/`necklace.mp4`/`mouthwash.mp4`/`generate-necklace.mp4`）。

\* **重要背景**：09:37:42 后端经历了一次部署重启（round4 提交于 09:05，三单当时在飞）。dog 在重启前（约 09:26）已完整跑完，无中断。necklace/mouthwash/generate 被标记 process_restarted 后由**系统自动续跑**（`auto_resumed_after_restart`，operation=resume_pre_picsart，各 2 条事件 + `pre_picsart_resume_succeeded`），全程无 failed、无人工介入。本轮因此顺带实证了重启自愈链路。轮询脚本在重启瞬间拿到一次非 JSON 响应而崩溃（已重启续轮），属客户端问题，与服务端无关。

## 新链路事件统计（trace.jsonl + 节点 output.json）

| 事件 | dog | necklace | mouthwash | generate |
|---|---|---|---|---|
| json_repairs | 0 | 0 | 0 | 0 |
| face_mask_degraded | 0 | 0 | 0 | 0 |
| voice_reference_default_fallback | 0 | 0 | 0 | 0 |
| picsart_terminal_failure_resubmit | 0 | 0 | 0 | 0 |
| auto_resumed_after_restart | 0 | 2 | 2 | 2 |
| pre_picsart_resume_started/succeeded | – | 1/1 | 1/1 | 1/1 |
| dialogue-tts-line 重试（TTS 抖动吸收） | 3 | 15 | 13 | 7 |

- 模型节点全部 attempt-1 通过，连续两轮（round3/round4）无 JSON 修正需求。
- 本轮四单 Picsart 回执全部一次 accepted，无终态失败重提。
- dog 单连续两轮未被音色兜底事件触发（模型均自行分配音色）；无人脸视频 fresh run 连续两轮成功。

## 对照组（只读观察，未触碰）

- 019b9d61（round3 mouthwash，曾死于 H3 prompt 尾部空格 + 续跑不自愈）：**续跑成功**——删中间层校验生效。
- dfe64795（round2 狗，曾死于无音色 + 续跑不补兜底）：**续跑成功**——续跑音色兜底生效。

round3 的两个续跑路径缺口在最终版均已修复并被实证。

## 产物（data/smoke-runs/public-api-e2e/round4/）

- 四个成片 mp4（sha256 全部校验通过）
- `frames/`：4 张 `*-reference-sheet.png`；每段首中尾各 1 帧（dog 2 段、necklace 4 段、mouthwash 5 段、generate-necklace 4 段，共 45 帧）
- `jobs.json`、`generate-necklace.json`、`generate-necklace-submit.json`、`submit-time.txt`、`public_api_e2e_round4.py`

## generate-necklace 台词原文（translated_voice_lines，8 段）

```text
[0.0-7.4s]   You know that feeling when your favorite necklace starts turning black after a couple of wears? It just makes everything feel cheap.
[7.4-10.0s]  That's exactly why I'm excited about this one.
[10.0-15.3s] This is made from titanium steel. It won't tarnish, even in water, so you can wear it in the shower and while swimming without worrying.
[15.3-18.6s] And the design is super minimal—it goes with literally every outfit you already own.
[18.6-24.1s] Look at how it sits on your collarbone. It's subtle, but it catches the light and makes your neck look longer and more elegant.
[24.1-28.4s] Whether you're heading to the office or out on a date, this necklace just works for any occasion.
[28.4-35.2s] And here's the real kicker—the price is less than a fraction of those big-name brands. And with stock this limited, you'll want to grab yours before they're gone. Just tap the link below.
[35.2-36.8s] Trust me, your future self will thank you.
```
