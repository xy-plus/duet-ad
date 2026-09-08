# 对外 v1 API 最终验收轮 round5（2026-09-08）——达标

目标：**新验证实例** https://8.166.140.227:3221（代码 c412085：删校验最终版 + 消字节点 + 灰块盖头发 + 按秒定价；数据目录 data-verify/）。线上 3211/3212 未触碰。
四单参数与 round2-4 完全一致（原始长度，幂等键 `pub-e2e-r5-*`），提交于 13:23:07（UTC+8）。本轮无部署重启干扰。

## 判定：达标 —— 四单全部一次性成功（无 failed、无续跑、无 process_restarted 介入）

| 单 | job id | 终态 | 耗时 | 结算（60 积分/秒） |
|---|---|---|---|---|
| dog（换木质气垫梳，中文改编） | vg_80e34503b56f4f1cb8c11b7f1ea9e04e | succeeded | 26.5 min | 1020（≈17s） |
| necklace（换项链，保持原台词） | vg_2be0b352e7874a8ba794ca3f11c4fa8b | succeeded | 122.1 min | 2220（≈37s） |
| mouthwash（换 Listerine，英文改编） | vg_2be14a82063e41e59146936a54a86027 | succeeded | 96.8 min | 2100（≈35s） |
| generate-necklace（generate + brief，英文） | vg_331ef2d132364e0b809a624cdb8fd25d | succeeded | 115.6 min | 2220（≈37s） |

四单成片 sha256 全部校验通过。按秒定价 price_version=`credits-per-second-60-v1` 生效，结算与时长吻合。

**耗时偏长的原因（环境问题，非链路问题）**：验证实例与线上实例同机，提交后 CPU 超载（5 核 load≈20+），三个并发 whisper-cli（各 -t 4）连续 900s 超时；whisper-asr 在 necklace/mouthwash/generate 各重试 4 次后被 never-fail 退避吸收、最终通过。无一单因此失败。

## 事件统计（trace.jsonl + 节点 output.json）

| 事件 | dog | necklace | mouthwash | generate |
|---|---|---|---|---|
| json_repairs | 0 | **1**（object-replace attempt-1 修复成功） | 0 | 0 |
| voice_reference_default_fallback | 0 | 0 | 0 | 0 |
| face_mask_degraded | 0 | 0 | 0 | 0 |
| picsart_terminal_failure_resubmit | 0 | 0 | 0 | 0 |
| auto_resumed / process_restarted | 0 | 0 | 0 | 0 |
| whisper-asr 重试（CPU 竞争超时） | 0 | 4 | 4 | 4 |
| dialogue-tts-line 重试 | 3 | 15 | 13 | 3 |
| erase-source-subtitles | ✓ attempt-1 | ✓ attempt-1 | ✓ attempt-1 | ✓ attempt-1 |

## 消字验证（重点）

- 四单 `erase-source-subtitles` 节点均 attempt-1 成功，产出 `work/source-nosubs.mp4`，并被下游 `segment-cut` 实际消费（input-media 为 nosubs 版本）；帧级 `mediakit-erase` 全部成功（18/36/45/36 帧）。
- 源片 vs nosubs 逐帧对照（frames/ 内 `*-SOURCE-orig.png` / `*-SOURCE-nosubs.png`）：
  - necklace / generate（同源）：5s 处 "Look For Less" 已消除 ✓
  - mouthwash：5s 处底部 "tetap tak selesai kan" 绿字已消除 ✓
  - **dog：1/3/6/10/14s 处顶部西班牙语烧录字幕在 nosubs 中原样保留 ✗**——节点跑了但对这条顶部白字未生效（疑似该区域检测未命中）。不影响本单成片（成片为全量再生成，无字幕），但消字节点本身对该输入无效，建议主代理关注。
- 成片抽帧（每段首中尾，共 45 张）：四单均无原片烧录字幕、也无 H3 自生成字幕 ✓。

## 换脸/替换物抽帧结论

- dog：成片人物（手）与替换物木质气垫梳均跟随本轮 tile（TILE_01/TILE_03）✓。注意：本轮 entity-01 图版把狗 canonicalize 成了棕色卷毛狗（源片为灰色蓬毛狗），成片跟随图版呈棕色狗——"跟随 tile"达标，但图版本身与源片品种有漂移，供主代理判断。
- necklace / generate-necklace：人物佩戴替换项链（密镶网球链），与图版一致 ✓；generate 单人物与项链单同源同 tile。
- mouthwash：人物与源片一致（黑衣男子），替换物段落供主代理逐帧复核（frames/mouthwash-seg*-*.png 全量已导出）。

## generate-necklace 台词原文（translated_voice_lines，4 段）

```text
[0.0-10.0s]   You know that annoying thing when your necklace turns black after two wears? It instantly looks cheap, right? I've been there.
[10.0-18.6s]  But this one? It's made of titanium steel. It won't tarnish, and you can shower or swim with it on. Plus, the minimalist design goes with everything.
[18.6-28.4s]  And the way it sits on your collarbone? It really flatters your neck. Whether it's a casual workday or a weekend date, it just works.
[28.4-36.8s]  The best part? It's a tiny fraction of the price of designer brands. Stock is limited, so grab yours now at the link below.
```

## 产物（data/smoke-runs/public-api-e2e/round5/）

- 四个成片 mp4（sha256 全部校验通过）
- `frames/`：4 张 reference-sheet、45 张分段首中尾帧、8 张源片 orig/nosubs 对照帧
- `jobs.json`、`generate-necklace.json`、`generate-necklace-submit.json`、`submit-time.txt`、`public_api_e2e_round5.py`
