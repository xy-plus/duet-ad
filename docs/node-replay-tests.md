# 节点重放测试（真实输入，E2E 前置门禁）

每轮 E2E 之前，先用这套测试并行验证除 H3 视频生成（picsart-submit/poll/
download/trim）之外的所有节点。**全部通过才发起 E2E。**

所有输入都是老项目对应节点真实出现过的输入（`data/projects/<id>/logs/
<node>/<scope>/attempt-N/input.json` + `input-media/`），没有 mock 数据；
测试调用管线里该节点对应的真实函数（`app.media` /
`app.replacement_sheet` / DeepSeek / Seedream 客户端），不是重发
整个 pipeline。

## 1. 采集 fixtures

fixtures 体积大（约 380MB，媒体按内容哈希去重），不进 git。运行测试前先
执行采集：

```bash
/home/xy/duet-ad1/.venv/bin/python scripts/harvest_node_fixtures.py
# 可选：把 round2 在跑项目已产出的节点也采进来（只读，不干扰在跑项目）
/home/xy/duet-ad1/.venv/bin/python scripts/harvest_node_fixtures.py --include-round2
```

产出：`tests/fixtures/node-replays/<project>/<node>/<scope>/{input.json,
input-media/, output.json}`，`boundary/` 子集收失败 attempt 和失败项目的
状态快照（如 dfe64795 的 TTS 缺音色现场），`manifest.json` 记录每条
fixture 的来源项目、attempt 和当时 input/output 的 sha256（字段格式见
`tests/fixtures/node-replays.manifest.template.json`）。

本地模型文件（whisper.cpp 权重、YAMNet）不复制，replay 时从宿主机
规范路径解析，manifest 里记 `skipped_media`。

## 2. 运行

```bash
cd /home/xy/duet-ad1-h3-max-r2v-release
NODE_REPLAY_LIVE=1 /home/xy/duet-ad1/.venv/bin/python -m pytest \
    tests/test_node_replay_local.py \
    tests/test_node_replay_external.py \
    tests/test_node_replay_boundary.py \
    -n auto -q
```

- `-n auto`（pytest-xdist）并行；套件设计为整体并行跑完。
- 凭证：`tests/conftest.py` 自动加载 `~/.config/duet-ad1/service.env`、
  `~/.config/claude/deepseek.env` 和主树 `.env`（与 deploy 脚本同源），
  不打印任何值。缺 DeepSeek 凭证时对应测试 skip。
- **重放门禁**：三个重放文件（local/external/boundary）全部标记为
  `node_replay`，**默认跳过**——日常 `pytest tests` 不跑真实媒体重放、
  不触发任何外部/付费调用；E2E 前置门禁如上显式加 `NODE_REPLAY_LIVE=1`
  才会执行整个重放套件。

### 成本粗估（每次全量运行）

| 节点 | 调用量 | 粗估成本 |
| --- | --- | --- |
| DeepSeek（video-maker/索引/object-replace/fusion/翻译/台词生成） | ~50 次（含约 20 次带 9 图的视觉调用） | 数元（flash 视觉模型） |
| Seedream 参考图版 | 5 张 2048×2048 | 约 1–2 元 |

配音 TTS 已切换到豆包语音合成 2.0（app/doubao_tts.py），AutoDL 节点与
fixtures 已随切换移除；如需回放 dialogue-tts-line，按新 client 重新采集。

## 3. 判定标准

除"不抛异常"外，每个节点还断言输出通过下游消费它的校验：

- video-maker 逐段分析 → 输出过 `normalize_visual_prompt`，且帧图片
  sha256 必须与 payload 里的帧回执一致（防绑定漂移）。
- video-maker-project-index → 过 `normalize_project_index`，产出
  people/entities/scenes/relations 稳定键。
- object-replace → 过 `normalize_object_replace` + 音色物化；有台词时
  必须物化出至少一条音色参考（无人分配音色时走目录兜底音色）。
- h3-max-r2v-prompt → 过 `normalize_fusion_prompt`（schema/sha/段号/权威
  约束全绑定）。
- deepseek-translation → 过 `_validate_translation_shape` 且归一化后行数
  与源台词一致；`与原视频相同` 项目走确定性跳过分支。
- 本地确定性节点（segment-planning、keyframe-selection、dialogue-localize、
  spoken-dialogue-filter、probe）与历史 output 做精确相等断言。
- replacement-sheet-inputs / user-reference → 重放的 canvas/prompt/参考图
  sha256 必须与 seedream-reference-submit 的 `input_receipts` 逐字节一致。
- whisper-asr / yamnet 互做下游门禁：ASR 行必须过 `filter_spoken_lines`
  （配合同项目 YAMNet 输出），反之亦然。

`test_fixture_coverage_complete` 失败关闭：fixtures 缺失或任何必需节点零
覆盖都会直接失败并提示先跑采集脚本，不会静默跳过。

## 4. 放行 E2E

上述三个测试文件全绿（允许凭证缺失导致的 skip，但 pre-E2E 门禁场景应
确保凭证齐全、零 skip）后，才发起 E2E。任何节点红了，先修节点再谈
E2E。

## 边界子集

`tests/test_node_replay_boundary.py` 重放 dfe64795 的真实失败现场（无音色
分配 + 4 条待合成台词）：断言当前管线对无音色 patch 走目录默认音色兜底、
每行台词都能解析出音色路径，而不是在 TTS 前抛
`dialogue TTS requires a voice reference`。
