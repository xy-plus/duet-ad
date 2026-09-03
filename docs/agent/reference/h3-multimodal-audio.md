# H3 current 音频合同

current v4 不是 source-audio multimodal。每个 segment 的 `H3Request` 固定满足：

- `mode=reference`，使用后端编译的唯一 Ref2VA prompt；
- exactly 3 张有序 Picture reference，并逐值绑定 source scene/time/transition；
- `reference_audios=()`，Fusion 输入 `voice_references=[]`，逐行 `voice_ref=null`；
- Context 为 `local:identity:<source_prompt_sha256>` 同字节 receipt，HTTP 调用数为 0。

源视频音轨和 `work/voice.mp3` 只供既有 ASR/YAMNet 分析。后端把冻结 `spoken` 文本、时间和 off-screen 呈现方式机械编译进 Ref2VA prompt；音频 path/bytes 不进入 Fusion workspace、Context request、H3 request 或 stitch source。

H3 输出音轨是成片声音的唯一真源。统一 EDL 对每个 segment：

- H3 有音轨：按该 segment 的视频帧预算裁补原生 H3 音频；
- H3 无音轨：在同一 EDL 插入等长有限静音；
- 任一情况都不回挂、覆盖、混入或 overlay 源音频/conditioning audio。

semantic score、speech/music diagnostics 和外部 A/B 观察只用于测试与 Skill 迭代；它们不阻断生产、不触发 retry，也不选择 source-audio 或其他 workflow 作为 fallback。媒体文件、PTS、时长、SHA 和 receipt 的技术完整性验证仍然有效。

## 历史只读

旧 `app.h3_multimodal`、`ref_audio_N`、Gateway `multimodal|multimodal_hd`、speaker/binding 与 source-audio receipt 只为历史已知 task 的原 receipt GET 恢复保留。它们不得用于 current create、迁移、重写 Ref2VA prompt、覆盖成片或备用 POST。
