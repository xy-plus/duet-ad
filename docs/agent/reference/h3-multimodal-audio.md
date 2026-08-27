# H3 原生多模态音频合同

`app.h3_multimodal.build_h3_request` 严格消费 video-maker 的
`h3_prompt_plan.json`：外部 Picture/Audio 编号保持 1-based，编译后的
Context-IR 冻结精确台词、语言、人物、图片和声线关系。静默 subject、不同
subject 复用 Picture、未绑定音频和不连续发声顺序均在任何 provider I/O 前失败。

音频通过 `app.h3.freeze_reference_audios` 一次读取并用 ffprobe 探测。只接受
1–3 段 MP3/WAV；每段 2–15 秒、总长不超过 15 秒。冻结 bytes、SHA-256、顺序、
用途和时长进入现有 H3 input/attempt/provider receipt。提交不重读源路径，而是将
冻结 bytes 物化到会话受控目录，再调用本机 Gateway：

```text
POST http://127.0.0.1:31000/v1/videos
{mode,prompt,duration_sec,aspect_ratio,resolution,images,audios}
```

`mode=multimodal` 由 Gateway 映射到
`minimax_h3_image_audio_to_video_v2_15s`；`multimodal_hd` 映射到
`minimax_h3_image_audio_to_video_v2`。应用不复制 Gateway 内部的 `ref_image_N` /
`ref_audio_N` 编码逻辑，也不把 `audio_required` 发给 Gateway。后者只绑定内部请求
receipt，并在下载后通过媒体 timeline/ffprobe 硬性要求 H3 成片含音轨。

付费安全沿用 `app.h3` 的 prepare/attempt/submitting/task-id 状态机：同一 request
重复 submit 不产生第二次 POST；`submission_unknown` 只能恢复为 GET，不会重发。
参考音频仅是 conditioning，不是目标音轨或 PTS 锁；H3 输出音轨才是成品真源。

限制来源：

- MiniMax 官方 V2 创建接口：<https://platform.minimax.io/docs/api-reference/video-generation-v2-create>
- MiniMax 官方视频生成指南：<https://platform.minimax.io/docs/guides/video-generation>
- 本机 Gateway HTTP 合同：`/home/xy/duet-video-v2/server/src/services/h3Client.service.ts`
- Gateway mode/workflow 投影：`/home/xy/duet-video-v2/deploy/h3-gateway/gateway.js`
