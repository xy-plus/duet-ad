# Lip-sync gateway（实验 B）

`app/lipsync.py` 是未接生产入口的实验 B gateway：先完成静音画面，再用统一冻结的
`target_audio` 驱动多人嘴型。`target_audio` 可以来自原音、改词、翻译、加台词或换声线；
它是最终 remux 的唯一音频真源，不能无条件回挂 source audio。

## 边界

- 默认 receipt 相对路径是 `work/lipsync/receipt.json`，schema 为
  `duet.lipsync.request` v1。API 接受的所有文件路径都必须是项目相对规范路径，解析后仍须
  位于项目根目录内。
- 模块没有默认 HTTP 客户端。协调层必须显式注入 `send`，因此本模块自身不会联网，也没有
  生产路由、后台任务或本机 lip-sync 模型。
- `freeze_request` 在任何 provider 调用前冻结并哈希：静音视频、上游视觉 receipt、统一
  `target_audio`、target-audio receipt、逐 speaker 音频段、speaker-to-face、一一对应参考帧、
  整数 PTS/timebase、provider 参数、workflow 和本地 idempotency key。
- 最多 3 个 speaker；每人最多 10 段、总计最多 30 段。缺 mapping、重复 face、跨 speaker
  区间冲突、非整数 PTS、超过毫秒精度或冻结文件漂移均在提交前 fail closed。
- 凭据只在内存中用于腾讯 aPaaS HMAC-SHA256 签名，receipt 不保存 app key、access token、
  动态 timestamp/signature、素材 URL 或 provider 原始错误。

## 状态机

```text
prepared --submit once--> accepted --query--> processing --query--> succeeded
    |                         |                           |
    +-- ambiguous ----------> submission_unknown         +--> failed
```

- `submitting` receipt 必须在付费提交前原子落盘并 fsync 文件及父目录。
- POST 超时、取消、5xx/成功响应但没有可靠 TaskId，均成为 `submission_unknown`；该状态永久禁止
  再次提交。
- 取得 TaskId 后只能执行 provider 的 `query` operation。腾讯官方进度查询端点本身使用 HTTP
  POST `/getprogress`，但它不是再次调用付费的 `/videomakenotrain`。
- 查询异常不改变 TaskId，可安全再次查询；provider 失败只公开稳定本地错误码，不公开
  `Message`/`FailMessage`。

## A/B 可比较 receipt

顶层 `comparison` 使用 `duet.av-generation` v1：

- `route=post_h3_lipsync` 与实验 A 的 `h3_native_audio_conditioned` 区分路线；
- `visual_input` 绑定上游视觉 receipt、静音视频和有序参考帧的 size/SHA；
- `target_audio_materials` 绑定 target-audio receipt、content/decoded SHA、采样率、声道、统一
  timebase/整数 PTS、dialogue SHA 和 speaker-face-map SHA；
- `upstream` 绑定视觉 attempt；
- `output` 在 gateway receipt 中固定为 `null`。

腾讯成功只证明 TaskId 对应一个临时 `MediaUrl`/声明时长，不能证明已下载输出 bytes、输出 SHA
或解码 PTS。后续协调层必须下载、探测并生成独立 output receipt，才能交给最终 remux/stitch；
禁止用本 gateway receipt 冒充输出媒体证据。

腾讯契约依据：

- <https://cloud.tencent.com/document/product/1240/116222>
- <https://cloud.tencent.com/document/product/1240/81270>
- <https://cloud.tencent.com/document/product/1240/107197>
