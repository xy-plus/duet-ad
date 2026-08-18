---
name: upload-create
type: behavior
status: done
owner: human
updated: 2026-08-18
links: []
---

# 上传创建会话

## 规则

| 当 | 则 |
| --- | --- |
| 登录后选择 mp4/mov/webm 视频并发送（备注可选） | 创建会话，立即返回 201，状态 `queued`，后台开始处理 |
| 选择口播转换（原文保持/原文改编/翻译为[语言]，默认原文保持） | 模式与目标语言随会话保存（meta 内部字段）；「翻译为」时目标语言必填 |
| 视频无音轨 | 422 `no audio track in video`，回滚；composer 提示「该视频没有音轨，本产品仅支持带口播的视频」 |
| 备注留空 | 标题取净化后的原文件名（去路径/控制字符、限 80 字，空则 `untitled`） |
| 同一 IP 1 分钟内上传超过 10 次 | 第 11 次起 429 `too many uploads` |
| 扩展名不是 .mp4/.mov/.webm | 422，指明不支持的扩展名 |
| 文件超过 500MB（MAX_UPLOAD_MB） | 422 `file exceeds ... bytes`，已写部分删除 |
| 视频打不开 / 时长超过 300s（MAX_DURATION_S） | 422（ffprobe 探测结果），不留会话 |
| 任一校验失败 | 整个会话目录回滚删除，列表中不出现 |

## 边界

- 未带文件也未给链接/两者都给：400；未登录：401
- 上传过程中前端禁止切换会话，避免打断；上传有进度条（XHR）
- 前端仅按 MIME `video/*` 或扩展名预筛，最终以后端校验链为准
- 本产品仅处理带口播视频：Web 端口播模式恒为三选一（默认「原文保持」），无音轨视频上传即 422；API 直接调用 `voice_mode=none` 可跳过音轨探测（兼容 curl 等存量用法）
- 视频流式落盘，不读进内存；校验在全部写完后进行（ffprobe）

## 例子

- 输入：`curl -H "Authorization: Bearer $TOKEN" -F file=@clip.mp4 -F note=厨房去油 http://localhost:3211/api/conversations` → 输出：201 `{"id":"<32位hex>","status":"queued"}`
- 输入：上传 `clip.mkv` → 输出：422 `{"detail":"unsupported extension: .mkv"}`
- 输入：`-F voice_mode=translate -F target_language=日语`（视频无音轨）→ 输出：422 `{"detail":"no audio track in video"}`
