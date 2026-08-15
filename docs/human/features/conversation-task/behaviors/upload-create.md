---
name: upload-create
type: behavior
status: done
owner: human
updated: 2026-08-15
links: []
---

# 上传创建会话

## 规则

| 当 | 则 |
| --- | --- |
| 登录后选择 mp4/mov/webm 视频并发送（备注可选） | 创建会话，立即返回 201，状态 `queued`，后台开始处理 |
| 备注留空 | 标题取净化后的原文件名（去路径/控制字符、限 80 字，空则 `untitled`） |
| 同一 IP 1 分钟内上传超过 10 次 | 第 11 次起 429 `too many uploads` |
| 扩展名不是 .mp4/.mov/.webm | 422，指明不支持的扩展名 |
| 文件超过 500MB（MAX_UPLOAD_MB） | 422 `file exceeds ... bytes`，已写部分删除 |
| 视频打不开 / 时长超过 300s（MAX_DURATION_S） | 422（ffprobe 探测结果），不留会话 |
| 任一校验失败 | 整个会话目录回滚删除，列表中不出现 |

## 边界

- 未带文件/未登录：422（表单校验）/ 401
- 上传过程中前端禁止切换会话，避免打断；上传有进度条（XHR）
- 前端仅按 MIME `video/*` 或扩展名预筛，最终以后端校验链为准
- 视频流式落盘，不读进内存；校验在全部写完后进行（ffprobe）

## 例子

- 输入：`curl -H "Authorization: Bearer $TOKEN" -F file=@clip.mp4 -F note=厨房去油 http://localhost:3211/api/conversations` → 输出：201 `{"id":"<32位hex>","status":"queued"}`
- 输入：上传 `clip.mkv` → 输出：422 `{"detail":"unsupported extension: .mkv"}`
