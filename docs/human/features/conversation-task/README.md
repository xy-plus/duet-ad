---
name: conversation-task
type: feature
status: done
owner: human
updated: 2026-08-20
tdd: N/A
links: []
---

# 会话式 H3 视频复刻

## 要什么

运营/创作者上传一支 1–15 秒参考视频，评审系统准备的关键帧、视觉提示词与自动台词，再显式选择台词和 9:16 画幅策略，提交 MiniMax Context IR → AutoDL H3 生成最终视频。

## 为什么

把看片、选帧、台词来源、提示词组合和付费生成固化成可审计流程；尤其要保证画面 OCR 不会变成角色台词、重复点击和服务重启不会静默重复扣费。

## 验收

- [x] 新会话使用 schema v2；无音轨视频合法，自动台词为空也能继续
- [x] 自动台词只来自 ASR，声学分类为 `spoken` 或 `sung` 的句子都可保留；OCR、字幕、画面文字和备注绝不成为台词
- [x] Context IR 以用户可见、可编辑的正文为准；不校验 `<d>` 台词内容或结构，用户确认后直接进入 H3
- [x] 生成前可选 `auto/edit/custom/none`；非 9:16 必须人工选择居中裁切或黑边留边
- [x] 界面展示实际时长；10–15 秒仍可提交但显示稳定性 warning，引擎时长按实际时长向上取整
- [x] 提交冻结版本化 receipt，随后异步执行 Context IR → H3，并持续显示 `queued/running/resume_required/succeeded/failed/submission_unknown`
- [x] 已知 task 故障只允许原参数继续同一 attempt；确定失败才用新请求 id 重试；`submission_unknown` 锁死提交并要求先核对供应商
- [x] 旧会话可查看但不能提交、重试或后处理
- [x] 可选去字幕水印/品牌后处理继续可用，但其结果不进入 H3 输入
- [x] `face_hold` 与 Seedance 生产提交/回退路径已删除
- [x] H3 成片下载只接受预解析地址与实际连接 peer 均为公网的 HTTPS、拒绝重定向、限制 200 MiB，并经 ffprobe 视频门禁后原子落盘

## 边界

- H3 是模型名，不代表服务启用了 HTTP/3。
- 当前新契约只接受最长 15 秒的单段视频，不能保证逐帧、文字、音频与源视频完全一致。
- 单共享口令，无用户级权限；状态用 2 秒轮询，不用 SSE/WebSocket。
- 供应商确定失败不触发自动重试；已知 task 的查询/超时/下载故障等待人工继续但不创建 attempt；`submission_unknown` 连人工继续也被拒绝，须先核对是否已创建任务。

## 取舍

- receipt 和 attempt 状态都落文件系统，以输入哈希、先落状态再 POST 和会话文件锁换取可恢复性；部署保持单 uvicorn 进程。
- 后处理与生成解耦：后处理只供人查看/下载，H3 始终绑定原始或显式画幅派生帧，避免旧图片编辑结果改变已确认的生成输入。

> 表现规格见本目录 `behaviors/`；纯文档变更不适用 TDD。
