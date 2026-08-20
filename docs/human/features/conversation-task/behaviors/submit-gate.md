---
name: submit-gate
type: behavior
status: done
owner: human
updated: 2026-08-20
tdd: N/A
links: [conversation-task, processing-state]
---

# H3 提交门控

## 规则

`POST /api/conversations/{id}/submit` 只接受：

```json
{
  "confirm": true,
  "client_request_id": "request-123456",
  "dialogue_mode": "auto",
  "fit_mode": "none"
}
```

`edit/custom` 还必须带非空 `lines`；`auto/none` 禁止带 `lines`。请求存在未知字段、无严格 `confirm:true`、id 不合规、台词时间越界或画幅选择不匹配时均拒绝。

| 条件 | 结果 |
| --- | --- |
| `ENABLE_H3_SUBMIT` 未开 | 501 `H3 submission is disabled.` |
| 会话不存在 | 404 `not found` |
| schema 不是 v2 | 409 `read_only` |
| 输入准备未 `done` | 409 `artifacts not ready` |
| MiniMax 或 AutoDL 凭据缺失 | 503 `h3_credentials_missing` |
| 冻结输入、receipt 或画幅派生失败 | 409 `prepared_input_invalid` / `frame_fit_failed` |
| 同一 generation active/succeeded，或确定失败后复用旧 id | 409；不创建新供应商任务 |
| generation 为 `resume_required` | 只接受原 id、原台词和原 fit；合法时返回 202 + 原 attempt，新 id/参数漂移分别 409 |
| generation 为 `submission_unknown` | 任意 id 均 409 `submission_outcome_unknown`；先核对供应商 |
| 全部门控通过 | 202 `{"status":"queued","attempt":N}`；后台异步执行 |

## 边界

- 首次 start 和确定失败后的 retry 会在锁内按人工选择冻结 receipt；`resume_required` 只加载既有 receipt，不重写它、不递增 attempt。
- H3 只使用 `work/keyframes/` 原图或 `work/h3_frames/{crop|pad}/` 派生图；不读取 Seedream `postprocessed/`。
- Context IR 不设置台词内容或标签结构门禁；用户确认的非空、限长正文可直接提交 H3。
- 成片下载先验证全部 DNS 解析地址，再在读取 status/body 前验证实际 socket peer 为公网；拒绝 userinfo 和重定向，限制 200 MiB，并在原子落盘前通过 ffprobe 正时长视频流验证。
- 旧 Seedance 提交实现和 `face_hold` 参数/提示词注入已删除，不是失败回退选项。
