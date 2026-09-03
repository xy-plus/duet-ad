# 第三方视频生成 API v1

该接口面向服务端合作方。它与内部前端的共享口令、路由和数据权限完全隔离；默认由 `PUBLIC_API_ENABLED=0` 关闭。

## 核心语义

- `100` 积分等于 `1 CNY`，每个成功视频固定消费 `1000` 积分（`10 CNY`）。
- 创建任务时预占 `1000` 积分；确定失败时释放；成功时消费。
- `submission_unknown` 表示供应商提交结果尚不能确定，积分继续冻结。合作方只能继续查询原任务，不能换幂等键重发。
- 积分归属于 `owner_id`。同一 owner 的多把 API Key 共用余额和历史任务，便于无中断轮换密钥。
- 成片没有自动过期，`expires_at` 固定为 `null`。服务端不暴露供应商 URL、本机路径或私有 receipt。

## 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/v1/video-generations/capabilities` | 查询参数、限制、默认值和价格 |
| `POST` | `/api/v1/video-generations` | 创建异步任务 |
| `GET` | `/api/v1/video-generations/{id}` | 查询状态、费用和最终结果 |
| `GET/HEAD` | `/api/v1/video-generations/{id}/content` | 下载成片，支持单段 Range |
| `GET` | `/api/v1/account/credits` | 查询可用、冻结和累计消费积分 |
| `GET` | `/api/v1/account/credit-transactions` | 查询最近积分流水 |
| `GET` | `/api/v1/openapi.json` | 仅包含公共 v1 路由的 OpenAPI |

所有业务接口使用：

```text
Authorization: Bearer duet_live_<key_id>.<secret>
```

创建任务还必须携带 `Idempotency-Key`，格式为 `[A-Za-z0-9_-]{8,64}`。它按 owner 作用域持久化：同键同输入返回原任务，同键不同输入返回 `409 idempotency_key_reused`。

## 创建任务

`POST /api/v1/video-generations` 使用 `multipart/form-data`：

- `source_video` 与 `source_video_url` 必须且只能提供一个。
- `aspect_ratio`：`9:16` 或 `16:9`，默认 `9:16`。
- `resolution`：`768p` 或 `480p`，默认 `768p`。
- `target_language`：省略表示与原视频相同；空字符串非法。
- `replacement_image` 与 `replacement_instruction` 必须同时提供或同时省略。
- 原视频仅支持 MP4、MOV、WebM，最多 500 MiB、300 秒；URL 必须为无凭据 HTTPS 公网地址。
- 替换图仅支持 JPEG、PNG、WebP，最多 10 MiB。

首次创建返回 `201`，幂等重放返回 `200`；两者都返回完整任务对象，并携带 `Location` 与 `Retry-After: 5`。

```bash
curl --request POST 'https://example.com/api/v1/video-generations' \
  --header 'Authorization: Bearer duet_live_example01.REPLACE_WITH_SECRET' \
  --header 'Idempotency-Key: partner-order-0001' \
  --form 'source_video=@/absolute/path/to/source.mp4;type=video/mp4' \
  --form 'aspect_ratio=9:16' \
  --form 'resolution=768p'
```

公共状态只有 `queued`、`running`、`succeeded`、`failed`、`submission_unknown`。`progress.percent` 是提示值，不是 ETA；客户端应遵循 `Retry-After`，随后采用带抖动的退避轮询，最长间隔 10 秒。

## 积分管理

余额不是可直接修改的字段，而是不可变事件的投影：

- `adjustment`：内部人工充值或扣减；
- `reserve`：创建任务时冻结；
- `capture`：成功后消费；
- `release`：确定失败后解冻。

每个事件都有确定的事件 ID并以 create-once 方式持久化。重复执行同一个内部调整幂等键不会重复加减；同一幂等键使用不同金额或原因会失败。管理员不能扣减已冻结积分，也不能把可用余额扣成负数。

内部管理只提供本机 CLI，不新增公网管理接口：

```bash
cd /home/xy/duet-ad1/.worktree/release-fusion-budget-439c4b5-next

/home/xy/duet-ad1/.venv/bin/python -m app.public_api_admin create-key \
  --registry /home/xy/.config/duet-ad1/public-api-clients.json \
  --owner partner_acme \
  --key-id acme0001 \
  --key-output /home/xy/.config/duet-ad1/public-api-test-credentials.json

/home/xy/duet-ad1/.venv/bin/python -m app.public_api_admin credits-adjust \
  --data-dir /absolute/public/api/data \
  --owner partner_acme \
  --credits 100000 \
  --reason '线下回款充值 1000 元' \
  --idempotency-key receipt_20260903_0001

/home/xy/duet-ad1/.venv/bin/python -m app.public_api_admin credits-show \
  --data-dir /absolute/public/api/data \
  --owner partner_acme
```

`create-key` 只显示一次明文密钥。注册表必须是绝对路径、权限 `0600`；目录权限为 `0700`。建议轮换流程为：创建同 owner 的新 key、合作方切换、再吊销旧 key。

## 启用边界

服务环境至少需要：

```text
PUBLIC_API_ENABLED=1
PUBLIC_API_CLIENTS_FILE=/home/xy/.config/duet-ad1/public-api-clients.json
DATA_DIR=/absolute/public/api/data
```

启用公共 API 时 `DATA_DIR` 必须为绝对路径。首版继续遵守项目现有的单进程约束；不能增加 Uvicorn worker 或让多个实例共享该数据目录。代理层仍需在 multipart 解析之前配置请求体、连接时长和并发限制。

本文件只描述代码合同，不代表接口已经部署。发布、重启、生成真实密钥和真实供应商 canary 均需单独授权。
