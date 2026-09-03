# 极简前端后端接口需求（v1）

## 目标

用户只提供原视频、输出配置、最终视频语种，以及可选的“参考图 + 替换说明”。后端自动识别原视频台词，由 Agent 完成改编和目标语种转换，并直接运行到成片。

本接口不增加用户确认节点，不要求用户提交台词，不向前端暴露分段、关键帧、模型、供应商或内部阶段。

## 1. 能力声明

现有 `GET /api/capabilities` 增加：

```json
{
  "minimal_creation": {
    "supported": true,
    "version": 1,
    "endpoint": "/api/conversations",
    "encoding": "multipart/form-data",
    "request_field": "generation_request",
    "replacement_image_field": "replacement_image",
    "aspect_ratios": ["16:9", "9:16"],
    "resolutions": ["480p", "768p"],
    "defaults": {
      "fit_mode": "auto",
      "optimize_image": true,
      "remove_subtitle": true,
      "remove_logo": true
    },
    "dialogue": {
      "mode": "auto_rewrite",
      "translation": true
    },
    "replacement": {
      "supported": true,
      "accept": ["image/jpeg", "image/png", "image/webp"],
      "max_bytes": 10485760,
      "max_instruction_chars": 1000
    }
  }
}
```

接口未全部就绪时不要返回 `supported: true`。前端不会回退到旧创建合同。

## 2. 创建项目

继续使用 `POST /api/conversations`，编码为 `multipart/form-data`。

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `reference_url` | 条件必填 | 默认来源；与 `file` 二选一 |
| `file` | 条件必填 | 本地上传；与 `reference_url` 二选一 |
| `client_request_id` | 是 | 创建请求幂等键 |
| `generation_request` | 是 | 下述 JSON 字符串 |
| `replacement_image` | 条件必填 | 仅在提供替换说明时上传 |

`generation_request`：

```json
{
  "version": 1,
  "output": {
    "aspect_ratio": "9:16",
    "resolution": "768p",
    "fit_mode": "auto"
  },
  "processing": {
    "optimize_image": true,
    "remove_subtitle": true,
    "remove_logo": true
  },
  "dialogue": {
    "mode": "auto_rewrite",
    "target_language": "中文"
  },
  "replacement_guidance": null
}
```

使用参考图时：

```json
{
  "replacement_guidance": {
    "instruction": "把画面中的白色水杯替换成参考图中的产品杯",
    "image_field": "replacement_image"
  }
}
```

必要约束只有：

1. `reference_url` 与 `file` 必须二选一。
2. `target_language` 去除首尾空白后必须非空，最长 80 个字符。
3. 三个 `processing` 值固定为 `true`，不能由用户关闭。
4. `replacement_image` 与 `replacement_guidance` 必须同时存在或同时不存在。
5. 请求中不接受 `script`、`lines` 或旧台词模式字段。

## 3. 后端执行语义

创建成功后，后端自动完成：

1. 获取并校验原视频。
2. 自动识别原视频台词。
3. 由 Agent 改编台词并转换为 `target_language`。
4. 按固定图片优化、去字幕、去 Logo 配置生成视频。
5. 直接运行到成功或失败，不进入中途台词校对。

这些步骤都是后端实现细节，不增加前端节点或新提交接口。

## 4. 创建响应与项目进度

首次创建返回 `201`；相同 `client_request_id` 的完全相同请求返回原项目和 `200`：

```json
{
  "id": "a4fce59c6ae84bfa8673d2337e990001",
  "title": "source.mp4",
  "has_video": false,
  "project_progress": {
    "percent": 0,
    "status": "queued"
  }
}
```

`GET /api/conversations` 和 `GET /api/conversations/{id}` 都需要返回：

```json
{
  "project_progress": {
    "percent": 47,
    "status": "running"
  }
}
```

- `percent`：整数 `0..100`。
- `status`：`queued | running | succeeded | failed`。
- `queued=0`，`running=1..99`，最终视频可读取后才允许 `succeeded=100`。
- 不返回内部阶段或分段进度。
- 成片可用时 `has_video=true`，并继续使用 `GET /api/conversations/{id}/files/generated.mp4`。

## 5. 最小错误合同

错误保持现有 `{"detail":{"code","message","field"}}` 结构。前端需要以下稳定机器码：

| HTTP | code | 条件 |
| --- | --- | --- |
| 400 | `source_exactly_one_required` | 视频来源不是恰好一个 |
| 409 | `client_request_id_conflict` | 同一幂等键对应不同输入 |
| 422 | `invalid_generation_request` | JSON 结构或固定配置错误 |
| 422 | `target_language_required` | 最终语种为空 |
| 422 | `replacement_pair_required` | 参考图与说明未成对提供 |
| 413/415 | `invalid_replacement_image` | 图片超限或格式不支持 |

校验失败不得创建项目或调用付费服务。

## 6. 联调验收

- 默认 `reference_url` 能直接创建，切换上传后 `file` 能创建。
- 请求与持久化结果均不包含用户 `script`。
- Agent 生成的台词满足 `target_language`，且不要求中途确认。
- 三个隐藏处理项始终为 `true`。
- 参考图与说明能一起冻结并进入图片优化链路。
- list/detail 返回同一项目级百分比，前端只显示一条进度条。
- 最终文件未就绪前不返回 `has_video=true` 或 `100%`。
