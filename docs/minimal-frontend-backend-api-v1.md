# 极简前端后端接口当前合同（v1）

## 目标

用户只提供原视频、输出配置、最终视频语种，以及可选的“参考图 + 替换说明”。后端自动识别原视频台词，由 Agent 完成改编和目标语种转换，并直接运行到成片。

本接口不增加用户确认节点，不要求用户提交台词，不向前端暴露分段、关键帧、模型、供应商或内部阶段。

## 1. 能力声明

`GET /api/capabilities` 必须返回：

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
      "remove_logo": true,
      "add_subtitles": false
    },
    "dialogue": {
      "modes": ["auto_rewrite", "generate"],
      "mode": "auto_rewrite",
      "translation": true,
      "max_brief_chars": 1000,
      "optional": true,
      "default": {
        "mode": "auto_rewrite",
        "target_language": "与原视频相同"
      }
    },
    "replacement": {
      "supported": true,
      "accept": ["image/jpeg", "image/png", "image/webp"],
      "max_bytes": 10485760,
      "max_images": 4,
      "max_instruction_chars": 1000
    }
  }
}
```

`minimal_creation` 的字段、固定值和数组顺序必须与上例精确一致；`version` 必须严格等于整数 `1`。缺失、额外字段、值或顺序不一致都必须 fail closed，不能推断、转换或选择其他创建合同。

## 2. 创建项目

唯一创建入口是 `POST /api/conversations`，编码为 `multipart/form-data`。

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `reference_url` | 条件必填 | 默认来源；与 `file` 二选一 |
| `file` | 条件必填 | 本地上传；与 `reference_url` 二选一 |
| `client_request_id` | 是 | 创建请求幂等键 |
| `generation_request` | 是 | 下述 JSON 字符串 |
| `replacement_image` | 条件必填 | 仅在提供替换说明时上传；多图时重复传该字段 |

上表五个字段构成当前客户端创建输入；服务端内部字段不属于公开合同。

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

`generation_request.version` 必须严格等于整数 `1`；其公开结构由上例中的 `output`、`processing`、`dialogue` 和 `replacement_guidance` 定义。

`dialogue` 只接受两种封闭形态：

- `{"mode": "auto_rewrite", "target_language": ...}`：自动识别原视频台词并改写为目标语种（上例）。
- `{"mode": "generate", "target_language": ..., "brief": ...}`：不参考原视频台词，由后端 Agent 按用户的口播描述直接生成目标语种台词。`brief` 必填，去除首尾空白后必须非空，最长 1000 个字符。

向后兼容的默认值：`dialogue` 整体可缺省，等价于能力声明中的 `default`（`{"mode": "auto_rewrite", "target_language": "与原视频相同"}`）；`dialogue` 内的 `mode` 键可缺省，默认 `auto_rewrite`。缺省值由服务端填充，归一化后的存储请求始终是完整形态。

使用参考图时：

```json
{
  "replacement_guidance": {
    "instruction": "把画面中的白色水杯替换成参考图中的产品杯",
    "image_field": "replacement_image"
  }
}
```

需要多张参考图时，在同一个 multipart 请求中重复传 `replacement_image` 字段，共 2 至 4 张；每张不超过 10 MB，格式限 `image/jpeg`、`image/png`、`image/webp`。`replacement_guidance` 仍只有一份，对全部参考图生效。

必要约束只有：

1. `reference_url` 与 `file` 必须二选一。
2. `target_language` 去除首尾空白后必须非空，最长 80 个字符。
3. 三个 `processing` 值固定为 `true`，不能由用户关闭。
4. `replacement_image` 与 `replacement_guidance` 必须同时存在或同时不存在。
5. `dialogue.mode` 为 `generate` 时 `brief` 必填；为 `auto_rewrite` 时不得携带 `brief`。
6. `generation_request` 不携带客户端台词文本或模型选择。

## 3. 后端执行语义

创建成功后，后端自动完成：

1. 获取并校验原视频。
2. 自动识别原视频台词（`generate` 模式下识别照常进行，但识别结果不作为台词来源）。
3. 由 Agent 改编（`auto_rewrite`）或按 `brief` 生成（`generate`）台词并转换为 `target_language`。
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

创建响应与 `GET /api/conversations/{id}` 必须返回同一份冻结的 `creation_input`。链接来源使用：

```json
{
  "creation_input": {
    "version": 1,
    "source": {
      "mode": "link",
      "reference_url": "https://media.example/video.mp4"
    },
    "replacement_image": null
  }
}
```

上传来源且带参考图时使用：

```json
{
  "creation_input": {
    "version": 1,
    "source": {
      "mode": "upload",
      "filename": "source.mp4",
      "bytes": 12345678
    },
    "replacement_image": {
      "filename": "product.png",
      "bytes": 345678,
      "media_type": "image/png",
      "preview_url": "/api/conversations/PROJECT_ID/creation-input/replacement-image"
    }
  }
}
```

`creation_input.version` 必须严格等于整数 `1`；`source.mode` 只能是 `link` 或 `upload`，并携带对应示例中的精确字段。`replacement_image` 未提供时必须为 `null`；提供时必须与冻结输入回执一致。`preview_url` 固定指向当前项目的鉴权只读接口 `GET /api/conversations/{id}/creation-input/replacement-image`，不得接受路径参数、覆盖或重新提交素材。

多图创建时 `creation_input` 额外携带 `replacement_images` 数组，与上传顺序一致；`replacement_image` 保持为首图投影不变。数组每项的 `preview_url` 指向同一形式的鉴权只读接口 `GET /api/conversations/{id}/creation-input/replacement-images/{index}`，`index` 从 1 开始。冻结的请求指纹回执 `input_receipt.version` 单图（含无参考图）为 `1`，与线上 v1 逐位兼容；多图为 `2`，并携带全部参考图的哈希回执。

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
- 成片可用时 `has_video=true`，成片读取接口固定为 `GET /api/conversations/{id}/files/generated.mp4`。

## 5. 最小错误合同

错误结构固定为 `{"detail":{"code","message","field"}}`。前端需要以下稳定机器码：

| HTTP | code | 条件 |
| --- | --- | --- |
| 400 | `invalid_client_request_id` | 幂等键为空或超过 64 个字符 |
| 400 | `source_exactly_one_required` | 视频来源不是恰好一个 |
| 409 | `client_request_id_conflict` | 同一幂等键对应不同输入 |
| 413 | `upload_too_large` | 上传的视频或参考图超过大小限制 |
| 415 | `unsupported_replacement_image` | 参考图格式不支持 |
| 422 | `invalid_creation_form` | 创建字段未知、重复，或参考图超过 4 张 |
| 422 | `invalid_generation_request` | JSON 结构或固定配置错误 |
| 422 | `target_language_required` | 最终语种为空 |
| 422 | `replacement_pair_required` | 参考图与说明未成对提供 |

校验失败不得创建项目或调用付费服务。

## 6. 联调验收

- 默认 `reference_url` 能直接创建，切换上传后 `file` 能创建。
- 请求与持久化结果均不包含用户 `script`。
- Agent 生成的台词满足 `target_language`，且不要求中途确认。
- 三个隐藏处理项始终为 `true`。
- 参考图与说明能一起冻结并进入图片优化链路。
- list/detail 返回同一项目级百分比，前端只显示一条进度条。
- 最终文件未就绪前不返回 `has_video=true` 或 `100%`。
