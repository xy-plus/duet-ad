# 极简前端配置回填：后端补充合同

## 目标

用户刷新页面、切换项目或换浏览器后，创建配置窗口必须展示该项目创建时冻结的真实配置，不能显示 HTML 默认值或上一个项目的草稿。

现有项目详情中的 `effective_request` 已足够恢复画幅、清晰度、最终语种和替换说明。缺少的是原视频来源信息和参考图展示信息。

## 最小接口改动

不新增写接口。创建项目时已经接收全部输入，只需将以下只读快照与 `effective_request`、`input_receipt` 原子保存，并在现有两个响应中返回同一个 `creation_input` 对象：

- `POST /api/conversations` 的成功响应
- `GET /api/conversations/{id}` 的详情响应

链接来源：

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

上传来源并使用参考图：

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

## 字段约束

- `version`：固定为整数 `1`。
- `source.mode`：只能是 `link` 或 `upload`。
- `link`：必须返回创建时的完整 `reference_url`。
- `upload`：必须返回原始 `filename` 和实际 `bytes`，不能返回服务器文件系统路径。
- `replacement_image`：未提供参考图时必须为 `null`；提供时返回原始文件名、实际字节数、服务端识别的媒体类型和稳定的鉴权预览 URL。
- `preview_url`：只允许读取该项目已经冻结的参考图，不接受路径参数，不允许目录穿越，也不能用于覆盖或重新提交素材。
- `creation_input`：创建成功后不可修改；其内容必须与 `input_receipt` 指向的冻结素材一致。
- 所有响应继续使用现有 Bearer 鉴权和项目访问边界。

如果安全策略不允许返回原始链接，后端必须明确返回 `reference_url: null`；此时前端只能显示“链接来源已提交”，无法做到原值回填，不能用项目标题猜测 URL。

## 兼容策略

- 新项目：必须返回完整 `creation_input`。
- 历史项目：没有该快照时可返回 `creation_input: null`，前端展示“历史项目未保存来源信息”，但仍使用 `effective_request` 恢复其他配置。
- 不要改变 `effective_request` 的结构或哈希；`creation_input` 是输入展示快照，不参与生成语义。
- 不要把内部字段（例如 `_minimal_replacement_image_path`）直接暴露给前端。

## 前端使用方式

前端恢复配置时按以下优先级读取：

1. `effective_request`：画幅、清晰度、语种、固定处理选项、替换说明。
2. `creation_input`：来源模式、原链接或上传文件信息、参考图信息及预览。
3. 同一浏览器本次会话的临时快照：仅用于后端尚未提供 `creation_input` 时保持刷新与项目切换体验。

轮询请求只更新项目状态，不应反复覆盖用户正在查看或编辑的配置；仅在用户主动打开或切换项目时回填。

既有项目的回填配置只用于查看，不能直接再次提交。用户必须点击“新建项目”进入可编辑状态，避免恢复出的原链接被误提交为新的付费任务。

## 验收用例

1. 链接创建后刷新：链接、画幅、清晰度、语种和替换说明完全一致。
2. 上传创建后刷新：显示原文件名和大小，原视频仍由现有 `source.mp4` 播放。
3. 使用参考图后刷新：显示原文件名和真实预览；未使用时严格返回 `null`。
4. A、B 项目使用不同配置，连续切换时各自配置不串用。
5. 换浏览器登录后，不依赖本地缓存也能恢复全部只读配置。
6. 历史项目缺少快照时明确降级，不猜测来源，也不把缺失值当作默认配置。
7. 打开既有项目时创建按钮不可用；点击“新建项目”后才恢复可编辑、可提交状态。

## 与三阶段进度的关系

三阶段文案完全由现有详情字段在前端映射，不需要新增状态接口或字段。
