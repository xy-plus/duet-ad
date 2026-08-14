# Volcengine Ark Seedance task reference

Verify these values against current official documentation if the API or model may have changed.

- Base URL: `https://ark.cn-beijing.volces.com/api/v3`
- Create: `POST /contents/generations/tasks`
- Status: `GET /contents/generations/tasks/{task_id}`
- Model used by the three successful runs: `doubao-seedance-2-0-260128`
- Authentication: `Authorization: Bearer $ARK_API_KEY`

Official references:

- https://developer.volcengine.com/articles/7641782568258306102
- https://developer.volcengine.com/articles/7606009619928449070
- https://www.volcengine.com/docs/82379/1795150

## Request body

```json
{
  "model": "doubao-seedance-2-0-260128",
  "content": [
    {"type": "text", "text": "<prompt>"},
    {
      "type": "image_url",
      "image_url": {"url": "<https, asset URI, or data URI>"},
      "role": "reference_image"
    }
  ],
  "generate_audio": true,
  "ratio": "9:16",
  "duration": 15,
  "resolution": "720p",
  "watermark": false
}
```

The content array order defines 图片1, 图片2, and so on. Preserve chronological reference order.

## Limits used by this workflow

- Submit at most nine image references in one task.
- Keep output duration within 4–15 seconds.
- Use local images only when request size is practical; encode them as data URIs.
- Prefer reachable HTTPS or trusted `asset://` inputs for large video or audio references.

## Mandatory two-phase gate

Phase A performs a dry run and writes `api_request.json`. It must not read a key or contact Ark. Mark the summary `PENDING_USER_CONFIRMATION` and ask the user to confirm.

Phase B starts only after a new, explicit confirmation for the unchanged payload. Require `--confirm-submit` mechanically. Read the key only from `ARK_API_KEY`. Never write or print its value.

A live creation may incur charges. One confirmation permits one task. A material input change or a second task requires another dry run and confirmation.

## Resume instead of duplicate

If local polling stops after task creation, read the saved task ID and use the status command. Do not create a replacement task merely because polling was interrupted. Download the successful result immediately because signed URLs expire.
