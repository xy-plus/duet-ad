# Image mask gateway

`app.image_masks` is the provider-neutral boundary for receipt-bound person
masks. It performs no implicit network access: callers inject a provider
adapter implementing `submit(ProviderMaskRequest)`, `get(task_id)`, and
`download(result_url)`.

The only supported purposes are `person` and
`protected_non_target_people`. A person mask must never be relabelled as a
scene mask. Missing scene masks are handled by the remote SAM2 path; there is no
full-frame or bounding-box fallback here.

## Frozen request

Before the provider POST, the attempt receipt freezes:

- provider, action, and model;
- purpose;
- project-relative source path, source SHA-256, width, and height;
- canonical frame PTS;
- provider params and their enclosing request SHA-256;
- cache version.

The request SHA binds all provider/cache semantics. Reusing the same receipt
with any changed value fails closed as `mask_receipt_mismatch`.

## Recovery states

The successful path is:

```text
prepared -> submitting -> accepted | response_received
accepted -> GET only -> response_received
response_received -> downloaded -> validated -> succeeded
```

`submitting` is persisted and fsynced before POST. A POST timeout or crash
without a task ID becomes `submission_unknown` and is never submitted again.
An accepted task is resumed only through `get(task_id)`. A result URL is
persisted as private receipt state and immediately downloaded to the private
landing file before validation.

All artifact paths are project-relative. Reads reject symlinks, traversal, and
non-regular files. Writes use no-follow directory traversal, a same-directory
temporary file, file `fsync`, atomic replacement, and directory `fsync`.

## Validation and consumer DTO

Provider output must be a decodable PNG with the exact source dimensions and
four channels. Its alpha support must contain at least one foreground pixel
and at least one transparent pixel. The gateway does not resize, synthesize,
or replace an invalid mask.

`MaskResult.producer_receipt` is a plain mapping intended for
`app.image_quality.mask_manifest_receipt` without importing that module. Its
schema is `duet.image-mask-producer` version 1:

```text
producer { provider, action, model }
purpose
source { path, sha256, width, height, frame_pts }
request_sha256
params
cache_version
mask {
  path, sha256, size, width, height, mime_type,
  alpha_nonzero_pixels, alpha_transparent_pixels
}
```

The first adapter is `AliyunVIAPISegmentHDBody`. It normalizes the official
`RequestId` and `Data.ImageURL` response while keeping the request client and
result downloader injected. The provider documents that `SegmentHDBody`
returns a same-size four-channel PNG via a temporary URL that expires after 30
minutes: <https://help.aliyun.com/zh/viapi/developer-reference/api-high-definition-human-body-segmentation/>.
