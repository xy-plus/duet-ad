# fal SAM 3.1 multi-target calibration (2026-09-16)

- Source video: `/home/xy/duet-ad1-h3-max-r2v-release/data/projects/26f64ded712348a0a07671a8459e0a82/work/segments/0004/source.mp4`
- Source duration (ffprobe): `8.423023` seconds
- Source SHA-256: `1ede236aaa3e0c49312f14c1f95430cf11455e38e72bd5dd15da5e9759201d84`
- Prompt sent verbatim: `face, necklace`
- Queue endpoint: `https://queue.fal.run/fal-ai/sam-3-1/video-rle`
- Model: `video-rle`
- Detection threshold: `0.3`
- Upload content type: `video/mp4`
- FAL proxy: configured (value deliberately not recorded)
- HTTP retry policy for this calibration: disabled; each HTTP request had one attempt.
- Queue submissions: `1`
- Request ID: `01a0a87d-fbf6-71e0-8de5-415f132cb022`
- Elapsed wall time (upload through result fetch): `53.745` seconds
- Complete unmodified result payload: `result.json`

## Observed multi-target response shape

- Top-level fields: `rle`, `boxes`, `metadata`, `scores`, and
  `boundingbox_frames_zip`.
- `rle`, `boxes`, and `metadata` are same-length, frame-indexed flat lists
  (252 entries each). Every observed frame has exactly one non-empty RLE
  string, one normalized `cxcywh` box, and one metadata object.
- Every metadata object is exactly `{index, score, box}`; `index` equals its
  list position, `score` is null, and `box` exactly equals `boxes[index]`.
- There is no label, target ID, nested object list, or other field that maps a
  mask/box to `face` versus `necklace`. This response therefore cannot be
  split into independently labelled face and necklace detections.

No credentials or HTTP exchange records are stored here, because those records contain
an Authorization header. `result.json` is the exact JSON object returned by the
result endpoint.
