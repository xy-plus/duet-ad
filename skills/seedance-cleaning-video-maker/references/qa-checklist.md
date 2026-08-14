# QA checklist

## Before Phase A handoff

- Source video was visually inspected, not inferred from its filename.
- Dense sampling covers every cut and action state.
- Selected keyframe count matches the request and does not exceed nine.
- Filenames sort in API submission order.
- No selected frame is a transition blend or severely blurred.
- The source is within this temporary version's clean-reference scope: user-owned, licensed, generic, or otherwise free of obvious third-party characters, trademarks, logos, and copyrighted artwork.
- If ordinary inspection exposed an obvious third-party element, work stopped and a clean source was requested; no masking or replacement was attempted.
- Subtitles and platform watermarks are avoided where possible.
- Each frame has one dominant role and maps to a prompt event.
- Every retained scene has a complete before/action/after loop.
- Timeline intervals are contiguous and sum to 15.0 seconds.
- Cause precedes effect; untouched regions do not change early.
- 图片1 through 图片9 match API content order.
- Dry run succeeds with the reviewed model, ratio, duration, resolution, audio, and watermark settings.
- Request JSON contains no API key, authorization header, or credential value.
- `api_submission.md` is marked `PENDING_USER_CONFIRMATION`.
- The turn ends without a live API request.

## Before Phase B submission

- A new user message explicitly confirms the latest unchanged dry run.
- No prompt, image, order, model, duration, ratio, resolution, audio, watermark, or task-count setting changed after confirmation.
- `ARK_API_KEY` exists in the environment without printing it.
- Live command contains `--confirm-submit`.
- Exactly one reviewed task will be created.

## After generation

- Task reached `succeeded`, `failed`, `cancelled`, or `expired`.
- A successful output was downloaded immediately to `generated.mp4`.
- No post-generation playback, probing, frame extraction, screenshotting, OCR, contact-sheet creation, visual inspection, subtitle check, scene check, metadata check, or hash calculation was performed.
- The Ark-downloaded `generated.mp4` was preserved byte-for-byte.
- No masking, cropping, trimming, editing, remuxing, or re-encoding was applied without a separate explicit user request.
- `api_submission.md` contains only task-response fields and the downloaded filename; it does not claim visual verification.
- `generated.mp4` is returned immediately as the primary deliverable, followed only by a concise task status.
- Final links point to user-facing deliverables only.
