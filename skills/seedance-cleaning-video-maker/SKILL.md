---
name: seedance-cleaning-video-maker
description: Analyze a clean, user-owned, licensed, or generic MP4, MOV, or WebM reference of a cleaning, before-and-after, product-demo, or short-form UGC video; compress it to a coherent 15-second structure; extract up to nine ordered keyframes while avoiding subtitles; write a frame-bound Seedance prompt; prepare, confirm, submit, poll, download, and immediately return one unmodified Volcengine Ark Seedance 2.0 result without post-generation inspection. Use when the user wants to reproduce a clean reference video, shorten a long montage for Seedance, choose alternate viral moments, or generate a Seedance package and original Ark video through the API. This temporary version does not remove, replace, blur, or pixelate third-party characters, trademarks, logos, or copyrighted artwork.
---

# Seedance Cleaning Video Maker

Turn a reference cleaning video into a complete, reviewable 15-second Seedance package. Base every description on inspected frames. Never infer scenes from filenames.

## Read the relevant references

- Read [references/prompt-and-compression.md](references/prompt-and-compression.md) before selecting final frames or drafting the prompt.
- Read [references/proven-patterns.md](references/proven-patterns.md) when deciding between one, two, or three scenes.
- Read [references/ark-api.md](references/ark-api.md) before building, submitting, polling, or diagnosing an API request.
- Read [references/qa-checklist.md](references/qa-checklist.md) before Phase A handoff and before Phase B submission.

## Follow this workflow

### 1. Inspect the source

Treat the source video as read-only. Use this temporary skill version only for material the user represents as user-owned, licensed, generic, or otherwise free of recognizable third-party characters, trademarks, logos, and copyrighted artwork. If ordinary inspection reveals an obvious third-party element, stop and request a clean source; do not edit, blur, pixelate, replace, or sanitize it in this version.

Confirm the source opens, then record duration, dimensions, orientation, frame rate, frame count, file size, and audio presence when the available probe can detect it.

Use native video tools when present. Otherwise run:

```powershell
python scripts/extract_keyframes.py <video> --out-dir <work-dir> --sample-count 40
```

Inspect the dense contact sheet. For rapid montages, increase sampling to 60–90 frames or extract exact times around ambiguous cuts. Do not select a frame until it has been visually inspected.

### 2. Compress to a coherent 15 seconds

Identify shot boundaries and write a source timeline before deciding what to keep. Prefer a complete causal loop:

`before state → tool enters → peak application → reaction → wipe/removal → final result`

Remove scenes that are subtitle-heavy, watermarked, motion-blurred, transition-blended, redundant, too short to show a result, or unrelated to the strongest visual story.

Choose one structure:

- one scene: use nine ordered action states and no cuts;
- two related scenes: allocate about five frames to the first and four to the second, with one direct cut;
- three high-contrast scenes: allocate three frames per scene, with two direct cuts and a complete mini-loop in every scene.

Do not exceed three scenes in 15 seconds. Prefer one or two scenes unless three subjects each have an exceptionally clear before/action/after contrast.

### 3. Select and export the final frames

When the user requests nine frames, deliver and submit exactly nine. Assign one dominant action or state to each image. Keep filenames in chronological submission order.

Prefer sharp, stable frames with visible product geometry, minimal hand occlusion, and no captions. Avoid choosing only the attractive result; every important cause must have a preceding frame.

Export exact PNGs:

```powershell
python scripts/extract_keyframes.py <video> `
  --out-dir <package>/keyframes `
  --times "0.33,1.66,2.33,3.55,4.15,4.55,7.66,8.32,8.85" `
  --prefix keyframe --columns 3
```

Move the generated `contact_sheet.jpg` and `manifest.json` to the package root. Preserve a dense inspection sheet separately.

### 4. Write the prompt

Write in the user's language. Use this order:

1. output duration, ratio, resolution, and capture style;
2. exact roles for 图片1 through 图片9;
3. invariant subject, product, environment, and camera details;
4. timestamped actions summing to 15 seconds;
5. physical cause and effect;
6. continuity constraints;
7. a short avoid list.

Make every image animate one main state or action. State that foam originates at the nozzle, marks appear only after contact, and clean regions advance only behind the wiping tool. Do not rely on negative prompts to repair contradictory positive instructions.

Avoid asking the model to synthesize subtitles or small packaging text. Add exact text in post-production only when the user explicitly requests it.

### 5. Create the Phase A package

Create a new versioned folder instead of overwriting an earlier generation:

```text
<video-stem>_seedance_package/
  keyframes/
  contact_sheet.jpg
  inspection_contact_sheet.jpg
  manifest.json
  shot_timeline.md
  seedance_prompt.txt
  api_request.json
  api_submission.md
```

Write `shot_timeline.md` with the source metadata, kept and removed scenes, exact frame roles, and final 15-second timing. Mark `api_submission.md` as `PENDING_USER_CONFIRMATION`.

Build a no-cost dry run:

```powershell
python scripts/seedance_task.py create `
  --prompt-file <package>/seedance_prompt.txt `
  --ref-images <image-1> <image-2> <image-3> <image-4> <image-5> <image-6> <image-7> <image-8> <image-9> `
  --model doubao-seedance-2-0-260128 `
  --ratio 9:16 --duration 15 --resolution 720p `
  --generate-audio --no-watermark `
  --payload-out <package>/api_request.json --dry-run
```

Verify one text item, the intended ordered image count, exact prompt round-trip, parameters, and absence of credentials. Present the files and ask for a separate explicit confirmation. End the turn without contacting Ark.

### 6. Submit only after a new confirmation

Treat the preparation request as Phase A only, even when it says “one click” or “submit.” Require a later user message such as “确认提交” after the latest dry run. Any material change to the prompt, references, model, duration, ratio, resolution, audio setting, watermark setting, or task count invalidates the confirmation.

For one confirmed task, read the key only from `ARK_API_KEY` and run:

```powershell
python scripts/seedance_task.py create `
  --prompt-file <package>/seedance_prompt.txt `
  --ref-images <image-1> <image-2> <image-3> <image-4> <image-5> <image-6> <image-7> <image-8> <image-9> `
  --model doubao-seedance-2-0-260128 `
  --ratio 9:16 --duration 15 --resolution 720p `
  --generate-audio --no-watermark --confirm-submit `
  --wait --state-file <package>/task.json `
  --download <package>/generated.mp4
```

Never accept, print, or store the API key in chat, arguments, Markdown, JSON, logs, or source files. One confirmation authorizes exactly one reviewed task. Do not retry a failed live creation as a second paid task without new authorization.

### 7. Download and immediately deliver the original result

After Ark reports `succeeded`, immediately download the temporary result URL to `<package>/generated.mp4`, then return that file to the user.

Do not open, play, probe, inspect, screenshot, sample, OCR, extract frames from, generate a contact sheet for, or otherwise analyze the generated video. Do not perform post-generation visual QA, subtitle checks, scene checks, media-metadata checks, or hash calculations. The successful Ark status plus completed download is the terminal condition for delivery.

Preserve the downloaded `generated.mp4` byte-for-byte as the Ark/Seedance original. Do not blur, pixelate, mask, crop, trim, edit, re-encode, remux, or otherwise post-process it unless the user gives a separate explicit instruction after receiving the original. If a separately requested derivative is later produced, keep it under a different filename and never present it as the Seedance original.

Update `api_submission.md` only with fields already returned by the task response, such as task ID, terminal status, requested duration, ratio, resolution, frame rate, seed, audio setting, and downloaded filename. Do not read the video to obtain additional fields. Deliver `generated.mp4` first and include only a concise task status afterward; generated contact sheets and post-generation review reports are not part of the final handoff.

## Handle failure safely

- For authentication errors, verify that `ARK_API_KEY` exists and is authorized; never request the key in chat.
- For request errors, inspect reference count, roles, media types, duration, model, and request size.
- For moderation rejection, isolate the first failing prompt or reference. If an obvious third-party element is involved, stop and request a clean source because this version contains no removal or sanitation workflow. Do not promise a bypass.
- For a running task interrupted locally, resume with `seedance_task.py status <task-id> --wait` rather than creating another task.

## Final constraints

- Submit no more than nine images in one task.
- Keep 图片1…图片9 aligned with API content order.
- Keep cause before effect and untouched regions unchanged.
- Keep the deliverable package versioned and reproducible.
- Return the exact Ark-downloaded `generated.mp4` as the primary video deliverable.
- Return the successful download immediately without inspecting or sampling the generated video.
- Never post-process a Seedance result without a separate explicit user request.
- Never install or modify this skill unless the user explicitly requests installation or edits.
