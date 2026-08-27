# Dual-target three-frame POC

`app.dual_target_poc` is an isolated paid Seedream probe. It does not update
`meta.json`, `work/postprocessed`, or any production generation receipt.

It requires a canonical eligible v2 `_image_continuity` plan and an already
published `duet.replacement-packs` receipt for the same plan, source inventory,
model, revision, person IDs, and scene IDs. Source frames are edit canvases;
only the published primary/alternate target identity and scene references are
sent as references. A frame with no observable main person receives no person
reference and still receives the frozen scene references.

The runner selects first/middle/last frames by default and never submits more
than three. Explicit positions use `--frame SEGMENT:FRAME`. Every paid request
has a durable Seedream v2 attempt receipt containing exact prompt, ordered input
roles and hashes, plan/profile/revision/model, and replacement-pack candidate
SHA. `submission_unknown` and deterministic failure are terminal and are never
automatically resubmitted; only the Seedream adapter's exact quota response may
use its bounded retry policy.

Run from the accepted integrated checkout with absolute paths:

```bash
/home/xy/duet-ad1/.venv/bin/python -m app.dual_target_poc \
  --project-dir /home/xy/duet-ad1/data/CONVERSATION_ID \
  --evaluation-dir /home/xy/duet-ad1/data/image-skill-eval-20260827/three-frame/RUN_ID
```

The evaluation directory must be outside the project directory. A successful
run writes `run.json`, three per-frame provider attempt receipts under `work/`,
and atomically exposes images under `results/`.
