# Legacy 3213 Phase 1

## Scope

This change keeps the current `web/` implementation and adds a read-only,
server-state-driven operation view. It does not change backend state,
provider submission, billing, or recovery behavior.

The page presents one project operation through these stages:

1. Source A
2. Analysis
3. Material index
4. Image processing
5. Prompt Fusion
6. Context IR
7. H3 generation
8. Stitching
9. committed output B

Only public GET response fields may determine a stage. Missing element-index or
Context IR segment detail is shown as unavailable instead of being inferred
from prompts or local files.

## Interaction model

- A sticky operation header owns current stage, truthful counts, project
  elapsed time, last server update, and the whole timeline.
- The history rail is enriched by bounded-concurrency detail GETs. It shows a
  short CID, first published keyframe when available, duration, server update
  time, current stage, and output-B state.
- Skill identity uses the CID-frozen milestone. Short hashes are primary;
  complete hashes remain in an expandable audit view.
- Material cards use a public element index when one exists. With the current
  API they show public frame/segment facts and explicitly state that element
  detail is not exposed. Segment joins are mapped to product language.
- Errors have a safe summary. Raw server text is available only in a collapsed,
  length-limited diagnostic block.
- Existing mutation actions remain unchanged. Labels distinguish new paid
  work, same-operation resume, local stitch retry, and ambiguous submission.

## Baseline captured before publication

- Capture time: `2026-08-30T13:37:50+0800`
- 3211/3212 loaded source:
  `/home/xy/duet-ad1/.worktree/good-preview-skills-r1`
- 3211/3212 source HEAD:
  `7a02e94f152f5584065c2135afe3a176adf32ea5`
- 3212 data directory:
  `/home/xy/duet-ad1/data/test-instances/three-skill-preview-3211/data`
- 3213 source worktree:
  `/home/xy/duet-ad1/.worktree/stable-audio-milestone-r1`
- 3213 source HEAD:
  `fcdd0691bdc3595f0a8262a424a5b90032fbe427`
- 3213 static root before publication:
  `/home/xy/duet-ad1/.worktree/stable-audio-milestone-r1/web-next/dist`
- 3213 `index.html` SHA-256 before publication:
  `b88baf1cec96cc836edffbab9ba7074901185cccd82cfff11b7870617d8b86b8`
- Caddy config:
  `/home/xy/duet-ad1/.deploy/caddy/config.json`
- Caddy config SHA-256 before publication:
  `f2b9017594fb5757a4c18cd01e11a65fd5255e5b68e4fbd020443988cdb9a350`
- Caddy service PID before publication: `3525693`
- duet-ad1 service PID before publication: `3903368`

## Publication and rollback contract

Publication must stage an immutable copy of `web/`, validate it, change only
the `srv3213` static root, and reload Caddy without restarting systemd units.

- Release directory:
  `/home/xy/duet-ad1/.deploy/legacy-3213/releases/20260830T060452Z`
- Pre-change Caddy backup:
  `/home/xy/duet-ad1/.deploy/caddy/config.pre-legacy-3213-20260830T060452Z.json`
- One-command rollback:
  `/home/xy/duet-ad1/.deploy/legacy-3213/rollback-to-web-next-b88baf1.sh`

The rollback script restores the exact pre-change config, validates it, reloads
Caddy, and GET-verifies ports 3211 and 3213. It does not restart Caddy or the
duet-ad1 service.

## Verification before publication

- `node --check` and `git diff --check`: passed.
- All legacy web tests: 104 passed.
- Headless Chromium at 1440 px and 390 px: passed. The mocked GET-only detail
  showed nine stages, current Context IR, a short CID, duration, server update,
  output-B state, Skill v3 short identity, and the explicit unavailable element
  index. There were no browser console errors; the mobile drawer was off-canvas.
- Repository-wide `pytest -q` was stopped after the shared suite made no
  progress past 23 percent. Its first failure,
  `tests/test_create_guards.py::test_same_client_request_id_dedup`, reproduces
  unchanged on the baseline worktree and is outside this static frontend change.
