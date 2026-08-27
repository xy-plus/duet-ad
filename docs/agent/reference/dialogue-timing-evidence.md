# Dialogue timing evidence gate

Status: the pre-H3 on-screen timing gate and sampled visibility producer are
wired into production. A new on-screen project without evidence returns
refresh-required and schedules the producer; it is never inferred valid. The
post-H3 validator remains `PREPARED_ONLY` research.

## Before H3

`work/speaker_timing.json` is required only when authoritative dialogue contains
an `on_screen` line. `none` and purely off-screen projects keep their original
path and skip probing/extraction. Historical v2 receipts remain readable; a new
paid on-screen request from such a receipt is refresh-required.
`h3_multimodal_source.json`, `multimodal_input.json`, and
`speaker_timing_production.json` bind the exact artifacts. The producer
enumerates exact decoded PTS, deterministically samples real frames at 8 fps
with one CPU worker, and sends only those frames, ordered contact sheets, and
frozen person identity references to the strict video-maker Skill phase.
Dialogue text and dialogue windows are not Skill inputs. The validator binds:

- the exact source-video SHA-256;
- the exact source duration/time base and complete decoded-PTS inventory hash;
- every ordered sampled-frame and identity-reference SHA-256 and integer PTS;
- the scene-cut artifact, sampling algorithm/version, cadence, and maximum
  unobserved sample gap;
- an integer time base and verified half-open `lip_verifiable` windows;
- evidence keyframes that belong to each window.

Only consecutive samples where the mapped person is both visible and
lip-verifiable may form a window. Unknown samples and cuts split windows, and
one sample interval is removed from each end. This is sampled evidence, not a
claim that every decoded frame was inspected. Ambiguous subject-to-person
mapping fails closed.

The project scheduler uses the same gate for a short project and for every
long-video segment. Each segment binds its own source bytes while PERSON
identity references remain bound to the exact segment/frame selected by the
project-level continuity receipt. A queued job is restartable only while those
bindings still match. A crashed running Skill job is adopted only when its
receipt-bound input, frozen Skill, and exact output are present; otherwise it
becomes `submission_unknown` and startup never reruns it.

For every `on_screen` line, the full authoritative dialogue interval must be a
subset of one verified window for the bound subject. Picture references and ASR
timestamps are never accepted as speaker-visibility proof. Failure occurs in
`h3_project.build_request_from_parts` before an H3 attempt can be created.
Factory output, Context-IR boundaries, and H3 paid/read boundaries revalidate
the same frozen on-screen-dialogue digest. Immediately before every H3
paid/read boundary, the producer receipt and its raw/sample evidence are
reloaded and compared with the pre-Context frozen receipt. Validation-cache
fingerprints include the production receipt, producer input, raw output,
frozen Skill, every sampled frame/contact sheet/identity reference/cut source,
their expected hashes, and their current raw-byte hashes.

## After H3

The standalone validator accepts an independently produced
`dialogue-av-acceptance.json` in the H3 work directory. The
`duet.dialogue-av-acceptance` v1 receipt binds the exact H3 output bytes, media
timeline, authoritative dialogue receipt, and speaker-timing receipt. It must
contain:

- complete ordered ASR intervals for all on-screen lines;
- zero unmatched speech intervals;
- independently verified lip windows for the bound subjects;
- fixed analyzer/model/result hashes;
- the fixed 250 ms maximum ASR boundary drift.

ASR start and end boundaries are compared independently with the authoritative
window. The verified lip window must cover that full authoritative interval.
No production stitch, reuse, or long-generation path consumes this artifact yet
because no real ASR/lip producer exists. It is not a publication guarantee and
does not add an optional field to stitch/provider receipts. No provider is
called by either validator.
