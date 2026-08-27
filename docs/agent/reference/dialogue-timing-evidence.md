# Dialogue timing evidence gate

Status: `PREPARED_ONLY`. The repository validates both receipts but does not
produce speaker-visibility, ASR, or lip-analysis evidence. Missing evidence is
therefore a closed result, never an inferred pass.

## Before H3

`work/speaker_timing.json` is a required `duet.speaker-timing` v1 artifact for
H3 multimodal projects. `h3_multimodal_source.json` and
`multimodal_input.json` bind its exact bytes. The validator additionally binds:

- the exact source-video SHA-256;
- every ordered H3 keyframe SHA-256 and its integer PTS;
- an integer time base and verified half-open `lip_verifiable` windows;
- evidence keyframes that belong to each window.

For every `on_screen` line, the full authoritative dialogue interval must be a
subset of one verified window for the bound subject. Picture references and ASR
timestamps are never accepted as speaker-visibility proof. Failure occurs in
`h3_project.build_request_from_parts` before an H3 attempt can be created.

## After H3

An independent analyzer must write
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
The receipt byte hash is included in the stitch receipt and is recomputed by
short- and long-video reuse validation. Missing, changed, unverified, early, or
out-of-tolerance evidence rejects publication/reuse. No provider is called by
these validators.
