# Dialogue timing evidence gate

Status: the pre-H3 on-screen timing gate is wired into production request
construction. The repository does not produce its authoritative visibility
artifact; a new on-screen project without one is therefore refresh-required,
never inferred valid. The post-H3 validator remains `PREPARED_ONLY` research.

## Before H3

`work/speaker_timing.json` is required only when authoritative dialogue contains
an `on_screen` line. `none` and purely off-screen projects keep their original
path. Historical v2 receipts remain readable; a new paid on-screen request from
such a receipt is refresh-required. `h3_multimodal_source.json` and
`multimodal_input.json` bind its exact bytes. The validator additionally binds:

- the exact source-video SHA-256;
- the source duration, every ordered H3 keyframe SHA-256, and integer PTS;
- an integer time base and verified half-open `lip_verifiable` windows;
- evidence keyframes that belong to each window.

For every `on_screen` line, the full authoritative dialogue interval must be a
subset of one verified window for the bound subject. Picture references and ASR
timestamps are never accepted as speaker-visibility proof. Failure occurs in
`h3_project.build_request_from_parts` before an H3 attempt can be created.
Factory output, Context-IR boundaries, and H3 paid/read boundaries revalidate
the same frozen on-screen-dialogue digest. Validation-cache fingerprints include
the artifact path, expected hash, and current raw-byte hash.

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
