# Duet AD 1.0 code release

This branch contains source snapshots of the versions running before service retirement. Each annotated tag identifies the commit at which a snapshot was added. See the README inside each directory for setup and architecture.

| Running units | Snapshot | Source commit | Tag |
| --- | --- | --- | --- |
| Production 3211/3212 and development 3252 | `production/` | `duet-ad1-h3-max-r2v@b387bde7023055b2df76778a0bc6144ef3bbbc46` | `release-1.0-production` |
| Seedance 3271/3272 | `seedance/` | `duet-ad1-h3-max-r2v@4796493490d06f83938eb2277f33142838f68578` | `release-1.0-seedance` |
| Hypit workers and studio | `hypit/` | `hypit@1df3413091e9b5f95e30864277ad976187578441` | `release-1.0-hypit` |
| Two-pass experiment (not running) | `two-pass/` | `duet-ad1-h3-max-r2v@929facd3476aa59b64c7245e2afc481491519fe5` | none |

Captured on 2026-10-15 UTC. Only code, configuration templates, tests, and source documentation are included. Runtime data, verification media, model binaries, credentials, generated archives, Hypit examples and sample projects are excluded. The original source commit IDs are provenance references; this branch intentionally does not contain their Git history.
