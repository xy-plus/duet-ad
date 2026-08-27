# Image mask gateway

`app.image_masks` is the provider-neutral boundary for receipt-bound person
masks. It performs no implicit network access: callers inject a provider
adapter implementing `submit(ProviderMaskRequest)`, `get(task_id)`, and
`download(result_url)`.

The gateway vocabulary contains `person` and `protected_non_target_people`, but
each provider capability narrows that set. A person mask must never be
relabelled as a scene mask. Missing scene masks are handled by the remote SAM2
path; there is no full-frame or bounding-box fallback here.

## Frozen request

Before the provider POST, the version 2 attempt receipt freezes:

- provider, action, and model;
- purpose;
- provider mask scope, identity-binding mechanism, person-ID parameter, and
  supported purposes;
- selected person ID, the complete visible-person roster, and the SHA-256 of
  the upstream roster receipt;
- project-relative source path, source SHA-256, width, and height;
- canonical frame PTS;
- provider params and their enclosing request SHA-256;
- cache version.

The request SHA binds request schema/version and all provider, capability,
person-instance, source, frame, parameter, and cache semantics. Reusing the
same receipt with any changed value fails closed as `mask_receipt_mismatch`.

`PersonInstanceRequest` is mandatory. `visible_person_ids` is canonicalized as
a unique sorted roster, must contain `person_id`, and is bound to a
project-relative `person_roster_receipt_path` plus its raw-byte SHA-256. Before
POST, the gateway reads that file without following symlinks and requires this
exact `duet.person-roster` version 1 content:

```text
source { path, sha256, width, height, frame_pts }
person_ids
```

The source fields must exactly match the current frozen image/frame, and
`person_ids` must exactly match `visible_person_ids`. The authoritative upstream
frame-inventory/person-detection layer owns this receipt. Downstream consumers
must match both its path and SHA rather than treating producer strings as
independent detector evidence.

## Provider identity capability

Providers declare one of two closed contracts:

- `all_people_union` + `sole_visible_person`: no person-ID provider parameter,
  only `purpose=person`, and the frozen roster must equal `[person_id]`.
- `person_instance` + `provider_person_id`: a declared `person_id_param` is
  mandatory, and the exact provider parameter value must equal `person_id`.

`AliyunVIAPISegmentHDBody` declares the first contract because its response is
the union of all people. It therefore supports only a single-person target POC.
Multi-person frames and `protected_non_target_people` fail before receipt
creation or POST. The same union output cannot be copied into multiple
person-ID receipts.

A future instance segmentation adapter must declare the second contract. Its
provider request, request SHA, and producer receipt then bind the selected
person ID independently. Cross-artifact quality validation must still reject
duplicate/overlapping instance outputs; provider capability proves distinct
requests, not that a provider returned correct pixels.

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
schema is `duet.image-mask-producer` version 2:

```text
producer { provider, action, model }
purpose
provider_capability {
  mask_scope, identity_binding, person_id_param, supported_purposes
}
person_instance {
  person_id, visible_person_ids, person_roster_receipt_path,
  person_roster_receipt_sha256, provider_person_id
}
source { path, sha256, width, height, frame_pts }
request_sha256
params
cache_version
mask {
  path, sha256, size, width, height, mime_type,
  alpha_nonzero_pixels, alpha_transparent_pixels
}
```

### Consumer loader

Consumers must not duplicate producer schema, path, request-SHA, roster, or PNG
validation. The public boundary is:

```python
def load_validated_mask(
    project_root: Path,
    artifact: Mapping[str, Any] | str | Path,
    *,
    expected_source: MaskSourceExpectation,
    expected_person_id: str,
    expected_visible_person_ids: tuple[str, ...],
    expected_roster: MaskRosterExpectation,
    expected_purpose: MaskPurpose,
) -> LoadedMaskArtifact
```

`artifact` may be an in-memory producer mapping or a project-relative path to
the producer/succeeded-attempt receipt. The loader independently revalidates:

- exact producer v2 schema and provider capability;
- the complete request SHA and expected purpose/person identity;
- source bytes, hash, dimensions, path, and frame PTS;
- authoritative roster bytes, hash, source binding, and person list;
- mask containment/nofollow, complete PNG structure, byte hash, dimensions,
  alpha support, and canonical producer metadata.

`LoadedMaskArtifact` is frozen. `canonical_receipt` is immutable canonical JSON
bytes and may contain private provider parameters, so it must not be logged.
`packed_mask` is an immutable row-major boolean mask encoded as
`numpy.packbits(alpha > 0, bitorder="little")`; its exact encoding identifier is
`row-major-alpha-gt-zero-packbits-little-v1`.

The first adapter is `AliyunVIAPISegmentHDBody`. It normalizes the official
`RequestId` and `Data.ImageURL` response while keeping the request client and
result downloader injected. The provider documents that `SegmentHDBody`
returns a same-size four-channel PNG via a temporary URL that expires after 30
minutes: <https://help.aliyun.com/zh/viapi/developer-reference/api-high-definition-human-body-segmentation/>.
