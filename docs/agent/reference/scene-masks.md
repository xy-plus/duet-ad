# Scene mask worker contract

`app.scene_masks` is the provider-neutral boundary for remote scene-component
mask jobs. Its default backend identity is `sam2_birefnet`; the endpoint URL is
runtime configuration, while `endpoint_identity`, model, version, frozen plan
SHA, scene, ordered frame SHA/dimensions/PTS, hard-cut chain, references,
box/point prompts, and people-protection inputs are receipt-bound.

Each component has one independent propagation job per hard-cut shot. Every
component/shot pair requires an in-shot reference with a positive prompt. A
missing prompt, incomplete people-protection input, or incomplete returned
frame set fails closed. No whole-frame or bounding-box fallback exists.

The client writes and fsyncs `submitting` before POST. A POST transport failure
without a durable task ID becomes terminal `submission_unknown`; it cannot be
resent through this API. Once a task ID is stored, all later calls are GET-only.
The HTTP client is injectable, so tests use `httpx.MockTransport` without real
network access.

Successful items use this exact semantic boundary:

```text
purpose = scene_component
channel = grayscale_alpha
producer_receipt.schema = duet.scene-mask.producer
```

Every item binds its component, shot, frame SHA, request SHA, propagation-job
SHA, backend/model/version/endpoint identity, project-relative PNG path, output
SHA, byte size, and dimensions. The PNG must be a contained non-symlink regular
file with both background and foreground pixels. Producer receipts must declare
`membership_engine=sam2`, hard-cut-only propagation, BiRefNet edge refinement
only inside SAM2's uncertain edge band, and `fallback=none`. Consumers must not
treat this schema/purpose as a person-protection mask.

Worker bodies and exception text are never copied into receipt or public
errors. Only stable `SceneMaskError.code` values cross the boundary.
