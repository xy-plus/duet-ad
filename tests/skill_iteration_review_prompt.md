# Blind review instructions

You are reviewing a frozen offline experiment. Do not edit the Skill, input, artifact, oracle, evaluator, or ratings after seeing a score. Review every scheduled run, including malformed and low-quality outputs.

Use the oracle only for predeclared identities and applicability. Judge actual semantics from the frozen source images, user reference image, raw Skill outputs, compiled outputs, and Fusion input/output. For image-postprocess, raw `global_plan` and raw `segment_frames` are the primary evidence; compiled outputs are secondary evidence because backend normalization can add language that the Skill did not produce.

For every rating axis, return an integer score from 0 through 4 and at least one structured evidence item containing an artifact SHA-256, an absolute JSON Pointer, and optional stable key, segment index, and frame order. Do not provide an unbound prose opinion. Record uncertainty as a failed required observation unless the oracle marked it not applicable before the run.

Image observations:

- Verify that the user prompt and reference image both bind to exactly the oracle stable key and do not spread to another key.
- Verify every indexed person is genuinely transformed. Apply face, clothing, visible gender expression, ethnic presentation, and style checks only to the stable keys predeclared evaluable by the oracle. Do not infer real identity or ancestry. Where visible, the face must differ slightly; clothing must keep palette and style while changing cut or design.
- Count non-person candidates from indexed entities and scenes with visible occurrences. Relations and parts of a person do not count separately. Require `min(2, candidate_count)` actual replacements. A user-bound non-person counts, but when another candidate exists at least one replacement must be autonomous.
- A rewritten source description is not a replacement. Record every materially identical source/target pair in `source_target_noop_keys`.
- A selected scene must remain the same category and narrative function but be a visibly different instance.
- Camera position, framing, lens, lighting, action, pose, occlusion, and contact remain unchanged.
- The same stable key must have one transformed design across all visible frames and segments.

Fusion observations:

- Expected visual count and order come from the frozen hard-cut intervals: the first frame starts one interval, each later `hard_cut` starts another, and `continuous` stays in the current interval.
- New keyframes are the only authority for static appearance. Record any old-only or contradictory face, clothing, object, or scene fact copied from `old_video_prompt`.
- Verify all visible replacement targets from the frozen image-optimization prompts are represented without reverting to their source appearance.
- Preserve supported action phase and direction, camera motion, timing, relation subject/object roles, and visible relation state.
- Do not project scene/action/relation state across a hard cut. Stable design may remain consistent when the same stable key reappears.
- Record dialogue/audio text and stable-key/tile/relation tokens that leak into final visual strings.

The scorer, not the reviewer, computes counts, caps, totals, policy status, and comparison recommendations. These results are offline evidence only and must never change a production project status or trigger a retry.

## Exact review JSON contract

Return one JSON object and nothing else. Its top-level keys are exactly
`schema_valid`, `facts`, and `ratings`. Do not repeat oracle-owned facts in
`facts`; the report builder injects them from the frozen oracle.

Every rating is exactly:

```json
{
  "score": 0,
  "evidence": [{
    "artifact_sha256": "64 lowercase hex characters",
    "json_pointer": "/an/existing/node",
    "stable_key": null,
    "segment_index": null,
    "frame_order": null
  }]
}
```

`score` is an integer from 0 through 4. Every evidence object contains all five
keys. `json_pointer` must resolve in the artifact whose exact SHA is supplied.
Use `null` only when a stable key, segment, or frame genuinely does not apply.

Every array in `facts` contains only unique, non-empty JSON strings; never put
objects, numbers, booleans, or `null` in those arrays.  For image-postprocess,
every array item is an indexed stable key.  For video-prompt-fusion,
`missing_replacement_keys` and `inconsistent_stable_keys` also contain stable
keys.  Items in the other Fusion arrays are deterministic violation identifiers
in the form `segment-<index>/frame-<order>/<short-label>`; use `0` only when a
segment or frame genuinely does not apply.  The scorer only uses these
identifiers to count whether a violation exists; put the supporting detail in
the rating evidence pointer, not in an object inside `facts`.

For `image-postprocess`, `facts` contains exactly:

```json
{
  "replaced_people_keys": [],
  "people_demographic_style_preserved_keys": [],
  "people_face_changed_keys": [],
  "people_clothing_palette_style_preserved_cut_changed_keys": [],
  "replaced_non_person_keys": [],
  "scene_replacement_keys": [],
  "same_kind_different_scene_keys": [],
  "user_prompt_binding_keys": [],
  "user_reference_binding_keys": [],
  "source_target_noop_keys": [],
  "camera_light_lens_changed": false,
  "inconsistent_stable_keys": []
}
```

Its `ratings` keys are exactly:

- `schema_and_binding`
- `user_replacement_binding`
- `all_people_replaced`
- `person_face_identity_shift`
- `person_style_and_clothing_similarity`
- `minimum_non_person_replacements`
- `source_target_material_difference`
- `scene_same_kind_different_instance`
- `camera_light_lens_preservation`
- `cross_frame_stable_consistency`

For `video-prompt-fusion`, `facts` contains exactly:

```json
{
  "actual_visual_count": 0,
  "new_frame_contradictions": [],
  "old_static_leaks": [],
  "missing_replacement_keys": [],
  "camera_light_lens_inventions": [],
  "action_direction_conflicts": [],
  "hard_cut_projection_count": 0,
  "relation_conflicts": [],
  "audio_text_leaks": [],
  "binding_token_leaks": [],
  "inconsistent_stable_keys": []
}
```

Its `ratings` keys are exactly:

- `schema_and_order`
- `new_keyframe_static_authority`
- `old_static_fact_exclusion`
- `replacement_target_propagation`
- `action_camera_rhythm_fidelity`
- `hard_cut_non_projection`
- `relation_preservation`
- `audio_visual_separation`
- `cross_segment_stable_consistency`
