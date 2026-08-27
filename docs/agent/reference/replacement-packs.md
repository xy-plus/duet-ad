# Replacement reference packs

`app.replacement_packs` is the durable pre-frame barrier for new person and
scene references. It generates one `primary, alternate` pack per v2 plan id,
then delegates all semantic acceptance to
`app.image_quality.evaluate_reference_packs`. It has no second verifier.

## Generation boundary

Every Seedream POST uses receipt v2 and ordered images:

- primary: `current_frame` (a deterministic neutral PNG), followed by
  `source_negative:{person|scene}:{PLAN_ID}:1..N`;
- alternate: `current_frame`, the atomically stored primary as
  `target_reference:{person|scene}:{PLAN_ID}:primary`, then the same ordered
  source-negative inputs.

Source PNGs are never the edit canvas or executor-facing target references.
They supply only soft attribute, relationship, composition, lighting/layout and
negative evidence. The alternate cannot start unless primary is durably stored.
`N <= 8` is enforced before provider work so Ark's ten-image limit cannot be
crossed.

Before each paid POST, the adapter fsyncs an exact request receipt containing
prompt and hash, model/mode, upstream plan SHA, profile, integer execution
revision, entity input SHA, and every ordered input role/byte SHA. It then calls
`seedream.edit(..., execution_binding=...)`; Seedream's own attempt receipt owns
provider recovery. `submission_unknown` is returned without publishing and a
repeat call cannot blindly POST. No v1 fallback exists.

## Public API

The concrete executor entry is:

```python
result = await prepare_replacement_packs_with_seedream(
    project_dir,
    project_plan,
    settings=settings,
    revision=execution["revision"],
    quality_gate=ImageQualityPackGate(
        plan=plan,
        frame_masks=frame_masks,
        profile=quality_profile,
        semantic_verifier=semantic_verifier,
    ),
)
```

For tests, `prepare_replacement_packs(..., generator=..., quality_gate=...)`
injects an async generator without calling a provider. A v2 construction is:

```python
project_plan = ProjectReplacementPlan(
    people=tuple(
        PersonPlan(
            item["id"],
            (source_by_plan_id[item["id"]],),
            item["replacement_identity"],
            item,
        )
        for item in plan["person_plans"]
    ),
    scenes=tuple(
        ScenePlan(
            item["id"],
            (source_by_plan_id[item["id"]],),
            item["replacement_scene"],
            item,
        )
        for item in plan["scene_plans"]
    ),
    upstream_plan_sha256=execution["plan_sha256"],
    upstream_source_inventory_sha256=canonical_source_inventory_sha256(
        execution["frames"]
    ),
    execution_profile=execution["profile"],
)
```

`source_by_plan_id` values are already loaded, project-relative authoritative
source-slot PNG paths; they are not inferred from plan text.

`PackBuildResult.status` is exact:

- `ready`: `pack` is a validated `ReplacementPackDTO` and may be consumed;
- `submission_unknown`: provider acceptance is ambiguous; no pack, no quality
  gate and no downstream POST;
- `unknown`: generator/quality availability or receipt result is unknown; no
  pack;
- `failed`: deterministic generation rejection or quality `fail`; no pack.

Input/path/schema errors raise `ReplacementPackError` before a partial DTO is
returned. Executors must reject every result except `ready` with non-null
`pack`.

## DTO and loader

The quality gate consumes `ReplacementPackCandidateDTO`, whose absolute
`project_dir`, `candidate_sha256`, upstream plan/source hashes, exact execution
profile/model/revision, ordered people/scenes, sources and images are all
revalidated from the candidate and per-entity producer receipts. Each source or
image exposes absolute `path` plus frozen `relative_path`, SHA-256, byte size and
dimensions. Image roles are exactly `primary, alternate`.

Only quality `pass` with a canonical receipt bound to both
`upstream_plan_sha256` and `reference_pack_candidate_sha256` publishes
`work/replacement-packs/pack.json`. `load_replacement_pack()` re-reads bytes,
PNG decode/dimensions, hashes, non-symlink project-relative paths, producer
receipt, candidate receipt and quality receipt. Its expected upstream hashes,
profile hash, model, revision, and exact ordered person/scene ids make stale
packs fail closed.

`canonical_source_inventory_sha256(execution["frames"])` preserves frame order,
projects each frame to `segment_index`, `frame_index`, `frame_name`, and
`source_sha256`, serializes canonical compact UTF-8 JSON, then hashes it. The
executor can therefore recompute the binding without trusting this module's
receipt.
