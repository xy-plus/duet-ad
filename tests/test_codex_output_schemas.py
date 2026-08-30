import json
import tempfile
from pathlib import Path

import jsonschema
import pytest

from app import codex_output_schemas, codex_runner
from app.codex_runner import CodexOutputValidationError, CodexRunner


_GLOBAL_KEYS = {
    "people": {"person-01"},
    "entities": set(),
    "scenes": set(),
    "relations": {"relation-01"},
}


def _assert_all_objects_closed(schema: object) -> None:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
            assert set(schema.get("required", [])) == set(schema.get("properties", {}))
        for value in schema.values():
            _assert_all_objects_closed(value)
    elif isinstance(schema, list):
        for value in schema:
            _assert_all_objects_closed(value)


def test_all_skill_output_schemas_are_valid_and_closed() -> None:
    schemas = [
        codex_output_schemas.PROJECT_INDEX_SCHEMA,
        codex_output_schemas.global_plan_schema(stable_keys=_GLOBAL_KEYS),
        codex_output_schemas.SEGMENT_FRAMES_SCHEMA,
        codex_output_schemas.prompt_fusion_schema(
            input_sha256="a" * 64, segment_count=3,
        ),
    ]
    for schema in schemas:
        jsonschema.Draft202012Validator.check_schema(schema)
        _assert_all_objects_closed(schema)


def test_project_index_dto_is_indexed_without_losing_relation_fields() -> None:
    model = {
        "people": [{
            "key": "person-01",
            "source_visual_description": "person",
            "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
            "replaceable": ["appearance"],
            "preserve": ["identity"],
        }],
        "entities": [{
            "key": "entity-01",
            "source_visual_description": "tool",
            "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
            "replaceable": ["appearance"],
            "preserve": ["function"],
        }],
        "scenes": [{
            "key": "scene-01",
            "source_visual_description": "room",
            "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
            "replaceable": ["environment"],
            "preserve": ["layout"],
        }],
        "relations": [{
            "key": "relation-01",
            "subject_key": "person-01",
            "predicate": "holds",
            "object_key": "entity-01",
            "occurrences": [{
                "segment_index": 1,
                "frames": [{
                    "frame_order": 1, "state": "held", "geometry": "in hand",
                }],
            }],
            "preserve": ["direction"],
            "replace_together": True,
        }],
    }
    jsonschema.validate(model, codex_output_schemas.PROJECT_INDEX_SCHEMA)
    normalized = codex_output_schemas.normalize_project_index(model)
    assert normalized["people"]["person-01"]["occurrences"][0]["frame_orders"] == [1]
    assert normalized["relations"]["relation-01"] == {
        key: value for key, value in model["relations"][0].items() if key != "key"
    }


def test_project_index_schema_requires_scene_but_not_people_or_entities() -> None:
    schema = codex_output_schemas.PROJECT_INDEX_SCHEMA
    assert schema["properties"]["scenes"]["minItems"] == 1
    assert schema["properties"]["people"]["minItems"] == 0
    assert schema["properties"]["entities"]["minItems"] == 0

    without_people_or_entities = {
        "people": [],
        "entities": [],
        "scenes": [{
            "key": "scene-01",
            "source_visual_description": "room",
            "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
            "replaceable": ["environment"],
            "preserve": ["layout"],
        }],
        "relations": [],
    }
    jsonschema.validate(without_people_or_entities, schema)
    without_people_or_entities["scenes"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(without_people_or_entities, schema)


def test_global_dto_uses_backend_frozen_properties_without_echoed_keys() -> None:
    global_model = {
        "people": {"person-01": {
            "source_identity": "source",
            "replacement_identity": "replacement", "wardrobe_change": "wardrobe",
            "local_color_change": "color",
        }},
        "entities": {},
        "scenes": {},
        "relations": {"relation-01": {
            "replacement_system": "compatible interface",
            "preserve": "directed roles",
        }},
    }
    schema = codex_output_schemas.global_plan_schema(stable_keys=_GLOBAL_KEYS)
    jsonschema.validate(global_model, schema)
    normalized_global = codex_output_schemas.normalize_global_plan(
        global_model, stable_keys=_GLOBAL_KEYS,
    )
    assert "person-01" in normalized_global["people"]
    assert normalized_global["relations"]["relation-01"] == {
        "replacement_system": "compatible interface",
        "preserve": "directed roles",
    }

    assert "key" not in schema["properties"]["people"]["properties"][
        "person-01"
    ]["properties"]


@pytest.mark.parametrize("mutation", ["wrong", "missing"])
def test_global_schema_rejects_wrong_or_missing_frozen_key(mutation: str) -> None:
    keys = {**_GLOBAL_KEYS, "entities": {"entity-01"}}
    schema = codex_output_schemas.global_plan_schema(stable_keys=keys)
    model = {
        "people": {"person-01": {
            "source_identity": "source", "replacement_identity": "replacement",
            "wardrobe_change": "wardrobe", "local_color_change": "color",
        }},
        "entities": {"entity-01": {
            "description": "entity", "owner": "project",
            "association": "member", "persistence": "persistent",
        }},
        "scenes": {},
        "relations": {"relation-01": {
            "replacement_system": "system", "preserve": "roles",
        }},
    }
    if mutation == "wrong":
        model["entities"]["wrong-entity"] = model["entities"].pop("entity-01")
    else:
        del model["entities"]["entity-01"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(model, schema)
    with pytest.raises(CodexOutputValidationError) as caught:
        codex_output_schemas.normalize_global_plan(model, stable_keys=keys)
    assert caught.value.reason == "global_plan_keys_invalid"
    assert caught.value.field_path == "/entities"


def test_global_schema_scales_to_real_inventory_and_empty_categories() -> None:
    keys = {
        "people": {f"person-{index:02d}" for index in range(1, 22)},
        "entities": {f"entity-{index:02d}" for index in range(1, 12)},
        "scenes": {f"scene-{index:02d}" for index in range(1, 4)},
        "relations": {f"relation-{index:02d}" for index in range(1, 32)},
    }
    schema = codex_output_schemas.global_plan_schema(stable_keys=keys)
    model = {
        "people": {key: {
            "source_identity": "source", "replacement_identity": "replacement",
            "wardrobe_change": "wardrobe", "local_color_change": "color",
        } for key in keys["people"]},
        "entities": {key: {
            "description": "entity", "owner": "project",
            "association": "member", "persistence": "persistent",
        } for key in keys["entities"]},
        "scenes": {key: {
            "source_scene": "source", "replacement_scene": "replacement",
            "semantic_change": "semantic", "geometry_change": "geometry",
            "depth_change": "depth", "layout_change": "layout",
            "local_color_change": "color",
        } for key in keys["scenes"]},
        "relations": {key: {
            "replacement_system": "system", "preserve": "roles",
        } for key in keys["relations"]},
    }
    jsonschema.validate(model, schema)
    normalized = codex_output_schemas.normalize_global_plan(
        model, stable_keys=keys,
    )
    assert tuple(map(len, normalized.values())) == (21, 11, 3, 31)

    empty_keys = {category: set() for category in keys}
    empty = {category: {} for category in keys}
    empty_schema = codex_output_schemas.global_plan_schema(stable_keys=empty_keys)
    jsonschema.validate(empty, empty_schema)
    assert codex_output_schemas.normalize_global_plan(
        empty, stable_keys=empty_keys,
    ) == empty


def test_global_normalizer_reports_safe_reason_and_field_path() -> None:
    empty_keys = {
        "people": set(), "entities": set(), "scenes": set(), "relations": set(),
    }
    with pytest.raises(CodexOutputValidationError) as caught:
        codex_output_schemas.normalize_global_plan(
            {"people": {}, "entities": {}, "scenes": {}},
            stable_keys=empty_keys,
        )
    assert caught.value.reason == "global_plan_shape_invalid"
    assert caught.value.field_path == "/global_plan"

    keys = {**empty_keys, "relations": {"relation-01"}}
    value = {
        "people": {}, "entities": {}, "scenes": {},
        "relations": {"relation-01": {
            "replacement_system": "", "preserve": "roles",
        }},
    }
    with pytest.raises(CodexOutputValidationError) as caught:
        codex_output_schemas.normalize_global_plan(value, stable_keys=keys)
    assert caught.value.reason == "global_plan_text_invalid"
    assert caught.value.field_path == "/relations/replacement_system"


def test_segment_dto_still_indexes_model_discovered_nested_keys() -> None:

    frame_model = {"frames": [{
        "key": "frame-001",
        "people": [{
            "key": "person-01", "visible_region": "full", "boundary": "outline",
            "body_and_pose": "standing", "derived_observations": [],
        }],
        "entities": [], "relationships": "none", "crop": "full",
    }]}
    jsonschema.validate(frame_model, codex_output_schemas.SEGMENT_FRAMES_SCHEMA)
    normalized = codex_output_schemas.normalize_segment_frames(frame_model)
    assert normalized["frames"]["frame-001"]["people"]["person-01"]["body_and_pose"] == "standing"


def test_fusion_schema_binds_hash_and_exact_segment_count() -> None:
    digest = "b" * 64
    schema = codex_output_schemas.prompt_fusion_schema(
        input_sha256=digest, segment_count=2,
    )
    valid = {
        "schema": "duet.video-prompt-fusion-output", "version": 2,
        "input_sha256": digest,
        "segments": [
            {"index": 1, "visual": ["first"]},
            {"index": 2, "visual": ["second"]},
        ],
    }
    jsonschema.validate(valid, schema)
    invalid = json.loads(json.dumps(valid))
    invalid["segments"].append({"index": 3, "visual": ["third"]})
    try:
        jsonschema.validate(invalid, schema)
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("fusion schema accepted an extra segment")


def test_codex_argv_receives_readonly_output_schema(monkeypatch, tmp_path: Path) -> None:
    session = tmp_path / "conversation"
    session.mkdir()
    monkeypatch.setattr(codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"))
    with tempfile.TemporaryDirectory(prefix="duet-schema-argv-", dir="/tmp") as raw:
        stage = Path(raw).resolve(strict=True)
        final = stage / ".codex-final-output.json"
        final.touch()
        schema = stage / ".codex-output-schema.json"
        schema.write_text('{"type":"object"}\n', encoding="utf-8")
        runner = CodexRunner(timeout_s=3, concurrency=1)
        token = codex_runner._ACTIVE_ISOLATED_STAGE.set(
            (id(runner), stage, session, (final,), final, True)
        )
        try:
            argv = runner.build_argv(stage, "prompt")
        finally:
            codex_runner._ACTIVE_ISOLATED_STAGE.reset(token)
        assert "--output-schema" in argv
        assert argv[argv.index("--output-schema") + 1] == str(schema)
        # The directory itself stays writable for Codex' nested sandbox
        # mountpoints.  Every staged input, including the schema, is overlaid
        # read-only; only the declared final transport is consumed.
        stage_mount = ["--bind", str(stage), str(stage)]
        assert any(
            argv[index:index + 3] == stage_mount
            for index in range(len(argv) - 2)
        )
        schema_mount = ["--ro-bind", str(schema), str(schema)]
        assert any(
            argv[index:index + 3] == schema_mount
            for index in range(len(argv) - 2)
        )
        final_mount = ["--ro-bind", str(final), str(final)]
        assert not any(
            argv[index:index + 3] == final_mount
            for index in range(len(argv) - 2)
        )
