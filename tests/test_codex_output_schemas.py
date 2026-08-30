import json
import tempfile
from pathlib import Path

import jsonschema

from app import codex_output_schemas, codex_runner
from app.codex_runner import CodexRunner


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
        codex_output_schemas.GLOBAL_PLAN_SCHEMA,
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
        "scenes": [],
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


def test_image_dtos_use_explicit_keys_and_backend_indexes_every_level() -> None:
    global_model = {
        "people": [{
            "key": "person-01", "source_identity": "source",
            "replacement_identity": "replacement", "wardrobe_change": "wardrobe",
            "local_color_change": "color",
        }],
        "entities": [],
        "scenes": [],
        "relations": [{
            "key": "relation-01",
            "replacement_system": "compatible interface",
            "preserve": "directed roles",
        }],
    }
    jsonschema.validate(global_model, codex_output_schemas.GLOBAL_PLAN_SCHEMA)
    normalized_global = codex_output_schemas.normalize_global_plan(global_model)
    assert "person-01" in normalized_global["people"]
    assert normalized_global["relations"]["relation-01"] == {
        "replacement_system": "compatible interface",
        "preserve": "directed roles",
    }

    invalid_global = json.loads(json.dumps(global_model))
    invalid_global["relations"][0]["subject_key"] = "person-01"
    try:
        jsonschema.validate(invalid_global, codex_output_schemas.GLOBAL_PLAN_SCHEMA)
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("global plan accepted an index-owned relation endpoint")

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
            (id(runner), stage, session, (final,), final)
        )
        try:
            argv = runner.build_argv(stage, "prompt")
        finally:
            codex_runner._ACTIVE_ISOLATED_STAGE.reset(token)
        assert "--output-schema" in argv
        assert argv[argv.index("--output-schema") + 1] == str(schema)
        schema_mount = ["--ro-bind", str(schema), str(schema)]
        assert any(
            argv[index:index + 3] == schema_mount
            for index in range(len(argv) - 2)
        )
