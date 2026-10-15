"""Many-to-one binding guards, retaining the legacy normalizer tolerance."""
import copy

import pytest

from app.object_replace_schema import (
    OBJECT_REPLACE_OUTPUT_NAME, OBJECT_REPLACE_OUTPUT_VERSION,
    merge_object_replace_index, normalize_object_replace, object_replace_schema,
)
from app.pipeline import Pipeline
from app.replacement_sheet import build_replacement_sheet_spec


def fixture(instruction="把包换成参考图中的物品", has_image=True):
    index = {"people": {}, "entities": {}, "scenes": {}, "relations": {}}
    for n in range(1, 4):
        index["entities"][f"entity-{n:02}"] = {
            "source_visual_description": f"source object {n}",
            "occurrences": [{"segment_index": n, "frame_orders": [1, 2]}],
            "preserve": ["contact"], "replaceable": ["appearance"],
        }
    index["relations"]["relation-01"] = {
        "subject_key": "entity-01", "object_key": "entity-03",
        "predicate": "touches", "replace_together": True,
    }
    schema, digest = object_replace_schema({
        "global_element_index": index, "translated_dialogue": [],
        "has_user_image": has_image, "user_replacement_instruction": instruction,
    }, ["voice-a"])
    output = {
        "schema": OBJECT_REPLACE_OUTPUT_NAME, "version": OBJECT_REPLACE_OUTPUT_VERSION,
        "input_sha256": digest, "people": {}, "entities": {}, "scenes": {},
        "dialogue_voices": [],
    }
    for stable_id in index["entities"]:
        output["entities"][stable_id] = {
            "matches_user_request": False,
            "replacement": None,
        }
    kwargs = dict(input_sha256=digest, element_index=index, voice_types=["voice-a"],
                  has_user_image=has_image, user_instruction=instruction,
                  dialogue_line_count=0)
    return index, schema, output, kwargs


def bind(output, *keys):
    for key in keys:
        output["entities"][key] = {
            "matches_user_request": True,
            "replacement": {
                "source": "image_1",
                "target_visual_description": "new object",
                "appearance_description": "isolated new object",
            },
        }


@pytest.mark.parametrize("instruction", ["把所有包替换成参考图", "把包换成参考图中的物品"])
def test_multiple_existing_entities_share_image_without_merging_source_facts(instruction):
    index, _, output, kwargs = fixture(instruction)
    original = copy.deepcopy(index)
    bind(output, "entity-01", "entity-02")
    normalized = normalize_object_replace(output, **kwargs)
    merged = merge_object_replace_index(index, normalized, ["voice-a"])
    assert index == original
    assert merged["relations"] == original["relations"]
    for key in index["entities"]:
        record = merged["entities"][key]
        assert {k: v for k, v in record.items() if k != "replacement"} == index["entities"][key]
    assert merged["entities"]["entity-03"]["replacement"] is None
    assert Pipeline._enhanced_index_uses_user_image(merged)
    spec = build_replacement_sheet_spec(merged)
    assert {tile["stable_key"] for tile in spec["tiles"] if tile["source"] == "image_1"} == {
        "entity-01", "entity-02"}
    assert len({tile["tile_id"] for tile in spec["tiles"]}) == 2


@pytest.mark.parametrize("has_image,instruction", [(False, "替换所有包"), (True, " \n ")])
def test_absent_image_or_instruction_still_requires_zero_bindings(has_image, instruction):
    _, _, output, kwargs = fixture(instruction, has_image)
    normalize_object_replace(output, **kwargs)
    bind(output, "entity-01", "entity-02")
    with pytest.raises(ValueError, match="binding count"):
        normalize_object_replace(output, **kwargs)


def test_explicit_request_allows_zero_and_single_binding_still_works():
    _, _, output, kwargs = fixture()
    assert normalize_object_replace(output, **kwargs)["entities"] == output["entities"]
    bind(output, "entity-01")
    assert normalize_object_replace(output, **kwargs)["entities"] == output["entities"]


def test_transport_schema_stays_closed_and_match_source_branches_stay_consistent():
    _, schema, _, _ = fixture()
    entities = schema["properties"]["entities"]
    assert entities["additionalProperties"] is False
    assert set(entities["properties"]) == {"entity-01", "entity-02", "entity-03"}
    for patch in entities["properties"].values():
        replacement_branches = [
            branch
            for branch in patch["anyOf"]
            if branch["properties"]["replacement"].get("type") != "null"
        ]
        pairs = {
            (
                branch["properties"]["matches_user_request"]["enum"][0],
                branch["properties"]["replacement"]["properties"]["source"]["enum"][0],
            )
            for branch in replacement_branches
        }
        assert pairs == {(True, "image_1")}
        assert any(
            branch["properties"]["matches_user_request"]["enum"] == [False]
            and branch["properties"]["replacement"] == {"type": "null"}
            for branch in patch["anyOf"]
        )


def test_legacy_normalizer_drops_unknown_keys_and_retains_match_tolerance():
    _, _, output, kwargs = fixture()
    bind(output, "entity-01", "entity-02")
    output["entities"]["entity-01"]["matches_user_request"] = False
    output["entities"]["entity-unknown"] = copy.deepcopy(output["entities"]["entity-02"])
    normalized = normalize_object_replace(output, **kwargs)
    assert "entity-unknown" not in normalized["entities"]
    assert normalized["entities"]["entity-01"]["matches_user_request"] is False
    assert normalized["entities"]["entity-01"]["replacement"]["source"] == "image_1"


def test_multiple_bindings_cannot_bypass_input_fingerprint():
    _, _, output, kwargs = fixture()
    bind(output, "entity-01", "entity-02")
    output["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not bound"):
        normalize_object_replace(output, **kwargs)
