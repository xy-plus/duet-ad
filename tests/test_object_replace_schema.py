from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

from app.fusion_schema import canonical_input
from app.object_replace_schema import (
    MAX_REFERENCE_AUDIO,
    OBJECT_REPLACE_OUTPUT_NAME,
    OBJECT_REPLACE_OUTPUT_VERSION,
    merge_object_replace_index,
    normalize_object_replace,
    object_replace_schema,
)


VOICE_IDS = ["voice-a", "voice-b", "voice-c", "voice-d"]


def _element(description: str) -> dict[str, Any]:
    return {
        "source_visual_description": description,
        "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
        "replaceable": ["appearance"],
        "preserve": ["geometry"],
    }


def _index(
    *,
    people: tuple[str, ...] = ("person-01", "person-02"),
    entities: tuple[str, ...] = ("entity-01", "entity-02", "entity-03"),
    scenes: tuple[str, ...] = ("scene-01",),
) -> dict[str, Any]:
    relations: dict[str, dict[str, Any]] = {}
    if people and entities:
        relations["relation-01"] = {
            "subject_key": people[0],
            "predicate": "holds",
            "object_key": entities[0],
            "occurrences": [{
                "segment_index": 1,
                "frames": [{
                    "frame_order": 1,
                    "state": "held",
                    "geometry": "right hand",
                }],
            }],
            "preserve": ["hand contact"],
            "replace_together": True,
        }
    return {
        "people": {stable_id: _element(stable_id) for stable_id in people},
        "entities": {stable_id: _element(stable_id) for stable_id in entities},
        "scenes": {stable_id: _element(stable_id) for stable_id in scenes},
        "relations": relations,
    }


def _replacement(source: str, description: str) -> dict[str, str]:
    return {
        "source": source,
        "target_visual_description": description,
    }


def _payload(element_index: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_element_index": element_index,
        "segment_visual_prompts": [{"segment_index": 1, "visual_prompt": "source"}],
        "user_replacement_instruction": "replace the held product",
        "has_user_image": True,
        "user_replacement_image": {"name": "Image 1", "sha256": "a" * 64},
        "target_language": "English",
        "translated_dialogue": [],
        "voice_catalog_sha256": "b" * 64,
        "voice_catalog_lines": "catalog text is model context, not a local gate",
    }


def _output(
    element_index: dict[str, Any],
    digest: str,
) -> dict[str, Any]:
    return {
        "schema": OBJECT_REPLACE_OUTPUT_NAME,
        "version": OBJECT_REPLACE_OUTPUT_VERSION,
        "input_sha256": digest,
        "people": {
            stable_id: {
                "matches_user_request": False,
                "replacement": _replacement(
                    "generated_variant", f"new appearance for {stable_id}",
                ),
                "voice_id": VOICE_IDS[index % len(VOICE_IDS)],
            }
            for index, stable_id in enumerate(element_index["people"])
        },
        "entities": {
            stable_id: {
                "matches_user_request": False,
                "replacement": _replacement(
                    "generated_variant", f"new appearance for {stable_id}",
                ),
            }
            for stable_id in element_index["entities"]
        },
        "scenes": {
            stable_id: {
                "matches_user_request": False,
                "replacement": _replacement(
                    "generated_variant", f"new layout for {stable_id}",
                ),
            }
            for stable_id in element_index["scenes"]
        },
    }


def _normalize(
    value: object,
    *,
    digest: str,
    element_index: dict[str, Any],
    voice_ids: list[str] = VOICE_IDS,
    has_user_image: bool = True,
    instruction: str = "replace the held product",
) -> dict[str, dict[str, dict[str, Any]]]:
    return normalize_object_replace(
        value,
        input_sha256=digest,
        element_index=element_index,
        voice_ids=voice_ids,
        has_user_image=has_user_image,
        user_instruction=instruction,
    )


def test_schema_is_closed_bound_and_all_quality_choices_are_nullable() -> None:
    element_index = _index()
    payload = _payload(element_index)
    schema, digest = object_replace_schema(payload, VOICE_IDS)

    assert digest == hashlib.sha256(canonical_input(payload)).hexdigest()
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "schema", "version", "input_sha256", "people", "entities", "scenes",
    ]
    assert schema["properties"]["input_sha256"]["enum"] == [digest]
    assert "relations" not in schema["properties"]

    for category in ("people", "entities", "scenes"):
        category_schema = schema["properties"][category]
        assert category_schema["additionalProperties"] is False
        assert list(category_schema["properties"]) == list(element_index[category])
        assert category_schema["required"] == list(element_index[category])
        for patch_schema in category_schema["properties"].values():
            assert patch_schema["additionalProperties"] is False
            replacement = patch_schema["properties"]["replacement"]
            assert replacement["anyOf"][1] == {"type": "null"}
            replacement_object = replacement["anyOf"][0]
            assert replacement_object["properties"]["source"]["enum"] == [
                "generated_variant", "image_1",
            ]
            assert replacement_object["properties"][
                "target_visual_description"
            ] == {"type": "string"}
            if category == "people":
                assert patch_schema["properties"]["voice_id"]["anyOf"] == [
                    {"type": "string", "enum": VOICE_IDS},
                    {"type": "null"},
                ]


def test_schema_does_not_revalidate_upstream_content_or_user_pairing() -> None:
    element_index = _index(scenes=())
    element_index["people"]["person-01"] = {
        "source_visual_description": "",
        "occurrences": "not locally reviewed",
        "unexpected": True,
    }
    element_index["relations"]["relation-01"]["object_key"] = "missing-object"
    element_index["extra_upstream_field"] = {"kept": True}
    payload = _payload(element_index)
    payload["has_user_image"] = False
    payload["user_replacement_image"] = None
    payload["segment_visual_prompts"] = []
    payload["extra_context"] = "forward this too"

    schema, digest = object_replace_schema(payload, VOICE_IDS)

    assert digest == hashlib.sha256(canonical_input(payload)).hexdigest()
    assert schema["properties"]["scenes"]["properties"] == {}


def test_normalizer_allows_semantically_imperfect_but_consumable_output() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)

    value["people"]["person-01"] = {
        "matches_user_request": True,
        "replacement": None,
        "voice_id": None,
    }
    value["people"]["person-02"] = {
        "matches_user_request": False,
        "replacement": _replacement("image_1", "person-02"),
        "voice_id": None,
    }
    value["entities"]["entity-01"] = {
        "matches_user_request": True,
        "replacement": _replacement("generated_variant", ""),
    }
    value["entities"]["entity-02"]["replacement"] = None
    value["entities"]["entity-03"]["replacement"] = None
    value["scenes"]["scene-01"] = {
        "matches_user_request": False,
        "replacement": None,
    }

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        has_user_image=False,
        instruction="",
    )

    assert normalized == {
        category: value[category] for category in ("people", "entities", "scenes")
    }


def test_normalizer_does_not_limit_description_length_or_compare_source() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"]["replacement"] = _replacement(
        "generated_variant", "person-01",
    )
    value["scenes"]["scene-01"]["replacement"] = _replacement(
        "image_1", "x" * 10_000,
    )

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert len(
        normalized["scenes"]["scene-01"]["replacement"][
            "target_visual_description"
        ]
    ) == 10_000


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"input_sha256": "0" * 64}),
        lambda value: value.update({"version": True}),
        lambda value: value.update({"extra": True}),
        lambda value: value["people"].pop("person-02"),
        lambda value: value["entities"].update({"entity-99": {}}),
        lambda value: value["scenes"]["scene-01"].update({"extra": True}),
        lambda value: value["people"]["person-01"].update(
            {"matches_user_request": None},
        ),
        lambda value: value["entities"]["entity-01"].update(
            {"replacement": {"source": "unknown", "target_visual_description": "x"}},
        ),
        lambda value: value["entities"]["entity-01"].update(
            {"replacement": {"source": "image_1", "target_visual_description": 1}},
        ),
    ],
)
def test_normalizer_rejects_only_unusable_transport_shapes(mutation: Any) -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    mutation(value)

    with pytest.raises(ValueError):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_requires_unambiguous_stable_id_maps() -> None:
    malformed = _index()
    malformed["people"] = []
    with pytest.raises(ValueError, match="global element index"):
        object_replace_schema(_payload(malformed), VOICE_IDS)

    ambiguous = _index(entities=("person-01",))
    with pytest.raises(ValueError, match="stable element id is ambiguous"):
        object_replace_schema(_payload(ambiguous), VOICE_IDS)


def test_normalizer_degrades_null_or_unmapped_voice_to_no_voice() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"]["voice_id"] = None
    _normalize(value, digest=digest, element_index=element_index)

    value["people"]["person-01"]["voice_id"] = "unknown"
    normalized = _normalize(value, digest=digest, element_index=element_index)
    assert normalized["people"]["person-01"]["voice_id"] is None


def test_normalizer_degrades_fourth_unique_voice_in_stable_person_order() -> None:
    four_people = _index(
        people=("person-01", "person-02", "person-03", "person-04"),
        entities=(),
    )
    _schema, digest = object_replace_schema(_payload(four_people), VOICE_IDS)
    value = _output(four_people, digest)
    assert MAX_REFERENCE_AUDIO == 3
    normalized = _normalize(value, digest=digest, element_index=four_people)

    assert [
        normalized["people"][stable_id]["voice_id"]
        for stable_id in four_people["people"]
    ] == ["voice-a", "voice-b", "voice-c", None]


def test_merge_safely_copies_without_semantic_review() -> None:
    element_index = _index()
    element_index["relations"]["relation-01"]["object_key"] = "unresolved"
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"] = {
        "matches_user_request": True,
        "replacement": None,
        "voice_id": None,
    }
    value["entities"]["entity-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("image_1", ""),
    }
    normalized = _normalize(value, digest=digest, element_index=element_index)

    original_snapshot = copy.deepcopy(element_index)
    patch_snapshot = copy.deepcopy(normalized)
    voice_metadata = {
        "voice-b": {"file": "voices/b.wav", "features": {"gender": "male"}},
    }
    metadata_snapshot = copy.deepcopy(voice_metadata)
    merged = merge_object_replace_index(element_index, normalized, voice_metadata)

    assert element_index == original_snapshot
    assert normalized == patch_snapshot
    assert voice_metadata == metadata_snapshot
    assert merged is not element_index
    assert merged["relations"] == element_index["relations"]
    assert merged["people"]["person-01"]["replacement"] is None
    assert merged["people"]["person-01"]["voice"] is None
    assert merged["people"]["person-02"]["voice"] == voice_metadata["voice-b"]
    assert merged["people"]["person-02"]["voice"] is not voice_metadata["voice-b"]
    assert merged["entities"]["entity-01"]["replacement"] == {
        "reason": "automatic",
        "source": "image_1",
        "target_visual_description": "",
    }


def test_merge_degrades_missing_backend_voice_metadata_to_no_voice() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    normalized = _normalize(
        _output(element_index, digest),
        digest=digest,
        element_index=element_index,
    )
    merged = merge_object_replace_index(element_index, normalized, {})
    assert all(person["voice"] is None for person in merged["people"].values())

    for person in normalized["people"].values():
        person["voice_id"] = None
    merged = merge_object_replace_index(element_index, normalized, {})
    assert all(person["voice"] is None for person in merged["people"].values())
