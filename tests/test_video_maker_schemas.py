from __future__ import annotations

from typing import Any

import pytest

from app.video_maker_schemas import (
    PROJECT_INDEX_SCHEMA,
    VISUAL_PROMPT_SCHEMA,
    normalize_project_index,
    normalize_visual_prompt,
    project_index_schema,
)


def _element(
    key: str, *, occurrences: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "source_visual_description": f"description for {key}",
        "occurrences": [] if occurrences is None else occurrences,
        "replaceable": [],
        "preserve": [],
    }


def _relation(
    key: str,
    *,
    subject_key: str,
    object_key: str,
    occurrences: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "subject_key": subject_key,
        "predicate": "beside",
        "object_key": object_key,
        "occurrences": [] if occurrences is None else occurrences,
        "preserve": [],
        "replace_together": False,
    }


def _schema_values(value: object, keyword: str) -> list[object]:
    found: list[object] = []
    if isinstance(value, dict):
        if keyword in value:
            found.append(value[keyword])
        for child in value.values():
            found.extend(_schema_values(child, keyword))
    elif isinstance(value, list):
        for child in value:
            found.extend(_schema_values(child, keyword))
    return found


def test_transport_schemas_do_not_cap_prose_or_collection_sizes() -> None:
    assert "maxLength" not in VISUAL_PROMPT_SCHEMA["properties"]["prompt"]
    assert PROJECT_INDEX_SCHEMA["properties"]["scenes"]["minItems"] == 0
    assert not ({20, 100, 200} & set(_schema_values(
        PROJECT_INDEX_SCHEMA, "maxItems",
    )))


def test_project_index_schema_binds_all_occurrences_to_exact_segments() -> None:
    schema = project_index_schema([1, 3, 5])
    element_segment_schemas = [
        schema["properties"][category]["items"]["properties"]
        ["occurrences"]["items"]["properties"]["segment_index"]
        for category in ("people", "entities", "scenes")
    ]
    relation_segment_schema = (
        schema["properties"]["relations"]["items"]["properties"]
        ["occurrences"]["items"]["properties"]["segment_index"]
    )

    for segment_schema in [*element_segment_schemas, relation_segment_schema]:
        assert segment_schema == {"type": "integer", "enum": [1, 3, 5]}
        assert all(index not in segment_schema["enum"] for index in (0, 2, 4, 6))


def test_static_project_index_schema_is_one_based() -> None:
    element_segment_schema = (
        PROJECT_INDEX_SCHEMA["properties"]["people"]["items"]["properties"]
        ["occurrences"]["items"]["properties"]["segment_index"]
    )
    relation_segment_schema = (
        PROJECT_INDEX_SCHEMA["properties"]["relations"]["items"]["properties"]
        ["occurrences"]["items"]["properties"]["segment_index"]
    )

    assert element_segment_schema == {"type": "integer", "minimum": 1}
    assert relation_segment_schema == {"type": "integer", "minimum": 1}


@pytest.mark.parametrize("indices", [[], [0], [1, 1], [True]])
def test_project_index_schema_rejects_invalid_input_identities(
    indices: list[int],
) -> None:
    with pytest.raises(ValueError, match="segment identities"):
        project_index_schema(indices)


def test_visual_prompt_normalizer_accepts_large_transport_valid_prose() -> None:
    prompt = "画面描述" * 20_000

    assert len(prompt.encode("utf-8")) > 32 * 1024
    assert normalize_visual_prompt({"prompt": prompt}) == prompt


def test_visual_prompt_normalizer_does_not_grade_empty_prose() -> None:
    assert normalize_visual_prompt({"prompt": ""}) == ""


def test_visual_prompt_normalizer_tolerates_extra_keys() -> None:
    # Models occasionally emit stray fields (e.g. is_project_index) next to
    # the contracted prompt; they are dropped instead of killing the output.
    assert (
        normalize_visual_prompt({"prompt": "画面", "is_project_index": False})
        == "画面"
    )


def test_visual_prompt_normalizer_requires_a_prompt_string() -> None:
    with pytest.raises(ValueError, match="visual prompt output is invalid"):
        normalize_visual_prompt({"is_project_index": False})
    with pytest.raises(ValueError, match="visual prompt output is invalid"):
        normalize_visual_prompt({"prompt": 3})
    with pytest.raises(ValueError, match="visual prompt output is invalid"):
        normalize_visual_prompt(["prompt"])


def test_project_index_normalizer_accepts_empty_collections() -> None:
    assert normalize_project_index({
        "people": [],
        "entities": [],
        "scenes": [],
        "relations": [],
    }) == {
        "people": {},
        "entities": {},
        "scenes": {},
        "relations": {},
    }


def test_duplicate_aliases_are_numbered_and_ambiguous_relations_pass_through() -> None:
    normalized = normalize_project_index({
        "people": [
            _element("person-local"),
            _element(
                "person-local",
                occurrences=[{"segment_index": 1, "frame_orders": [1]}],
            ),
        ],
        "entities": [_element("prop-local")],
        "scenes": [],
        "relations": [
            _relation(
                "relation-local",
                subject_key="person-local",
                object_key="prop-local",
            ),
            _relation(
                "relation-local",
                subject_key="missing-local",
                object_key="prop-local",
            ),
        ],
    })

    assert list(normalized["people"]) == ["person-01", "person-02"]
    assert normalized["people"]["person-01"]["occurrences"] == []
    assert list(normalized["relations"]) == ["relation-01", "relation-02"]
    assert normalized["relations"]["relation-01"]["subject_key"] == (
        "unresolved:person-local"
    )
    assert normalized["relations"]["relation-01"]["object_key"] == "entity-01"
    assert normalized["relations"]["relation-02"]["subject_key"] == (
        "unresolved:missing-local"
    )

def test_project_index_schema_has_no_segment_metrics() -> None:
    schema = project_index_schema([1, 3, 5])

    assert set(schema["properties"]) == {
        "people", "entities", "scenes", "relations",
    }
    assert "segment_metrics" not in PROJECT_INDEX_SCHEMA["properties"]


def test_project_index_schema_binds_optional_target_reference_ids() -> None:
    schema = project_index_schema(
        [1, 3],
        reference_image_ids=["image_1", "image_2", "image_3", "image_4"],
    )
    expected = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ["image_1", "image_2", "image_3", "image_4"],
        },
        "minItems": 0,
    }

    for category in ("people", "entities", "scenes"):
        item = schema["properties"][category]["items"]
        assert item["properties"]["target_reference_image_ids"] == expected
        assert "target_reference_image_ids" not in item["required"]
    assert "target_reference_image_ids" not in schema["properties"][
        "relations"
    ]["items"]["properties"]


def test_project_index_schema_keeps_empty_reference_lists_legacy_compatible() -> None:
    schema = project_index_schema([1], reference_image_ids=[])

    for category in ("people", "entities", "scenes"):
        item = schema["properties"][category]["items"]
        assert "target_reference_image_ids" not in item["properties"]


def test_project_index_normalizer_keeps_target_references_without_source_facts() -> None:
    value = {
        "people": [{
            **_element(
                "person-local",
                occurrences=[{"segment_index": 1, "frame_orders": [1]}],
            ),
            "target_reference_image_ids": ["image_1", "image_3"],
        }],
        "entities": [{
            **_element(
                "bag-local",
                occurrences=[{"segment_index": 3, "frame_orders": [2]}],
            ),
            "target_reference_image_ids": ["image_2"],
        }],
        "scenes": [],
        "relations": [],
    }

    normalized = normalize_project_index(
        value,
        reference_image_ids=["image_1", "image_2", "image_3", "image_4"],
    )

    assert normalized["people"]["person-01"]["target_reference_image_ids"] == [
        "image_1", "image_3",
    ]
    assert normalized["entities"]["entity-01"]["target_reference_image_ids"] == [
        "image_2",
    ]
    assert normalized["people"]["person-01"]["occurrences"] == [
        {"segment_index": 1, "frame_orders": [1]},
    ]
    assert normalized["entities"]["entity-01"]["occurrences"] == [
        {"segment_index": 3, "frame_orders": [2]},
    ]


@pytest.mark.parametrize(
    ("first_target_ids", "second_target_ids", "expected_target_ids"),
    [
        (["image_1"], ["image_2"], ["image_2", "image_1"]),
        (None, ["image_2"], ["image_2"]),
    ],
)
def test_project_index_identity_merge_unions_target_reference_ids_in_input_order(
    first_target_ids: list[str] | None,
    second_target_ids: list[str] | None,
    expected_target_ids: list[str],
) -> None:
    first = {
        **_element(
            "person-a",
            occurrences=[{"segment_index": 1, "frame_orders": [1]}],
        ),
        "canonical_id": "person-cv-001",
        "identity_observation_ids": ["face-a"],
    }
    second = {
        **_element(
            "person-b",
            occurrences=[{"segment_index": 2, "frame_orders": [1]}],
        ),
        "canonical_id": "person-cv-001",
        "identity_observation_ids": ["face-b"],
    }
    if first_target_ids is not None:
        first["target_reference_image_ids"] = first_target_ids
    if second_target_ids is not None:
        second["target_reference_image_ids"] = second_target_ids
    evidence = {
        "status": "complete",
        "persons": [{
            "canonical_id": "person-cv-001",
            "observation_ids": ["face-a", "face-b"],
        }],
        "observations": [
            {
                "observation_id": "face-a",
                "canonical_id": "person-cv-001",
                "kind": "keyframe",
                "segment_index": 1,
                "frame_order": 1,
            },
            {
                "observation_id": "face-b",
                "canonical_id": "person-cv-001",
                "kind": "keyframe",
                "segment_index": 2,
                "frame_order": 1,
            },
        ],
    }

    normalized = normalize_project_index(
        {
            "people": [first, second],
            "entities": [],
            "scenes": [],
            "relations": [],
        },
        evidence,
        reference_image_ids=["image_2", "image_1", "image_3"],
    )

    person = normalized["people"]["person-01"]
    assert person["target_reference_image_ids"] == expected_target_ids
    assert person["occurrences"] == [
        {"segment_index": 1, "frame_orders": [1]},
        {"segment_index": 2, "frame_orders": [1]},
    ]


@pytest.mark.parametrize(
    "target_reference_image_ids",
    [["image_3"], ["unknown-image"]],
)
def test_project_index_normalizer_rejects_invalid_target_reference_ids(
    target_reference_image_ids: list[str],
) -> None:
    value = {
        "people": [{
            **_element("person-local"),
            "target_reference_image_ids": target_reference_image_ids,
        }],
        "entities": [],
        "scenes": [],
        "relations": [],
    }

    with pytest.raises(ValueError, match="reference image IDs"):
        normalize_project_index(
            value,
            reference_image_ids=["image_1", "image_2"],
        )


@pytest.mark.parametrize("references, expected", [
    ([], None),
    (["image_2", "image_1", "image_2"], ["image_2", "image_1"]),
])
def test_project_index_normalizes_only_redundant_model_reference_ids(references, expected):
    record = {**_element("local"), "target_reference_image_ids": references}
    value = {"people": [record], "entities": [], "scenes": [], "relations": []}
    normalized = normalize_project_index(value, reference_image_ids=["image_1", "image_2"])
    person = normalized["people"]["person-01"]
    assert person.get("target_reference_image_ids") == expected
    assert ("target_reference_image_ids" in person) == (expected is not None)
    assert person["occurrences"] == record["occurrences"]
    assert record["target_reference_image_ids"] == references


def test_project_index_normalizer_drops_target_references_for_legacy_calls() -> None:
    normalized = normalize_project_index({
        "people": [{
            **_element("person-local"),
            "target_reference_image_ids": ["image_1"],
        }],
        "entities": [],
        "scenes": [],
        "relations": [],
    })

    assert "target_reference_image_ids" not in normalized["people"]["person-01"]


def test_project_index_normalizer_tolerates_extra_segment_metrics_key() -> None:
    normalized = normalize_project_index({
        "people": [],
        "entities": [],
        "scenes": [],
        "relations": [],
        "segment_metrics": [],
    })

    assert set(normalized) == {"people", "entities", "scenes", "relations"}
