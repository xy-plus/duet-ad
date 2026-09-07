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


def test_project_index_normalizer_accepts_empty_collections() -> None:
    assert normalize_project_index({
        "people": [],
        "entities": [],
        "scenes": [],
        "relations": [],
        "segment_metrics": [],
    }) == {
        "people": {},
        "entities": {},
        "scenes": {},
        "relations": {},
        "segment_metrics": {},
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
        "segment_metrics": [
            {"segment_index": 1, "max_face_share_percent": 42},
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
    assert normalized["segment_metrics"] == {1: 42}


def test_project_index_schema_covers_segment_metrics_for_each_input() -> None:
    schema = project_index_schema([1, 3, 5])
    metrics = schema["properties"]["segment_metrics"]

    assert metrics["minItems"] == 3
    assert metrics["maxItems"] == 3
    item = metrics["items"]
    assert item["additionalProperties"] is False
    assert set(item["properties"]) == {
        "segment_index", "max_face_share_percent",
    }
    assert item["properties"]["segment_index"] == {
        "type": "integer", "enum": [1, 3, 5],
    }
    assert item["properties"]["max_face_share_percent"] == {
        "type": "integer", "minimum": 0, "maximum": 100,
    }
    assert set(schema["properties"]) == {
        "people", "entities", "scenes", "relations", "segment_metrics",
    }
    assert PROJECT_INDEX_SCHEMA["properties"]["segment_metrics"]["minItems"] == 0


def test_project_index_normalizer_collects_segment_metrics() -> None:
    normalized = normalize_project_index({
        "people": [],
        "entities": [],
        "scenes": [],
        "relations": [],
        "segment_metrics": [
            {"segment_index": 1, "max_face_share_percent": 0},
            {"segment_index": 3, "max_face_share_percent": 100},
        ],
    })

    assert normalized["segment_metrics"] == {1: 0, 3: 100}


@pytest.mark.parametrize(
    "segment_metrics",
    [
        [{"segment_index": 1}],
        [{"segment_index": 1, "max_face_share_percent": 10, "extra": 1}],
        [{"segment_index": True, "max_face_share_percent": 10}],
        [{"segment_index": "1", "max_face_share_percent": 10}],
        [{"segment_index": 1, "max_face_share_percent": -1}],
        [{"segment_index": 1, "max_face_share_percent": 101}],
        [{"segment_index": 1, "max_face_share_percent": 12.5}],
        [{"segment_index": 1, "max_face_share_percent": True}],
        [
            {"segment_index": 1, "max_face_share_percent": 10},
            {"segment_index": 1, "max_face_share_percent": 20},
        ],
    ],
)
def test_project_index_normalizer_rejects_invalid_segment_metrics(
    segment_metrics: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="segment metric"):
        normalize_project_index({
            "people": [],
            "entities": [],
            "scenes": [],
            "relations": [],
            "segment_metrics": segment_metrics,
        })


def test_project_index_normalizer_requires_segment_metrics_key() -> None:
    with pytest.raises(ValueError, match="project index output is invalid"):
        normalize_project_index({
            "people": [],
            "entities": [],
            "scenes": [],
            "relations": [],
        })
