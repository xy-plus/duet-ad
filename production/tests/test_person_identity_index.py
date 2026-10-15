from copy import deepcopy

import pytest

from app.video_maker_schemas import normalize_project_index, project_index_schema


def sample():
    # Reduced from abc09e15.../work/element_index.json: the same source person
    # was split because the language model described two different hats.
    people = [
        {"key": "person-01", "source_visual_description":
         "Man wearing a gray baseball cap, black-framed glasses, and a gray crewneck sweater",
         "occurrences": [{"segment_index": 1, "frame_orders": [7, 8, 9]}],
         "preserve": ["person identity", "visible body proportions"],
         "replaceable": ["clothing color/style", "headwear", "glasses"]},
        {"key": "person-02", "source_visual_description":
         "Person wearing a gray knit beanie, gray crewneck sweater, black wide-leg pants, and black shoes",
         "occurrences": [{"segment_index": 2, "frame_orders": [5, 6, 7, 8, 9]}],
         "preserve": ["person identity", "body proportions"],
         "replaceable": ["beanie style", "clothing color/style", "shoe style"]},
    ]
    observations = []
    for person in people:
        person["canonical_id"] = "person-cv-001"
        person["identity_observation_ids"] = []
        for occurrence in person["occurrences"]:
            for frame in occurrence["frame_orders"]:
                oid = f"s{occurrence['segment_index']}-f{frame}"
                person["identity_observation_ids"].append(oid)
                observations.append({"observation_id": oid, "canonical_id": "person-cv-001",
                                     "kind": "keyframe", "frame_order": frame,
                                     "segment_index": occurrence["segment_index"],
                                     "bbox": [10, 10, 20, 20]})
    value = {"people": people, "entities": [{"key": "bag", "source_visual_description": "black bag",
             "occurrences": [], "preserve": [], "replaceable": []}], "scenes": [],
             "relations": [{"key": "r1", "subject_key": "person-01", "object_key": "person-02",
                            "predicate": "same source", "occurrences": [], "preserve": [],
                            "replace_together": False},
                           {"key": "r2", "subject_key": "person-02", "object_key": "bag",
                            "predicate": "wearing", "occurrences": [], "preserve": [],
                            "replace_together": False}]}
    evidence = {"status": "complete", "observations": observations,
                "persons": [{"canonical_id": "person-cv-001",
                             "observation_ids": [obs["observation_id"] for obs in observations]}]}
    return value, evidence


def test_real_split_person_merges_and_rewrites_both_relation_endpoints():
    value, evidence = sample()
    original = deepcopy(value)
    result = normalize_project_index(value, evidence)
    assert value == original
    assert list(result["people"]) == ["person-01"]
    person = result["people"]["person-01"]
    assert len(person["occurrences"]) == 2
    assert "baseball cap" in person["source_visual_description"]
    assert "knit beanie" in person["source_visual_description"]
    assert person["replaceable"].count("clothing color/style") == 1
    assert result["relations"]["relation-01"]["subject_key"] == "person-01"
    assert result["relations"]["relation-01"]["object_key"] == "person-01"
    assert result["relations"]["relation-02"]["subject_key"] == "person-01"
    assert result["relations"]["relation-02"]["object_key"] == "entity-01"


@pytest.mark.parametrize("fault,reason", [
    ("forged", "unknown_observation"), ("conflict", "conflicting_identity"),
    ("uncovered", "occurrence_coverage_mismatch"),
])
def test_untrusted_claim_keeps_original_person(fault, reason):
    value, evidence = sample()
    person = value["people"][1]
    if fault == "forged":
        person["identity_observation_ids"].append("invented")
    elif fault == "conflict":
        person["canonical_id"] = "person-cv-002"
    elif fault == "uncovered":
        person["occurrences"][0]["frame_orders"].remove(5)
    result = normalize_project_index(value, evidence)
    assert len(result["people"]) == 2
    assert result["identity_resolution"][1]["reason"] == reason
    assert result["relations"]["relation-02"]["subject_key"] == "person-02"


def test_missing_claim_recovers_only_exclusive_single_person_frames():
    value, evidence = sample()
    for person in value["people"]:
        person.pop("canonical_id")
        person.pop("identity_observation_ids")
    result = normalize_project_index(value, evidence)
    assert len(result["people"]) == 1
    assert result["identity_resolution"][1]["reason"] == "recovered_single_person_frames"


def test_body_only_frames_do_not_discard_confirmed_face_identity():
    value, evidence = sample()
    value["people"][1]["identity_observation_ids"] = ["s2-f8", "s2-f9"]
    evidence["observations"] = [obs for obs in evidence["observations"]
                                if obs["segment_index"] != 2 or obs["frame_order"] >= 8]
    assert len(normalize_project_index(value, evidence)["people"]) == 1


def test_uncovered_person_is_retained_as_unknown():
    value, evidence = sample()
    person = deepcopy(value["people"][0])
    person.update(key="back-view", canonical_id=None, identity_observation_ids=[],
                  occurrences=[{"segment_index": 3, "frame_orders": [1, 2]}])
    value["people"].append(person)
    result = normalize_project_index(value, evidence)
    assert len(result["people"]) == 2
    assert result["people"]["person-03"]["canonical_id"] is None


def test_two_faces_same_frame_are_not_merged_by_frame_coordinates():
    value, evidence = sample()
    first = evidence["observations"][0]
    second = {**first, "observation_id": "other-face", "canonical_id": "person-cv-002",
              "bbox": [60, 10, 20, 20]}
    evidence["observations"].append(second)
    evidence["persons"].append({"canonical_id": "person-cv-002", "observation_ids": ["other-face"]})
    for person, obs in zip(value["people"], [first, second]):
        person["canonical_id"] = obs["canonical_id"]
        person["identity_observation_ids"] = [obs["observation_id"]]
        person["occurrences"] = [{"segment_index": 1, "frame_orders": [7]}]
    assert len(normalize_project_index(value, evidence)["people"]) == 2
    for person in value["people"]:
        person.update(canonical_id=None, identity_observation_ids=[])
    assert len(normalize_project_index(value, evidence)["people"]) == 2


def test_unclaimed_overlapping_person_cannot_borrow_visible_face():
    value, evidence = sample()
    value["people"][1]["occurrences"] = deepcopy(value["people"][0]["occurrences"])
    for person in value["people"]:
        person.update(canonical_id=None, identity_observation_ids=[])
    result = normalize_project_index(value, evidence)
    assert len(result["people"]) == 2
    assert all(person["canonical_id"] is None for person in result["people"].values())


def test_two_people_cannot_explicitly_claim_the_same_observation():
    value, evidence = sample()
    first = evidence["observations"][0]
    evidence["observations"].append({**first, "observation_id": "other-face",
                                     "canonical_id": "person-cv-002"})
    evidence["persons"].append({"canonical_id": "person-cv-002", "observation_ids": ["other-face"]})
    for person in value["people"]:
        person.update(canonical_id="person-cv-001", identity_observation_ids=[first["observation_id"]],
                      occurrences=[{"segment_index": 1, "frame_orders": [7]}])
    result = normalize_project_index(value, evidence)
    assert len(result["people"]) == 2
    assert all(person["canonical_id"] is None for person in result["people"].values())
    assert all(item["reason"] == "shared_observation_claim" for item in result["identity_resolution"])


def test_single_keyframe_may_bind_already_confirmed_backend_identity():
    value, evidence = sample()
    for person in value["people"]:
        person.update(canonical_id=None, identity_observation_ids=[])
    evidence["observations"] = [evidence["observations"][0], evidence["observations"][-1]]
    assert len(normalize_project_index(value, evidence)["people"]) == 1


def test_schema_only_extends_people_and_binds_ids_to_keyframes():
    _, evidence = sample()
    evidence["observations"].append({"observation_id": "sampled", "kind": "sample",
                                    "canonical_id": "person-cv-001"})
    old = project_index_schema([1, 2])
    schema = project_index_schema([1, 2], evidence)
    for category in ("entities", "scenes", "relations"):
        assert schema["properties"][category] == old["properties"][category]
    properties = schema["properties"]["people"]["items"]["properties"]
    assert properties["canonical_id"]["enum"] == ["person-cv-001", None]
    assert "sampled" not in properties["identity_observation_ids"]["items"]["enum"]


def test_skipped_evidence_and_absent_sidecar_do_not_merge():
    value, evidence = sample()
    assert len(normalize_project_index(value)["people"]) == 2
    assert "identity_resolution" not in normalize_project_index(value)
    evidence["status"] = "skipped"
    assert len(normalize_project_index(value, evidence)["people"]) == 2
