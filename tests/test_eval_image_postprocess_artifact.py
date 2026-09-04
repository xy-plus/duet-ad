import json
from pathlib import Path

import cv2
import numpy as np

from eval_image_postprocess_artifact import evaluate


def _png(value: int = 127) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((8, 6, 3), value, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def test_artifact_evaluator_uses_source_frame_evidence(tmp_path: Path):
    source = tmp_path / "source"
    frames = source / "work" / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    (frames / "01.png").write_bytes(_png())

    index_path = tmp_path / "element_index.json"
    index_path.write_text(
        json.dumps(
            {
                "people": {
                    "person-01": {
                        "occurrences": [
                            {"segment_index": 1, "frame_orders": [1]}
                        ]
                    }
                },
                "entities": {},
                "scenes": {"scene-01": {"occurrences": []}},
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "image_optimization.json"
    output_path.write_text(
        json.dumps(
            {
                "version": 4,
                "person_plans": [
                    {
                        "id": "PERSON_01",
                        "identity": "stable_key=person-01",
                    }
                ],
                "scene_plans": [
                    {
                        "id": "SCENE_01",
                        "scene": "stable_key=scene-01",
                    }
                ],
                "segments": [
                    {
                        "segment_index": 1,
                        "persons": [
                            {
                                "id": "PERSON_01",
                                "observable_frames": [1],
                            }
                        ],
                        "frame_constraints": [
                            {
                                "frame_index": 1,
                                "non_person_entity_ledger": {"entities": []},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate(source, index_path, output_path)

    assert report["source"]["frame_count"] == 1
    assert report["scores"]["per_frame_visible_element_binding"] == 100.0
    assert report["scores"]["scene_and_frame_key_closure"] == 100.0
    assert report["score_interpretation"] == {
        "structural_scores_establish_replacement_success": False,
        "replacement_semantics_assessed": False,
        "note": (
            "Stable-key coverage and membership scores only describe structural "
            "binding. Raw exact no-op evidence can prove a failure, but neither "
            "coverage nor lexical inequality proves replacement success."
        ),
    }
    assert report["decision"] is None


def test_hard_cut_scope_does_not_reward_missing_expected_elements(tmp_path: Path):
    source = tmp_path / "source"
    frames = source / "work" / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    (frames / "01.png").write_bytes(_png())
    index_path = tmp_path / "element_index.json"
    index_path.write_text(
        json.dumps({
            "people": {
                "person-01": {
                    "occurrences": [{"segment_index": 1, "frame_orders": [1]}]
                }
            },
            "entities": {
                "entity-01": {
                    "occurrences": [{"segment_index": 1, "frame_orders": [1]}]
                }
            },
            "scenes": {},
        }),
        encoding="utf-8",
    )
    output_path = tmp_path / "image_optimization.json"
    output_path.write_text(
        json.dumps({
            "person_plans": [],
            "scene_plans": [],
            "segments": [{
                "segment_index": 1,
                "persons": [],
                "frame_constraints": [{
                    "frame_index": 1,
                    "non_person_entity_ledger": {"entities": []},
                }],
            }],
        }),
        encoding="utf-8",
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({
            "segments": [{
                "transition_skeleton": [{
                    "segment_index": 1,
                    "frame_index": 1,
                    "source_transition_from_previous": "hard_cut",
                }]
            }]
        }),
        encoding="utf-8",
    )

    report = evaluate(
        source,
        index_path,
        output_path,
        request_path=request_path,
    )

    assert report["scores"]["hard_cut_direct_evidence_scope"] == 0.0
    assert report["structural_evidence"]["hard_cut_scope_frames"] == [
        {"segment": 1, "frame": 1, "membership_exact": False}
    ]


def test_raw_global_plan_reports_exact_noops_without_claiming_success(
    tmp_path: Path,
):
    source = tmp_path / "source"
    frames = source / "work" / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    (frames / "01.png").write_bytes(_png())
    index_path = tmp_path / "element_index.json"
    index_path.write_text(
        json.dumps({
            "people": {
                "person-01": {
                    "source_visual_description": "Woman in a coral blouse",
                    "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                }
            },
            "entities": {
                "entity-01": {
                    "source_visual_description": "Thin gold necklace",
                    "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                }
            },
            "scenes": {
                "scene-01": {
                    "source_visual_description": "Residential lawn",
                    "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                }
            },
        }),
        encoding="utf-8",
    )
    output_path = tmp_path / "image_optimization.json"
    output_path.write_text(
        json.dumps({
            "person_plans": [{"id": "P", "identity": "stable_key=person-01"}],
            "scene_plans": [{"id": "S", "scene": "stable_key=scene-01"}],
            "segments": [{
                "segment_index": 1,
                "persons": [{"id": "P", "observable_frames": [1]}],
                "frame_constraints": [{
                    "frame_index": 1,
                    "non_person_entity_ledger": {"entities": [{
                        "description": "stable_key=entity-01",
                    }]},
                }],
            }],
        }),
        encoding="utf-8",
    )
    raw_global_plan_path = tmp_path / "global_plan.json"
    raw_global_plan_path.write_text(
        json.dumps({
            "people": {
                "person-01": {
                    "source_identity": "Woman in a coral blouse",
                    "replacement_identity": "  WOMAN   IN A CORAL BLOUSE  ",
                }
            },
            "entities": {
                "entity-01": {
                    "description": "Thin gold necklace",
                }
            },
            "scenes": {
                "scene-01": {
                    "source_scene": "Residential lawn",
                    "replacement_scene": "source-preserve/no-invention",
                }
            },
        }),
        encoding="utf-8",
    )

    report = evaluate(
        source,
        index_path,
        output_path,
        raw_global_plan_path=raw_global_plan_path,
    )

    raw = report["raw_global_plan"]
    assert raw["exact_normalized_source_target_noop_keys"] == [
        "people/person-01",
        "entities/entity-01",
    ]
    assert raw["source_preserve_target_keys"] == ["scenes/scene-01"]
    assert raw["missing_comparison_keys"] == []
    assert report["score_interpretation"][
        "structural_scores_establish_replacement_success"
    ] is False
