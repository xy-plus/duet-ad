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
    assert report["decision"] is None
