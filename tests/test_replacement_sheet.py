from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.replacement_sheet import (
    REPLACEMENT_SHEET_HEIGHT,
    REPLACEMENT_SHEET_WIDTH,
    build_replacement_sheet_spec,
    finalize_replacement_sheet,
    materialize_replacement_sheet_inputs,
    replacement_sheet_spec_sha256,
)


def _write_png(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((72, 128, 3), color, dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", image)
    assert encoded
    path.write_bytes(payload.tobytes())
    return path


def _replacement(
    target: str,
    *,
    source: str = "generated_variant",
    occurrences: object = None,
) -> dict[str, object]:
    return {
        "source_visual_description": f"source for {target}",
        "occurrences": [] if occurrences is None else occurrences,
        "replacement": {
            "source": source,
            "target_visual_description": target,
        },
    }


def test_spec_is_globally_ordered_without_guessing_missing_occurrences(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    first = _write_png(project / "work/segments/0001/analysis-01.png", (10, 20, 30))
    second = _write_png(project / "work/segments/0002/analysis-01.png", (40, 50, 60))
    third = _write_png(project / "work/segments/0002/analysis-02.png", (70, 80, 90))
    prepared = [
        {"index": 2, "analysis_paths": [second, third]},
        {"index": 1, "analysis_paths": [first]},
    ]
    enhanced = {
        "people": {
            "person-10": _replacement(
                "new tenth person",
                occurrences=[{"segment_index": 999, "frame_orders": [1]}],
            ),
            "person-03": {"replacement": None},
            "person-02": _replacement(
                "new second person",
                source="image_1",
                occurrences=[{"segment_index": 2, "frame_orders": [2]}],
            ),
        },
        "entities": {
            "entity-01": _replacement(
                "new object",
                occurrences=[{"segment_index": 2, "frame_orders": [1]}],
            ),
        },
        "scenes": {
            "scene-01": _replacement(
                "new room",
                occurrences=[{"segment_index": "bad", "frame_orders": [1]}],
            ),
        },
        "relations": {
            "relation-01": {
                "replacement": {
                    "source": "generated_variant",
                    "target_visual_description": "must not get a tile",
                }
            }
        },
    }

    spec = build_replacement_sheet_spec(project, enhanced, prepared)

    assert [tile["stable_key"] for tile in spec["tiles"]] == [
        "person-02",
        "person-10",
        "entity-01",
        "scene-01",
    ]
    assert [tile["tile_id"] for tile in spec["tiles"]] == [
        "TILE_01", "TILE_02", "TILE_03", "TILE_04",
    ]
    assert [(tile["row"], tile["column"]) for tile in spec["tiles"]] == [
        (1, 1), (1, 2), (2, 1), (2, 2),
    ]
    assert (spec["columns"], spec["rows"]) == (2, 2)
    assert spec["tiles"][0]["source"] == "image_1"
    assert spec["tiles"][0]["source_frame_receipt"]["path"] == (
        "work/segments/0002/analysis-02.png"
    )
    assert spec["tiles"][1]["source_frame_receipt"] is None
    assert spec["tiles"][2]["source_frame_receipt"]["path"] == (
        "work/segments/0002/analysis-01.png"
    )
    assert spec["tiles"][3]["source_frame_receipt"] is None
    assert spec["spec_sha256"] == replacement_sheet_spec_sha256(spec)
    canonical_index = json.dumps(
        enhanced,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert spec["enhanced_element_index_sha256"] == hashlib.sha256(
        canonical_index
    ).hexdigest()


def test_spec_with_no_replacement_is_empty_and_does_not_require_frames(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        project,
        {
            "people": {"person-01": {"replacement": None}},
            "entities": {},
            "scenes": {},
            "relations": {},
        },
        [],
    )

    assert spec["tiles"] == []
    assert spec["columns"] == 0
    assert spec["rows"] == 0
    assert spec["spec_sha256"] == replacement_sheet_spec_sha256(spec)


def test_missing_analysis_frames_do_not_become_a_quality_gate(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        project,
        {
            "people": {
                "person-01": _replacement(
                    "new person",
                    occurrences=[{"segment_index": 1, "frame_orders": [9]}],
                ),
            },
            "entities": {},
            "scenes": {},
            "relations": {},
        },
        [{"index": 1, "analysis_paths": [project / "missing.png"]}],
    )

    assert len(spec["tiles"]) == 1
    assert spec["tiles"][0]["source_frame_receipt"] is None


def test_materialize_writes_one_bound_layout_and_evidence_sheet_atomically(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    frame = _write_png(
        project / "work/segments/0001/analysis-01.png",
        (12, 80, 170),
    )
    enhanced = {
        "people": {
            "person-01": _replacement(
                "a distinctly different lead with a silver braided bob",
                occurrences=[{"segment_index": 1, "frame_orders": [1]}],
            ),
        },
        "entities": {
            "entity-01": _replacement(
                "a matte green ceramic device",
                source="image_1",
                occurrences=[{"segment_index": 1, "frame_orders": [1]}],
            ),
        },
        "scenes": {
            "scene-01": _replacement(
                "a different sunlit studio instance",
                occurrences=[{"segment_index": 1, "frame_orders": [1]}],
            ),
        },
        "relations": {},
    }
    spec = build_replacement_sheet_spec(
        project,
        enhanced,
        [{"index": 1, "analysis_paths": [frame]}],
    )
    directory = project / "work/replacement-sheet"

    first = materialize_replacement_sheet_inputs(
        project,
        directory,
        spec,
        "Base object-replace design direction.",
    )
    first_bytes = {
        name: (directory / name).read_bytes()
        for name in ("canvas.png", "source-evidence.png", "prompt.txt", "spec.json")
    }
    second = materialize_replacement_sheet_inputs(
        project,
        directory,
        spec,
        "Base object-replace design direction.",
    )

    assert first["spec_sha256"] == spec["spec_sha256"]
    assert first["prompt"] == second["prompt"]
    assert first["canvas_path"] == directory / "canvas.png"
    assert first["source_evidence_path"] == directory / "source-evidence.png"
    assert (directory / "prompt.txt").read_text(encoding="utf-8") == first["prompt"]
    assert hashlib.sha256((directory / "prompt.txt").read_bytes()).hexdigest() == (
        hashlib.sha256(first["prompt"].encode("utf-8")).hexdigest()
    )
    for name, payload in first_bytes.items():
        assert (directory / name).read_bytes() == payload
        assert stat.S_IMODE((directory / name).stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert not list(directory.glob(".*.tmp"))
    assert json.loads((directory / "spec.json").read_text()) == spec

    canvas = cv2.imread(str(directory / "canvas.png"))
    evidence = cv2.imread(str(directory / "source-evidence.png"))
    assert canvas.shape[:2] == (
        REPLACEMENT_SHEET_HEIGHT,
        REPLACEMENT_SHEET_WIDTH,
    )
    assert evidence.shape == canvas.shape
    assert not np.array_equal(canvas, evidence)
    prompt = (directory / "prompt.txt").read_text(encoding="utf-8")
    assert prompt.startswith("Base object-replace design direction.")
    assert f"frozen global-index binding is {spec['spec_sha256']}" in prompt
    assert "Input Image 1 is the blank 2048x2048" in prompt
    assert "Input Image 2 is source-frame evidence" in prompt
    assert "No user-reference image is supplied" in prompt
    assert "Input Image 3" not in prompt
    assert "TILE_01 | row=1 | column=1 | category=people" in prompt
    assert 'stable_key="entity-01" | SOURCE="image_1"' in prompt


def test_materialize_numbers_user_image_before_canvas_and_evidence(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        project,
        {
            "people": {},
            "entities": {
                "entity-01": _replacement(
                    "the exact replacement product design",
                    source="image_1",
                ),
            },
            "scenes": {},
            "relations": {},
        },
        [],
    )

    result = materialize_replacement_sheet_inputs(
        project,
        project / "work/replacement-sheet",
        spec,
        "Use the object-replace design without changing its stable key.",
        user_image_present=True,
    )

    prompt = result["prompt"]
    image_1 = prompt.index("Input Image 1 is the user-reference image")
    image_2 = prompt.index("Input Image 2 is the blank 2048x2048")
    image_3 = prompt.index("Input Image 3 is source-frame evidence")
    assert image_1 < image_2 < image_3
    assert "only a tile whose SOURCE is exactly image_1" in prompt
    assert "No user-reference image is supplied" not in prompt
    assert (
        'TILE_01 | row=1 | column=1 | category=entities | '
        'stable_key="entity-01" | SOURCE="image_1"'
    ) in prompt


def test_materialize_rejects_drifted_bound_source_evidence(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    frame = _write_png(
        project / "work/segments/0001/analysis-01.png",
        (12, 80, 170),
    )
    spec = build_replacement_sheet_spec(
        project,
        {
            "people": {},
            "entities": {
                "entity-01": _replacement(
                    "a matte green ceramic device",
                    occurrences=[{"segment_index": 1, "frame_orders": [1]}],
                ),
            },
            "scenes": {},
            "relations": {},
        },
        [{"index": 1, "analysis_paths": [frame]}],
    )
    _write_png(frame, (170, 80, 12))

    with pytest.raises(ValueError, match="source-frame receipt does not match"):
        materialize_replacement_sheet_inputs(
            project,
            project / "work/replacement-sheet",
            spec,
            "Use the frozen target.",
        )


def test_materialize_skips_empty_spec_for_caller_control(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        project,
        {"people": {}, "entities": {}, "scenes": {}, "relations": {}},
        [],
    )

    with pytest.raises(ValueError, match="no replacement tiles"):
        materialize_replacement_sheet_inputs(
            project,
            project / "work/replacement-sheet",
            spec,
            "unused",
        )


def test_spec_never_truncates_large_replacement_sets(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    enhanced = {
        "people": {},
        "entities": {
            f"entity-{index:04d}": _replacement(f"variant {index}")
            for index in range(1, 258)
        },
        "scenes": {},
        "relations": {},
    }

    spec = build_replacement_sheet_spec(project, enhanced, [])

    assert len(spec["tiles"]) == 257
    assert spec["tiles"][0]["tile_id"] == "TILE_01"
    assert spec["tiles"][-1]["tile_id"] == "TILE_257"
    assert spec["tiles"][-1]["stable_key"] == "entity-0257"
    assert (spec["columns"], spec["rows"]) == (17, 16)


def test_finalize_normalizes_to_one_2048_png_and_restores_labels(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        project,
        {
            "people": {"person-01": _replacement("new person")},
            "entities": {"entity-01": _replacement("new entity")},
            "scenes": {},
            "relations": {},
        },
        [],
    )
    provider_image = np.full((180, 320, 3), (31, 63, 127), dtype=np.uint8)
    encoded, payload = cv2.imencode(".jpg", provider_image)
    assert encoded
    destination = project / "work/replacement-sheet/reference-sheet.png"

    first = finalize_replacement_sheet(payload.tobytes(), spec, destination)
    first_bytes = destination.read_bytes()
    second = finalize_replacement_sheet(destination, spec, destination)
    second_bytes = destination.read_bytes()

    decoded = cv2.imdecode(np.frombuffer(first_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (2048, 2048)
    assert decoded.std() > 0
    assert first == {
        "path": destination.resolve().as_posix(),
        "size_bytes": len(first_bytes),
        "sha256": hashlib.sha256(first_bytes).hexdigest(),
        "width": 2048,
        "height": 2048,
    }
    assert second["path"] == first["path"]
    assert cv2.imdecode(
        np.frombuffer(second_bytes, np.uint8), cv2.IMREAD_COLOR
    ).shape[:2] == (2048, 2048)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_materialize_rejects_a_mutated_binding_spec(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        project,
        {
            "people": {"person-01": _replacement("new person")},
            "entities": {},
            "scenes": {},
            "relations": {},
        },
        [],
    )
    spec["tiles"][0]["stable_key"] = "person-99"

    with pytest.raises(ValueError, match="spec binding is invalid"):
        materialize_replacement_sheet_inputs(
            project,
            project / "work/replacement-sheet",
            spec,
            "base",
        )
