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
    REPLACEMENT_SHEET_SPEC_VERSION,
    REPLACEMENT_SHEET_WIDTH,
    build_replacement_sheet_spec,
    finalize_replacement_sheet,
    materialize_replacement_sheet_inputs,
    replacement_sheet_spec_sha256,
    validate_replacement_sheet_spec,
)


def _replacement(
    appearance: str,
    *,
    source: str = "generated_variant",
    target: str = "TARGET_ACTION_RELATION_SENTINEL",
    occurrences: object = None,
) -> dict[str, object]:
    return {
        "source_visual_description": "SOURCE_VISUAL_SENTINEL",
        "occurrences": (
            ["OCCURRENCE_SENTINEL"] if occurrences is None else occurrences
        ),
        "replacement": {
            "source": source,
            "target_visual_description": target,
            "appearance_description": appearance,
        },
    }


def test_v2_spec_is_globally_ordered_and_appearance_only(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    enhanced = {
        "people": {
            "person-10": _replacement("silver braided bob and oval face"),
            "person-03": {"replacement": None},
            "person-02": _replacement(
                "warm brown freckled forearm with a narrow scar",
                source="image_1",
            ),
        },
        "entities": {
            "entity-01": _replacement("matte green ceramic grooming tool"),
        },
        "scenes": {
            "scene-01": _replacement("sunlit oak walk-in wardrobe"),
        },
        "relations": {
            "RELATION_SENTINEL": {
                "description": "person holds entity beside another entity"
            }
        },
    }

    spec = build_replacement_sheet_spec(enhanced)

    assert spec["version"] == REPLACEMENT_SHEET_SPEC_VERSION == 2
    assert [tile["stable_key"] for tile in spec["tiles"]] == [
        "person-02",
        "person-10",
        "entity-01",
        "scene-01",
    ]
    assert [tile["tile_id"] for tile in spec["tiles"]] == [
        "TILE_01",
        "TILE_02",
        "TILE_03",
        "TILE_04",
    ]
    assert [(tile["row"], tile["column"]) for tile in spec["tiles"]] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]
    assert (spec["columns"], spec["rows"]) == (2, 2)
    assert spec["tiles"][0] == {
        "tile_id": "TILE_01",
        "category": "people",
        "stable_key": "person-02",
        "row": 1,
        "column": 1,
        "source": "image_1",
        "appearance_description": (
            "warm brown freckled forearm with a narrow scar"
        ),
    }
    assert all("target" not in tile for tile in spec["tiles"])
    assert all("target_visual_description" not in tile for tile in spec["tiles"])
    assert all("source_frame_receipt" not in tile for tile in spec["tiles"])
    assert all("occurrences" not in tile for tile in spec["tiles"])
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
    assert validate_replacement_sheet_spec(spec, allow_empty=False) == spec["tiles"]


def test_spec_with_no_replacement_is_empty(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    spec = build_replacement_sheet_spec(
        {
            "people": {"person-01": {"replacement": None}},
            "entities": {},
            "scenes": {},
            "relations": {},
        },
    )

    assert spec["tiles"] == []
    assert spec["columns"] == 0
    assert spec["rows"] == 0
    assert spec["spec_sha256"] == replacement_sheet_spec_sha256(spec)
    assert validate_replacement_sheet_spec(spec, allow_empty=True) == []


@pytest.mark.parametrize(
    "replacement",
    [
        {"source": "generated_variant", "appearance_description": ""},
        {"source": "generated_variant", "appearance_description": "   "},
        {"source": "generated_variant"},
        {"source": "unknown", "appearance_description": "new design"},
    ],
)
def test_spec_rejects_invalid_appearance_contract(
    tmp_path: Path,
    replacement: dict[str, object],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    enhanced = {
        "people": {"person-01": {"replacement": replacement}},
        "entities": {},
        "scenes": {},
        "relations": {},
    }

    with pytest.raises(ValueError, match="replacement sheet"):
        build_replacement_sheet_spec(enhanced)


def test_materialize_writes_only_bound_layout_prompt_and_spec(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    enhanced = {
        "people": {
            "person-01": _replacement("silver braided bob and angular face"),
        },
        "entities": {
            "entity-01": _replacement("matte green ceramic grooming tool"),
        },
        "scenes": {
            "scene-01": _replacement("empty sunlit oak walk-in wardrobe"),
        },
        "relations": {
            "RELATION_PROMPT_LEAK_SENTINEL": {
                "description": "hand grips tool while brushing animal"
            }
        },
    }
    spec = build_replacement_sheet_spec(enhanced)
    directory = project / "work/replacement-sheet"

    first = materialize_replacement_sheet_inputs(directory, spec)
    first_bytes = {
        name: (directory / name).read_bytes()
        for name in ("canvas.png", "prompt.txt", "spec.json")
    }
    second = materialize_replacement_sheet_inputs(directory, spec)

    assert first == {
        "spec": spec,
        "spec_sha256": spec["spec_sha256"],
        "prompt": first["prompt"],
        "canvas_path": directory / "canvas.png",
    }
    assert first["prompt"] == second["prompt"]
    assert not (directory / "source-evidence.png").exists()
    assert (directory / "prompt.txt").read_text(encoding="utf-8") == first["prompt"]
    for name, payload in first_bytes.items():
        assert (directory / name).read_bytes() == payload
        assert stat.S_IMODE((directory / name).stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert not list(directory.glob(".*.tmp"))
    assert json.loads((directory / "spec.json").read_text()) == spec

    canvas = cv2.imread(str(directory / "canvas.png"))
    assert canvas.shape[:2] == (
        REPLACEMENT_SHEET_HEIGHT,
        REPLACEMENT_SHEET_WIDTH,
    )
    prompt = first["prompt"]
    assert f"frozen global-index binding is {spec['spec_sha256']}" in prompt
    assert "Input Image 1 is the blank 2048x2048" in prompt
    assert "Input Image 2" not in prompt
    assert "source-frame evidence" not in prompt
    assert "silver braided bob and angular face" in prompt
    assert "matte green ceramic grooming tool" in prompt
    assert "empty sunlit oak walk-in wardrobe" in prompt
    assert "TARGET_ACTION_RELATION_SENTINEL" not in prompt
    assert "RELATION_PROMPT_LEAK_SENTINEL" not in prompt
    assert "SOURCE_VISUAL_SENTINEL" not in prompt
    assert "OCCURRENCE_SENTINEL" not in prompt
    assert "MISSING_FRAME_SENTINEL" not in prompt


def test_prompt_enforces_per_category_cell_isolation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        {
            "people": {
                "person-01": _replacement("freckled left forearm with a scar"),
            },
            "entities": {
                "entity-01": _replacement("red-brown dog with a white chest"),
            },
            "scenes": {
                "scene-01": _replacement("board-formed concrete studio"),
            },
            "relations": {},
        },
    )

    prompt = materialize_replacement_sheet_inputs(
        project / "work/replacement-sheet",
        spec,
    )["prompt"]

    assert "appearance atlas, not a reenacted video frame" in prompt
    assert "each listed stable key exactly once" in prompt
    assert "Only that cell's stable key may appear in it" in prompt
    assert "PEOPLE tiles: show exactly one replacement person" in prompt
    assert "Show no props, held objects" in prompt
    assert "ENTITIES tiles: show exactly one isolated replacement object or animal" in prompt
    assert "Show no person, hand, other entity" in prompt
    assert "SCENES tiles: show one empty, unoccupied replacement environment" in prompt
    assert "show no indexed person, animal, movable entity" in prompt
    assert "APPEARANCE=\"freckled left forearm with a scar\"" in prompt
    assert "APPEARANCE=\"red-brown dog with a white chest\"" in prompt
    assert "APPEARANCE=\"board-formed concrete studio\"" in prompt


def test_materialize_uses_canvas_first_and_scopes_user_image(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        {
            "people": {},
            "entities": {
                "entity-01": _replacement(
                    "exact wooden paddle brush design",
                    source="image_1",
                ),
                "entity-02": _replacement("matte steel comb"),
            },
            "scenes": {},
            "relations": {},
        },
    )

    result = materialize_replacement_sheet_inputs(
        project / "work/replacement-sheet",
        spec,
        user_image_present=True,
    )

    prompt = result["prompt"]
    image_1 = prompt.index("Input Image 1 is the blank 2048x2048")
    image_2 = prompt.index("Input Image 2 is the user-reference image")
    assert image_1 < image_2
    assert "SOURCE=image_1 maps only to this Seedream Input Image 2" in prompt
    assert "SOURCE=generated_variant must ignore it completely" in prompt
    assert "Input Image 3" not in prompt
    assert "source-frame evidence" not in prompt
    assert (
        'TILE_01 | row=1 | column=1 | category=entities | '
        'stable_key="entity-01" | SOURCE="image_1" | '
        'APPEARANCE="exact wooden paddle brush design"'
    ) in prompt


def test_materialize_rejects_missing_user_image_for_bound_tile(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        {
            "people": {},
            "entities": {
                "entity-01": _replacement("wooden brush", source="image_1"),
            },
            "scenes": {},
            "relations": {},
        },
    )

    with pytest.raises(ValueError, match="user-reference image is missing"):
        materialize_replacement_sheet_inputs(
            project / "work/replacement-sheet",
            spec,
        )


def test_materialize_skips_empty_spec_for_caller_control(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        {"people": {}, "entities": {}, "scenes": {}, "relations": {}}
    )

    with pytest.raises(ValueError, match="no replacement tiles"):
        materialize_replacement_sheet_inputs(
            project / "work/replacement-sheet",
            spec,
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

    spec = build_replacement_sheet_spec(enhanced)

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
        {
            "people": {"person-01": _replacement("new person")},
            "entities": {"entity-01": _replacement("new entity")},
            "scenes": {},
            "relations": {},
        },
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


def test_rejects_unsupported_or_non_closed_v2_binding(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = build_replacement_sheet_spec(
        {
            "people": {"person-01": _replacement("new person")},
            "entities": {},
            "scenes": {},
            "relations": {},
        },
    )
    unsupported = dict(spec)
    unsupported["version"] = 1
    unsupported["spec_sha256"] = replacement_sheet_spec_sha256(unsupported)
    with pytest.raises(ValueError, match="spec binding is invalid"):
        validate_replacement_sheet_spec(unsupported, allow_empty=False)

    for invalid_version in (True, 2.0):
        invalid_type = dict(spec)
        invalid_type["version"] = invalid_version
        invalid_type["spec_sha256"] = replacement_sheet_spec_sha256(invalid_type)
        with pytest.raises(ValueError, match="spec binding is invalid"):
            validate_replacement_sheet_spec(invalid_type, allow_empty=False)

    extra_field = dict(spec)
    extra_field["source_evidence_path"] = "old-source-evidence.png"
    extra_field["spec_sha256"] = replacement_sheet_spec_sha256(extra_field)
    with pytest.raises(ValueError, match="spec binding is invalid"):
        validate_replacement_sheet_spec(extra_field, allow_empty=False)

    uppercase_digest = dict(spec)
    uppercase_digest["enhanced_element_index_sha256"] = "A" * 64
    uppercase_digest["spec_sha256"] = replacement_sheet_spec_sha256(
        uppercase_digest
    )
    with pytest.raises(ValueError, match="spec binding is invalid"):
        validate_replacement_sheet_spec(uppercase_digest, allow_empty=False)

    uppercase_spec_digest = dict(spec)
    uppercase_spec_digest["spec_sha256"] = spec["spec_sha256"].upper()
    with pytest.raises(ValueError, match="spec binding is invalid"):
        validate_replacement_sheet_spec(uppercase_spec_digest, allow_empty=False)

    spec["tiles"][0]["stable_key"] = "person-99"
    with pytest.raises(ValueError, match="spec binding is invalid"):
        materialize_replacement_sheet_inputs(
            project / "work/replacement-sheet",
            spec,
        )
