import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import pipeline
from app.codex_runner import CodexOutputValidationError


class _IndexRunner:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[tuple[Path, str]] = []

    def run_isolated_until_output(
        self, cdir: Path, prompt: str, *, session_dir: Path, output_path: Path,
        max_output_bytes: int, validate_output, output_schema: dict,
    ):
        self.calls.append((cdir, prompt))
        assert session_dir.name == "conversation"
        assert (cdir / "SKILL.md").is_file()
        isolated_work = cdir / "work"
        assert not (isolated_work / "prompt.txt").exists()
        request = json.loads(
            (isolated_work / "project_index_request.json").read_text(
                encoding="utf-8"
            )
        )
        assert request["phase"] == "project_index"
        assert [item["segment_index"] for item in request["segments"]] == [1]
        frame_items = request["segments"][0]["frames"]
        assert [item["frame_order"] for item in frame_items] == [1, 2]
        assert [item["path"] for item in frame_items] == [
            "work/segments/1/keyframes/01.png",
            "work/segments/1/keyframes/02.png",
        ]
        for item in frame_items:
            data = (cdir / item["path"]).read_bytes()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image.shape[:2] == (4, 6)
            assert item["sha256"] == hashlib.sha256(data).hexdigest()
        assert sorted(
            path.relative_to(isolated_work).as_posix()
            for path in isolated_work.rglob("*")
            if path.is_file()
        ) == [
            "project_index_request.json",
            "segments/1/keyframes/01.png",
            "segments/1/keyframes/02.png",
        ]
        assert output_schema["properties"]["people"]["type"] == "array"
        model_payload = {
            category: [{"key": key, **item} for key, item in values.items()]
            for category, values in self.payload.items()
        }
        raw = json.dumps(model_payload, ensure_ascii=False).encode("utf-8")
        assert len(raw) <= max_output_bytes
        return validate_output(raw)


def _element_index() -> dict:
    return {
        "people": {
            "person-01": {
                "source_visual_description": "红色外套女性",
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1, 2]}
                ],
                "replaceable": ["face", "hair", "wardrobe"],
                "preserve": ["one identity", "body shape", "relationships"],
            }
        },
        "entities": {
            "entity-01": {
                "source_visual_description": "handheld component",
                "occurrences": [{"segment_index": 1, "frame_orders": [1, 2]}],
                "replaceable": ["appearance"],
                "preserve": ["function", "interface"],
            }
        },
        "scenes": {
            "scene-01": {
                "source_visual_description": "indoor activity area",
                "occurrences": [{"segment_index": 1, "frame_orders": [1, 2]}],
                "replaceable": ["environment"],
                "preserve": ["layout", "camera geometry"],
            }
        },
        "relations": {
            "relation-01": {
                "subject_key": "entity-01",
                "predicate": "held_by",
                "object_key": "person-01",
                "occurrences": [{"segment_index": 1, "frames": [
                    {"frame_order": 1, "state": "held", "geometry": "inside hand"},
                    {"frame_order": 2, "state": "released", "geometry": "apart"},
                ]}],
                "preserve": ["roles", "state sequence"],
                "replace_together": True,
            }
        },
    }


def _png(width: int = 12, height: int = 8, value: int = 96) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_project_index_call_is_once_isolated_and_preserves_segment_outputs(tmp_path):
    cdir = tmp_path / "conversation"
    work = cdir / "work"
    frames = work / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    frame_paths = [frames / "01.png", frames / "02.png"]
    originals = [_png(value=64), _png(value=128)]
    frame_paths[0].write_bytes(originals[0])
    frame_paths[1].write_bytes(originals[1])
    segment_prompt = work / "segments" / "1" / "work" / "prompt.txt"
    segment_prompt.write_text("original segment prompt", encoding="utf-8")
    runner = _IndexRunner(_element_index())

    result = pipeline._generate_project_element_index(
        runner,
        cdir,
        {1: frame_paths},
        skill_bytes=b"frozen video-maker skill",
    )

    assert result == work / "element_index.json"
    assert json.loads(result.read_text(encoding="utf-8")) == _element_index()
    assert segment_prompt.read_text(encoding="utf-8") == "original segment prompt"
    assert [path.read_bytes() for path in frame_paths] == originals
    assert len(runner.calls) == 1
    assert runner.calls[0][0] != cdir
    assert 'phase="project_index"' in runner.calls[0][1]
    assert not runner.calls[0][0].exists()


def test_project_index_call_has_no_retry_or_fallback(tmp_path):
    cdir = tmp_path / "conversation"
    frame = cdir / "work" / "keyframes" / "01.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(_png())

    class FailingRunner:
        calls = 0

        def run_isolated_until_output(
            self, _cdir: Path, _prompt: str, *, session_dir: Path, **_kwargs
        ):
            self.calls += 1
            assert session_dir == cdir
            raise RuntimeError("project index failed")

    runner = FailingRunner()
    with pytest.raises(RuntimeError, match="project index failed"):
        pipeline._generate_project_element_index(
            runner, cdir, {0: [frame]}, skill_bytes=b"frozen video-maker skill"
        )
    assert runner.calls == 1
    assert not (cdir / "work" / "element_index.json").exists()


@pytest.mark.parametrize(
    "invalid_case",
    [
        "element_segment_out_of_range",
        "element_frame_out_of_range",
        "relation_segment_out_of_range",
        "relation_frame_out_of_range",
    ],
)
def test_project_index_rejects_unbound_frame_references(
    tmp_path, invalid_case,
):
    cdir = tmp_path / "conversation"
    frames = cdir / "work" / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    frame_paths = [frames / "01.png", frames / "02.png"]
    for path in frame_paths:
        path.write_bytes(_png())
    payload = copy.deepcopy(_element_index())
    element_occurrences = payload["people"]["person-01"]["occurrences"]
    relation_occurrences = payload["relations"]["relation-01"]["occurrences"]
    if invalid_case == "element_segment_out_of_range":
        element_occurrences[0]["segment_index"] = 2
    elif invalid_case == "element_frame_out_of_range":
        element_occurrences[0]["frame_orders"] = [1, 3]
    elif invalid_case == "relation_segment_out_of_range":
        relation_occurrences[0]["segment_index"] = 2
    else:
        relation_occurrences[0]["frames"][1]["frame_order"] = 3

    with pytest.raises(ValueError, match="project index output is invalid"):
        pipeline._generate_project_element_index(
            _IndexRunner(payload),
            cdir,
            {1: frame_paths},
            skill_bytes=b"frozen video-maker skill",
        )
    assert not (cdir / "work" / "element_index.json").exists()


def test_project_index_canonicalizes_equivalent_same_segment_occurrences(
    tmp_path,
):
    cdir = tmp_path / "conversation"
    frames = cdir / "work" / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    frame_paths = [frames / "01.png", frames / "02.png"]
    for path in frame_paths:
        path.write_bytes(_png())
    payload = copy.deepcopy(_element_index())
    payload["people"]["person-01"]["occurrences"] = [
        {"segment_index": 1, "frame_orders": [2, 1, 1]},
    ]
    payload["entities"]["entity-01"]["occurrences"] = [
        {"segment_index": 1, "frame_orders": [2]},
        {"segment_index": 1, "frame_orders": [1]},
    ]
    payload["scenes"]["scene-01"]["occurrences"] = [
        {"segment_index": 1, "frame_orders": [2]},
        {"segment_index": 1, "frame_orders": [1]},
    ]
    relation = payload["relations"]["relation-01"]
    relation["occurrences"] = [
        {
            "segment_index": 1,
            "frames": list(reversed(relation["occurrences"][0]["frames"])),
        },
        {
            "segment_index": 1,
            "frames": [{
                "frame_order": 1,
                "state": " held ",
                "geometry": " inside hand ",
            }],
        },
    ]

    result = pipeline._generate_project_element_index(
        _IndexRunner(payload),
        cdir,
        {1: frame_paths},
        skill_bytes=b"frozen video-maker skill",
    )

    frozen = json.loads(result.read_text(encoding="utf-8"))
    for category in ("people", "entities", "scenes"):
        item = next(iter(frozen[category].values()))
        assert item["occurrences"] == [
            {"segment_index": 1, "frame_orders": [1, 2]},
        ]
    assert frozen["relations"]["relation-01"]["occurrences"] == [{
        "segment_index": 1,
        "frames": [
            {"frame_order": 1, "state": "held", "geometry": "inside hand"},
            {"frame_order": 2, "state": "released", "geometry": "apart"},
        ],
    }]


def test_project_index_rejects_conflicting_relation_frame_facts(tmp_path):
    cdir = tmp_path / "conversation"
    frames = cdir / "work" / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    frame_paths = [frames / "01.png", frames / "02.png"]
    for path in frame_paths:
        path.write_bytes(_png())
    payload = copy.deepcopy(_element_index())
    payload["relations"]["relation-01"]["occurrences"].append({
        "segment_index": 1,
        "frames": [{
            "frame_order": 1,
            "state": "released",
            "geometry": "apart",
        }],
    })

    with pytest.raises(CodexOutputValidationError) as caught:
        pipeline._generate_project_element_index(
            _IndexRunner(payload),
            cdir,
            {1: frame_paths},
            skill_bytes=b"frozen video-maker skill",
        )

    assert caught.value.reason == "relation_frame_conflict"
    assert caught.value.field_path == "/relations/0/occurrences/1/frames/0"


@pytest.mark.parametrize(
    "invalid_case",
    ["empty", "missing_frame", "duplicate_frame", "unknown_frame"],
)
def test_project_index_requires_exactly_one_scene_for_every_input_frame(
    tmp_path, invalid_case,
):
    cdir = tmp_path / "conversation"
    frames = cdir / "work" / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    frame_paths = [frames / "01.png", frames / "02.png"]
    for path in frame_paths:
        path.write_bytes(_png())
    payload = copy.deepcopy(_element_index())
    scene = payload["scenes"]["scene-01"]
    if invalid_case == "empty":
        payload["scenes"] = {}
    elif invalid_case == "missing_frame":
        scene["occurrences"][0]["frame_orders"] = [1]
    elif invalid_case == "duplicate_frame":
        payload["scenes"]["scene-02"] = {
            **copy.deepcopy(scene),
            "occurrences": [{"segment_index": 1, "frame_orders": [2]}],
        }
    else:
        scene["occurrences"][0]["frame_orders"] = [1, 3]

    with pytest.raises(ValueError, match="project index output is invalid"):
        pipeline._generate_project_element_index(
            _IndexRunner(payload),
            cdir,
            {1: frame_paths},
            skill_bytes=b"frozen video-maker skill",
        )
    assert not (cdir / "work" / "element_index.json").exists()


def test_project_index_allows_empty_people_and_entities(tmp_path):
    cdir = tmp_path / "conversation"
    frames = cdir / "work" / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    frame_paths = [frames / "01.png", frames / "02.png"]
    for path in frame_paths:
        path.write_bytes(_png())
    payload = copy.deepcopy(_element_index())
    payload["people"] = {}
    payload["entities"] = {}
    payload["relations"] = {}

    result = pipeline._generate_project_element_index(
        _IndexRunner(payload),
        cdir,
        {1: frame_paths},
        skill_bytes=b"frozen video-maker skill",
    )

    assert json.loads(result.read_text(encoding="utf-8")) == payload


def test_project_index_filters_invalid_relation_endpoints_with_diagnostics(
    tmp_path,
):
    cdir = tmp_path / "conversation"
    frames = cdir / "work" / "segments" / "1" / "work" / "keyframes"
    frames.mkdir(parents=True)
    frame_paths = [frames / "01.png", frames / "02.png"]
    for path in frame_paths:
        path.write_bytes(_png())
    payload = copy.deepcopy(_element_index())
    valid_relation = copy.deepcopy(payload["relations"]["relation-01"])
    payload["relations"]["relation-02"] = {
        **copy.deepcopy(valid_relation),
        "object_key": "entity-99",
    }
    payload["relations"]["relation-03"] = {
        **copy.deepcopy(valid_relation),
        "subject_key": "entity-01",
        "object_key": "entity-01",
    }
    payload["relations"]["relation-04"] = {
        **copy.deepcopy(valid_relation),
    }
    payload["relations"]["relation-05"] = {
        **copy.deepcopy(valid_relation),
        "subject_key": "entity-99",
    }

    result = pipeline._generate_project_element_index(
        _IndexRunner(payload),
        cdir,
        {1: frame_paths},
        skill_bytes=b"frozen video-maker skill",
    )

    frozen = json.loads(result.read_text(encoding="utf-8"))
    assert frozen["relations"] == {
        "relation-01": valid_relation,
        "relation-04": valid_relation,
    }
    diagnostic = json.loads(
        (cdir / "work" / "errors" / "project-index-filtered.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["error"] == {
        "code": "project_index_fields_filtered",
        "dropped_paths": [
            "/relations/1/object_key",
            "/relations/2/object_key",
            "/relations/4/subject_key",
        ],
        "dropped_count": 3,
        "filters": [
            {
                "path": "/relations/1/object_key",
                "reason": "relation_object_unknown",
                "count": 1,
            },
            {
                "path": "/relations/2/object_key",
                "reason": "relation_endpoints_identical",
                "count": 1,
            },
            {
                "path": "/relations/4/subject_key",
                "reason": "relation_subject_unknown",
                "count": 1,
            },
        ],
    }


def test_segmented_image_prompt_generation_passes_element_index_once(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    keyframes = work / "segments" / "1" / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(b"frozen")
    element_index_path = work / "element_index.json"
    element_index_path.write_text(
        json.dumps(_element_index(), ensure_ascii=False), encoding="utf-8"
    )
    captured = []

    def generate(
        _settings,
        _runner,
        specs,
        *,
        session_dir,
        step,
        element_index_path,
        **_kwargs,
    ):
        captured.append((specs, session_dir, step, element_index_path))
        return {"version": 4}, {1: {}}

    monkeypatch.setattr(pipeline, "_generate_image_optimization_project", generate)

    result = pipeline._generate_segmented_image_prompts(
        object(),
        object(),
        [{"index": 1, "chain_id": "chain-1", "join_mode": "hard_cut"}],
        [{"index": 1, "keyframes": ["01.png"]}],
        work,
        session_dir=tmp_path,
        element_index_path=element_index_path,
        skill_bytes=b"frozen image-postprocess skill",
    )

    assert result == ({"version": 4}, {1: {}})
    assert len(captured) == 1
    assert captured[0][3] == element_index_path


def test_image_project_creates_index_once_outside_existing_image_retries(
    tmp_path, monkeypatch
):
    keyframes = tmp_path / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(b"frozen")
    element_index_path = tmp_path / "work" / "element_index.json"
    index_calls = []
    image_calls = []

    def make_index(_runner, session_dir, frame_paths, **_kwargs):
        index_calls.append((session_dir, frame_paths))
        element_index_path.write_text(
            json.dumps(_element_index(), ensure_ascii=False), encoding="utf-8"
        )
        return element_index_path

    def generate(_runner, _segments, _mode, **kwargs):
        image_calls.append(kwargs["element_index_path"])
        if len(image_calls) == 1:
            raise pipeline.image_optimization.ImageOptimizationOutputError("retry")
        return {"version": 4}, {0: {}}

    monkeypatch.setattr(pipeline, "_generate_project_element_index", make_index)
    monkeypatch.setattr(
        pipeline.image_optimization, "generate_project_prompts", generate
    )
    settings = SimpleNamespace(
        retry_count=1,
        retry_interval_s=0,
        seedream_edit_mode="conservative",
    )

    result = pipeline._generate_image_optimization_project(
        settings,
        object(),
        [
            {
                "index": 0,
                "keyframes_dir": keyframes,
                "chain_id": "short-000",
                "join_mode": "hard_cut",
            }
        ],
        session_dir=tmp_path,
        step="image project",
        skill_bytes=b"frozen image-postprocess skill",
        video_skill_bytes=b"frozen video-maker skill",
    )

    assert result == ({"version": 4}, {0: {}})
    assert len(index_calls) == 1
    assert image_calls == [element_index_path, element_index_path]
