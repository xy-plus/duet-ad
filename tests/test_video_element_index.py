import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import pipeline


class _IndexRunner:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[tuple[Path, str]] = []

    def run_isolated(
        self, cdir: Path, prompt: str, *, session_dir: Path
    ) -> None:
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
        (isolated_work / "element_index.json").write_text(
            json.dumps(self.payload, ensure_ascii=False), encoding="utf-8"
        )


def _element_index() -> dict:
    return {
        "people": {
            "woman-red-coat": {
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
        "scenes": {},
        "relations": {
            "relation-01": {
                "subject_key": "entity-01",
                "predicate": "held_by",
                "object_key": "woman-red-coat",
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

        def run_isolated(
            self, _cdir: Path, _prompt: str, *, session_dir: Path
        ) -> None:
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
