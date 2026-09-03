import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import generation_config, image_optimization, mediakit, pipeline, postprocess
from conftest import make_settings


def _png(value: int) -> bytes:
    image = np.full((48, 64, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _sampling(raw: tuple[bytes, ...]) -> dict:
    return {
        "schema": "duet.backend-keyframe-sampling",
        "version": 1,
        "selection_method": "scene-anchor-capacity-hamilton-v1",
        "keyframes": [
            {
                "order": order,
                "path": f"keyframes/{order:02d}.png",
                "sha256": hashlib.sha256(data).hexdigest(),
                "source_scene_id": "SCENE_01",
                "source_time_s": float(order - 1),
                "repeated": False,
            }
            for order, data in enumerate(raw, 1)
        ],
    }


def _install_fake_mediakit(monkeypatch, calls: list[tuple[str, str]]) -> None:
    async def erase(_settings, _cdir, image, out, _confirm, scenes, *, gate=None):
        del gate
        scene = scenes[0]
        calls.append((scene, str(image)))
        source = cv2.imread(str(image), cv2.IMREAD_COLOR)
        assert source is not None
        source[:] = 100 if scene == mediakit.TEXT_SCENE else 200
        out.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(out), source)
        receipt_path = out.parent / ".mediakit" / f"{out.name}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        inspected = mediakit._inspect_input(image)
        output_sha256 = mediakit._inspect_input(
            out, max_bytes=mediakit.MAX_OUTPUT_BYTES,
        )["sha256"]
        receipt_path.write_text(json.dumps({
            "version": mediakit.RECEIPT_VERSION,
            "state": "succeeded",
            "source": inspected,
            "scenes": [scene],
            "output": out.name,
            "output_sha256": output_sha256,
            "stages": [{
                "scene": scene,
                "state": "succeeded",
                "output_sha256": output_sha256,
            }],
        }), encoding="utf-8")
        return out

    monkeypatch.setattr(mediakit, "erase_image", erase)


def test_three_frames_are_cleaned_in_stage_order_and_reused(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cdir = settings.data_dir / "cid"
    work = cdir / "work/segments/1/work"
    keyframes = work / "keyframes"
    keyframes.mkdir(parents=True)
    raw = tuple(_png(order) for order in range(1, 4))
    for order, data in enumerate(raw, 1):
        (keyframes / f"{order:02d}.png").write_bytes(data)
    calls: list[tuple[str, str]] = []
    _install_fake_mediakit(monkeypatch, calls)

    canonical, receipt, paths = pipeline._prepare_previsual_keyframes(
        settings,
        cdir,
        1,
        work,
        _sampling(raw),
        {"remove_subtitle": True, "remove_watermark": True},
        None,
    )

    assert len(canonical) == len(paths) == 3
    assert [scene for scene, _path in calls] == (
        [mediakit.TEXT_SCENE] * 3 + [mediakit.ICON_SCENE] * 3
    )
    assert all("/text/" in source for _scene, source in calls[3:])
    assert tuple((keyframes / f"{order:02d}.png").read_bytes() for order in range(1, 4)) == raw
    assert receipt["version"] == 3
    assert receipt["keyframe_count"] == 3
    assert receipt["preprocess"] == {
        "remove_subtitle": True,
        "remove_watermark": True,
    }
    assert [item["artifact"]["sha256"] for item in receipt["keyframes"]] == [
        hashlib.sha256(data).hexdigest() for data in canonical
    ]

    pipeline._prepare_previsual_keyframes(
        settings,
        cdir,
        1,
        work,
        _sampling(raw),
        {"remove_subtitle": True, "remove_watermark": True},
        None,
    )
    assert len(calls) == 6


def test_previsual_failure_stops_before_any_visual_model(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cdir = settings.data_dir / "cid"
    work = cdir / "work/segments/1/work"
    keyframes = work / "keyframes"
    keyframes.mkdir(parents=True)
    raw = tuple(_png(order) for order in range(1, 4))
    for order, data in enumerate(raw, 1):
        (keyframes / f"{order:02d}.png").write_bytes(data)

    async def fail(_settings, _cdir, _image, out, *_args, **_kwargs):
        if out.name == "01.png":
            raise mediakit.MediaKitError(502, "erase failed")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_png(90))
        return out

    monkeypatch.setattr(mediakit, "erase_image", fail)
    with pytest.raises(pipeline.PipelineError, match="erase failed"):
        pipeline._prepare_previsual_keyframes(
            settings,
            cdir,
            1,
            work,
            _sampling(raw),
            {"remove_subtitle": True, "remove_watermark": False},
            None,
        )
    assert not (work / "keyframe_sampling.json").is_file()


def test_video_maker_sees_only_cleaned_proxies(tmp_path):
    cdir = tmp_path / "conversation"
    work = cdir / "work"
    work.mkdir(parents=True)
    raw = tuple(_png(10) for _order in range(3))
    cleaned = tuple(_png(200) for _order in range(3))
    (work / "01_frame_000.000s.png").write_bytes(raw[0])
    (work / "contact_sheet_01.jpg").write_bytes(b"raw-overview")

    class Runner:
        def run_isolated_until_output(
            self, stage, _prompt, *, session_dir, output_path,
            max_output_bytes, validate_output, output_schema,
        ):
            del output_path, max_output_bytes, output_schema
            assert session_dir == cdir
            assert not list((stage / "work").glob("*_frame_*.png"))
            assert not list((stage / "work").glob("contact_sheet*.jpg"))
            for order in range(1, 4):
                image = cv2.imread(
                    str(stage / "work/keyframes" / f"{order:02d}.png"),
                    cv2.IMREAD_COLOR,
                )
                assert image is not None
                assert image.mean() > 190
            return validate_output(b'{"prompt":"complete natural language prompt"}')

    names, prompt = pipeline._run_visual_attempt(
        Runner(),
        cdir,
        "analyze",
        work,
        isolate_dialogue=False,
        frozen_keyframes=raw,
        analysis_keyframes=cleaned,
        skill_bytes=b"frozen video-maker skill",
    )
    assert names == [f"{order:02d}.png" for order in range(1, 4)]
    assert prompt == "complete natural language prompt"
    assert tuple((work / "keyframes" / name).read_bytes() for name in names) == raw


def test_project_index_and_image_canonical_share_the_cleaned_frame_set(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, retry_count=0)
    cdir = settings.data_dir / "cid"
    cleaned = cdir / "work/.postprocess-private/v4-canvases/0001/brand"
    cleaned.mkdir(parents=True)
    for order in range(1, 4):
        (cleaned / f"{order:02d}.png").write_bytes(_png(200))
    element_index = cdir / "work/element_index.json"
    element_index.parent.mkdir(parents=True, exist_ok=True)
    element_index.write_text("{}", encoding="utf-8")
    observed: list[str] = []

    def index(_runner, session_dir, frame_paths, **_kwargs):
        assert session_dir == cdir
        assert list(frame_paths) == [1]
        assert [path.parent for path in frame_paths[1]] == [cleaned] * 3
        observed.append("project-index")
        return element_index

    def canonical(_runner, specs, _mode, **kwargs):
        assert specs[0]["keyframes_dir"] == cleaned
        assert kwargs["element_index_path"] == element_index
        observed.append("image-canonical")
        return {"version": 4}, {1: "prompt"}

    monkeypatch.setattr(pipeline, "_generate_project_element_index", index)
    monkeypatch.setattr(image_optimization, "generate_project_prompts", canonical)
    continuity, prompts = pipeline._generate_image_optimization_project(
        settings,
        object(),
        [{
            "index": 1,
            "chain_id": "chain-001",
            "join_mode": "hard_cut",
            "keyframes_dir": cleaned,
        }],
        session_dir=cdir,
        step="image canonical",
        skill_bytes=b"image skill",
        video_skill_bytes=b"video skill",
        render_options={"remove_subtitle": True, "remove_watermark": True},
    )
    assert observed == ["project-index", "image-canonical"]
    assert continuity == {"version": 4}
    assert prompts == {1: "prompt"}


def test_seedream_uses_cleaned_frames_and_postprocess_does_not_erase_again(tmp_path):
    cdir = tmp_path / "cid"
    work = cdir / "work"
    cleaned = work / ".postprocess-private/v4-canvases/0001/brand"
    cleaned.mkdir(parents=True)
    relative_paths = []
    for order in range(1, 4):
        path = cleaned / f"{order:02d}.png"
        path.write_bytes(_png(200))
        relative_paths.append(str(path.relative_to(work)))
    grouped = postprocess._group_targets(cdir, {
        "segments": [{
            "index": 1,
            "keyframe_paths": relative_paths,
            "keyframe_sampling": {
                "version": 3,
                "keyframe_count": 3,
                "keyframes": [
                    {"artifact": {"sha256": hashlib.sha256(
                        (cleaned / f"{order:02d}.png").read_bytes()
                    ).hexdigest()}}
                    for order in range(1, 4)
                ],
            },
        }],
    })
    assert [source for source, _target in grouped[1]] == [
        cleaned / f"{order:02d}.png" for order in range(1, 4)
    ]
    assert generation_config.postprocess_options({
        "optimize_image": True,
        "remove_subtitle": True,
        "remove_watermark": True,
    }) == {
        "optimize_image": True,
        "remove_subtitle": False,
        "remove_brand": False,
    }
