"""任务 B：处理流水线（extract --fps 4 → codex 沙箱 → 白名单校验 → meta 落盘）。"""
import asyncio
import base64
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest
import httpx
import cv2
import numpy as np
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import codex_runner, h3, h3_project, image_optimization, long_generation, long_video, pipeline, postprocess, prepared_input, scenes as scene_planner, seedream, storage, vocal, voice
from app.codex_runner import CodexError, CodexRunner
from app.main import create_app


_GENERATE_IMAGE_OPTIMIZATION_PROJECT = image_optimization.generate_project_prompts
_GENERATE_PROJECT_ELEMENT_INDEX = pipeline._generate_project_element_index
_ANALYSIS_PROVENANCE_KEYS = {
    "analysis_audio_path",
    "analysis_audio_sha256",
    "analysis_has_bgm",
    "classification_evidence_sha256",
}


def _assert_analysis_bound_provenance(
    actual: list[dict], expected: list[dict], *, audio: bytes, has_bgm: bool,
) -> None:
    assert [
        {key: value for key, value in line.items()
         if key not in _ANALYSIS_PROVENANCE_KEYS}
        for line in actual
    ] == expected
    assert actual
    evidence = {line["classification_evidence_sha256"] for line in actual}
    assert len(evidence) == 1
    assert all(
        line["analysis_audio_path"] == "work/voice.mp3"
        and line["analysis_audio_sha256"] == hashlib.sha256(audio).hexdigest()
        and line["analysis_has_bgm"] is has_bgm
        and len(line["classification_evidence_sha256"]) == 64
        for line in actual
    )


def _short_dual_target_plan():
    return {
        "version": 2,
        "phase": "plan",
        "segment_indices": [0],
        "eligible": True,
        "reason": None,
        "person_plans": [{
            "id": "PERSON_01",
            "source_identity": "source person",
            "replacement_identity": "replacement person",
            "wardrobe_change": "different realistic wardrobe",
            "local_color_change": "different local clothing colors",
            "reference": {"segment_index": 0, "frame_index": 1},
            "observable_segments": [0],
        }],
        "scene_plans": [{
            "id": "SCENE_01",
            "source_scene": "source environment",
            "replacement_scene": "different realistic environment",
            "semantic_change": "replace the environment semantics",
            "geometry_changes": ["different structural geometry"],
            "depth_changes": ["different spatial depth"],
            "layout_changes": ["different spatial layout"],
            "local_color_change": "different local surface colors",
            "reference": {"segment_index": 0, "frame_index": 1},
            "segments": [0],
        }],
        "segments": [{
            "segment_index": 0,
            "persons": [{
                "id": "PERSON_01",
                "state": "replace",
                "observable_frames": [1],
                "target_region": "the complete observable person",
                "boundary": "the exact person boundary",
            }],
            "scene": {
                "scene_id": "SCENE_01",
                "target_region": "the complete observable environment",
                "boundary": "all visible scene surfaces and boundaries",
                "layout_reference_frame_index": 1,
            },
            "protected_non_target_people": [],
            "protected_relations": ["all physical and interaction relations"],
        }],
    }


def _short_dual_target_plan_v3(*, frame_count: int = 3):
    plan = deepcopy(_short_dual_target_plan())
    plan["version"] = 3
    plan["segments"][0]["persons"][0]["observable_frames"] = list(
        range(1, frame_count + 1)
    )
    plan["segments"][0].update(
        frame_constraints=[
            {
                "frame_index": index,
                "visible_body_parts": f"frame {index} visible body parts",
                "pose_skeleton": f"frame {index} pose skeleton",
                "contact_points": f"frame {index} contact points",
                "occlusion_order": f"frame {index} occlusion order",
                "out_of_frame_crop": f"frame {index} out of frame crop",
                "non_person_entity_ledger": {
                    "entities": [{
                        "entity_id": "ENTITY_01",
                        "description": f"frame {index} visible non-person entity",
                        "visibility": "full",
                    }],
                    "relations": [{
                        "subject_id": "ENTITY_01",
                        "predicate": "contacts",
                        "object_id": "PERSON_01",
                    }],
                },
                "dominant_palette_contract": {
                    "area_weighted_warm_cool_family": "balanced",
                    "saturation_style": "muted",
                },
            }
            for index in range(1, frame_count + 1)
        ],
        photometric_contract={
            "light_direction": "preserve source light direction",
            "light_quality": "preserve source light quality",
            "exposure_or_intensity": "preserve source exposure",
            "wb_cct": "preserve source white balance",
            "global_contrast": "preserve source contrast",
            "tone_curve": "preserve source tone curve",
        },
    )
    return plan


def _long_dual_target_plan_v3(*, frame_count: int = 3):
    plan = _short_dual_target_plan_v3(frame_count=frame_count)
    plan["segment_indices"] = [1, 2]
    plan["person_plans"][0].update(
        reference={"segment_index": 1, "frame_index": 1},
        observable_segments=[1, 2],
    )
    plan["scene_plans"][0].update(
        reference={"segment_index": 1, "frame_index": 1},
        segments=[1, 2],
    )
    plan["segments"] = [
        {
            **deepcopy(plan["segments"][0]),
            "segment_index": index,
            "persons": [{
                **deepcopy(plan["segments"][0]["persons"][0]),
                "observable_frames": list(range(1, frame_count + 1)),
            }],
        }
        for index in (1, 2)
    ]
    return plan


@pytest.fixture(autouse=True)
def _stub_image_postprocess_codex(monkeypatch):
    def project_index(_runner, cdir, _frame_paths):
        path = Path(cdir) / "work" / "element_index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"people": {}, "entities": {}, "scenes": {}}),
            encoding="utf-8",
        )
        return path

    def generate(_runner, segments, mode, **_kwargs):
        segment_specs = [
            {
                "index": segment["index"],
                "chain_id": segment["chain_id"],
                "join_mode": segment["join_mode"],
                "transition_skeleton": segment["transition_skeleton"],
            }
            for segment in segments
        ]
        source_frames = {
            segment["index"]: sorted(segment["keyframes_dir"].glob("*.png"))
            for segment in segments
        }
        request = {
            "semantic_slots": image_optimization.semantic_slot_manifest(
                segment_specs
            )
        }
        plan, _diagnostics = image_optimization.compile_semantic_plan(
            _semantic_image_output(request),
            segment_specs,
            source_frames=source_frames,
        )
        return plan, image_optimization.compile_frame_prompts(plan, mode)

    monkeypatch.setattr(
        pipeline.image_optimization, "generate_project_prompts", generate
    )
    monkeypatch.setattr(pipeline, "_generate_project_element_index", project_index)

ROOT = Path(pipeline.__file__).resolve().parent.parent
EXTRACT_SCRIPT = ROOT / "skills" / "video-maker" / "scripts" / "extract_keyframes.py"
CROP_SCRIPT = ROOT / "skills" / "video-maker" / "scripts" / "crop_image.py"

PROMPT_TEXT = "生成一支 15 秒、9:16 竖屏、720p、写实手机实拍风格的清洁短视频。"

# 1×1 真实 PNG（validate_work_dir 会用 cv2 解码校验，占位字节过不了）
_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _semantic_image_output(request: dict, *, omit_wardrobe: bool = False) -> dict:
    people = {
        "subject": {
            "source_identity": "当前可见源人物",
            "replacement_identity": "明显不同且跨帧稳定的新人物",
            "wardrobe_change": "不同款式且保持用途的服装",
            "local_color_change": "人物局部固有色明显变化",
        }
    }
    if omit_wardrobe:
        people["subject"].pop("wardrobe_change")
    return {
        "people": people,
        "scenes": {
            slot["key"]: {
                "source_scene": "当前可见源环境",
                "replacement_scene": "同用途且设计不同的真实新环境",
                "semantic_change": "环境语义明显变化",
                "geometry_change": "可见形状和空间结构明显变化",
                "depth_change": "前中后景纵深明显变化",
                "layout_change": "功能区域和实体布局明显变化",
                "local_color_change": "局部材质固有色明显变化",
            }
            for slot in request["semantic_slots"].get("scenes", [])
        },
        "frames": {
            slot["key"]: {
                "people": {"subject": {
                    "visible_region": f"{slot['key']} 当前可见人物区域",
                    "boundary": f"{slot['key']} 当前可见边界",
                    "body_and_pose": f"{slot['key']} 当前可见身体与姿态",
                }},
                "relationships": f"{slot['key']} 当前可见接触与遮挡关系",
                "entities": f"{slot['key']} 当前可见非人物实体",
                "crop": f"{slot['key']} 当前画外裁切",
                "palette_description": "warm-neutral and natural-muted",
            }
            for slot in request["semantic_slots"].get("frames", [])
        },
    }


def _write_valid_package(work: Path, frames: int = 3, prompt: str = PROMPT_TEXT):
    """按约定文件名造一套合法产物，返回关键帧文件名列表。"""
    kdir = work / "keyframes"
    kdir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(1, frames + 1):
        name = f"{i:02d}.png"
        (kdir / name).write_bytes(_PX_PNG)
        names.append(name)
    (work / "prompt.txt").write_text(prompt, encoding="utf-8")
    return names


def _make_conversation(settings, video_1s):
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    shutil.copy(video_1s, settings.data_dir / meta["id"] / "source.mp4")
    # 本文件既有用例模拟旧 voice_mode 会话；新 prepared-input 用例会显式补
    # dialogue_mode + duration_s + 新 voice_mode。
    return storage.update_meta(settings.data_dir, meta["id"], voice_mode="none")


def _replace_source_with_duration(settings, cid: str, duration_s: float) -> Path:
    source = settings.data_dir / cid / "source.mp4"
    subprocess.run(
        [
            "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=16x16:r=24",
            "-t", str(duration_s), "-pix_fmt", "yuv420p", "-an", "-y",
            str(source),
        ],
        check=True,
    )
    assert storage.probe_video(source).duration_s == pytest.approx(
        duration_s, abs=0.05,
    )
    return source


def test_backend_materializes_exact_nine_in_frozen_order_with_repeat_receipt(
    tmp_path, video_1s,
):
    work = tmp_path / "work"
    work.mkdir()
    scenes = [{
        "index": 1,
        "start_decode_frame_index": 0,
        "end_decode_frame_index": 3,
        "start_s": 0.0,
        "end_s": 0.201,
        "frames": [
            {
                "decode_frame_index": index,
                "pts": index,
                "time_base_num": 1,
                "time_base_den": 10,
            }
            for index in range(3)
        ],
    }]
    selection = scene_planner.select_segment_keyframes(
        scenes,
        {"index": 1, "start_s": 0.0, "end_s": 0.201},
    )

    names, receipt, frozen = pipeline._materialize_backend_keyframes(
        video_1s, work, selection
    )

    assert names == [f"{index:02d}.png" for index in range(1, 10)]
    assert len(frozen) == 9
    assert [item["decode_frame_index"] for item in receipt["keyframes"]] == [
        item["decode_frame_index"] for item in selection
    ]
    assert [item["repeated"] for item in receipt["keyframes"]] == [
        item["repeated"] for item in selection
    ]
    assert receipt["keyframes"][0]["sha256"] == receipt["keyframes"][1]["sha256"]
    assert receipt["keyframes"][0]["path"] == "keyframes/01.png"
    assert receipt["keyframes"][1]["path"] == "keyframes/02.png"
    persisted = json.loads((work / "keyframe_sampling.json").read_text(encoding="utf-8"))
    assert persisted == receipt
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 0.201,
        "chain_id": "chain-001",
        "join_mode": "hard_cut",
    }
    segwork = work / "segments" / "1" / "work"
    segwork.mkdir(parents=True)
    shutil.copytree(work / "keyframes", segwork / "keyframes")
    bound_receipt = deepcopy(receipt)
    for item in bound_receipt["keyframes"]:
        item["path"] = f"keyframes/{Path(item['path']).name}"
    bound = pipeline._bind_keyframe_source_timeline(
        work,
        [segment],
        [{
            **segment,
            "keyframes": names,
            "keyframe_sampling": bound_receipt,
        }],
        scenes,
    )
    assert len(bound[0]["keyframe_sources"]) == 9
    assert [item["source_time_s"] for item in bound[0]["keyframe_sources"]] == [
        item["source_time_s"] for item in selection
    ]


def test_visual_attempt_restores_backend_frozen_frames_after_codex_mutation(tmp_path):
    cdir = tmp_path / "conversation"
    work = cdir / "work"
    keyframes = work / "keyframes"
    keyframes.mkdir(parents=True)
    frozen = tuple(_PX_PNG for _index in range(9))

    class MutatingRunner:
        def run_isolated(
            self, stage, _prompt, *, session_dir, writable_paths,
        ):
            assert session_dir == cdir
            isolated_work = stage / "work"
            (keyframes / "01.png").write_bytes(b"mutated")
            (keyframes / "09.png").unlink()
            (isolated_work / "prompt.txt").write_text(
                PROMPT_TEXT, encoding="utf-8"
            )

    names, _prompt = pipeline._run_visual_attempt(
        MutatingRunner(),
        cdir,
        "backend frozen",
        work,
        isolate_dialogue=False,
        frozen_keyframes=frozen,
        skill_bytes=b"frozen video-maker skill",
    )

    assert names == [f"{index:02d}.png" for index in range(1, 10)]
    assert tuple((keyframes / name).read_bytes() for name in names) == frozen


def test_visual_analyzer_receives_half_resolution_proxies_and_restores_originals(
    tmp_path,
):
    cdir = tmp_path / "conversation"
    work = cdir / "work"
    work.mkdir(parents=True)
    source = np.full((8, 12, 3), 96, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", source)
    assert ok
    frozen = tuple(encoded.tobytes() for _index in range(9))
    (work / "01_frame_000.000s.png").write_bytes(encoded.tobytes())
    (work / "contact_sheet_01.jpg").write_bytes(b"overview")

    class InspectingRunner:
        def run_isolated(
            self, stage, _prompt, *, session_dir, writable_paths,
        ):
            assert session_dir == cdir
            assert not list((stage / "work").glob("*_frame_*.png"))
            assert (stage / "work" / "contact_sheet_01.jpg").read_bytes() == b"overview"
            for order in range(1, 10):
                data = (stage / "work" / "keyframes" / f"{order:02d}.png").read_bytes()
                image = cv2.imdecode(
                    np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                assert image.shape[:2] == (4, 6)
            (stage / "work" / "prompt.txt").write_text(
                PROMPT_TEXT, encoding="utf-8"
            )

    names, _prompt = pipeline._run_visual_attempt(
        InspectingRunner(),
        cdir,
        "analyze proxies",
        work,
        isolate_dialogue=False,
        frozen_keyframes=frozen,
        skill_bytes=b"frozen video-maker skill",
    )

    assert names == [f"{index:02d}.png" for index in range(1, 10)]
    assert tuple(
        (work / "keyframes" / name).read_bytes() for name in names
    ) == frozen


def test_source_scene_cut_is_frozen_on_the_first_post_cut_keyframe(tmp_path):
    work = tmp_path / "work"
    segwork = work / "segments" / "1" / "work"
    raw_dir = segwork / "frames"
    selected_dir = segwork / "keyframes"
    raw_dir.mkdir(parents=True)
    selected_dir.mkdir()
    source_times = [0.0, 0.75, 2.0, 2.5, 4.0, 6.0, 8.0, 11.0, 14.0]
    manifest_frames = []
    names = []
    for order, source_time_s in enumerate(source_times, 1):
        data = f"source-frame-{order}".encode()
        raw_name = f"frames/{order:03d}.png"
        (segwork / raw_name).write_bytes(data)
        selected_name = f"{order:02d}.png"
        (selected_dir / selected_name).write_bytes(data)
        manifest_frames.append({
            "index": order,
            "time_seconds": source_time_s,
            "file": raw_name,
        })
        names.append(selected_name)
    (segwork / "manifest.json").write_text(
        json.dumps({"duration_seconds": 14.5, "frames": manifest_frames}),
        encoding="utf-8",
    )
    segments = [{
        "index": 1,
        "start_s": 0.0,
        "end_s": 14.5,
        "chain_id": "chain-001",
        "join_mode": "hard_cut",
    }]
    metas = [{**segments[0], "keyframes": names}]

    bound = pipeline._bind_keyframe_source_timeline(
        work,
        segments,
        metas,
        [
            {"index": 1, "start_s": 0.0, "end_s": 2.267},
            {"index": 2, "start_s": 2.267, "end_s": 14.5},
        ],
    )

    timeline = bound[0]["keyframe_sources"]
    assert [item["source_time_s"] for item in timeline] == source_times
    assert [item["source_scene_id"] for item in timeline] == [
        "SCENE_01", "SCENE_01", "SCENE_01",
        "SCENE_02", "SCENE_02", "SCENE_02",
        "SCENE_02", "SCENE_02", "SCENE_02",
    ]
    assert timeline[0]["transition"] == {"type": "start", "at_s": 0.0}
    assert timeline[3]["transition"] == {
        "type": "hard_cut", "at_s": 2.267,
    }
    assert all(
        item["transition"] == {"type": "continuous", "at_s": None}
        for item in timeline[1:3] + timeline[4:]
    )

    with pytest.raises(
        pipeline.PipelineError,
        match="keyframe source timeline misses scene anchor",
    ):
        pipeline._bind_keyframe_source_timeline(
            work,
            segments,
            metas,
            [
                {"index": 1, "start_s": 0.0, "end_s": 1.0},
                {
                    "index": 2,
                    "start_s": 1.0,
                    "end_s": 2.0,
                    "has_source_frames": False,
                },
                {"index": 3, "start_s": 2.0, "end_s": 14.5},
            ],
        )


@pytest.mark.parametrize(
    ("source_times", "cut_at_s"),
    [
        ([0.0, 0.75, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], 14.25),
        ([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], 0.25),
    ],
    ids=("trailing-cut-14.25", "leading-cut-0.25"),
)
def test_source_timeline_rejects_any_scene_without_selected_anchor(
    tmp_path, source_times, cut_at_s,
):
    work = tmp_path / "work"
    segwork = work / "segments" / "1" / "work"
    raw_dir = segwork / "frames"
    selected_dir = segwork / "keyframes"
    raw_dir.mkdir(parents=True)
    selected_dir.mkdir()
    manifest_frames = []
    names = []
    for order, source_time_s in enumerate(source_times, 1):
        data = f"source-frame-{order}".encode()
        raw_name = f"frames/{order:03d}.png"
        (segwork / raw_name).write_bytes(data)
        selected_name = f"{order:02d}.png"
        (selected_dir / selected_name).write_bytes(data)
        manifest_frames.append({
            "index": order,
            "time_seconds": source_time_s,
            "file": raw_name,
        })
        names.append(selected_name)
    (segwork / "manifest.json").write_text(
        json.dumps({"frames": manifest_frames}), encoding="utf-8",
    )
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 14.5,
        "chain_id": "chain-001",
        "join_mode": "hard_cut",
    }

    with pytest.raises(
        pipeline.PipelineError,
        match="keyframe source timeline misses scene anchor",
    ):
        pipeline._bind_keyframe_source_timeline(
            work,
            [segment],
            [{**segment, "keyframes": names}],
            [
                {"index": 1, "start_s": 0.0, "end_s": cut_at_s},
                {"index": 2, "start_s": cut_at_s, "end_s": 14.5},
            ],
        )


def test_two_generation_segments_without_source_cut_remain_continuous(
    tmp_path,
):
    work = tmp_path / "work"
    local_times = [0.0, 1.75, 3.5, 5.25, 7.0, 8.75, 10.5, 12.25, 14.0]
    segments = []
    metas = []
    for segment_index, start_s in enumerate((0.0, 14.5), 1):
        segwork = work / "segments" / str(segment_index) / "work"
        raw_dir = segwork / "frames"
        selected_dir = segwork / "keyframes"
        raw_dir.mkdir(parents=True)
        selected_dir.mkdir()
        manifest_frames = []
        names = []
        for order, local_time in enumerate(local_times, 1):
            data = f"segment-{segment_index}-frame-{order}".encode()
            raw_name = f"frames/{order:03d}.png"
            (segwork / raw_name).write_bytes(data)
            selected_name = f"{order:02d}.png"
            (selected_dir / selected_name).write_bytes(data)
            manifest_frames.append({
                "index": order,
                "time_seconds": local_time,
                "file": raw_name,
            })
            names.append(selected_name)
        (segwork / "manifest.json").write_text(
            json.dumps({
                "duration_seconds": 14.5,
                "frames": manifest_frames,
            }),
            encoding="utf-8",
        )
        segment = {
            "index": segment_index,
            "start_s": start_s,
            "end_s": start_s + 14.5,
            "chain_id": "chain-001",
            "join_mode": "hard_cut" if segment_index == 1 else "continue",
        }
        segments.append(segment)
        metas.append({**segment, "keyframes": names})

    bound = pipeline._bind_keyframe_source_timeline(
        work,
        segments,
        metas,
        [{"index": 1, "start_s": 0.0, "end_s": 29.0}],
    )

    assert bound[0]["keyframe_sources"][0]["transition"] == {
        "type": "start", "at_s": 0.0,
    }
    assert all(
        source["source_scene_id"] == "SCENE_01"
        and source["transition"] == {"type": "continuous", "at_s": None}
        for source in (
            bound[0]["keyframe_sources"][1:]
            + bound[1]["keyframe_sources"]
        )
    )
    assert bound[1]["keyframe_sources"][0]["source_time_s"] == 14.5


def test_source_timeline_is_an_explicit_image_optimization_input():
    assert "keyframe_sources" in inspect.signature(
        pipeline._frame_inventory
    ).parameters
    assert "keyframe_sources" in inspect.signature(
        pipeline._freeze_image_optimization
    ).parameters


def test_segmented_image_prompts_come_from_one_project_call(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    work = tmp_path / "session" / "work"
    segments = [
        {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
        {"index": 2, "chain_id": "chain-001", "join_mode": "continue"},
    ]
    for segment in segments:
        (work / "segments" / str(segment["index"]) / "work" / "keyframes").mkdir(
            parents=True
        )
    global_plan = {
        "version": 1,
        "segment_indices": [1, 2],
        "elements": [],
    }
    calls = []

    def fake_project(_settings, _runner, specs, **_kwargs):
        calls.append(specs)
        return global_plan, {1: "prompt-1", 2: "prompt-2"}

    monkeypatch.setattr(
        pipeline, "_generate_image_optimization_project", fake_project
    )

    frozen, prompts = pipeline._generate_segmented_image_prompts(
        settings,
        object(),
        segments,
        [dict(segment) for segment in segments],
        work,
        session_dir=work.parent,
    )

    assert frozen is global_plan
    assert prompts == {1: "prompt-1", 2: "prompt-2"}
    assert len(calls) == 1
    assert [item["index"] for item in calls[0]] == [1, 2]
    assert (work / "segments" / "1" / "work" / "image_optimization_prompt.txt").read_text() == "prompt-1\n"
    assert (work / "segments" / "2" / "work" / "image_optimization_prompt.txt").read_text() == "prompt-2\n"


def test_segmented_v4_prompts_remain_frame_dicts_and_are_not_written_as_text(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    work = tmp_path / "session" / "work"
    segments = [
        {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
        {"index": 2, "chain_id": "chain-001", "join_mode": "continue"},
    ]
    for segment in segments:
        (work / "segments" / str(segment["index"]) / "work" / "keyframes").mkdir(
            parents=True
        )
    prompts = {1: {1: "frame-one"}, 2: {1: "frame-two"}}
    monkeypatch.setattr(
        pipeline,
        "_generate_image_optimization_project",
        lambda *_args, **_kwargs: ({"version": 4}, prompts),
    )
    monkeypatch.setattr(
        pipeline,
        "_write_image_optimization_prompt",
        lambda *_args: pytest.fail("v4 prompt dict must not be string-written"),
    )

    continuity, actual = pipeline._generate_segmented_image_prompts(
        settings,
        object(),
        segments,
        [dict(segment) for segment in segments],
        work,
        session_dir=work.parent,
    )

    assert continuity == {"version": 4}
    assert actual == prompts
    assert not list(work.rglob("image_optimization_prompt.txt"))


def test_v4_frame_inventory_derives_transition_from_lineage_and_frame_pair(tmp_path):
    frames = {}
    for index, values in ((1, (b"one", b"two")), (2, (b"three",))):
        directory = tmp_path / str(index)
        directory.mkdir()
        frames[index] = []
        for frame_index, value in enumerate(values, 1):
            path = directory / f"{frame_index:02d}.png"
            path.write_bytes(value)
            frames[index].append(path)

    inventory = pipeline._frame_inventory(
        frames,
        segment_lineage={
            1: {"chain_id": "chain-001", "join_mode": "hard_cut"},
            2: {"chain_id": "chain-001", "join_mode": "continue"},
        },
    )

    assert [item["source_transition_from_previous"] for item in inventory] == [
        "start", "same_camera", "same_camera",
    ]
    assert all(re.fullmatch(r"[0-9a-f]{64}", item[
        "source_transition_evidence_sha256"
    ]) for item in inventory)
    changed = pipeline._frame_inventory(
        frames,
        segment_lineage={
            1: {"chain_id": "chain-001", "join_mode": "hard_cut"},
            2: {"chain_id": "chain-001", "join_mode": "hard_cut"},
        },
    )
    assert changed[-1]["source_transition_from_previous"] == "hard_cut"
    assert changed[-1]["source_transition_evidence_sha256"] != inventory[-1][
        "source_transition_evidence_sha256"
    ]


def test_v4_frame_inventory_prefers_bound_source_cut_timeline(tmp_path):
    frames = {}
    for index in (1, 2):
        directory = tmp_path / str(index)
        directory.mkdir()
        frames[index] = []
        for frame_index in (1, 2):
            path = directory / f"{frame_index:02d}.png"
            path.write_bytes(f"{index}-{frame_index}".encode())
            frames[index].append(path)
    sources = {
        1: [
            {"order": 1, "source_time_s": 0.0, "source_scene_id": "SCENE_01",
             "transition": {"type": "start", "at_s": 0.0}},
            {"order": 2, "source_time_s": 1.0, "source_scene_id": "SCENE_02",
             "transition": {"type": "hard_cut", "at_s": 1.0}},
        ],
        2: [
            {"order": 1, "source_time_s": 2.0, "source_scene_id": "SCENE_02",
             "transition": {"type": "continuous", "at_s": None}},
            {"order": 2, "source_time_s": 3.0, "source_scene_id": "SCENE_03",
             "transition": {"type": "hard_cut", "at_s": 3.0}},
        ],
    }
    lineage = {
        1: {"chain_id": "chain-001", "join_mode": "hard_cut"},
        2: {"chain_id": "chain-001", "join_mode": "continue"},
    }
    inventory = pipeline._frame_inventory(
        frames, segment_lineage=lineage, keyframe_sources=sources,
    )
    assert [item["source_transition_from_previous"] for item in inventory] == [
        "start", "hard_cut", "same_camera", "hard_cut",
    ]
    changed = deepcopy(sources)
    changed[2][0]["source_time_s"] = 2.25
    rebound = pipeline._frame_inventory(
        frames, segment_lineage=lineage, keyframe_sources=changed,
    )
    assert rebound[2]["source_transition_evidence_sha256"] != inventory[2][
        "source_transition_evidence_sha256"
    ]


def test_v4_plan_input_receives_backend_transition_skeleton_before_codex(tmp_path, monkeypatch):
    work = tmp_path / "work"
    segments = [
        {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
        {"index": 2, "chain_id": "chain-001", "join_mode": "continue"},
    ]
    metas = []
    for segment in segments:
        keyframes = ["01.png", "02.png"]
        directory = work / "segments" / str(segment["index"]) / "work" / "keyframes"
        directory.mkdir(parents=True)
        for name in keyframes:
            (directory / name).write_bytes(_PX_PNG)
        first_transition = (
            {"type": "start", "at_s": 0.0}
            if segment["index"] == 1
            else {"type": "hard_cut", "at_s": 2.0}
        )
        metas.append({
            "index": segment["index"],
            "keyframes": keyframes,
            "keyframe_sources": [
                {
                    "order": 1,
                    "source_time_s": float((segment["index"] - 1) * 2),
                    "source_scene_id": f"SCENE_{segment['index']:02d}",
                    "transition": first_transition,
                },
                {
                    "order": 2,
                    "source_time_s": float((segment["index"] - 1) * 2 + 1),
                    "source_scene_id": f"SCENE_{segment['index']:02d}",
                    "transition": {"type": "continuous", "at_s": None},
                },
            ],
        })
    captured = []

    def generate(_settings, _runner, specs, **_kwargs):
        captured.extend(specs)
        return {"version": 1, "segment_indices": [1, 2], "elements": []}, {
            1: "prompt-1", 2: "prompt-2",
        }

    monkeypatch.setattr(pipeline, "_generate_image_optimization_project", generate)
    continuity, _prompts = pipeline._generate_segmented_image_prompts(
        make_settings(tmp_path), object(), segments, metas, work, session_dir=tmp_path,
    )

    assert continuity["version"] == 1
    assert [item["transition_skeleton"][0]["source_transition_from_previous"]
            for item in captured] == ["start", "hard_cut"]
    assert all(
        item["transition_skeleton"][-1]["source_transition_evidence_sha256"]
        for item in captured
    )


def test_v4_semantic_diagnostics_do_not_retry_or_switch_compilers(
    tmp_path, monkeypatch, caplog,
):
    settings = make_settings(tmp_path, retry_count=1, retry_interval_s=0)
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
    frame = keyframes / "01.png"
    frame.write_bytes(_PX_PNG)
    skeleton = pipeline._frame_inventory(
        {0: [frame]},
        segment_lineage={0: {"chain_id": "short-000", "join_mode": "hard_cut"}},
    )
    class Runner:
        def __init__(self):
            self.calls = 0

        def run_isolated(self, workdir, _prompt, *, session_dir):
            self.calls += 1
            work = Path(workdir) / "work"
            if (work / "project_index_request.json").is_file():
                (work / "element_index.json").write_text(
                    json.dumps({"people": {}, "entities": {}, "scenes": {}}),
                    encoding="utf-8",
                )
                return
            request = json.loads(
                (work / "request.json").read_text(
                    encoding="utf-8"
                )
            )
            semantic = _semantic_image_output(request, omit_wardrobe=True)
            if request["phase"] == "global_plan":
                output = work / "global_plan.json"
                semantic = {
                    key: semantic.get(key, {})
                    for key in ("people", "entities", "scenes")
                }
            else:
                output = work / "segment_frames.json"
                semantic = {"frames": semantic["frames"]}
            output.write_text(
                json.dumps(semantic),
                encoding="utf-8",
            )

    runner = Runner()
    monkeypatch.setattr(
        pipeline.image_optimization,
        "generate_project_prompts",
        _GENERATE_IMAGE_OPTIMIZATION_PROJECT,
    )
    monkeypatch.setattr(
        pipeline, "_generate_project_element_index", _GENERATE_PROJECT_ELEMENT_INDEX,
    )
    caplog.set_level("INFO", logger="app.image_optimization")
    plan, prompts = pipeline._generate_image_optimization_project(
        settings,
        runner,
        [{
            "index": 0,
            "chain_id": "short-000",
            "join_mode": "hard_cut",
            "keyframes_dir": keyframes,
            "transition_skeleton": skeleton,
        }],
        session_dir=tmp_path,
        step="project image plan",
    )

    assert runner.calls == 3
    assert not hasattr(image_optimization, "generic_project_prompts")
    assert plan["version"] == 4
    assert list(prompts) == [0]
    assert [item["id"] for item in plan["person_plans"]] == ["PERSON_01"]
    assert plan["segments"][0]["persons"][0]["state"] == "replace"
    assert plan["segments"][0]["persons"][0]["observable_frames"] == [1]
    assert plan["scene_plans"][0]["continuity_graph"]["views"][0][
        "transition_from_previous"
    ] == skeleton[0]["source_transition_from_previous"]
    assert "替换人物" in prompts[0][1]
    assert "score=" in caplog.text
    assert "missing:people.subject.wardrobe_change" in caplog.text
    assert "palette_description" in caplog.text


def test_v4_pipeline_freezes_authoritative_transitions_and_anchor_schedule(tmp_path):
    settings = make_settings(tmp_path)
    frame_dir = tmp_path / "work" / "keyframes"
    frame_dir.mkdir(parents=True)
    frames = []
    for frame_index in (1, 2):
        path = frame_dir / f"{frame_index:02d}.png"
        path.write_bytes(_PX_PNG)
        frames.append(path)
    plan = _short_dual_target_plan_v3(frame_count=2)
    plan["version"] = 4
    for frame in plan["segments"][0]["frame_constraints"]:
        frame["dominant_palette_contract"] = {
            "area_weighted_warm_cool_family": "balanced",
            "saturation_style": "muted",
        }
    plan["scene_plans"][0]["continuity_graph"] = {
        "components": [{"component_id": "COMPONENT_01", "target_spec": "target"}],
        "topology": [],
        "views": [
            {
                "segment_index": 0,
                "frame_index": 1,
                "transition_from_previous": "start",
                "observations": [{"component_id": "COMPONENT_01", "visibility": "full"}],
                "view_relations": [],
            },
            {
                "segment_index": 0,
                "frame_index": 2,
                    "transition_from_previous": "same_camera",
                "observations": [{"component_id": "COMPONENT_01", "visibility": "full"}],
                "view_relations": [],
            },
        ],
    }
    prompts = image_optimization.compile_frame_prompts(plan, settings.seedream_edit_mode)
    meta = {"keyframes": [path.name for path in frames]}

    continuity, frozen = pipeline._freeze_image_optimization(
        settings, meta, plan, prompts, {0: frames}, require_dual_target=True,
        segment_lineage={0: {"chain_id": "chain-001", "join_mode": "hard_cut"}},
    )

    assert continuity["_image_continuity"]["version"] == 4
    private = frozen["_image_optimization"]
    assert private["version"] == 4
    assert private["execution_inputs"]["frames"][1][
        "source_transition_from_previous"
    ] == "same_camera"
    assert private["scene_anchor_schedule"]["scenes"][0]["global_anchor"][
        "frame_index"
    ] == 1
    assert private["frames"][1]["frame_name"] == "02.png"
    assert isinstance(private["frames"][1]["current"], str)


def test_invalid_project_output_stops_before_writing_segment_prompts(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    def fail_project(*_args, **_kwargs):
        raise pipeline.PipelineError("image optimization output is missing or invalid")

    monkeypatch.setattr(
        pipeline, "_generate_image_optimization_project", fail_project
    )

    with pytest.raises(pipeline.PipelineError, match="image optimization"):
        pipeline._generate_segmented_image_prompts(
            settings,
            object(),
            [
                {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
                {"index": 2, "chain_id": "chain-001", "join_mode": "continue"},
            ],
            [{"index": 1}, {"index": 2}],
            tmp_path / "work",
            session_dir=tmp_path,
        )

    assert not list((tmp_path / "work").rglob("image_optimization_prompt.txt"))


def test_long_scene_bounds_normalize_detector_millisecond_rounding(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "scenes.json").write_text(
        json.dumps(
            {
                "duration_s": 36.733,
                "scenes": [
                    {"start_s": 0.0, "end_s": 20.467},
                    {"start_s": 20.467, "end_s": 36.733},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert pipeline._scene_bounds_for_long_plan(work, 36.733333) == [
        {"start_s": 0.0, "end_s": 20.467},
        {"start_s": 20.467, "end_s": 36.733333},
    ]


def test_long_scene_bounds_include_the_one_millisecond_limit(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "scenes.json").write_text(
        json.dumps({"scenes": [{"start_s": 0.0, "end_s": 36.733}]}),
        encoding="utf-8",
    )

    assert pipeline._scene_bounds_for_long_plan(work, 36.734) == [
        {"start_s": 0.0, "end_s": 36.734}
    ]


def _planner_test_scenes(duration_s: float) -> list[dict]:
    scenes = []
    start_s = 0.0
    while start_s < duration_s:
        index = len(scenes) + 1
        end_s = min(start_s + long_video.SEGMENT_PROVIDER_MAX_DURATION_S, duration_s)
        scenes.append({
            "index": index,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "start_time": {
                "pts": int(round(start_s * 1000)),
                "time_base_num": 1,
                "time_base_den": 1000,
            },
            "end_time": {
                "pts": int(round(end_s * 1000)),
                "time_base_num": 1,
                "time_base_den": 1000,
            },
            "frames": [{
                "decode_frame_index": index,
                "pts": int(round((start_s + 1) * 1000)),
                "pts_origin": 0,
                "time_base_num": 1,
                "time_base_den": 1000,
            }],
        })
        start_s = end_s
    return scenes


@pytest.mark.parametrize(
    "planner",
    (long_video.plan_segments, scene_planner.plan_segments),
    ids=("provider", "exact-scene"),
)
def test_auto_planner_ignores_source_spanning_sung_and_keeps_scene_cuts(planner):
    sung = {
        "text": "sung lyrics",
        "start_s": 0.0,
        "end_s": 14.5,
        "classification": "sung",
        "kept": True,
    }
    dialogue = pipeline._planner_dialogue({
        "dialogue_mode": "auto",
        "voice_line_provenance": [sung],
    }, [sung])

    segments = planner(14.5, _planner_test_scenes(14.5), dialogue)

    assert dialogue == []
    assert [(item["start_s"], item["end_s"]) for item in segments] == [
        (0.0, 10.0),
        (10.0, 14.5),
    ]


@pytest.mark.parametrize(
    "planner",
    (long_video.plan_segments, scene_planner.plan_segments),
    ids=("provider", "exact-scene"),
)
def test_auto_planner_assigns_source_spanning_spoken_without_blocking(planner):
    spoken = {
        "text": "spoken line",
        "start_s": 0.0,
        "end_s": 20.0,
        "classification": "spoken",
        "kept": True,
    }
    dialogue = pipeline._planner_dialogue({
        "dialogue_mode": "auto",
        "voice_line_provenance": [spoken],
    }, [spoken])

    segments = planner(20.0, _planner_test_scenes(20), dialogue)

    assert [(item["start_s"], item["end_s"]) for item in segments] == [
        (0.0, 10.0), (10.0, 20.0),
    ]
    assert long_video.localize_dialogue(
        dialogue, segments[0], segments=segments
    ) == [{
        "text": "spoken",
        "start_s": 0.0,
        "end_s": 10.0,
        "classification": "spoken",
    }]
    assert long_video.localize_dialogue(
        dialogue, segments[1], segments=segments
    ) == [{
        "text": " line",
        "start_s": 0.0,
        "end_s": 10.0,
        "classification": "spoken",
    }]


@pytest.mark.parametrize(
    "planner",
    (long_video.plan_segments, scene_planner.plan_segments),
    ids=("provider", "exact-scene"),
)
def test_auto_planner_mixed_vocals_only_delivers_spoken_dialogue(planner):
    sung = {
        "text": "sung lyrics",
        "start_s": 0.0,
        "end_s": 20.0,
        "classification": "sung",
        "kept": True,
    }
    spoken = {
        "text": "spoken line",
        "start_s": 2.0,
        "end_s": 3.0,
        "classification": "spoken",
        "kept": True,
    }
    dialogue = pipeline._planner_dialogue({
        "dialogue_mode": "auto",
        "voice_line_provenance": [sung, spoken],
    }, [sung, spoken])

    segments = planner(20.0, _planner_test_scenes(20), dialogue)

    assert dialogue == [{
        "text": "spoken line",
        "start_s": 2.0,
        "end_s": 3.0,
        "classification": "spoken",
    }]
    assert [(item["start_s"], item["end_s"]) for item in segments] == [
        (0.0, 10.0),
        (10.0, 20.0),
    ]


@pytest.mark.parametrize("duration_s", (10, 20), ids=("n1", "n2"))
def test_planner_projection_keeps_final_dialogue_consistent_for_n1_n2(duration_s):
    sung = {
        "text": "sung lyrics",
        "start_s": 0.0,
        "end_s": float(duration_s),
        "classification": "sung",
        "kept": True,
    }
    spoken = {
        "text": "spoken line",
        "start_s": 2.0,
        "end_s": 3.0,
        "classification": "spoken",
        "kept": True,
    }
    meta = {
        "dialogue_mode": "auto",
        "voice_line_provenance": [sung, spoken],
    }
    dialogue = pipeline._planner_dialogue(meta, [sung, spoken])
    segments = long_video.plan_segments(
        float(duration_s), _planner_test_scenes(duration_s), dialogue
    )

    localized = [
        line
        for segment in segments
        for line in long_video.localize_dialogue(
            dialogue, segment, segments=segments
        )
    ]
    assert localized == [{
        "text": "spoken line",
        "start_s": 2.0,
        "end_s": 3.0,
        "classification": "spoken",
    }]
    assert pipeline._planner_dialogue(
        {"dialogue_mode": "none"}, [spoken]
    ) == []
    for mode in ("edit", "custom"):
        assert pipeline._planner_dialogue(
            {"dialogue_mode": mode}, [spoken]
        ) == [spoken]


def test_long_scene_bounds_reject_just_over_one_millisecond(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "scenes.json").write_text(
        json.dumps({"scenes": [{"start_s": 0.0, "end_s": 36.733}]}),
        encoding="utf-8",
    )

    bounds = pipeline._scene_bounds_for_long_plan(work, 36.734001)
    with pytest.raises(long_video.LongVideoError) as caught:
        long_video.plan_segments(36.734001, bounds, [])
    assert caught.value.code == "long_video_invalid_scenes"


def test_long_scene_bounds_do_not_hide_a_real_terminal_gap(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "scenes.json").write_text(
        json.dumps(
            {
                "scenes": [
                    {"start_s": 0.0, "end_s": 20.467},
                    {"start_s": 20.467, "end_s": 36.7},
                ],
            }
        ),
        encoding="utf-8",
    )

    bounds = pipeline._scene_bounds_for_long_plan(work, 36.733333)
    with pytest.raises(long_video.LongVideoError) as caught:
        long_video.plan_segments(36.733333, bounds, [])
    assert caught.value.code == "long_video_invalid_scenes"


def test_run_converges_container_duration_to_visual_manifest_timeline(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"fake")
    storage.update_meta(
        settings.data_dir, meta["id"], duration_s=16.787007,
        dialogue_mode="auto", voice_mode="keep",
    )
    captured = {}

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 16.766667}), encoding="utf-8"
            )
            (work / "contact_sheet.jpg").write_bytes(b"sheet")

    def fake_write(root, *, source, duration_s, segments, workflow):
        captured["duration_s"] = duration_s
        captured["segments"] = segments
        path = root / long_video.PLAN_RECEIPT_FILENAME
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(
        storage, "probe_video",
        lambda _path: storage.VideoProbe(16.766667, 1080, 1920),
    )
    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    def fake_voice_step(_settings, cid, *_args, **_kwargs):
        storage.update_meta(
            _settings.data_dir, cid, voice_line_provenance=[]
        )
        return []

    monkeypatch.setattr(pipeline, "_voice_step", fake_voice_step)
    monkeypatch.setattr(pipeline, "_detect_segments", lambda *_a, **_kw: [])
    (cdir / "work" / "scenes.json").write_text(
        json.dumps({
            "duration_s": 16.767,
            "scenes": [{"start_s": 0.0, "end_s": 16.767}],
        }),
        encoding="utf-8",
    )
    def fake_process_segment(
        _settings, work, _source, seg, _runner, lines, _lang,
        new_input_contract,
        ):
            anchors = work / "segments" / str(seg["index"]) / "work" / "anchors"
            anchors.mkdir(parents=True, exist_ok=True)
            (anchors / "first.png").write_bytes(_PX_PNG)
            (anchors / "last.png").write_bytes(_PX_PNG)
            (anchors.parent / "image_optimization_prompt.txt").write_text(
                "Codex 分段图片二次编辑提示词", encoding="utf-8"
            )
            keyframes = _write_valid_package(anchors.parent, frames=9, prompt="p")
            return {
                **seg,
                "keyframes": keyframes,
                "prompt": "p",
                "dialogue": lines or [],
            }

    monkeypatch.setattr(pipeline, "_process_segment", fake_process_segment)
    monkeypatch.setattr(
        pipeline,
        "_bind_keyframe_source_timeline",
        lambda _work, _segments, metas, _scenes: metas,
    )
    monkeypatch.setattr(long_video, "write_plan_receipt", fake_write)
    expected_settings = settings

    def fake_freeze(
        root, meta, receipt, fit_mode, dialogue_mode, *, settings,
    ):
        assert settings is expected_settings
        seg = meta["segments"][0]
        segdir = Path(root) / "work" / "segments" / "1"
        first = segdir / "work" / "anchors" / "first.png"
        last = segdir / "work" / "anchors" / "last.png"
        return long_generation.FrozenPlan(
            Path(root), Path(root) / "source.mp4", receipt,
            (long_generation.FrozenSegment(
                seg["index"], seg["start_s"], seg["end_s"], seg["chain_id"],
                seg["join_mode"], segdir, first, first.read_bytes(), last,
                last.read_bytes(), "p",
            ),),
        )

    monkeypatch.setattr(long_generation, "freeze_plan", fake_freeze)

    pipeline.run(settings, meta["id"], object())

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["duration_s"] == 16.766667
    assert captured["duration_s"] == 16.766667
    assert captured["segments"][-1]["end_s"] == 16.766667
    assert stored["_image_continuity"]["segment_indices"] == [
        segment["index"] for segment in captured["segments"]
    ]


def test_run_does_not_reprobe_or_rewrite_frozen_generation(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"fake")
    original_generation = {"status": "succeeded", "attempt": 1}
    storage.update_meta(
        settings.data_dir, meta["id"], status="done", duration_s=9.5,
        generation=original_generation,
    )
    monkeypatch.setattr(
        storage, "probe_video",
        lambda _path: (_ for _ in ()).throw(AssertionError("must stay frozen")),
    )

    pipeline.run(settings, meta["id"], object())

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done"
    assert stored["duration_s"] == 9.5
    assert stored["generation"] == original_generation


def test_run_rechecks_300_second_gate_after_manifest(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"fake")
    storage.update_meta(settings.data_dir, meta["id"], duration_s=300.0)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        assert step == "extract"
        work = Path(argv[argv.index("--out-dir") + 1])
        (work / "manifest.json").write_text(
            json.dumps({"duration_seconds": 300.001}), encoding="utf-8"
        )

    monkeypatch.setattr(
        storage, "probe_video",
        lambda _path: storage.VideoProbe(300.0, 320, 240),
    )
    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)

    pipeline.run(settings, meta["id"], object())

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "failed"
    assert stored["error"] == "long_video_duration_exceeded"


def test_run_rejects_rewrite_when_source_probe_crosses_long_threshold(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"fake")
    storage.update_meta(
        settings.data_dir, meta["id"], duration_s=13.9, voice_mode="rewrite",
    )
    operations = []
    monkeypatch.setattr(
        storage, "probe_video",
        lambda _path: storage.VideoProbe(15.001, 320, 240),
    )
    monkeypatch.setattr(
        pipeline, "_run_cmd",
        lambda *_a, **_kw: operations.append("extract"),
    )

    pipeline.run(settings, meta["id"], object())

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "failed"
    assert stored["error"] == "long_video_audio_mode_unsupported"
    assert operations == []


def test_run_rejects_translate_when_manifest_crosses_ten_second_threshold_before_voice(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(settings.data_dir, "", "clip.mp4")
    cdir = settings.data_dir / meta["id"]
    (cdir / "source.mp4").write_bytes(b"fake")
    storage.update_meta(
            settings.data_dir, meta["id"], duration_s=9.9, voice_mode="translate",
        target_language="English",
    )
    operations = []

    def fake_cmd(argv, *, timeout, step, cwd=None):
        operations.append(step)
        work = Path(argv[argv.index("--out-dir") + 1])
        (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 10.001}), encoding="utf-8"
        )

    monkeypatch.setattr(
        storage, "probe_video",
            lambda _path: storage.VideoProbe(9.9, 320, 240),
    )
    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(
        pipeline, "_voice_step",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("voice must not run")),
    )

    pipeline.run(settings, meta["id"], object())

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "failed"
    assert stored["error"] == "long_video_audio_mode_unsupported"
    assert operations == ["extract"]


def test_duration_calibration_accepts_keep_after_crossing_long_threshold():
    assert pipeline._validate_calibrated_duration(
        {"voice_mode": "keep"}, 10.001
    ) == 10.001


def test_duration_calibration_keeps_short_rewrite_valid():
    assert pipeline._validate_calibrated_duration(
        {"voice_mode": "rewrite"}, 10.0
    ) == 10.0


@pytest.fixture
def fake_steps(monkeypatch):
    """mock 掉 extract 子进程与 codex；返回调用记录。"""
    calls = {"cmd": [], "codex": []}

    def fake_cmd(argv, *, timeout, step, cwd=None):
        calls["cmd"].append({"argv": list(argv), "timeout": timeout, "step": step, "cwd": cwd})
        if _run_fake_media_step(argv, step):
            return
        if step == "extract":
            duration_s = storage.probe_video(Path(argv[2])).duration_s
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": duration_s})
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            duration_s = json.loads(
                (work / "manifest.json").read_text(encoding="utf-8")
            )["duration_seconds"]
            _write_exact_scene_inventory(work, Path(argv[2]), duration_s)

    def fake_codex(self, workdir, prompt):
        calls["codex"].append({"workdir": Path(workdir), "prompt": prompt})
        _write_valid_package(
            Path(workdir) / "work",
            frames=9 if "最终台词由后端" in prompt else 3,
        )

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    return calls


@pytest.fixture(autouse=True)
def fake_vocal_analysis(monkeypatch):
    """旧口播流水线测试只验证编排；声学算法由 test_vocal.py 和专门集成测试覆盖。"""
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 1000, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )


# ---------- 产物白名单校验 ----------


class TestValidateWorkDir:
    def test_valid(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        names = _write_valid_package(work, frames=3)
        got_names, prompt = pipeline.validate_work_dir(work)
        assert got_names == names
        assert prompt == PROMPT_TEXT

    def test_zero_keyframes(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work, frames=0)
        with pytest.raises(pipeline.PipelineError, match="keyframe"):
            pipeline.validate_work_dir(work)

    def test_ten_keyframes(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work, frames=10)
        with pytest.raises(pipeline.PipelineError, match="keyframe"):
            pipeline.validate_work_dir(work)

    def test_prompt_missing(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        (work / "prompt.txt").unlink()
        with pytest.raises(pipeline.PipelineError, match="prompt"):
            pipeline.validate_work_dir(work)

    def test_prompt_empty(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work, prompt="  \n ")
        with pytest.raises(pipeline.PipelineError, match="prompt"):
            pipeline.validate_work_dir(work)

    def test_prompt_too_large(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        _write_valid_package(work)
        (work / "prompt.txt").write_bytes(b"x" * (32 * 1024 + 1))
        with pytest.raises(pipeline.PipelineError, match="prompt"):
            pipeline.validate_work_dir(work)

    def test_non_png_not_counted(self, tmp_path):
        """keyframes/ 里的非 PNG 文件不计入帧数。"""
        work = tmp_path / "work"
        work.mkdir()
        names = _write_valid_package(work, frames=2)
        (work / "keyframes" / "notes.txt").write_text("x", encoding="utf-8")
        got_names, _ = pipeline.validate_work_dir(work)
        assert got_names == names


# ---------- _run_cmd：子进程包装 ----------


class TestRunCmd:
    def test_timeout(self, monkeypatch):
        def slow(argv, **kw):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kw["timeout"])

        monkeypatch.setattr(pipeline.subprocess, "run", slow)
        with pytest.raises(pipeline.PipelineError, match="timed out"):
            pipeline._run_cmd(["whatever"], timeout=1, step="extract")

    def test_missing_executable(self, monkeypatch):
        def nope(argv, **kw):
            raise FileNotFoundError(2, "No such file or directory", argv[0])

        monkeypatch.setattr(pipeline.subprocess, "run", nope)
        with pytest.raises(pipeline.PipelineError, match="not found"):
            pipeline._run_cmd(["no-such-bin"], timeout=1, step="extract")

    def test_stderr_scrubbed_and_truncated(self, monkeypatch):
        stderr = (
            "PATH=/home/xy/.local/bin:/usr/bin\n"
            "ARK_API_KEY=supersecretvalue\n"
            + "y" * 1200 + "\nreal error line\n"
        )
        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)
        monkeypatch.setattr(
            pipeline.subprocess, "run", lambda *a, **kw: fake
        )
        with pytest.raises(pipeline.PipelineError) as exc_info:
            pipeline._run_cmd(["x"], timeout=1, step="extract")
        msg = str(exc_info.value)
        assert "real error line" in msg
        assert "ARK_API_KEY" not in msg and "supersecretvalue" not in msg
        assert "PATH=" not in msg
        assert len(msg) <= 560  # 500 截断 + 步骤/退出码前缀


# ---------- CodexRunner ----------


@pytest.fixture
def captured_codex(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append({"argv": list(argv), "kw": kw})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    return calls


class TestCodexRunner:
    def test_argv_sandbox(self, captured_codex, tmp_path):
        runner = CodexRunner(timeout_s=600, concurrency=1)
        runner.run(tmp_path, "提示词")
        (call,) = captured_codex
        argv, kw = call["argv"], call["kw"]

        assert argv[:2] == ["codex", "exec"]
        assert argv[argv.index("-C") + 1] == str(tmp_path)
        assert argv[argv.index("-s") + 1] == "workspace-write"
        assert "use_legacy_landlock" not in argv
        assert "--skip-git-repo-check" in argv
        assert "--ephemeral" in argv
        assert argv[argv.index("--color") + 1] == "never"
        assert argv[argv.index("-o") + 1] == str(tmp_path / "codex_last_message.txt")
        configs = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
        assert 'model_reasoning_effort="medium"' in configs
        assert "sandbox_workspace_write.network_access=false" in configs
        assert 'shell_environment_policy.inherit="core"' in configs
        assert any(
            c.startswith("shell_environment_policy.exclude=")
            and "*KEY*" in c and "*TOKEN*" in c and "*SECRET*" in c and "*PASSWORD*" in c
            for c in configs
        )
        assert not any("dangerously-bypass" in a for a in argv)
        assert argv[-1] == "提示词"
        assert kw["timeout"] == 600
        assert kw.get("shell") is not True
        assert kw["capture_output"] is True and kw["text"] is True

    def test_env_scrubbed(self, captured_codex, monkeypatch, tmp_path):
        """调起 codex 的进程环境不得携带秘密变量；PATH/HOME 保留。"""
        monkeypatch.setenv("ARK_API_KEY", "topsecret")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "topsecret")
        monkeypatch.setenv("MY_DB_PASSWORD", "topsecret")
        monkeypatch.setenv("SAFE_VAR", "ok")
        CodexRunner(timeout_s=1, concurrency=1).run(tmp_path, "p")
        env = captured_codex[0]["kw"]["env"]
        assert env is not None
        for key in env:
            assert not re.search(r"KEY|TOKEN|SECRET|PASSWORD", key, re.IGNORECASE), key
        assert env["SAFE_VAR"] == "ok"
        assert "PATH" in env and "HOME" in env

    def test_timeout(self, monkeypatch):
        def slow(argv, **kw):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kw["timeout"])

        monkeypatch.setattr(codex_runner.subprocess, "run", slow)
        with pytest.raises(CodexError, match="timed out") as caught:
            CodexRunner(timeout_s=7, concurrency=1).run(Path("/wd"), "p")
        assert caught.value.retryable is True

    def test_nonzero_stderr_scrubbed(self, monkeypatch):
        stderr = (
            "PATH=/usr/bin\nAWS_SECRET_ACCESS_KEY=abc123\n" + "z" * 1200 + "\nreal codex failure\n"
        )
        fake = subprocess.CompletedProcess(args=[], returncode=3, stdout="", stderr=stderr)
        monkeypatch.setattr(codex_runner.subprocess, "run", lambda *a, **kw: fake)
        with pytest.raises(CodexError) as exc_info:
            CodexRunner(timeout_s=1, concurrency=1).run(Path("/wd"), "p")
        msg = str(exc_info.value)
        assert "real codex failure" in msg
        assert "abc123" not in msg and "AWS_SECRET_ACCESS_KEY" not in msg
        assert len(msg) <= 560
        assert exc_info.value.retryable is True

    def test_missing_codex_binary(self, monkeypatch):
        def nope(argv, **kw):
            raise FileNotFoundError(2, "No such file or directory", argv[0])

        monkeypatch.setattr(codex_runner.subprocess, "run", nope)
        with pytest.raises(CodexError, match="codex") as caught:
            CodexRunner(timeout_s=1, concurrency=1).run(Path("/wd"), "p")
        assert caught.value.retryable is False

    def test_concurrency_serialized(self, monkeypatch):
        runner = CodexRunner(timeout_s=30, concurrency=1)
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_run(argv, **kw):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
        threads = [
            threading.Thread(target=runner.run, args=(Path("/wd"), "p")) for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert max_active == 1

    def test_voice_run_stages_only_audio_inputs_behind_outer_bwrap(
        self, monkeypatch, tmp_path
    ):
        """音频 agent 的可见输入只有 voice.mp3 与最小 manifest；输出不直接落主 work。"""
        cdir = tmp_path / "conversation"
        work = cdir / "work"
        frames = work / "frames"
        frames.mkdir(parents=True)
        (cdir / "source.mp4").write_text("SOURCE_VISUAL_SECRET", encoding="utf-8")
        (frames / "000001.png").write_text("FRAME_VISUAL_SECRET", encoding="utf-8")
        (work / "contact_sheet.jpg").write_text("OCR_FAKE_DIALOGUE", encoding="utf-8")
        (work / "visual_prompt.txt").write_text("OCR_FAKE_DIALOGUE", encoding="utf-8")
        (work / "voice.mp3").write_bytes(b"audio-only")
        (work / "manifest.json").write_text(
            json.dumps({"duration_seconds": 99, "frames": ["000001.png"]}),
            encoding="utf-8",
        )

        calls = []

        def fake_run(argv, **kwargs):
            calls.append((list(argv), kwargs))
            stage = Path(argv[argv.index("-C") + 1])
            visible = {
                path.relative_to(stage).as_posix()
                for path in stage.rglob("*")
                if path.is_file()
            }
            assert visible == {"work/voice.mp3", "work/manifest.json"}
            assert json.loads((stage / "work" / "manifest.json").read_text()) == {
                "duration_seconds": 1.25
            }
            (stage / "work" / "voice_lines.json").write_text(
                json.dumps([{"text": "真实口播", "start_s": 0.0, "end_s": 1.0}]),
                encoding="utf-8",
            )
            (stage / "work" / "rogue.txt").write_text("ignore me", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setenv("ARK_API_KEY", "must-not-leak")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-leak")
        monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
        lines = CodexRunner(timeout_s=7, concurrency=1).run_voice(
            work,
            "voice prompt",
            duration_s=1.25,
            validate_output=lambda raw: voice.validate_voice_lines(raw, 1.25),
        )

        assert lines == [{"text": "真实口播", "start_s": 0.0, "end_s": 1.0}]
        assert not (work / "voice_lines.json").exists()
        assert not (work / "rogue.txt").exists()
        (argv, kwargs), = calls
        assert Path(argv[0]).name == "bwrap"
        assert any(argv[index:index + 2] == ["--tmpfs", "/tmp"] for index in range(len(argv)))
        assert "--bind" in argv and "--chdir" in argv
        inner = argv.index("codex")
        assert argv[inner:inner + 2] == ["codex", "exec"]
        assert argv[argv.index("-s", inner) + 1] == "workspace-write"
        assert "sandbox_workspace_write.network_access=false" in argv
        assert kwargs["timeout"] == 7
        assert all(
            secret not in kwargs["env"]
            for secret in ("ARK_API_KEY", "ANTHROPIC_AUTH_TOKEN")
        )

    def test_voice_run_fails_closed_without_bwrap(self, monkeypatch, tmp_path):
        work = tmp_path / "conversation" / "work"
        work.mkdir(parents=True)
        (work / "voice.mp3").write_bytes(b"audio")
        monkeypatch.setattr(codex_runner.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            codex_runner.subprocess,
            "run",
            lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("must not run")),
        )

        with pytest.raises(CodexError, match="bwrap"):
            CodexRunner(timeout_s=1, concurrency=1).run_voice(
                work,
                "voice prompt",
                duration_s=1.0,
                validate_output=lambda raw: raw,
            )

    def test_voice_run_rejects_symlinked_audio(self, tmp_path):
        work = tmp_path / "conversation" / "work"
        work.mkdir(parents=True)
        outside = tmp_path / "outside.mp3"
        outside.write_bytes(b"audio")
        (work / "voice.mp3").symlink_to(outside)

        with pytest.raises(CodexError, match="voice.mp3"):
            CodexRunner(timeout_s=1, concurrency=1).run_voice(
                work,
                "voice prompt",
                duration_s=1.0,
                validate_output=lambda raw: raw,
            )

    def test_real_nonpaid_voice_sandbox_probe_blocks_session_and_repo(self, tmp_path):
        """真实 bwrap + `codex sandbox` 探针，不调用模型/API。"""
        if not shutil.which("bwrap") or not shutil.which("codex"):
            pytest.skip("bwrap/codex unavailable")
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        (cdir / "source.mp4").write_text("outside", encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="duet-voice-probe-", dir="/tmp") as raw_stage:
            stage = Path(raw_stage)
            (stage / "work").mkdir()
            (stage / "work" / "voice.mp3").write_bytes(b"audio")
            (stage / "work" / "manifest.json").write_text(
                json.dumps({"duration_seconds": 1.0}), encoding="utf-8"
            )
            script = (
                f"test ! -r {str(cdir / 'source.mp4')!r} && "
                f"test ! -r {str(ROOT / 'app' / 'pipeline.py')!r} && "
                "test -r work/voice.mp3 && test -r work/manifest.json && "
                "printf '[]' > work/voice_lines.json"
            )
            inner = [
                "codex", "sandbox", "-P", ":workspace", "-C", str(stage),
                "bash", "-c", script,
            ]
            argv = codex_runner._isolated_outer_argv(stage, cdir, inner)
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=20)

            assert proc.returncode == 0, proc.stderr
            assert (stage / "work" / "voice_lines.json").read_text() == "[]"

    def test_real_readonly_stage_allows_atomic_exact_output_publish(self, tmp_path):
        """真实 bwrap 探针：冻结输入不可写，声明输出可原子替换。"""
        if not shutil.which("bwrap"):
            pytest.skip("bwrap unavailable")
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        with tempfile.TemporaryDirectory(
            prefix="duet-fusion-permission-probe-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            work = stage / "work"
            image = work / "selected" / "01.png"
            image.parent.mkdir(parents=True)
            skill = stage / "SKILL.md"
            descriptor = work / "multimodal_input.json"
            output = work / "h3_prompt_plan.json"
            skill.write_text("frozen-skill", encoding="utf-8")
            descriptor.write_text("frozen-input", encoding="utf-8")
            image.write_bytes(b"frozen-png")
            output.touch(mode=0o600)
            script = (
                "for target in SKILL.md work/multimodal_input.json "
                "work/selected/01.png; do "
                "if printf tampered > \"$target\" 2>/dev/null; then exit 41; fi; "
                "done; "
                "printf generated > work/.h3_prompt_plan.tmp && "
                "mv -f work/.h3_prompt_plan.tmp work/h3_prompt_plan.json"
            )
            argv = codex_runner._isolated_outer_argv(
                stage,
                cdir,
                ["/usr/bin/bash", "-c", script],
                writable_paths=(output,),
            )

            assert any(
                argv[index:index + 3]
                == ["--bind", str(stage), str(stage)]
                for index in range(len(argv) - 2)
            )
            for frozen in (skill, descriptor, image):
                assert any(
                    argv[index:index + 3]
                    == ["--ro-bind", str(frozen), str(frozen)]
                    for index in range(len(argv) - 2)
                )
            assert not any(
                argv[index:index + 3] == ["--bind", str(output), str(output)]
                for index in range(len(argv) - 2)
            )
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=20,
            )

            assert proc.returncode == 0, proc.stderr
            assert skill.read_text(encoding="utf-8") == "frozen-skill"
            assert descriptor.read_text(encoding="utf-8") == "frozen-input"
            assert image.read_bytes() == b"frozen-png"
            assert output.read_text(encoding="utf-8") == "generated"

    def test_run_isolated_threads_exact_writable_file_into_bwrap_argv(
        self, captured_codex, tmp_path,
    ):
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        with tempfile.TemporaryDirectory(
            prefix="duet-fusion-argv-probe-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            work = stage / "work"
            work.mkdir()
            output = work / "h3_prompt_plan.json"
            output.touch(mode=0o600)

            CodexRunner(timeout_s=3, concurrency=1).run_isolated(
                stage,
                "fusion prompt",
                session_dir=cdir,
                writable_paths=(output,),
            )

        (call,) = captured_codex
        argv = call["argv"]
        assert any(
            argv[index:index + 3] == ["--bind", str(stage), str(stage)]
            for index in range(len(argv) - 2)
        )
        assert not any(
            argv[index:index + 3] == ["--bind", str(output), str(output)]
            for index in range(len(argv) - 2)
        )
        assert argv[argv.index("-C") + 1] == str(stage)

    def test_isolated_atomic_output_is_completion_signal_before_process_exit(
        self, monkeypatch, tmp_path,
    ):
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        monkeypatch.setattr(
            codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
        )
        with tempfile.TemporaryDirectory(
            prefix="duet-output-completion-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            work = stage / "work"
            work.mkdir()
            output = work / "result.json"
            runner = CodexRunner(timeout_s=10, concurrency=1)
            monkeypatch.setattr(
                runner,
                "build_argv",
                lambda _workdir, _prompt: [
                    "/usr/bin/bash", "-c",
                    f"printf '{{\"ok\":true}}' > '{work / '.result.tmp'}'; "
                    f"mv -f '{work / '.result.tmp'}' '{output}'; /usr/bin/sleep 8",
                ],
            )

            started = time.monotonic()
            result = runner.run_isolated_until_output(
                stage,
                "prompt",
                session_dir=cdir,
                output_path=output,
                max_output_bytes=1024,
                validate_output=lambda raw: json.loads(raw.decode("utf-8")),
            )

            assert result == {"ok": True}
            assert time.monotonic() - started < 3

    def test_isolated_direct_write_is_adopted_after_clean_exit(
        self, monkeypatch, tmp_path,
    ):
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        monkeypatch.setattr(
            codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
        )
        with tempfile.TemporaryDirectory(
            prefix="duet-output-partial-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            work = stage / "work"
            work.mkdir()
            output = work / "result.json"
            runner = CodexRunner(timeout_s=3, concurrency=1)
            monkeypatch.setattr(
                runner,
                "build_argv",
                lambda _workdir, _prompt: [
                    "/usr/bin/bash", "-c", f"printf '{{\"ok\":true}}' > '{output}'",
                ],
            )

            assert runner.run_isolated_until_output(
                stage,
                "prompt",
                session_dir=cdir,
                output_path=output,
                max_output_bytes=1024,
                validate_output=lambda raw: json.loads(raw.decode("utf-8")),
            ) == {"ok": True}

    def test_isolated_direct_write_is_not_adopted_before_clean_exit(
        self, monkeypatch, tmp_path,
    ):
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        monkeypatch.setattr(
            codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
        )
        with tempfile.TemporaryDirectory(
            prefix="duet-output-in-place-live-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            work = stage / "work"
            work.mkdir()
            output = work / "result.json"
            runner = CodexRunner(timeout_s=3, concurrency=1)
            monkeypatch.setattr(
                runner,
                "build_argv",
                lambda _workdir, _prompt: [
                    "/usr/bin/bash", "-c",
                    f"printf '{{\"ok\":true}}' > '{output}'; "
                    f"/usr/bin/sleep 0.3; printf '{{' > '{output}'",
                ],
            )

            with pytest.raises(CodexError, match="without publishing valid output"):
                runner.run_isolated_until_output(
                    stage,
                    "prompt",
                    session_dir=cdir,
                    output_path=output,
                    max_output_bytes=1024,
                    validate_output=lambda raw: json.loads(raw.decode("utf-8")),
                )

    def test_isolated_direct_write_does_not_hide_nonzero_exit(
        self, monkeypatch, tmp_path,
    ):
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        monkeypatch.setattr(
            codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
        )
        with tempfile.TemporaryDirectory(
            prefix="duet-output-in-place-error-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            work = stage / "work"
            work.mkdir()
            output = work / "result.json"
            runner = CodexRunner(timeout_s=3, concurrency=1)
            monkeypatch.setattr(
                runner,
                "build_argv",
                lambda _workdir, _prompt: [
                    "/usr/bin/bash", "-c",
                    f"printf '{{\"ok\":true}}' > '{output}'; exit 7",
                ],
            )

            with pytest.raises(CodexError, match="codex exit 7"):
                runner.run_isolated_until_output(
                    stage,
                    "prompt",
                    session_dir=cdir,
                    output_path=output,
                    max_output_bytes=1024,
                    validate_output=lambda raw: json.loads(raw.decode("utf-8")),
                )

    def test_isolated_completion_kills_term_ignoring_descendants(
        self, monkeypatch, tmp_path,
    ):
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        marker = tmp_path / "descendant-survived"
        monkeypatch.setattr(
            codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
        )
        with tempfile.TemporaryDirectory(
            prefix="duet-output-descendant-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            work = stage / "work"
            work.mkdir()
            output = work / "result.json"
            runner = CodexRunner(timeout_s=5, concurrency=1)
            monkeypatch.setattr(
                runner,
                "build_argv",
                lambda _workdir, _prompt: [
                    "/usr/bin/bash", "-c",
                    "trap '' TERM; "
                    f"(trap '' TERM; /usr/bin/sleep 2; /usr/bin/touch '{marker}') & "
                    f"printf '{{\"ok\":true}}' > '{work / '.result.tmp'}'; "
                    f"mv -f '{work / '.result.tmp'}' '{output}'; wait",
                ],
            )

            assert runner.run_isolated_until_output(
                stage,
                "prompt",
                session_dir=cdir,
                output_path=output,
                max_output_bytes=1024,
                validate_output=lambda raw: json.loads(raw.decode("utf-8")),
            ) == {"ok": True}
            time.sleep(1.2)
            assert not marker.exists()

    def test_isolated_completion_rejects_symlinked_output_parent(
        self, monkeypatch, tmp_path,
    ):
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setattr(
            codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
        )
        with tempfile.TemporaryDirectory(
            prefix="duet-output-parent-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            (stage / "work").symlink_to(outside, target_is_directory=True)
            runner = CodexRunner(timeout_s=3, concurrency=1)
            monkeypatch.setattr(
                runner,
                "build_argv",
                lambda *_args: (_ for _ in ()).throw(AssertionError("must not spawn")),
            )

            with pytest.raises(CodexError, match="output path"):
                runner.run_isolated_until_output(
                    stage,
                    "prompt",
                    session_dir=cdir,
                    output_path=stage / "work" / "result.json",
                    max_output_bytes=1024,
                    validate_output=lambda raw: raw,
                )

    def test_readonly_isolated_fails_closed_on_namespace_error(
        self, monkeypatch, tmp_path,
    ):
        cdir = tmp_path / "conversation"
        cdir.mkdir()
        monkeypatch.setattr(
            codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
        )
        monkeypatch.setattr(
            codex_runner.subprocess,
            "run",
            lambda argv, **_kwargs: subprocess.CompletedProcess(
                argv, 1, "", "bwrap: Creating new namespace failed"
            ),
        )
        with tempfile.TemporaryDirectory(
            prefix="duet-fusion-namespace-probe-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            work = stage / "work"
            work.mkdir()
            output = work / "h3_prompt_plan.json"
            output.touch(mode=0o600)

            with pytest.raises(CodexError, match="namespace failed"):
                CodexRunner(timeout_s=3, concurrency=1).run_isolated(
                    stage,
                    "fusion prompt",
                    session_dir=cdir,
                    writable_paths=(output,),
                )

            assert output.read_bytes() == b""


# ---------- 流水线编排（状态机） ----------


def test_run_done(tmp_path, video_1s, fake_steps):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    cid = meta["id"]
    cdir = settings.data_dir / cid

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    done = storage.load_meta(settings.data_dir, cid)
    assert done["status"] == "done" and done["error"] is None
    assert done["keyframes"] == ["01.png", "02.png", "03.png"]
    # 单段模式不添加任何 workaround 前缀，meta 与磁盘同步
    assert done["prompt"] == PROMPT_TEXT
    assert (cdir / "work" / "prompt.txt").read_text(encoding="utf-8") == done["prompt"]
    assert not (cdir / "preview.mp4").exists()  # 新契约不再生成占位预览

    # extract 调用契约：venv python 绝对路径、argv 列表、--fps 4、120s 超时
    extract = fake_steps["cmd"][0]
    assert extract["step"] == "extract"
    assert extract["argv"][0] == sys.executable
    assert str(EXTRACT_SCRIPT) in extract["argv"]
    assert "--fps" in extract["argv"] and "4" in extract["argv"]
    assert extract["timeout"] == 120

    # codex 运行前 skill 的 scripts/ 拷进会话目录（crop_image.py 相对引用）
    assert (cdir / "scripts" / "crop_image.py").read_bytes() == CROP_SCRIPT.read_bytes()
    assert (cdir / "scripts" / "extract_keyframes.py").is_file()

    # codex：工作目录=会话目录；prompt 指向 SKILL.md 且含硬性禁令
    (codex_call,) = fake_steps["codex"]
    assert codex_call["workdir"] == cdir
    prompt = codex_call["prompt"]
    for needle in (
        str(pipeline.SKILL_MD),
        "work/",
        sys.executable,
        str(cdir),
        "禁止联网",
        "环境变量",
    ):
        assert needle in prompt, needle


def test_run_status_sequence_processing_then_done(tmp_path, video_1s, fake_steps, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    seen = []
    original_claim = storage.claim_pipeline_input
    original_finish = storage.finish_input_claim

    def claim(data_dir, cid):
        claimed = original_claim(data_dir, cid)
        if claimed is not None:
            seen.append(claimed["status"])
        return claimed

    def finish(data_dir, cid, owner, **changes):
        if "status" in changes:
            seen.append(changes["status"])
        return original_finish(data_dir, cid, owner, **changes)

    monkeypatch.setattr(storage, "claim_pipeline_input", claim)
    monkeypatch.setattr(storage, "finish_input_claim", finish)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    assert seen == ["processing", "done"]


def test_run_extract_failure(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def boom(argv, *, timeout, step, cwd=None):
        raise pipeline.PipelineError(f"{step} exit 1: codec missing")

    monkeypatch.setattr(pipeline, "_run_cmd", boom)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "extract" in m["error"]


def test_run_codex_failure(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
            )

    def bad_codex(self, workdir, prompt):
        raise CodexError("codex exit 2: agent crashed")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", bad_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "codex" in m["error"]


def test_run_codex_timeout(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
            )

    def slow_codex(self, workdir, prompt):
        raise CodexError("codex timed out after 600s")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", slow_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "timed out" in m["error"]


def test_run_visual_retries_invalid_output_and_transient_timeout(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    calls = 0

    def flaky_codex(self, workdir, prompt):
        nonlocal calls
        calls += 1
        work = Path(workdir) / "work"
        assert not (work / "keyframes").exists()
        assert not (work / "prompt.txt").exists()
        if calls == 1:
            (work / "keyframes").mkdir()
            (work / "keyframes" / "stale.txt").write_text("stale")
            return
        if calls == 2:
            raise CodexError("codex timed out after 600s", retryable=True)
        _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", flaky_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert calls == 3
    assert not (settings.data_dir / meta["id"] / "work" / "keyframes" / "stale.txt").exists()


def test_run_codex_timeout_salvages_complete_output(tmp_path, video_1s, monkeypatch):
    """codex 超时被杀但产物已完整落盘 → 收养为 done。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
            )

    def slow_codex(self, workdir, prompt):
        _write_valid_package(Path(workdir) / "work")  # 被杀前产物已写完
        raise CodexError("codex timed out after 600s")

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", slow_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done"
    assert m["keyframes"] == ["01.png", "02.png", "03.png"]
    assert m["prompt"] == PROMPT_TEXT


def test_run_validation_failure(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)

    def fake_cmd(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text("{}")
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 1.0, "scenes": [], "segments": []})
            )

    def noop_codex(self, workdir, prompt):
        pass  # 一个产物都不写

    monkeypatch.setattr(pipeline, "_run_cmd", fake_cmd)
    monkeypatch.setattr(CodexRunner, "run", noop_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert m["error"] == (
        "codex visual output invalid: required keyframes/prompt artifacts "
        "are missing or invalid"
    )
    assert "keyframe count 0" not in m["error"]


# ---------- 口播步（ASR，抽帧之后） ----------

VOICE_LINES = [
    {"text": "第一句。", "start_s": 0.0, "end_s": 0.5},
    {"text": "第二句。", "start_s": 0.5, "end_s": 1.0},
]


def _write_exact_nine_extract_manifest(work: Path, duration_s: float) -> None:
    work.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(1, 10):
        time_s = duration_s * (index - 1) / 8
        name = f"{index:03d}_frame_{time_s:07.3f}s.png"
        (work / name).write_bytes(_PX_PNG)
        frames.append({
            "index": index,
            "time_seconds": round(time_s, 6),
            "file": name,
        })
    (work / "manifest.json").write_text(
        json.dumps({"duration_seconds": duration_s, "frames": frames}),
        encoding="utf-8",
    )


def _write_exact_scene_inventory(
    work: Path, source: Path, duration_s: float,
) -> None:
    """Freeze the exact decode inventory emitted by the current scene detector."""
    capture = pipeline.cv2.VideoCapture(str(source))
    assert capture.isOpened()
    frame_count = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            frame_count += 1
    finally:
        capture.release()
    assert frame_count > 0
    frames = [
        {
            "decode_frame_index": index,
            "pts": index,
            "time_base_num": round(duration_s * 1_000_000),
            "time_base_den": frame_count * 1_000_000,
        }
        for index in range(frame_count)
    ]
    effective_scene = {
        "index": 1,
        "source_scene_indices": [1],
        "start_decode_frame_index": 0,
        "end_decode_frame_index": frame_count,
        "start_s": 0.0,
        "end_s": duration_s,
        "frames": frames,
    }
    (work / "scenes.json").write_text(
        json.dumps({
            "duration_s": duration_s,
            "scenes": [{
                "index": 1,
                "start_s": 0.0,
                "end_s": duration_s,
            }],
            "effective_scenes": [effective_scene],
            "segments": [],
        }),
        encoding="utf-8",
    )


def _run_fake_media_step(argv, step: str) -> bool:
    if step == "segment cut":
        subprocess.run(argv, check=True, capture_output=True)
        return True
    if step.startswith("segment ") and step.endswith(" extract"):
        source = Path(argv[2])
        work = Path(argv[argv.index("--out-dir") + 1])
        _write_exact_nine_extract_manifest(
            work, storage.probe_video(source).duration_s,
        )
        return True
    return False


def _fake_extract_ok(argv, *, timeout, step, cwd=None):
    """extract/scenes 假子进程：写 manifest（含 duration_seconds，口播步要读）与空拆段 scenes.json。"""
    if _run_fake_media_step(argv, step):
        return
    if step == "extract":
        duration_s = storage.probe_video(Path(argv[2])).duration_s
        work = Path(argv[argv.index("--out-dir") + 1])
        (work / "contact_sheet.jpg").write_bytes(b"sheet")
        (work / "manifest.json").write_text(
            json.dumps({"duration_seconds": duration_s}), encoding="utf-8"
        )
    elif step == "scenes":
        work = Path(argv[argv.index("--work-dir") + 1])
        duration_s = json.loads(
            (work / "manifest.json").read_text(encoding="utf-8")
        )["duration_seconds"]
        _write_exact_scene_inventory(work, Path(argv[2]), duration_s)


def _set_voice_mode(settings, meta, voice_mode, target_language=""):
    changes = {"voice_mode": voice_mode}
    if target_language:
        changes["target_language"] = target_language
    storage.update_meta(settings.data_dir, meta["id"], **changes)


def _no_codex(self, workdir, prompt):
    raise AssertionError("codex must not run")


def test_run_voice_none_skips_asr(tmp_path, video_1s, fake_steps):
    """voice_mode=none（默认）不跑口播步：codex 只被调一次、无 voice 产物。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    cdir = settings.data_dir / meta["id"]

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done"
    assert "voice_lines" not in m
    (codex_call,) = fake_steps["codex"]
    assert "voice.mp3" not in codex_call["prompt"] and "听写" not in codex_call["prompt"]
    assert not (cdir / "work" / "voice.mp3").exists()


def test_run_voice_none_does_not_call_vocal_analyze(tmp_path, video_1s, fake_steps, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    called = []

    def unexpected(audio):
        called.append(audio)
        raise AssertionError("vocal.analyze must not run when voice_mode=none")

    monkeypatch.setattr(vocal, "analyze", unexpected)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    assert called == []
    assert storage.load_meta(settings.data_dir, meta["id"])["status"] == "done"


def test_run_voice_vocal_filter_records_bgm_and_dropped_count(
    tmp_path, video_1s, monkeypatch
):
    """句级过滤：spoken 与 sung 都保留（sung = 吟唱/唱词型台词），只丢 None 假转录。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    lines = [
        {"text": "口播", "start_s": 0.0, "end_s": 0.3},
        {"text": "唱歌", "start_s": 0.3, "end_s": 0.6},
        {"text": "幻觉", "start_s": 0.6, "end_s": 0.9},
    ]
    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps(lines), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[
                vocal.VocalWindow(0, 300, sung=0.0, spoken=0.3, music=0.2),
                vocal.VocalWindow(300, 600, sung=0.1, spoken=0.01, music=0.2),
                vocal.VocalWindow(600, 900, sung=0.01, spoken=0.01, music=0.2),
            ],
            has_bgm=True,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done"
    assert stored["voice_lines"] == [lines[0], lines[1]]
    assert stored["has_bgm"] is True
    assert stored["voice_lines_vocal_dropped"] == 1
    assert stored["vocal_filter_enabled"] is True
    _assert_analysis_bound_provenance(stored["voice_line_provenance"], [
        {**lines[0], "classification": "spoken", "provenance": "asr", "kept": True},
        {**lines[1], "classification": "sung", "provenance": "asr", "kept": True},
        {**lines[2], "classification": None, "provenance": "asr", "kept": False},
    ], audio=b"mp3-bytes", has_bgm=True)


def test_run_voice_drops_spoken_subtitle_credit_before_prompt(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    credit = {"text": "(字幕製作:貝爾)", "start_s": 0.0, "end_s": 0.8}

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(
                json.dumps([credit]), encoding="utf-8"
            )
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 800, sung=0.0, spoken=0.8, music=0.0)],
            has_bgm=False,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done"
    assert stored["voice_lines"] == []
    assert stored["voice_lines_credit_dropped"] == 1
    _assert_analysis_bound_provenance(stored["voice_line_provenance"], [{
        **credit,
        "classification": "spoken",
        "provenance": "asr",
        "kept": False,
        "drop_reason": "subtitle_credit",
    }], audio=b"mp3-bytes", has_bgm=False)


def test_run_voice_vocal_filter_off_bypasses_but_records_decisions(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    lines = [
        {"text": "口播", "start_s": 0.0, "end_s": 0.4},
        {"text": "幻觉", "start_s": 0.4, "end_s": 0.9},
    ]

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps(lines), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setenv("VOCAL_FILTER", "off")
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[
                vocal.VocalWindow(0, 400, sung=0.0, spoken=0.3, music=0.2),
                vocal.VocalWindow(400, 900, sung=0.01, spoken=0.01, music=0.2),
            ],
            has_bgm=True,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == lines
    assert stored["vocal_filter_enabled"] is False
    _assert_analysis_bound_provenance(stored["voice_line_provenance"], [
        {**lines[0], "classification": "spoken", "provenance": "asr", "kept": True},
        {**lines[1], "classification": None, "provenance": "asr", "kept": True},
    ], audio=b"mp3-bytes", has_bgm=True)


def test_run_voice_vocal_failure_fails_pipeline(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        if "voice.mp3" in prompt:
            (Path(workdir) / "work" / "voice_lines.json").write_text(
                json.dumps([{"text": "口播", "start_s": 0.0, "end_s": 0.5}]),
                encoding="utf-8",
            )

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _audio: (_ for _ in ()).throw(
        vocal.VocalError("模型校验失败")
    ))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "failed"
    assert "vocal classification unavailable" in stored["error"]
    assert "模型校验失败" in stored["error"]


def test_run_voice_rejects_audio_replaced_during_vocal_analysis(
    tmp_path, video_1s, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"analyzed-bytes")
        return out

    def replace_during_analysis(audio):
        audio.write_bytes(b"replacement-bytes")
        return _spoken_analysis(False)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(vocal, "analyze", replace_during_analysis)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "failed"
    assert stored["error"] == "vocal analysis audio drifted"


def test_run_voice_audio_longer_than_container(tmp_path, video_1s, monkeypatch):
    """音频长 36ms 时 ASR 先按音频通过，再把最终台词裁回视频时间轴。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    line = {"text": "台词", "start_s": 0.224, "end_s": 27.936}  # end_s 超容器 27.9 但等于音频时长

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([line]), encoding="utf-8")
        else:
            _write_valid_package(work)

    def fake_extract_ok(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 27.9}), encoding="utf-8"
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 27.9, "scenes": [], "segments": []})
            )

    monkeypatch.setattr(pipeline, "_run_cmd", fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 27.936)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 30_000, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done"
    assert stored["voice_lines"] == [
        {"text": "台词", "start_s": 0.224, "end_s": 27.9}
    ]
    assert any("video duration 27.900s" in item for item in stored["voice_warnings"])


def test_prepared_duration_has_no_project_upper_bound():
    assert pipeline._prepared_durations({"duration_s": 3600.1}) == (3600.1, 3601)


def test_run_voice_keep_runs_asr_and_stores_lines(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    cdir = settings.data_dir / meta["id"]

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    codex_calls = []

    def fake_codex(self, workdir, prompt):
        codex_calls.append({"workdir": Path(workdir), "prompt": prompt})
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:  # ASR 调用
            (work / "voice_lines.json").write_text(json.dumps(VOICE_LINES), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done" and m["error"] is None
    assert m["voice_lines"] == VOICE_LINES
    assert (cdir / "work" / "voice.mp3").read_bytes() == b"mp3-bytes"

    # 两次 codex 调用：ASR 在前（抽帧之后）、video-maker 在后；ASR 不带 SKILL.md
    assert len(codex_calls) == 2
    (asr_call, maker_call) = codex_calls
    asr_prompt = asr_call["prompt"]
    assert "work/voice.mp3" in asr_prompt
    assert "work/manifest.json" in asr_prompt
    assert "1.000" in asr_prompt  # 时长数字直传 prompt
    assert "原文保持" in asr_prompt
    assert str(pipeline.SKILL_MD) not in asr_prompt
    for needle in ("禁止联网", "环境变量"):
        assert needle in asr_prompt, needle
    assert sys.executable not in asr_prompt
    assert str(cdir) not in asr_prompt
    assert asr_call["workdir"] != cdir
    assert asr_call["workdir"].parent == Path("/tmp")
    assert str(pipeline.SKILL_MD) in maker_call["prompt"]


def test_run_voice_agent_cannot_see_visual_or_ocr_inputs(
    tmp_path, video_1s, monkeypatch
):
    """独立污染反例：原会话有 OCR 假台词和视觉输入，ASR stage 仍只有音频输入。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    cdir = settings.data_dir / meta["id"]
    work = cdir / "work"
    work.mkdir(exist_ok=True)
    (work / "visual_prompt.txt").write_text("画面字：买一送一（不要朗读）", encoding="utf-8")
    (work / "frame_note.txt").write_text("OCR_FAKE_DIALOGUE", encoding="utf-8")
    asr_seen = []
    real_line = {"text": "Jumpa geng sekelapa.", "start_s": 0.0, "end_s": 0.8}

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        stage_work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            visible = {
                path.relative_to(Path(workdir)).as_posix()
                for path in Path(workdir).rglob("*")
                if path.is_file()
            }
            asr_seen.append((Path(workdir), visible))
            assert visible == {"work/voice.mp3", "work/manifest.json"}
            assert "OCR_FAKE_DIALOGUE" not in "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in Path(workdir).rglob("*")
                if path.is_file()
            )
            (stage_work / "voice_lines.json").write_text(
                json.dumps([real_line]), encoding="utf-8"
            )
        else:
            _write_valid_package(stage_work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == [real_line]
    assert len(asr_seen) == 1
    assert asr_seen[0][0] != cdir


def test_run_voice_translate_prompt_has_target_language(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "translate", target_language="英文")
    calls = []

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        calls.append(prompt)
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps(VOICE_LINES), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done"
    assert m["voice_lines"] == VOICE_LINES
    assert "翻译成英文" in calls[0]
    # 目标语言由后端注入 maker prompt（codex 不从台词反推）
    assert "提示词与台词使用目标语言：英文" in calls[1]


def test_run_voice_rewrite_prompt_has_rule_and_lines(tmp_path, video_1s, monkeypatch):
    """rewrite 模式：ASR prompt 含洗稿规则（句数/句序/时间边界不变）且 voice_lines 落 meta。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "rewrite")
    calls = []
    rewritten_lines = [
        {"text": "直接使用金刚刷。", "start_s": 0.0, "end_s": 1.0}
    ]

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        calls.append(prompt)
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(
                json.dumps(rewritten_lines), encoding="utf-8"
            )
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done"
    assert m["voice_lines"] == [
        {"text": "直接使用金刚刷。", "start_s": 0.0, "end_s": 1.0}
    ]
    assert m["voice_text_normalizations"] == []
    asr_prompt = calls[0]
    assert "洗稿" in asr_prompt
    assert "句数不变" in asr_prompt
    assert "句序不变" in asr_prompt
    assert "时间边界不变" in asr_prompt
    assert "通用称呼" in asr_prompt


def test_run_voice_mode_unknown_fails(tmp_path, video_1s, monkeypatch):
    """绕过入口校验直改 meta 的非法 voice_mode → failed 且 error 含 unknown voice_mode。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "dub")
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", _no_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "unknown voice_mode" in m["error"]


def test_run_voice_translate_whitespace_target_fails(tmp_path, video_1s, monkeypatch):
    """target_language 为纯空白串 → 视为缺失 failed，不生成「翻译成   」prompt。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "translate", target_language="   ")
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", _no_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "target_language" in m["error"]


def test_run_voice_translate_requires_target_language(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "translate")  # 契约要求 translate 必带 target_language
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", _no_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "target_language" in m["error"]


def test_run_voice_no_audio_track_fails(tmp_path, video_1s, monkeypatch):
    """无音轨兜底：extract_audio 探测返回 None → failed（上传校验只查时长，不查音轨）。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(CodexRunner, "run", _no_codex)
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))
    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "audio" in m["error"]


def test_run_dialogue_auto_no_audio_is_valid_and_writes_visual_plan_receipt(
    tmp_path, video_1s, monkeypatch
):
    """新 H3 auto：无音轨等价于空台词，exact-nine 视觉计划仍会冻结。"""

    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _replace_source_with_duration(settings, meta["id"], 10.0)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )

    def fake_codex(self, workdir, prompt):
        _write_valid_package(Path(workdir) / "work", frames=9)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", lambda _cdir: None)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    cdir = settings.data_dir / meta["id"]
    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []
    assert stored["voice_line_provenance"] == []
    assert stored["vocal_filter_enabled"] is True
    assert stored["long_video_plan_receipt"] == long_video.PLAN_RECEIPT_FILENAME
    receipt_path = cdir / long_video.PLAN_RECEIPT_FILENAME
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["video"]["duration_s"] == 10.0
    assert len(receipt["segments"]) == 1
    assert receipt["segments"][0]["dialogue"]["count"] == 0
    assert stored["segments"][0]["keyframes"] == [
        f"{index:02d}.png" for index in range(1, 10)
    ]
    frozen = stored["_image_optimization"]
    continuity = stored["_image_continuity"]
    assert continuity["version"] == 4
    assert continuity["segment_indices"] == [1]
    assert continuity["eligible"] is True
    assert pipeline.image_optimization.dual_target_plan_receipt(stored) == continuity
    assert frozen["version"] == 4
    assert [(item["segment_index"], item["frame_name"])
            for item in frozen["frames"]] == [
        (1, f"{index:02d}.png") for index in range(1, 10)
    ]


@pytest.mark.parametrize(
    ("dialogue_mode", "provenance"),
    [("edit", "asr+edited"), ("custom", "manual")],
)
def test_run_manual_dialogue_uses_frozen_lines_without_source_audio(
    tmp_path, video_1s, monkeypatch, dialogue_mode, provenance
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _replace_source_with_duration(settings, meta["id"], 10.0)
    frozen_lines = [{
        "text": "用户冻结台词。",
        "start_s": 0.25,
        "end_s": 1.25,
        "classification": None,
        "provenance": provenance,
    }]
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode=dialogue_mode,
        voice_mode="keep",
        voice_lines=frozen_lines,
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )

    def fake_codex(self, workdir, prompt):
        assert "voice.mp3" not in prompt
        _write_valid_package(Path(workdir) / "work", frames=9)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(
        voice,
        "extract_audio",
        lambda *_args, **_kwargs: pytest.fail(
            "manual dialogue must not extract source audio"
        ),
    )
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    cdir = settings.data_dir / meta["id"]
    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == frozen_lines
    assert stored["segments"][0]["dialogue"] == frozen_lines
    assert "用户冻结台词。" in stored["segments"][0]["prompt"]
    receipt = json.loads(
        (cdir / long_video.PLAN_RECEIPT_FILENAME).read_text(encoding="utf-8")
    )
    assert receipt["segments"][0]["dialogue"]["count"] == 1


def test_run_none_dialogue_skips_asr_and_source_audio(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _replace_source_with_duration(settings, meta["id"], 10.0)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="none",
        voice_mode="keep",
        voice_lines=[],
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(
        voice,
        "extract_audio",
        lambda *_args, **_kwargs: pytest.fail(
            "none dialogue must not extract source audio"
        ),
    )
    monkeypatch.setattr(
        CodexRunner,
        "run",
        lambda self, workdir, prompt: _write_valid_package(
            Path(workdir) / "work", frames=9
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []
    assert stored["segments"][0]["dialogue"] == []


def test_exact_nine_freezes_a_complete_v4_frame_bound_prompt_receipt(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _replace_source_with_duration(settings, meta["id"], 10.0)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", lambda _cdir: None)
    monkeypatch.setattr(
        CodexRunner,
        "run",
        lambda self, workdir, prompt: _write_valid_package(
            Path(workdir) / "work", frames=9,
        ),
    )
    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["_image_continuity"]["version"] == 4
    frozen = stored["_image_optimization"]
    assert frozen["version"] == 4
    assert [item["frame_name"] for item in frozen["frames"]] == [
        f"{index:02d}.png" for index in range(1, 10)
    ]
    assert all(item["segment_index"] == 1 for item in frozen["frames"])
    assert all(item["current"].strip() for item in frozen["frames"])


def test_short_pipeline_semantic_compiler_continues_to_frozen_v4(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, retry_interval_s=0)
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
    frames = []
    for index in (1, 2):
        frame = keyframes / f"{index:02d}.png"
        frame.write_bytes(_PX_PNG)
        frames.append(frame)
    skeleton = pipeline._frame_inventory(
        {0: frames},
        segment_lineage={
            0: {"chain_id": "short-000", "join_mode": "hard_cut"}
        },
    )
    segment_specs = [{
        "index": 0,
        "chain_id": "short-000",
        "join_mode": "hard_cut",
        "keyframes_dir": keyframes,
        "transition_skeleton": skeleton,
    }]

    class Runner:
        def __init__(self):
            self.calls = 0

        def run_isolated(self, workdir, _prompt, *, session_dir):
            self.calls += 1
            request = json.loads(
                (Path(workdir) / "work" / "request.json").read_text(
                    encoding="utf-8"
                )
            )
            semantic = _semantic_image_output(request, omit_wardrobe=True)
            output = Path(workdir) / "work" / (
                "global_plan.json"
                if request["phase"] == "global_plan" else "segment_frames.json"
            )
            output.write_text(
                json.dumps(
                    {key: semantic.get(key, {}) for key in ("people", "entities", "scenes")}
                    if request["phase"] == "global_plan"
                    else {"frames": semantic["frames"]}
                ),
                encoding="utf-8",
            )

    runner = Runner()
    monkeypatch.setattr(
        pipeline.image_optimization,
        "generate_project_prompts",
        _GENERATE_IMAGE_OPTIMIZATION_PROJECT,
    )
    plan, prompts = pipeline._generate_image_optimization_project(
        settings,
        runner,
        segment_specs,
        session_dir=tmp_path,
        step="project image plan",
    )
    continuity, frozen = pipeline._freeze_image_optimization(
        settings,
        {"keyframes": [frame.name for frame in frames]},
        plan,
        prompts,
        {0: frames},
        require_dual_target=True,
        segment_lineage={
            0: {"chain_id": "short-000", "join_mode": "hard_cut"}
        },
    )

    assert runner.calls == 2
    assert continuity["_image_continuity"]["version"] == 4
    receipt = frozen["_image_optimization"]
    assert receipt["version"] == 4
    assert [item["frame_name"] for item in receipt["frames"]] == [
        "01.png", "02.png"
    ]
    assert [
        item["source_transition_from_previous"]
        for item in receipt["execution_inputs"]["frames"]
    ] == ["start", "same_camera"]


@pytest.mark.parametrize(
    "plan",
    [
        None,
        {"version": 1, "segment_indices": [1], "elements": []},
        {
            "version": 2,
            "phase": "plan",
            "segment_indices": [0],
            "eligible": False,
            "reason": "person_replacement_unsafe",
            "person_plans": [],
            "scene_plans": [],
            "segments": [],
        },
    ],
    ids=["missing", "legacy-v1", "ineligible"],
)
def test_short_new_input_fails_closed_without_eligible_dual_target_plan(
    tmp_path, video_1s, monkeypatch, plan,
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )

    def fake_codex(self, workdir, prompt):
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", lambda _cdir: None)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        pipeline,
        "_generate_image_optimization_project",
        lambda *_args, **_kwargs: (plan, {0: "compiled prompt"}),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "failed"
    assert "_image_continuity" not in stored
    assert "_image_optimization" not in stored


def test_run_dialogue_auto_ignores_external_lines_and_isolates_visual_codex(
    tmp_path, video_1s, monkeypatch
):
    """auto 只收养 _voice_step：外部/OCR 文字不能进入唯一发声块。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _replace_source_with_duration(settings, meta["id"], 10.0)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=10.0,
        voice_lines=[{"text": "外部伪台词", "start_s": 0.0, "end_s": 0.8}],
        ratio="9:16",
        fit_mode="crop",
    )
    asr_line = {"text": "真实口播。", "start_s": 0.1, "end_s": 0.8}
    maker_saw_voice_file = []

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(
                json.dumps([asr_line]), encoding="utf-8"
            )
            return
        maker_saw_voice_file.append((work / "voice_lines.json").exists())
        assert "最终台词由后端" in prompt
        _write_valid_package(
            work, frames=9, prompt="画面包装上可见 OCR ONLY。",
        )

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    cdir = settings.data_dir / meta["id"]
    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert maker_saw_voice_file == [False]
    assert stored["voice_lines"] == [asr_line]
    segment_prompt = stored["segments"][0]["prompt"]
    assert '说出台词："真实口播。"，嘴型与画面同步' in segment_prompt
    assert '说出台词："OCR ONLY"' not in segment_prompt
    assert "外部伪台词" not in segment_prompt
    assert stored["segments"][0]["dialogue"] == [
        {**asr_line, "classification": "spoken"}
    ]
    assert (cdir / "work" / "voice.mp3").read_bytes() == b"normalized-audio"


@pytest.mark.parametrize(
    ("voice_mode", "target_language", "expected"),
    [
        ("rewrite", "", "洗稿"),
        ("translate", "日语", "翻译成日语"),
    ],
)
def test_run_dialogue_auto_preserves_requested_voice_processing_mode(
    tmp_path, video_1s, monkeypatch, voice_mode, target_language, expected
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _replace_source_with_duration(settings, meta["id"], 10.0)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode=voice_mode,
        target_language=target_language,
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )
    asr_prompts = []

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            asr_prompts.append(prompt)
            (work / "voice_lines.json").write_text(
                json.dumps([{"text": "台词。", "start_s": 0.1, "end_s": 0.8}]),
                encoding="utf-8",
            )
        else:
            _write_valid_package(work, frames=9)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert len(asr_prompts) == 1
    assert expected in asr_prompts[0]


def test_run_dialogue_auto_routes_explicit_empty_15_4s_scene_result_to_long_plan(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _replace_source_with_duration(settings, meta["id"], 15.4)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=15.4,
        ratio="9:16",
        fit_mode="pad",
    )
    line = {"text": "第十六秒台词。", "start_s": 9.5, "end_s": 10.5}

    def fake_steps(argv, *, timeout, step, cwd=None):
        if _run_fake_media_step(argv, step):
            return
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 15.4}), encoding="utf-8"
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            _write_exact_scene_inventory(work, Path(argv[2]), 15.4)

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([line]), encoding="utf-8")
        else:
            _write_valid_package(work, frames=9)

    monkeypatch.setattr(pipeline, "_run_cmd", fake_steps)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 15.4)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
                windows=[vocal.VocalWindow(0, 15_400, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    cdir = settings.data_dir / meta["id"]
    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["fit_required"] is True
    assert stored["voice_lines"] == [line]
    receipt = json.loads((cdir / "long_video_plan.json").read_text(encoding="utf-8"))
    assert receipt["video"]["duration_s"] == 15.4
    assert len(receipt["segments"]) == 2
    assert [
        long_video.provider_duration_s(item["start_s"], item["end_s"])
        for item in receipt["segments"]
    ] == [10, 6]
    assert stored["segments"][0]["dialogue"] == [
        {
            "text": "第十六秒",
            "start_s": 9.5,
            "end_s": 10.0,
            "classification": "spoken",
        }
    ]
    assert stored["segments"][1]["dialogue"] == [
        {
            "text": "台词。",
            "start_s": 0.0,
            "end_s": 0.5,
            "classification": "spoken",
        }
    ]
    assert "".join(
        item["text"]
        for segment in stored["segments"]
        for item in segment["dialogue"]
    ) == line["text"]
    assert stored["voice_lines_dropped"] == 0
    assert all(
        len(segment["keyframe_sampling"]["keyframes"]) == 9
        for segment in stored["segments"]
    )
    assert [
        item["source_time_s"]
        for segment in stored["segments"]
        for item in segment["keyframe_sources"]
    ] == sorted(
        item["source_time_s"]
        for segment in stored["segments"]
        for item in segment["keyframe_sources"]
    )


def test_long_v3_freezes_every_segment_source_frame_before_postprocess(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=15.4,
        ratio="9:16",
        fit_mode="pad",
    )

    def fake_steps(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 15.4}), encoding="utf-8"
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 15.4, "scenes": [], "segments": []}),
                encoding="utf-8",
            )
        elif step.startswith("segment ") and step.endswith(" extract"):
            work = Path(argv[argv.index("--out-dir") + 1])
            work.mkdir(parents=True, exist_ok=True)
            (work / "001_frame_000.000s.png").write_bytes(_PX_PNG)
            (work / "002_frame_015.400s.png").write_bytes(_PX_PNG)
            (work / "manifest.json").write_text(
                json.dumps({
                    "frames": [
                        {"index": 1, "time_seconds": 0.0, "file": "001_frame_000.000s.png"},
                        {"index": 2, "time_seconds": 15.4, "file": "002_frame_015.400s.png"},
                    ]
                }),
                encoding="utf-8",
            )

    plan = _long_dual_target_plan_v3(frame_count=3)
    prompts = image_optimization.compile_frame_prompts(
        plan, settings.seedream_edit_mode
    )
    monkeypatch.setattr(pipeline, "_run_cmd", fake_steps)
    monkeypatch.setattr(
        pipeline,
        "_cut_segment",
        lambda _source, _start, _end, segdir: (
            segdir.mkdir(parents=True, exist_ok=True),
            (segdir / "source.mp4").write_bytes(b"segment"),
        ),
    )
    monkeypatch.setattr(voice, "extract_audio", lambda _cdir: None)
    monkeypatch.setattr(
        CodexRunner,
        "run",
        lambda self, workdir, prompt: _write_valid_package(
            Path(workdir) / "work", frames=3
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_image_optimization_project",
        lambda *_args, **_kwargs: (plan, prompts),
    )
    monkeypatch.setattr(
        pipeline,
        "_bind_keyframe_source_timeline",
        lambda _work, _segments, metas, _scenes: metas,
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["_image_continuity"]["version"] == 3
    frozen = stored["_image_optimization"]
    assert frozen["version"] == 3
    assert {
        (item["segment_index"], item["frame_name"])
        for item in frozen["frames"]
    } == {
        (segment_index, f"{frame_index:02d}.png")
        for segment_index in (1, 2)
        for frame_index in (1, 2, 3)
    }


def test_run_dialogue_auto_clips_mp3_encoder_tail_to_video_timeline(
    tmp_path, video_1s, monkeypatch
):
    """线上复现：10.080s MP3 不能把 10.000s 视频 receipt 的台词时间轴撑长。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _replace_source_with_duration(settings, meta["id"], 10.0)
    storage.update_meta(
        settings.data_dir,
        meta["id"],
        dialogue_mode="auto",
        voice_mode="keep",
        duration_s=10.0,
        ratio="9:16",
        fit_mode="none",
    )
    asr_line = {"text": "完整十秒口播。", "start_s": 0.0, "end_s": 10.08}
    normalized = {"text": "完整十秒口播。", "start_s": 0.0, "end_s": 10.0}

    def fake_steps(argv, *, timeout, step, cwd=None):
        if _run_fake_media_step(argv, step):
            return
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 10.0}), encoding="utf-8"
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            _write_exact_scene_inventory(work, Path(argv[2]), 10.0)

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(
                json.dumps([asr_line]), encoding="utf-8"
            )
        else:
            _write_valid_package(work, frames=9)

    monkeypatch.setattr(pipeline, "_run_cmd", fake_steps)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 10.08)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 10_080, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    cdir = settings.data_dir / meta["id"]
    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == [normalized]
    _assert_analysis_bound_provenance(stored["voice_line_provenance"], [
        {
            **normalized,
            "classification": "spoken",
            "provenance": "asr",
            "kept": True,
            "asr_start_s": 0.0,
            "asr_end_s": 10.08,
            "time_adjustment": "clipped_to_video_duration",
        }
    ], audio=b"normalized-audio", has_bgm=False)
    assert any("video duration 10.000s" in item for item in stored["voice_warnings"])
    receipt = json.loads((cdir / "long_video_plan.json").read_text(encoding="utf-8"))
    assert receipt["video"]["duration_s"] == 10.0
    assert receipt["segments"][0]["dialogue"]["count"] == 1
    assert stored["segments"][0]["dialogue"] == [
        {**normalized, "classification": "spoken"}
    ]


def test_run_voice_drops_lines_starting_in_mp3_only_tail(tmp_path, video_1s, monkeypatch):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    kept = {"text": "视频内台词", "start_s": 9.5, "end_s": 9.9}
    tail = {"text": "编码尾部伪句", "start_s": 10.0, "end_s": 10.08}

    def fake_steps(argv, *, timeout, step, cwd=None):
        if step == "extract":
            work = Path(argv[argv.index("--out-dir") + 1])
            (work / "contact_sheet.jpg").write_bytes(b"sheet")
            (work / "manifest.json").write_text(
                json.dumps({"duration_seconds": 10.0}), encoding="utf-8"
            )
        elif step == "scenes":
            work = Path(argv[argv.index("--work-dir") + 1])
            (work / "scenes.json").write_text(
                json.dumps({"duration_s": 10.0, "scenes": [], "segments": []}),
                encoding="utf-8",
            )

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"normalized-audio")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(
                json.dumps([kept, tail]), encoding="utf-8"
            )
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", fake_steps)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 10.08)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(
        vocal,
        "analyze",
        lambda _audio: vocal.VocalAnalysis(
            windows=[vocal.VocalWindow(0, 10_080, sung=0.0, spoken=0.3, music=0.0)],
            has_bgm=False,
        ),
    )

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == [kept]
    _assert_analysis_bound_provenance(stored["voice_line_provenance"], [
        {
            **kept,
            "classification": "spoken",
            "provenance": "asr",
            "kept": True,
        },
        {
        **tail,
        "classification": "spoken",
        "provenance": "asr",
        "kept": False,
        "drop_reason": "starts_at_or_after_video_duration",
        },
    ], audio=b"normalized-audio", has_bgm=False)
    assert any("dropped 1" in item for item in stored["voice_warnings"])


@pytest.mark.parametrize(
    "lines",
    [
        [{"text": "Jumpa geng sekelapa.", "start_s": 0.0, "end_s": 2.5}],
        [
            {
                "text": "¿Tu perro tiene nudos y demasiado pelo suelto?",
                "start_s": 0.05,
                "end_s": 3.35,
            },
            {
                "text": "Este peine de doble diente es la solución.",
                "start_s": 3.84,
                "end_s": 6.65,
            },
            {
                "text": "Desenreda suavemente y sin tirones.",
                "start_s": 7.09,
                "end_s": 9.32,
            },
        ],
        [
            {
                "text": "Finally found the cheapest cheese squishy!",
                "start_s": 0.2,
                "end_s": 2.0,
            }
        ],
    ],
)
def test_voice_timeline_normalization_preserves_11_12_13_text_and_order(lines):
    decisions = [
        {
            **line,
            "classification": "spoken" if index % 2 == 0 else "sung",
            "provenance": "asr",
            "kept": True,
        }
        for index, line in enumerate(lines)
    ]

    normalized, normalized_decisions, warnings = pipeline._normalize_voice_timeline(
        decisions, 10.0
    )

    assert normalized == lines
    assert normalized_decisions == decisions
    assert warnings == []


def test_single_voice_line_uses_strong_track_evidence_when_asr_timestamp_misses():
    """temp/11 形态：唯一台词文本正确，但 ASR 区间早于真实口播，不能被误删。"""
    line = {"text": "Jumpa geng sekelapa.", "start_s": 0.0, "end_s": 2.5}
    analysis = vocal.VocalAnalysis(
        windows=[
            vocal.VocalWindow(0, 975, sung=0.0, spoken=0.015625, music=0.0),
            vocal.VocalWindow(4875, 5850, sung=0.0, spoken=0.33203125, music=0.0),
        ],
        has_bgm=False,
    )
    bgm_only = vocal.VocalAnalysis(
        windows=[
            vocal.VocalWindow(0, 975, sung=0.0, spoken=0.05859375, music=0.2),
        ],
        has_bgm=True,
    )

    assert pipeline._classify_voice_line(line, analysis, only_line=True) == "spoken"
    assert pipeline._classify_voice_line(line, bgm_only, only_line=True) is None


def test_voice_prompt_requires_multilingual_transcript_and_forbids_placeholders(tmp_path):
    prompt = pipeline._voice_prompt(tmp_path, "keep", "", 10.08)
    assert "自动识别实际语言" in prompt
    assert "[无法辨识]" in prompt
    assert "[inaudible]" in prompt
    assert "输出空数组" in prompt


def test_run_voice_placeholder_retries_then_continues_without_fake_dialogue(
    tmp_path, video_1s, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    calls = {"voice": 0}

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            calls["voice"] += 1
            (work / "voice_lines.json").write_text(
                json.dumps([
                    {"text": "[无法辨识]", "start_s": 0.0, "end_s": 0.9}
                ]),
                encoding="utf-8",
            )
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(voice, "probe_audio_duration", lambda _path: 1.0)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _spoken_analysis(True))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert calls["voice"] == 3
    assert stored["voice_lines"] == []
    assert "[无法辨识]" not in stored["prompt"]
    assert any("占位符" in warning for warning in stored["voice_warnings"])


def test_run_voice_codex_timeout_salvages_complete_lines(tmp_path, video_1s, monkeypatch):
    """ASR 的 codex 超时被杀但 voice_lines.json 已完整 → 收养，继续 video-maker。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    calls = {"asr": 0, "maker": 0}

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            calls["asr"] += 1
            (work / "voice_lines.json").write_text(json.dumps(VOICE_LINES), encoding="utf-8")
            raise CodexError("codex timed out after 600s")
        calls["maker"] += 1
        _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "done" and m["error"] is None
    assert m["voice_lines"] == VOICE_LINES
    assert calls == {"asr": 1, "maker": 1}


def test_run_voice_codex_failure_no_product(tmp_path, video_1s, monkeypatch):
    """ASR 失败且无完整产物 → 报原始 CodexError。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def bad_codex(self, workdir, prompt):
        if "voice.mp3" in prompt:  # 只有 ASR 调用失败；video-maker 正常则验证断言只针对 ASR
            raise CodexError("codex timed out after 600s")
        _write_valid_package(Path(workdir) / "work")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", bad_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert "timed out" in m["error"]


def test_run_voice_validation_failure(tmp_path, video_1s, monkeypatch):
    """ASR 返回 0 但产物非法 → failed，错误归因于 Codex 输出阶段。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def bad_codex(self, workdir, prompt):
        (Path(workdir) / "work" / "voice_lines.json").write_bytes(b"not json")

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", bad_codex)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert m["error"] == (
        "codex voice output invalid: required voice_lines artifact "
        "is missing or invalid"
    )


def test_run_voice_missing_output_reports_codex_stage(tmp_path, video_1s, monkeypatch):
    """ASR 返回 0 但没写产物时，不把底层文件缺失误报成输入视频问题。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", lambda self, workdir, prompt: None)

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    m = storage.load_meta(settings.data_dir, meta["id"])
    assert m["status"] == "failed"
    assert m["error"].startswith("codex voice output invalid:")
    assert "voice_lines.json missing" not in m["error"]


# ---------- HTTP 接线 ----------


@pytest.mark.parametrize("partial_prompt", [False, True])
def test_startup_recovers_stale_unfrozen_pipeline_claim_without_h3(
    tmp_path, monkeypatch, partial_prompt,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    meta = storage.new_conversation(settings.data_dir, "n", "a.mp4")
    if partial_prompt:
        (settings.data_dir / meta["id"] / "work" / "prompt.txt").write_text(
            "partial, not frozen", encoding="utf-8"
        )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_pipeline_input(settings.data_dir, meta["id"])
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    called = threading.Event()

    def fake_run(_settings, cid, _runner, **_kwargs):
        assert cid == meta["id"]
        owner = _kwargs["claimed_owner"]
        assert owner["process_generation"] == "boot-new"
        assert storage.load_pipeline_claim(settings.data_dir, cid, owner)
        assert storage.finish_input_claim(
            settings.data_dir, cid, owner, status="failed", error="recovered-test"
        )
        called.set()

    monkeypatch.setattr(pipeline, "run", fake_run)
    with TestClient(create_app(settings)):
        assert called.wait(timeout=1)


def test_parallel_pipeline_continuations_share_application_event_loop(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path,
        enable_pipeline=True,
        enable_h3_submit=False,
        codex_concurrency=3,
        seedream_concurrency=1,
    )
    metas = [
        storage.new_conversation(settings.data_dir, f"n-{index}", "a.mp4")
        for index in range(3)
    ]
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert all(
        storage.claim_pipeline_input(settings.data_dir, meta["id"])
        for meta in metas
    )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")

    def fake_pipeline_run(_settings, cid, _runner, **kwargs):
        owner = kwargs["claimed_owner"]
        assert storage.finish_input_claim(
            settings.data_dir,
            cid,
            owner,
            status="done",
            error=None,
            postprocess={"status": "running", "segments": []},
        )

    loop_ids = []
    completed = threading.Event()
    completed_count = 0
    completed_lock = threading.Lock()

    async def fake_postprocess_run_task(
        _settings, _cid, _mediakit_sem, seedream_sem, **_kwargs,
    ):
        nonlocal completed_count
        loop_ids.append(id(asyncio.get_running_loop()))
        async with seedream_sem:
            await asyncio.sleep(0.03)
        with completed_lock:
            completed_count += 1
            if completed_count == len(metas):
                completed.set()

    monkeypatch.setattr(pipeline, "run", fake_pipeline_run)
    monkeypatch.setattr(long_generation, "plan_receipt", lambda *_args: {})
    monkeypatch.setattr(postprocess, "run_task", fake_postprocess_run_task)

    with TestClient(create_app(settings)):
        assert completed.wait(timeout=3)

    assert len(loop_ids) == len(metas)
    assert len(set(loop_ids)) == 1


@pytest.mark.parametrize("frozen", ["generation", "receipt", "plan", "fit"])
def test_startup_does_not_resume_stale_pipeline_after_input_freezes(
    tmp_path, monkeypatch, frozen,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    meta = storage.new_conversation(settings.data_dir, "n", "a.mp4")
    cdir = settings.data_dir / meta["id"]
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_pipeline_input(settings.data_dir, meta["id"])
    if frozen == "generation":
        storage.update_meta(
            settings.data_dir, meta["id"], generation={"status": "queued"}
        )
    elif frozen == "receipt":
        (cdir / "prepared_input.json").write_bytes(b"frozen-receipt")
    elif frozen == "plan":
        (cdir / "long_video_plan.json").write_bytes(b"frozen-plan")
    else:
        fitted = cdir / "work" / "h3_frames" / "crop" / "01.png"
        fitted.parent.mkdir(parents=True)
        fitted.write_bytes(b"frozen-fit")
    before = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    called = []
    monkeypatch.setattr(pipeline, "run", lambda *_args, **_kwargs: called.append(1))

    with TestClient(create_app(settings)):
        pass

    after = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file()
    }
    assert called == []
    if frozen == "generation":
        assert after == before
    else:
        assert {key: value for key, value in after.items() if key != "meta.json"} == {
            key: value for key, value in before.items() if key != "meta.json"
        }
        recovered = storage.load_meta(settings.data_dir, meta["id"])
        assert recovered["status"] == "failed"
        assert recovered["error"] == "input_recovery_required"


def test_startup_does_not_duplicate_current_process_pipeline_claim(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    meta = storage.new_conversation(settings.data_dir, "n", "a.mp4")
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-current")
    assert storage.claim_pipeline_input(settings.data_dir, meta["id"])
    called = []
    monkeypatch.setattr(pipeline, "run", lambda *_args, **_kwargs: called.append(1))

    with TestClient(create_app(settings)):
        pass

    assert called == []


def _wait_for_pipeline_terminal(settings, cid):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        meta = storage.load_meta(settings.data_dir, cid)
        if meta["status"] in {"done", "failed"}:
            return meta
        time.sleep(0.01)
    pytest.fail("pipeline recovery did not reach a terminal status")


def test_startup_pipeline_recovery_replaces_stale_short_scripts(
    tmp_path, video_1s, fake_steps, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    meta = _make_conversation(settings, video_1s)
    cdir = settings.data_dir / meta["id"]
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_pipeline_input(settings.data_dir, meta["id"])
    scripts = cdir / "scripts"
    scripts.mkdir()
    (scripts / "untrusted.py").write_text("raise SystemExit", encoding="utf-8")
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")

    with TestClient(create_app(settings)):
        terminal = _wait_for_pipeline_terminal(settings, meta["id"])

    assert terminal["status"] == "done", terminal.get("error")
    assert not (scripts / "untrusted.py").exists()
    assert (scripts / "crop_image.py").read_bytes() == CROP_SCRIPT.read_bytes()


def test_startup_pipeline_recovery_replaces_stale_segment_scripts(
    tmp_path, video_1s, fake_steps, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    meta = _make_conversation(settings, video_1s)
    cdir = settings.data_dir / meta["id"]
    segment = {"index": 1, "start_s": 0.0, "end_s": 1.0}
    monkeypatch.setattr(pipeline, "_detect_segments", lambda *_args: [segment])

    def fake_cut(source, _start_s, _end_s, segdir):
        segdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, segdir / "source.mp4")

    monkeypatch.setattr(pipeline, "_cut_segment", fake_cut)
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_pipeline_input(settings.data_dir, meta["id"])
    scripts = cdir / "work" / "segments" / "1" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "untrusted.py").write_text("raise SystemExit", encoding="utf-8")
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")

    with TestClient(create_app(settings)):
        terminal = _wait_for_pipeline_terminal(settings, meta["id"])

    assert terminal["status"] == "done", terminal.get("error")
    assert not (scripts / "untrusted.py").exists()
    assert (scripts / "crop_image.py").read_bytes() == CROP_SCRIPT.read_bytes()


@pytest.mark.parametrize("meta_pointer", [False, True])
def test_startup_reconciles_half_committed_prepared_receipt_without_rewrite(
    tmp_path, video_1s, monkeypatch, meta_pointer,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    meta = _make_conversation(settings, video_1s)
    cdir = settings.data_dir / meta["id"]
    work = cdir / "work"
    names = _write_valid_package(work)
    visual = work / "visual_prompt.txt"
    visual.write_text(PROMPT_TEXT, encoding="utf-8")
    storage.update_meta(
        settings.data_dir, meta["id"], duration_s=1.0,
        dialogue_mode="auto", voice_mode="keep", voice_lines=[],
    )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_pipeline_input(settings.data_dir, meta["id"])
    frozen = prepared_input.write_prepared_input(
        root=cdir,
        source=cdir / "source.mp4",
        audio=None,
        keyframes=[work / "keyframes" / name for name in names],
        visual=visual,
        final=work / "prompt.txt",
        dialogue_mode="auto",
        dialogue=[],
        vocal_filter_enabled=True,
        duration_s=1.0,
        ratio="9:16",
        fit_mode="none",
        engine_request={"h3": {"workflow": pipeline.H3_ENGINE_WORKFLOW,
                               "duration": 1, "resolution": h3.H3_RESOLUTION}},
    )
    if meta_pointer:
        storage.update_meta(
            settings.data_dir, meta["id"],
            prompt=frozen.prompt_text,
            prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
        )
    before = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    }
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    monkeypatch.setattr(
        pipeline, "run", lambda *_args, **_kwargs: pytest.fail("must not rerun")
    )

    with TestClient(create_app(settings)):
        pass

    recovered = storage.load_meta(settings.data_dir, meta["id"])
    after = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    }
    assert recovered["status"] == "done"
    assert recovered["prepared_input_receipt"] == prepared_input.RECEIPT_FILENAME
    assert after == before


@pytest.mark.parametrize("meta_pointer", [False, True])
def test_startup_reconciles_half_committed_long_plan_without_rewrite(
    tmp_path, video_1s, monkeypatch, meta_pointer,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    meta = _make_conversation(settings, video_1s)
    cdir = settings.data_dir / meta["id"]
    segdir = cdir / "work" / "segments" / "1"
    segwork = segdir / "work"
    key = segwork / "keyframes" / "01.png"
    first = segwork / "anchors" / "first.png"
    last = segwork / "anchors" / "last.png"
    for path in (key, first, last):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_PX_PNG)
    (segdir / "source.mp4").write_bytes(b"segment")
    visual = segwork / "visual_prompt.txt"
    visual_text = "产品保持在画面中央。"
    visual.write_text(visual_text, encoding="utf-8")
    final = segwork / "prompt.txt"
    final.write_text(
        pipeline.NO_BGM_LINE + "\n" + prepared_input.compose_final_prompt(
            long_video.compose_segment_visual_prompt(visual_text), []
        ),
        encoding="utf-8",
    )
    (segwork / "voice_lines.json").write_text("[]", encoding="utf-8")
    duration = 14.000000000000004
    public = {
        "index": 1, "start_s": 0.0, "end_s": duration,
        "chain_id": "chain-001", "join_mode": "hard_cut",
        "source": "segments/1/source.mp4", "keyframes": ["01.png"],
        "keyframe_paths": ["segments/1/work/keyframes/01.png"],
        "first_frame_path": "segments/1/work/anchors/first.png",
        "last_frame_path": "segments/1/work/anchors/last.png",
        "visual_prompt": visual_text, "prompt": final.read_text(encoding="utf-8"),
        "dialogue": [], "lines": [],
    }
    storage.update_meta(
        settings.data_dir, meta["id"], duration_s=duration,
        dialogue_mode="auto", voice_mode="keep", voice_lines=[],
    )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_pipeline_input(settings.data_dir, meta["id"])
    with monkeypatch.context() as historical:
        historical.setattr(long_video, "SHORT_VIDEO_MAX_S", 14.0)
        historical.setattr(long_video, "SEGMENT_PROVIDER_MAX_DURATION_S", 14)
        receipt = long_video.write_plan_receipt(
            cdir, source=cdir / "source.mp4", duration_s=duration,
            segments=[{
                **public, "source_path": segdir / "source.mp4",
                "keyframe_paths": [key], "first_frame_path": first,
                "last_frame_path": last, "visual_prompt_path": visual,
                "final_prompt_path": final,
            }],
            workflow=pipeline.H3_BOUNDARY_WORKFLOW,
        )
    if meta_pointer:
        storage.update_meta(
            settings.data_dir, meta["id"], segments=[public],
            long_video_plan_receipt=receipt.name,
        )
    before = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    }
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    monkeypatch.setattr(
        pipeline, "run", lambda *_args, **_kwargs: pytest.fail("must not rerun")
    )

    with TestClient(create_app(settings)):
        pass

    recovered = storage.load_meta(settings.data_dir, meta["id"])
    after = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    }
    assert recovered["status"] == "done", recovered.get("error")
    assert recovered["long_video_plan_receipt"] == receipt.name
    assert recovered["fit_required"] is True
    assert after == before


def test_startup_marks_corrupt_half_frozen_pipeline_as_recovery_required(
    tmp_path, video_1s, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_pipeline=True, enable_h3_submit=False
    )
    meta = _make_conversation(settings, video_1s)
    cdir = settings.data_dir / meta["id"]
    storage.update_meta(
        settings.data_dir, meta["id"], duration_s=1.0,
        dialogue_mode="auto", voice_mode="keep",
    )
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_pipeline_input(settings.data_dir, meta["id"])
    receipt = cdir / prepared_input.RECEIPT_FILENAME
    receipt.write_bytes(b"not-json")
    before = receipt.read_bytes()
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    monkeypatch.setattr(
        pipeline, "run", lambda *_args, **_kwargs: pytest.fail("must not rerun")
    )

    with TestClient(create_app(settings)):
        pass

    recovered = storage.load_meta(settings.data_dir, meta["id"])
    assert recovered["status"] == "failed"
    assert recovered["error"] == "input_recovery_required"
    assert recovered["_input_owner"] is None
    assert receipt.read_bytes() == before


def test_post_triggers_pipeline_and_detail_filled(tmp_path, video_1s, fake_steps):
    settings = make_settings(tmp_path, enable_pipeline=True)
    with TestClient(create_app(settings)) as c:
        with open(video_1s, "rb") as f:
            r = c.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", f, "video/mp4")},
            )
        assert r.status_code == 201
        cid = r.json()["id"]
        r = c.get(f"/api/conversations/{cid}", headers=AUTH)
    body = r.json()
    assert body["status"] == "done", body.get("error")
    assert body["segments"][0]["keyframes"] == [
        f"{index:02d}.png" for index in range(1, 10)
    ]
    assert body["segments"][0]["visual_prompt"] == PROMPT_TEXT
    assert body["aspect_ratio"] == "16:9"
    assert body["resolution"] == "480p"
    assert body["fit_profiles"] == {
        "16:9": {"fit_required": True, "default_fit_mode": "crop"},
        "9:16": {"fit_required": True, "default_fit_mode": "crop"},
    }
    assert "has_preview" not in body
    assert body["error"] is None


def test_done_fit_requirement_uses_actual_keyframes_not_source_dimensions(
    tmp_path, video_1s, fake_steps, monkeypatch
):
    settings = make_settings(tmp_path, enable_pipeline=True)
    monkeypatch.setattr(storage, "probe_video", lambda *_args: storage.VideoProbe(1.0, 90, 160))

    with TestClient(create_app(settings)) as client:
        with open(video_1s, "rb") as file:
            created = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", file, "video/mp4")},
            )
        detail = client.get(
            f"/api/conversations/{created.json()['id']}", headers=AUTH
        ).json()

    assert detail["status"] == "done", detail.get("error")
    assert detail["fit_required"] is True  # fake Codex 产出 1x1 关键帧
    assert detail["aspect_ratio"] == "9:16"
    assert detail["resolution"] == "480p"


def test_pipeline_off_by_default(client, video_1s, monkeypatch):
    """Settings 直建（旧测试路径）默认不触发流水线，保持 queued。"""
    called = []
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: called.append(1))
    with open(video_1s, "rb") as f:
        r = client.post(
            "/api/conversations", headers=AUTH, files={"file": ("clip.mp4", f, "video/mp4")}
        )
    assert r.status_code == 201
    assert called == []
    r = client.get(f"/api/conversations/{r.json()['id']}", headers=AUTH)
    assert r.json()["status"] == "queued"


# ---------- config 新字段 ----------


def test_config_pipeline_fields(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("ACCESS_TOKEN", "t")
    monkeypatch.setenv("CODEX_TIMEOUT_S", "42")
    monkeypatch.setenv("CODEX_CONCURRENCY", "3")
    monkeypatch.setenv("MAX_QUEUED", "7")
    monkeypatch.setenv("ENABLE_PIPELINE", "0")
    s = get_settings()
    assert s.codex_timeout_s == 42
    assert s.codex_concurrency == 3
    assert s.max_queued == 7
    assert s.enable_pipeline is False

    monkeypatch.delenv("CODEX_TIMEOUT_S")
    monkeypatch.delenv("CODEX_CONCURRENCY")
    monkeypatch.delenv("MAX_QUEUED")
    monkeypatch.delenv("ENABLE_PIPELINE")
    s = get_settings()
    assert s.codex_timeout_s == 1800
    assert s.codex_concurrency == 10
    assert s.max_queued == 100
    assert s.enable_pipeline is True  # 生产路径默认开


# ---------- storage.update_meta ----------


def test_update_meta(tmp_path):
    meta = storage.new_conversation(tmp_path, "", "a.mp4")
    updated = storage.update_meta(tmp_path, meta["id"], status="done", keyframes=["k.png"])
    assert updated["status"] == "done" and updated["keyframes"] == ["k.png"]
    assert updated["updated_at"] >= meta["updated_at"]
    assert storage.load_meta(tmp_path, meta["id"])["status"] == "done"
    assert storage.update_meta(tmp_path, "0" * 32, status="x") is None
    assert storage.update_meta(tmp_path, "..", status="x") is None


# ---------- 假 codex 桩：全编排真实子进程 e2e（无 mock） ----------


def _write_stub_codex(bin_dir: Path, frames: int) -> Path:
    """生成一个按新契约直产合法产物的假 codex：从 work/ 抽好的帧里挑 frames 张复制进 keyframes/。"""
    stub = bin_dir / "codex"
    stub.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            f"""\
            import shutil, sys
            from pathlib import Path

            argv = sys.argv[1:]
            workdir = Path(argv[argv.index("-C") + 1])
            out = Path(argv[argv.index("-o") + 1])
            work = workdir / "work"
            kdir = work / "keyframes"
            kdir.mkdir(exist_ok=True)
            frames = sorted(work.glob("*_frame_*.png"))[:{frames}]
            assert frames, "no extracted frames in work/"
            for i, src in enumerate(frames, start=1):
                shutil.copy(src, kdir / f"{{i:02d}}.png")
            (work / "prompt.txt").write_text({PROMPT_TEXT!r} + "（桩产物）", encoding="utf-8")
            out.write_text("stub done", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def _write_stub_codex_voice(bin_dir: Path, frames: int) -> Path:
    """桩 codex 兼处理两种调用：ASR 调用写 voice_lines.json，video-maker 调用挑帧写 prompt。"""
    stub = bin_dir / "codex"
    stub.write_text(
        "#!/usr/bin/python3\n"
        + textwrap.dedent(
            f"""\
            import json, shutil, sys
            from pathlib import Path

            argv = sys.argv[1:]
            workdir = Path(argv[argv.index("-C") + 1])
            out = Path(argv[argv.index("-o") + 1])
            work = workdir / "work"
            if "voice.mp3" in argv[-1]:
                (work / "voice_lines.json").write_text(
                    json.dumps([{{"text": "你好，世界。", "start_s": 0.0, "end_s": 1.0}}]),
                    encoding="utf-8",
                )
                out.write_text("asr done", encoding="utf-8")
                raise SystemExit(0)
            kdir = work / "keyframes"
            kdir.mkdir(exist_ok=True)
            frames = sorted(work.glob("*_frame_*.png"))[:{frames}]
            assert frames, "no extracted frames in work/"
            for i, src in enumerate(frames, start=1):
                shutil.copy(src, kdir / f"{{i:02d}}.png")
            (work / "prompt.txt").write_text({PROMPT_TEXT!r} + "（桩产物）", encoding="utf-8")
            out.write_text("stub done", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


@pytest.fixture
def voice_stub_bin():
    """外层 voice bwrap 会隐藏 /tmp 和仓库，桩程序必须放在仍可见的位置。"""
    with tempfile.TemporaryDirectory(prefix="duet-codex-stub-", dir="/var/tmp") as raw:
        yield Path(raw)


def test_full_pipeline_voice_with_stub_codex(tmp_path, monkeypatch, voice_stub_bin):
    """真 subprocess 全链路含口播：extract → ffmpeg 抽音轨 → 桩 codex ASR → 桩 codex 选帧 → done。"""
    video = tmp_path / "talk.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-shortest", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True, capture_output=True,
    )
    bin_dir = voice_stub_bin
    _write_stub_codex_voice(bin_dir, 3)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video)
    _set_voice_mode(settings, meta, "keep")
    cid = meta["id"]
    cdir = settings.data_dir / cid

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert m["voice_lines"] == [{"text": "你好，世界。", "start_s": 0.0, "end_s": 1.0}]
    assert (cdir / "work" / "voice.mp3").is_file()
    assert (cdir / "work" / "voice_lines.json").is_file()
    assert m["keyframes"] == ["01.png", "02.png", "03.png"]


def test_full_pipeline_with_stub_codex(tmp_path, video_1s, monkeypatch):
    """真 subprocess 全链路：extract --fps 4 → 桩 codex → 校验 → done（不再生成 preview）。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_codex(bin_dir, 3)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    cid = meta["id"]
    cdir = settings.data_dir / cid

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert m["keyframes"] == ["01.png", "02.png", "03.png"]
    assert "15 秒" in m["prompt"]
    assert (cdir / "codex_last_message.txt").is_file()
    assert (cdir / "work" / "contact_sheet.jpg").is_file()  # 1s×4fps=5 帧，单页联系表
    assert (cdir / "work" / "manifest.json").is_file()
    assert (cdir / "scripts" / "crop_image.py").is_file()  # skill 脚本已拷进会话目录
    assert not (cdir / "preview.mp4").exists()


def test_full_pipeline_relative_data_dir(tmp_path, video_1s, monkeypatch):
    """回归：DATA_DIR 为相对路径（生产默认 "data"）时流水线也必须成功。

    子进程带 cwd 时相对 data_dir 会错位，run() 入口须先把会话目录解析为绝对路径。
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_codex(bin_dir, 3)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)

    settings = make_settings(tmp_path, data_dir=Path("data"))
    meta = _make_conversation(settings, video_1s)
    cid = meta["id"]

    pipeline.run(settings, cid, CodexRunner(settings.codex_timeout_s, settings.codex_concurrency))

    m = storage.load_meta(settings.data_dir, cid)
    assert m["status"] == "done", m["error"]
    assert m["keyframes"] == ["01.png", "02.png", "03.png"]
    assert (tmp_path / "data" / cid / "work" / "prompt.txt").is_file()


def _spoken_analysis(spoken: bool) -> vocal.VocalAnalysis:
    """声学预判桩：spoken=True 表示音轨含人声（12 窗口 spoken≥0.2 的形态）。"""
    return vocal.VocalAnalysis(
        windows=[
            vocal.VocalWindow(0, 30_000, sung=0.0, spoken=0.3 if spoken else 0.01, music=0.0),
        ],
        has_bgm=False,
    )


def _sung_analysis(sung: float) -> vocal.VocalAnalysis:
    """声学预判桩：纯唱/吟唱音轨（spoken≈0，sung 给定为强或弱）。"""
    return vocal.VocalAnalysis(
        windows=[
            vocal.VocalWindow(0, 30_000, sung=sung, spoken=0.01, music=0.8),
        ],
        has_bgm=True,
    )


def test_run_voice_empty_lines_with_sung_retries_then_warns(tmp_path, video_1s, monkeypatch):
    """真实量化边界 51/256 算人声；三次空则明确警告并按用户确认继续无台词。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([]), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _sung_analysis(0.19921875))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []
    assert stored["voice_warnings"] == [
        "voice_lines.json empty after automatic retries despite vocal evidence; continuing without dialogue"
    ]


def test_run_voice_empty_lines_with_weak_sung_passes(tmp_path, video_1s, monkeypatch):
    """弱 sung（纯 BGM 里蹭出的 <0.2 唱分）+ 听写为空：仍属合法「无台词」。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([]), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _sung_analysis(0.059))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []


def test_run_voice_empty_lines_with_spoken_retries_and_succeeds(tmp_path, video_1s, monkeypatch):
    """音轨有人声但 codex 第一次输出空数组（随机摆烂）：重试一次听出台词 → done。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")
    line = {"text": "台词", "start_s": 0.0, "end_s": 0.9}
    codex_calls = []
    voice_stages = []

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        codex_calls.append(prompt)
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            visible_before = {
                path.relative_to(Path(workdir)).as_posix()
                for path in Path(workdir).rglob("*")
                if path.is_file()
            }
            assert visible_before == {"work/voice.mp3", "work/manifest.json"}
            voice_stages.append(Path(workdir))
            (work / "voice_lines.json").write_text(
                json.dumps([] if len(codex_calls) == 1 else [line]), encoding="utf-8"
            )
            (work / "stale-from-attempt.txt").write_text("must not survive", encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _spoken_analysis(True))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == [line]
    # voice 调用 = 第一次听写 + 重试共 2 次（prompt 步另有 1 次非 voice 调用）
    assert sum(1 for p in codex_calls if "voice.mp3" in p) == 2
    assert len(voice_stages) == 2 and voice_stages[0] != voice_stages[1]


def test_run_voice_empty_lines_with_spoken_retry_still_empty_warns(tmp_path, video_1s, monkeypatch):
    """音轨有人声、重试后仍空：明确 warning 后继续，且不会伪造台词。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([]), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _spoken_analysis(True))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []
    assert stored["voice_warnings"]


def test_run_voice_empty_lines_without_spoken_passes(tmp_path, video_1s, monkeypatch):
    """音轨无人声（纯 BGM/静音）且听写为空：合法「无台词」，done 且 voice_lines=[]。"""
    settings = make_settings(tmp_path)
    meta = _make_conversation(settings, video_1s)
    _set_voice_mode(settings, meta, "keep")

    def fake_extract_audio(cdir_arg):
        out = cdir_arg / "work" / "voice.mp3"
        out.write_bytes(b"mp3-bytes")
        return out

    def fake_codex(self, workdir, prompt):
        work = Path(workdir) / "work"
        if "voice.mp3" in prompt:
            (work / "voice_lines.json").write_text(json.dumps([]), encoding="utf-8")
        else:
            _write_valid_package(work)

    monkeypatch.setattr(pipeline, "_run_cmd", _fake_extract_ok)
    monkeypatch.setattr(voice, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(CodexRunner, "run", fake_codex)
    monkeypatch.setattr(vocal, "analyze", lambda _a: _spoken_analysis(False))

    pipeline.run(settings, meta["id"], CodexRunner(1, 1))

    stored = storage.load_meta(settings.data_dir, meta["id"])
    assert stored["status"] == "done", stored.get("error")
    assert stored["voice_lines"] == []
