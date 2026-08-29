"""后处理编排：HTTP 门控、MediaKit 场景映射、失败保留和并发限流。"""

import asyncio
import base64
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings

from app import (
    context_ir_bridge,
    h3,
    image_optimization,
    long_generation,
    long_video,
    mediakit,
    pipeline,
    postprocess,
    prepared_input,
    stitch,
    storage,
)
from app import main as main_module
from app.codex_runner import CodexRunner
from app.main import create_app

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

OPTIONS_SUB = {"remove_subtitle": True, "remove_brand": False}
OPTIONS_BRAND = {"remove_subtitle": False, "remove_brand": True}


def _solid_png(path: Path, bgr: tuple[int, int, int], *, local_bgr=None) -> None:
    image = np.full((100, 100, 3), bgr, dtype=np.uint8)
    if local_bgr is not None:
        image[:5, :5] = local_bgr
    assert cv2.imwrite(str(path), image)


@pytest.fixture
def enabled(tmp_path):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    with TestClient(create_app(settings)) as c:
        yield settings, c


def _make_conv(
    settings, status="done", segments=False, frame_count=2,
    segment_frame_count=None, segment_total=2,
):
    """造会话：单段 = work/keyframes + work/prompt.txt；多段 = meta.segments + 段目录产物。"""
    meta = storage.new_conversation(settings.data_dir, "n", "a.mp4")
    cid = meta["id"]
    cdir = settings.data_dir / cid
    if segments:
        counts = (
            (segment_frame_count, segment_frame_count)
            if segment_frame_count is not None else (1, 2)
        )
        if segment_total not in {1, 2}:
            raise ValueError("test segment_total must be 1 or 2")
        segs = [{
            "index": index,
            "start_s": 8.0 * (index - 1),
            "end_s": 8.0 * index,
            "keyframes": [
                f"{i:02d}.png" for i in range(1, counts[index - 1] + 1)
            ],
            "prompt": f"段{index}提示词",
            "lines": ["台词。"] if index == 1 else [],
        } for index in range(1, segment_total + 1)]
        meta["segments"] = segs
        for seg in segs:
            segdir = cdir / "work" / "segments" / str(seg["index"])
            (segdir / "work" / "keyframes").mkdir(parents=True)
            for name in seg["keyframes"]:
                (segdir / "work" / "keyframes" / name).write_bytes(PNG)
            (segdir / "work" / "prompt.txt").write_text(seg["prompt"], encoding="utf-8")
    else:
        (cdir / "work" / "keyframes").mkdir(parents=True)
        for i in range(1, frame_count + 1):
            (cdir / "work" / "keyframes" / f"{i:02d}.png").write_bytes(PNG)
        (cdir / "work" / "prompt.txt").write_text("单段提示词", encoding="utf-8")
        meta["keyframes"] = [f"{i:02d}.png" for i in range(1, frame_count + 1)]
        meta["prompt"] = "单段提示词"
    meta["status"] = status
    prompts = (
        {seg["index"]: f"第 {seg['index']} 段 Codex 图片优化提示词" for seg in meta["segments"]}
        if segments else {0: "当前视频 Codex 图片优化提示词"}
    )
    meta.update(image_optimization.freeze_prompts(settings, meta, prompts))
    (cdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cid


def _completed_v4_project(
    settings, monkeypatch, *, frame_count=2, segments=False,
    postprocess_options=None, segment_total=2,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    cid = _make_conv(
        settings,
        segments=segments,
        frame_count=frame_count,
        segment_frame_count=frame_count if segments else None,
        segment_total=segment_total,
    )
    cdir = settings.data_dir / cid
    grouped_frames = (
        {
            index: sorted(
                (cdir / "work" / "segments" / str(index) / "work" / "keyframes").glob("*.png")
            )
            for index in range(1, segment_total + 1)
        }
        if segments else {
            0: sorted((cdir / "work" / "keyframes").glob("*.png"))
        }
    )
    frames = [frame for index in sorted(grouped_frames) for frame in grouped_frames[index]]
    skeletons = {
        segment_index: [{
            "segment_index": segment_index,
            "frame_index": index,
            "frame_name": frame.name,
            "source_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            "source_transition_from_previous": (
                "start"
                if index == 1 and segment_index in {0, 1}
                else "same_camera"
            ),
            "source_transition_evidence_sha256": (
                (
                    "a"
                    if index == 1 and segment_index in {0, 1}
                    else "b"
                ) * 64
            ),
        } for index, frame in enumerate(segment_frames, 1)]
        for segment_index, segment_frames in grouped_frames.items()
    }
    segment_specs = [{
            "index": segment_index,
            "chain_id": (
                "short-000" if segment_index == 0 else "chain-001"
            ),
            "join_mode": "hard_cut" if segment_index in {0, 1} else "continue",
            "keyframes_dir": segment_frames[0].parent,
            "transition_skeleton": skeletons[segment_index],
        } for segment_index, segment_frames in grouped_frames.items()]
    slots = image_optimization.semantic_slot_manifest(segment_specs)
    semantic = {
        "people": {"subject": {
            "source_identity": "当前可见源人物",
            "replacement_identity": "明显不同且跨帧稳定的新人物",
            "wardrobe_change": "不同款式且保持用途的服装",
            "local_color_change": "人物局部固有色明显变化",
        }},
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
            for slot in slots["scenes"]
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
            }
            for slot in slots["frames"]
        },
    }
    plan, diagnostics = image_optimization.compile_semantic_plan(
        semantic,
        segment_specs,
        source_frames=grouped_frames,
    )
    assert diagnostics["score"] == 1.0
    assert "blocking" not in diagnostics
    prompts = image_optimization.compile_frame_prompts(
        plan, settings.seedream_edit_mode,
    )
    execution = image_optimization.freeze_execution_inputs(
        plan,
        revision=1,
        profile={"id": "dual-target", "revision": 4},
        model=settings.seedream_model,
        frame_inventory=[
            item
            for segment_index in sorted(skeletons)
            for item in skeletons[segment_index]
        ],
    )
    frozen = image_optimization.freeze_frame_prompts(
        settings, execution, prompts, plan=plan,
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        **image_optimization.freeze_continuity(
            plan,
            frame_counts={
                index: len(segment_frames)
                for index, segment_frames in grouped_frames.items()
            },
        ),
        **frozen,
    )

    async def edit(_settings, images, _prompt, output, *, receipt_path):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.seedream, "edit", edit)
    options = postprocess_options or {
        "remove_subtitle": False,
        "remove_brand": False,
        "optimize_image": True,
    }
    if options.get("remove_subtitle") or options.get("remove_brand"):
        monkeypatch.setattr(
            postprocess.mediakit, "erase_image", FakeEdit(receipts=True)
        )
    asyncio.run(postprocess.start(
        settings,
        cid,
        {"confirm": True, "options": options},
        {},
    ))
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1),
    ))
    latest = storage.load_meta(settings.data_dir, cid)
    assert latest["postprocess"]["status"] == "done"
    return cid, cdir, frames


def _write_root_timeline_authority(
    cdir, originals, *, duration_s, authority_work=None,
):
    authority_work = authority_work or cdir / "work"
    times = [round(1.5 * index, 3) for index in range(9)]
    samples = []
    for index, (source, time_s) in enumerate(zip(originals, times), 1):
        name = f"source-frame-{index:02d}.png"
        (authority_work / name).write_bytes(source.read_bytes())
        samples.append({"index": index, "time_seconds": time_s, "file": name})
    source = cdir / "source.mp4"
    (authority_work / "manifest.json").write_text(json.dumps({
        "source": str(source),
        "file_size_bytes": source.stat().st_size,
        "duration_seconds": duration_s,
        "frames": samples,
    }), encoding="utf-8")
    split = 7.0
    (authority_work / "scenes.json").write_text(json.dumps({
        "duration_s": duration_s,
        "scenes": [
            {
                "index": 1, "start_s": 0.0, "end_s": split,
                "frames": [item["file"] for item in samples
                           if item["time_seconds"] < split],
            },
            {
                "index": 2, "start_s": split, "end_s": duration_s,
                "frames": [item["file"] for item in samples
                           if item["time_seconds"] >= split],
            },
        ],
        "segments": [],
    }), encoding="utf-8")


def test_normalized_n1_detail_projects_root_media_and_all_frame_prompts(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid, cdir, originals = _completed_v4_project(
        settings, monkeypatch, frame_count=9,
    )
    (cdir / "source.mp4").write_bytes(b"source-video")
    (cdir / "work" / "visual_prompt.txt").write_text(
        "冻结的旧视频提示词", encoding="utf-8"
    )
    _write_root_timeline_authority(cdir, originals, duration_s=14.5)
    storage.update_meta(
        settings.data_dir,
        cid,
        duration_s=14.5,
        voice_line_provenance=[],
    )
    meta = storage.load_meta(settings.data_dir, cid)
    acceptance = postprocess.image_acceptance_status(settings, cid, meta)
    postprocess.accept_images(settings, cid, {
        "confirm": True,
        "expected_meta_sha256": acceptance["expected_meta_sha256"],
    })
    normalized = long_generation.normalize_single_segment_project(
        settings,
        cid,
        storage.load_meta(settings.data_dir, cid),
    )
    assert normalized["segments"][0]["keyframe_paths"] == [
        f"keyframes/{path.name}" for path in originals
    ]
    keyframe_sources = normalized["segments"][0]["keyframe_sources"]
    assert len(keyframe_sources) == 9
    assert [item["source_time_s"] for item in keyframe_sources] == [
        round(1.5 * index, 3) for index in range(9)
    ]
    assert keyframe_sources[0]["transition"] == {
        "type": "start", "at_s": 0.0,
    }
    assert keyframe_sources[5]["transition"] == {
        "type": "hard_cut", "at_s": 7.0,
    }
    receipt = json.loads(
        (cdir / long_video.PLAN_RECEIPT_FILENAME).read_text(encoding="utf-8")
    )
    assert receipt["version"] == long_video.VISUAL_PLAN_RECEIPT_VERSION
    assert receipt["segments"][0]["keyframe_sources"] == keyframe_sources

    private_frames = normalized["_image_optimization"]["frames"]
    assert {item["segment_index"] for item in private_frames} == {0}
    with TestClient(create_app(settings)) as client:
        detail_response = client.get(
            f"/api/conversations/{cid}", headers=AUTH
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert client.get(
            f"/api/conversations/{cid}/files/keyframes/01.png"
        ).status_code == 401
        assert client.get(
            f"/api/conversations/{cid}/files/keyframes/01.png", headers=AUTH
        ).content == PNG
        assert client.get(
            f"/api/conversations/{cid}/files/postprocessed/01.png", headers=AUTH
        ).content == PNG
        assert client.get(
            f"/api/conversations/{cid}/files/keyframes/..%2Fmeta.json", headers=AUTH
        ).status_code == 404

    assert detail["segment_count"] == 1
    segment = detail["segments"][0]
    assert segment["index"] == 1
    assert segment["keyframe_paths"] == [
        f"keyframes/{path.name}" for path in originals
    ]
    assert detail["postprocess"]["frames"] == [
        path.name for path in originals
    ]
    projected = segment["image_optimization_prompts"]
    assert len(projected) == 9
    assert projected == [{
        "frame_name": item["frame_name"],
        "text": item["current"],
        "default_text": item["default"],
        "sha256": item["sha256"],
    } for item in private_frames]
    assert "_image_optimization" not in json.dumps(detail)


def test_normalized_n1_rebuilds_missing_root_timeline_authority(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid, cdir, originals = _completed_v4_project(
        settings, monkeypatch, frame_count=9,
    )
    (cdir / "source.mp4").write_bytes(b"source-video")
    (cdir / "work" / "visual_prompt.txt").write_text(
        "冻结的旧视频提示词", encoding="utf-8"
    )
    storage.update_meta(
        settings.data_dir, cid, duration_s=14.5, voice_line_provenance=[],
    )
    meta = storage.load_meta(settings.data_dir, cid)
    acceptance = postprocess.image_acceptance_status(settings, cid, meta)
    postprocess.accept_images(settings, cid, {
        "confirm": True,
        "expected_meta_sha256": acceptance["expected_meta_sha256"],
    })
    calls = []

    def rebuild(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        if str(argv[1]).endswith("extract_keyframes.py"):
            authority_work = Path(argv[argv.index("--out-dir") + 1])
            _write_root_timeline_authority(
                cdir, originals, duration_s=14.5,
                authority_work=authority_work,
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(long_generation.subprocess, "run", rebuild)
    normalized = long_generation.normalize_single_segment_project(
        settings, cid, storage.load_meta(settings.data_dir, cid),
    )

    assert [Path(call[0][1]).name for call in calls] == [
        "extract_keyframes.py", "scenes.py",
    ]
    assert len(normalized["segments"][0]["keyframe_sources"]) == 9
    receipt = json.loads(
        (cdir / long_video.PLAN_RECEIPT_FILENAME).read_text(encoding="utf-8")
    )
    assert receipt["version"] == long_video.VISUAL_PLAN_RECEIPT_VERSION


def test_n2_frame_prompt_projection_is_exact_and_fails_closed_on_index_drift(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid, _cdir, _originals = _completed_v4_project(
        settings,
        monkeypatch,
        frame_count=9,
        segments=True,
        segment_total=2,
    )
    with TestClient(create_app(settings)) as client:
        detail = client.get(
            f"/api/conversations/{cid}", headers=AUTH
        ).json()
        assert [
            len(segment["image_optimization_prompts"])
            for segment in detail["segments"]
        ] == [9, 9]

        current = storage.load_meta(settings.data_dir, cid)
        storage.update_meta(
            settings.data_dir, cid, segments=current["segments"][:1]
        )
        drifted = client.get(
            f"/api/conversations/{cid}", headers=AUTH
        ).json()

    assert all(
        "image_optimization_prompts" not in segment
        for segment in drifted["segments"]
    )


def test_v4_manual_acceptance_receipt_enables_h3_without_image_verification(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid, cdir, originals = _completed_v4_project(settings, monkeypatch)
    meta = storage.load_meta(settings.data_dir, cid)
    before = postprocess.image_acceptance_status(settings, cid, meta)
    assert before == {
        "required": True,
        "accepted": False,
        "expected_meta_sha256": before["expected_meta_sha256"],
    }
    assert len(before["expected_meta_sha256"]) == 64
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(cdir, meta, originals, settings=settings)

    accepted = postprocess.accept_images(settings, cid, {
        "confirm": True,
        "expected_meta_sha256": before["expected_meta_sha256"],
    })

    assert accepted["required"] is True and accepted["accepted"] is True
    latest = storage.load_meta(settings.data_dir, cid)
    assert "_image_verification" not in latest
    assert postprocess.generation_keyframes(
        cdir, latest, originals, settings=settings,
    ) == sorted((cdir / "work" / "postprocessed").glob("*.png"))


def _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
    tmp_path, monkeypatch, segment_count, *, postprocess_options=None,
    historical_pre_unification=False, forbid_legacy_short=False,
    complete_generation=False, dialogue_mode="auto",
    silent_segment_indices=(),
    web_output_validation=False, dialogue_classification="spoken",
    has_bgm=False, automatic_operation=False,
):
    settings = make_settings(
        tmp_path,
        retry_interval_s=0,
        enable_mediakit_erase=bool(
            postprocess_options
            and (
                postprocess_options.get("remove_subtitle")
                or postprocess_options.get("remove_brand")
            )
        ),
        enable_h3_submit=True,
        autodl_art_token="art",
        minimax_api_key="minimax",
        h3_poll_interval_s=0,
    )
    cid, cdir, originals = _completed_v4_project(
        settings,
        monkeypatch,
        frame_count=9,
        segments=not historical_pre_unification,
        postprocess_options=postprocess_options,
        segment_total=segment_count,
    )
    (cdir / "source.mp4").write_bytes(b"source-video")
    if historical_pre_unification:
        assert segment_count == 1
        _write_root_timeline_authority(cdir, originals, duration_s=14.5)
    project_duration = (
        14.5 if historical_pre_unification else 10.0 * segment_count
    )
    old_visual_prompt = "九张已验收图片中的人物保持静默，歌声来自画外。"
    (cdir / "work" / "visual_prompt.txt").write_text(
        old_visual_prompt, encoding="utf-8"
    )
    manual_mode = dialogue_mode in {"edit", "custom"}
    manual_provenance = "asr+edited" if dialogue_mode == "edit" else "manual"
    global_manual_lines = [
        {
            "text": f"用户冻结台词{index}",
            "start_s": 10.0 * (index - 1) + 1.0,
            "end_s": 10.0 * (index - 1) + 2.0,
            "classification": None,
            "provenance": manual_provenance,
        }
        for index in range(1, segment_count + 1)
        if index not in silent_segment_indices
    ]
    if manual_mode:
        provenance = []
        voice_sha256 = None
        evidence = None
    else:
        subprocess.run(
            [
                "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                "-t", str(project_duration + 0.08), "-q:a", "9", "-y",
                str(cdir / "work" / "voice.mp3"),
            ],
            check=True,
        )
        voice_sha256 = hashlib.sha256(
            (cdir / "work" / "voice.mp3").read_bytes()
        ).hexdigest()
        provenance = [{
            "text": "تجربة صوتية",
            "start_s": 0.0,
            "end_s": project_duration - 0.06,
            "classification": dialogue_classification,
            "provenance": "asr",
            "kept": True,
        }]
        evidence = long_generation.classification_evidence_sha256(
            audio_path="work/voice.mp3",
            audio_sha256=voice_sha256,
            has_bgm=has_bgm,
            decisions=provenance,
        )
    common_changes = dict(
        duration_s=project_duration,
        vocal_filter_enabled=True,
        voice_mode="keep",
        dialogue_mode=dialogue_mode,
        fit_required=False,
        fit_profiles={
            "9:16": {"fit_required": False, "default_fit_mode": "none"},
            "16:9": {"fit_required": True, "default_fit_mode": "crop"},
        },
        aspect_ratio="9:16",
        resolution="768p",
        voice_lines=global_manual_lines if manual_mode else [],
        voice_line_provenance=[{
            **line,
            "analysis_audio_path": "work/voice.mp3",
            "analysis_audio_sha256": voice_sha256,
            "analysis_has_bgm": has_bgm,
            "classification_evidence_sha256": evidence,
        } for line in provenance],
        has_bgm=has_bgm,
    )
    expected_receipt = None
    has_frozen_segment_plan = not historical_pre_unification
    if has_frozen_segment_plan:
        public_segments = []
        receipt_segments = []
        for index in range(1, segment_count + 1):
            start_s = 10.0 * (index - 1)
            end_s = 10.0 * index
            segdir = cdir / "work" / "segments" / str(index)
            work = segdir / "work"
            source = segdir / "source.mp4"
            source.write_bytes(f"segment-{index}".encode())
            visual_text = f"第{index}段九张优化图片视觉动作"
            visual = work / "visual_prompt.txt"
            visual.write_text(visual_text, encoding="utf-8")
            dialogue = (
                []
                if index in silent_segment_indices
                else (
                    [{
                        **global_manual_lines[index - 1],
                        "start_s": 1.0,
                        "end_s": 2.0,
                    }]
                    if manual_mode
                    else [{
                        "text": f"画外口播{index}",
                        "start_s": 1.0,
                        "end_s": 2.0,
                        "classification": dialogue_classification,
                    }]
                )
            )
            final_text = (
                "不要生成背景音乐\n"
                + prepared_input.compose_final_prompt(
                    long_video.compose_segment_visual_prompt(visual_text),
                    dialogue,
                )
            )
            final = work / "prompt.txt"
            final.write_text(final_text, encoding="utf-8")
            segment_frames = sorted((work / "keyframes").glob("*.png"))
            segment = {
                "index": index,
                "start_s": start_s,
                "end_s": end_s,
                "chain_id": "chain-001",
                "join_mode": "hard_cut" if index == 1 else "continue",
                "source": f"segments/{index}/source.mp4",
                "keyframes": [path.name for path in segment_frames],
                "keyframe_paths": [
                    path.relative_to(cdir / "work").as_posix()
                    for path in segment_frames
                ],
                "keyframe_sources": [{
                    "order": order,
                    "source_time_s": round(start_s + 1.0 * (order - 1), 3),
                    "source_scene_id": "SCENE_01",
                    "transition": (
                        {
                            "type": "start",
                            "at_s": round(start_s, 3),
                        }
                        if index == 1 and order == 1
                        else {"type": "continuous", "at_s": None}
                    ),
                } for order in range(1, 10)],
                "first_frame_path": segment_frames[0].relative_to(
                    cdir / "work"
                ).as_posix(),
                "last_frame_path": segment_frames[-1].relative_to(
                    cdir / "work"
                ).as_posix(),
                "visual_prompt": visual_text,
                "prompt": final_text,
                "dialogue": dialogue,
                "lines": [line["text"] for line in dialogue],
            }
            public_segments.append(segment)
            receipt_segments.append({
                **segment,
                "source_path": source,
                "keyframe_paths": segment_frames,
                "first_frame_path": segment_frames[0],
                "last_frame_path": segment_frames[-1],
                "visual_prompt_path": visual,
                "final_prompt_path": final,
            })
        receipt_path = long_video.write_plan_receipt(
            cdir,
            source=cdir / "source.mp4",
            duration_s=10.0 * segment_count,
            segments=receipt_segments,
            workflow=h3.H3_WORKFLOW,
        )
        assert json.loads(receipt_path.read_text(encoding="utf-8"))[
            "version"
        ] == long_video.VISUAL_PLAN_RECEIPT_VERSION
        expected_receipt = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        common_changes.update(
            duration_s=10.0 * segment_count,
            segments=public_segments,
            long_video_plan_receipt=receipt_path.name,
        )
    storage.update_meta(settings.data_dir, cid, **common_changes)
    current = storage.load_meta(settings.data_dir, cid)
    acceptance = postprocess.image_acceptance_status(settings, cid, current)
    if acceptance["required"]:
        postprocess.accept_images(settings, cid, {
            "confirm": True,
            "expected_meta_sha256": acceptance["expected_meta_sha256"],
        })

    skill_file = tmp_path / "video-prompt-fusion-SKILL.md"
    skill_file.write_text("strict video prompt fusion", encoding="utf-8")
    monkeypatch.setattr(pipeline, "PROMPT_FUSION_SKILL_MD", skill_file)
    monkeypatch.setattr(
        long_generation, "PROMPT_FUSION_SKILL_SOURCE", skill_file
    )
    fusion_calls = []
    fusion_prompts = []

    def skill(
        self, scope, _prompt, *, session_dir, writable_paths=(),
    ):
        fusion_calls.append(session_dir)
        assert session_dir == cdir
        assert writable_paths == (scope / "work" / "h3_prompt_plan.json",)
        input_data = (scope / "work" / "multimodal_input.json").read_bytes()
        input_payload = json.loads(input_data.decode("utf-8"))
        output_segments = []
        for segment in input_payload["segments"]:
            timeline = long_generation._freeze_local_keyframe_sources(
                [{
                    key: frame[key]
                    for key in (
                        "order", "segment_time_s", "source_scene_id",
                        "transition",
                    )
                } for frame in segment["new_keyframes"]],
            )
            visual = [
                f"Use only the accepted optimized storyboard for shot {index}."
                for index in range(
                    1,
                    2 + sum(
                        frame["transition"]["type"] == "hard_cut"
                        for frame in timeline
                    ),
                )
            ]
            final_prompt = long_generation._compile_fusion_ref2va_prompt(
                visual=visual,
                timeline=timeline,
                lines=json.loads(segment["audio_content"]["lines_json"]),
                music_policy=segment["audio_content"]["music_policy"],
            )
            fusion_prompts.append(final_prompt)
            output_segments.append({
                "index": segment["index"], "visual": visual,
            })
        plan = {
            "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
            "version": long_generation.PROMPT_FUSION_VERSION,
            "input_sha256": hashlib.sha256(input_data).hexdigest(),
            "segments": output_segments,
        }
        (scope / "work" / "h3_prompt_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False), encoding="utf-8"
        )

    monkeypatch.setattr(CodexRunner, "run_isolated", skill)
    context_sources = []
    context_effective = {}
    context_requests = []

    def context_gateway(request):
        context_requests.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/files/upload":
            return httpx.Response(200, json={
                "file": {"file_id": "427752006353318"},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            })
        if request.method == "POST" and request.url.path == "/v2/h3_context_ir":
            body = json.loads(request.content)
            source_prompt = body["content"][0]["text"]
            context_sources.append(source_prompt)
            task_id = f"context-task-{len(context_sources)}"
            context_effective[task_id] = f"EFFECTIVE::{source_prompt}"
            return httpx.Response(200, json={"task_id": task_id})
        if request.method == "GET":
            task_id = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"task": {
                "id": task_id,
                "task_type": "h3_context_ir",
                "status": "succeeded",
                "modality": "text",
                "content": {"prompt": context_effective[task_id]},
            }})
        raise AssertionError(request.url)

    real_optimize = context_ir_bridge.optimize_h3_prompt

    def optimize(frozen):
        with httpx.Client(
            transport=httpx.MockTransport(context_gateway)
        ) as context_client:
            return real_optimize(frozen, client=context_client)

    monkeypatch.setattr(context_ir_bridge, "optimize_h3_prompt", optimize)
    h3_requests = []
    stitch_calls = []
    reuse_calls = []
    attempts = {}
    timeline_calls = []
    if complete_generation:
        def start_h3(request):
            h3_requests.append(request)
            attempt_id = f"{len(h3_requests):06d}"
            attempts[request.client_request_id] = attempt_id
            subprocess.run([
                "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=16x16:r=24",
                "-t", "0.2", "-pix_fmt", "yuv420p", "-y",
                str(request.workdir / "generated.mp4"),
            ], check=True)
            return h3.H3Result("succeeded", attempt_id)

        monkeypatch.setattr(h3, "start", start_h3)
        monkeypatch.setattr(h3, "output_is_reusable", lambda *_a, **_k: True)
        monkeypatch.setattr(
            h3,
            "inspect",
            lambda request: h3.H3Result(
                "succeeded", attempts[request.client_request_id]
            ),
        )
        def timeline_receipt(request, attempt_id):
            timeline_calls.append((request, attempt_id))
            return {
                "attempt_id": attempt_id,
                "audio": {"mode": "provider_generated"},
            }

        monkeypatch.setattr(h3, "load_media_timeline_receipt", timeline_receipt)

        def fake_stitch(**kwargs):
            stitch_calls.append(kwargs)
            kwargs["output"].write_bytes(b"provider-generated-stitch")

        monkeypatch.setattr(stitch, "stitch_video", fake_stitch)

        def reusable(plan, dialogue_mode, *, generation=None,
                     provider_media=None):
            reuse_calls.append((plan, dialogue_mode, generation, provider_media))
            return True

        monkeypatch.setattr(
            long_generation, "stitched_output_is_reusable", reusable
        )
    else:
        monkeypatch.setattr(
            h3,
            "start",
            lambda request: h3_requests.append(request)
            or h3.H3Result("failed", "000001", error_code="stubbed"),
        )
    if forbid_legacy_short:
        monkeypatch.setattr(
            main_module,
            "_freeze_submission",
            lambda *_args, **_kwargs: pytest.fail(
                "current v4 segment project reached legacy short freeze"
            ),
        )
    payload = {
        "confirm": True,
        "client_request_id": "short-off-screen-123",
        "dialogue_mode": dialogue_mode,
        "dialogue_delivery": "off_screen",
        "fit_mode": "none",
        "aspect_ratio": "9:16",
        "resolution": "768p",
    }
    if expected_receipt is not None:
        payload.update(
            expected_plan_receipt=expected_receipt,
            fast_mode=False,
        )
    real_continue_after_fusion = main_module._continue_after_prompt_fusion
    deferred_continuations = []

    def defer_after_fusion(settings_arg, cid_arg, owner_arg):
        deferred_continuations.append((settings_arg, cid_arg, owner_arg))
        return False

    monkeypatch.setattr(
        main_module, "_continue_after_prompt_fusion", defer_after_fusion,
    )
    real_start_automatic = main_module._start_automatic_v4_generation
    monkeypatch.setattr(
        main_module,
        "_start_automatic_v4_generation",
        lambda *_args, **_kwargs: None,
    )
    with TestClient(create_app(settings)) as client:
        monkeypatch.setattr(
            main_module,
            "_start_automatic_v4_generation",
            real_start_automatic,
        )
        normalized = storage.load_meta(settings.data_dir, cid)
        assert postprocess.image_acceptance_status(
            settings, cid, normalized
        )["accepted"] is acceptance["required"]
        assert postprocess.generation_keyframes(
            cdir, normalized, originals, settings=settings,
        ) == (
            sorted((cdir / "work" / "postprocessed").glob("*.png"))
            if not has_frozen_segment_plan else [
                path
                for index in range(1, segment_count + 1)
                for path in sorted((
                    cdir / "work" / "segments" / str(index)
                    / "work" / "postprocessed"
                ).glob("*.png"))
            ]
        )
        if automatic_operation:
            real_start_automatic(settings, cid, CodexRunner(1, 1))
            final_submit = None
        else:
            first = client.post(
                f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
            )
            assert first.status_code == 202, first.json()
            final_submit = first
        fused_meta = storage.load_meta(settings.data_dir, cid)
        fusion_receipt = fused_meta["_prompt_fusion"]
        assert fusion_receipt["status"] == "done"
        assert fusion_receipt["error"] is None
        assert len(fusion_receipt["manifest_sha256"]) == 64
        assert fused_meta.get("generation") is None
        assert h3_requests == []
        assert len(deferred_continuations) == 1
        assert deferred_continuations[0][1:] == (
            cid, fused_meta["_input_owner"],
        )
        frozen_fusion = long_generation.load_prompt_fusion_manifest(
            root=cdir, skill_source_path=skill_file,
        )
        assert frozen_fusion.final_prompts == tuple(fusion_prompts)
        monkeypatch.setattr(
            main_module,
            "_continue_after_prompt_fusion",
            real_continue_after_fusion,
        )
        assert real_continue_after_fusion(
            settings, cid, fused_meta["_input_owner"],
        ) is True
        if web_output_validation:
            assert complete_generation is True
            detail = client.get(
                f"/api/conversations/{cid}", headers=AUTH
            )
            assert detail.status_code == 200
            assert detail.json()["has_video"] is True
            if manual_mode:
                assert detail.json()["dialogue"]["lines"] == [
                    {
                        "text": line["text"],
                        "start_s": line["start_s"],
                        "end_s": line["end_s"],
                    }
                    for line in global_manual_lines
                ]
            generated = client.get(
                f"/api/conversations/{cid}/files/generated.mp4", headers=AUTH
            )
            assert generated.status_code == 200
            assert generated.content == b"provider-generated-stitch"

            storage.update_meta(
                settings.data_dir, cid, dialogue_delivery="on_screen"
            )
            wrong_delivery = client.get(
                f"/api/conversations/{cid}", headers=AUTH
            )
            assert wrong_delivery.json()["has_video"] is False
            storage.update_meta(
                settings.data_dir, cid, dialogue_delivery="off_screen"
            )

            fusion_output = cdir / "work" / "h3_prompt_plan.json"
            fusion_output_data = fusion_output.read_bytes()
            fusion_output.write_bytes(fusion_output_data + b" ")
            tampered = client.get(
                f"/api/conversations/{cid}", headers=AUTH
            )
            assert tampered.json()["has_video"] is False
            fusion_output.write_bytes(fusion_output_data)

    if final_submit is not None:
        assert final_submit.status_code == 202, final_submit.json()
    assert fusion_calls == [cdir]
    if historical_pre_unification:
        migrated = storage.load_meta(settings.data_dir, cid)
        assert migrated["generation"]["status"] == "failed"
        assert migrated["generation"]["segments"][0]["error"] == (
            "long_video_legacy_plan_read_only"
        )
        assert h3_requests == []
        assert json.loads(
            (cdir / long_video.PLAN_RECEIPT_FILENAME).read_text(encoding="utf-8")
        )["version"] == long_video.VISUAL_MULTIMODAL_PLAN_RECEIPT_VERSION
        return
    assert len(h3_requests) == (segment_count if complete_generation else 1)
    for bound_request in h3_requests:
        assert bound_request.context_ir_required is True
        assert bound_request.context_ir_receipt_path is not None
        assert bound_request.context_ir_receipt_sha256 is not None
        h3._require_context_ir_receipt(bound_request)
    request = h3_requests[0]
    assert context_sources == fusion_prompts[:len(context_sources)]
    assert request.prompt == fusion_prompts[0]
    assert "subject_definitions:" in request.prompt
    assert "<AUDIO_CONTENT_JSON>" not in request.prompt
    assert "[AUDIO_CONTENT_JSON]" not in request.prompt
    assert request.on_screen_dialogue == ()
    assert len(request.keyframes) == 9
    first_has_spoken_prompt = (
        "the off-screen narrator (S1) says" in fusion_prompts[0]
    )
    assert request.reference_audios == ()
    assert request.workflow == h3.H3_WORKFLOW
    assert request.multimodal_compiler_version is None
    assert (
        "the off-screen narrator (S1) says" in request.prompt
    ) is first_has_spoken_prompt
    assert len(json.loads(
        (cdir / "work" / "multimodal_input.json").read_text(encoding="utf-8")
    )["segments"]) == segment_count
    assert all(old_visual_prompt not in prompt for prompt in context_sources)
    if complete_generation:
        assert [request.prompt for request in h3_requests] == fusion_prompts
        assert len(stitch_calls) == 1
        provider_generated_indices = set(range(1, segment_count + 1))
        assert stitch_calls[0]["audio_mode"] == "provider_generated"
        validation_passes = 1 + int(web_output_validation)
        assert len(reuse_calls) == validation_passes
        assert all(call[1] == dialogue_mode for call in reuse_calls)
        assert all(
            set(call[3]) == provider_generated_indices for call in reuse_calls
        )
        assert len(timeline_calls) == (
            len(provider_generated_indices) * validation_passes
        )
        completed = storage.load_meta(settings.data_dir, cid)
        assert completed["generation"]["status"] == "succeeded"
        for state in completed["generation"]["segments"]:
            assert "h3_attempt_id" in state
    for index in range(1, segment_count + 1):
        segment_work = cdir / "work" / "segments" / str(index) / "work"
        assert not (segment_work / "multimodal_input.json").exists()
        assert not (segment_work / "h3_prompt_plan.json").exists()
        assert not (segment_work / "h3_multimodal_source.json").exists()
    if segment_count == 1 and postprocess_options is None and not automatic_operation:
        frozen_meta = storage.load_meta(settings.data_dir, cid)
        frozen_meta.pop("_prompt_fusion", None)
        fusion_output = cdir / "work" / "h3_prompt_plan.json"
        fusion_output.write_bytes(fusion_output.read_bytes() + b"\n")
        with pytest.raises(
            long_generation.LongGenerationError,
            match="prompt_fusion_manifest_invalid",
        ):
            long_generation.freeze_plan(
                cdir,
                frozen_meta,
                long_generation.plan_receipt(cdir, frozen_meta),
                "none",
                dialogue_mode,
                dialogue_delivery="off_screen",
                aspect_ratio="9:16",
                resolution="768p",
                prepare_fit=False,
                settings=settings,
            )


@pytest.mark.parametrize("segment_count", [1, 2])
def test_n1_n2_off_screen_fusion_bootstraps_then_enters_context_h3(
    tmp_path, monkeypatch, segment_count,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path, monkeypatch, segment_count,
    )


def test_historical_pre_unification_n1_migrates_but_never_starts_provider(
    tmp_path, monkeypatch,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        1,
        historical_pre_unification=True,
    )


def test_n2_off_screen_fusion_completes_context_h3_and_native_stitch(
    tmp_path, monkeypatch,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path, monkeypatch, 2, complete_generation=True,
    )


def test_current_custom_dialogue_uses_one_automatic_fusion_h3_native_stitch(
    tmp_path, monkeypatch,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        1,
        complete_generation=True,
        dialogue_mode="custom",
        automatic_operation=True,
        web_output_validation=True,
    )


@pytest.mark.parametrize("segment_count", [1, 2], ids=("n1", "n2"))
def test_current_sung_only_uses_prompt_fusion_provider_output(
    tmp_path, monkeypatch, segment_count,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        segment_count,
        complete_generation=True,
        dialogue_classification="sung",
        has_bgm=True,
    )


@pytest.mark.parametrize("has_bgm", [True, None], ids=("bgm", "unknown"))
def test_current_spoken_without_no_bgm_proof_continues_to_fusion_h3(
    tmp_path, monkeypatch, has_bgm,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        1,
        dialogue_classification="spoken",
        has_bgm=has_bgm,
    )


def test_off_screen_native_output_is_visible_and_delivery_bound(
    tmp_path, monkeypatch,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        1,
        complete_generation=True,
        web_output_validation=True,
    )


def test_single_operation_uses_bound_local_context_identity_receipt(
    tmp_path, monkeypatch,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        1,
        complete_generation=True,
    )


@pytest.mark.parametrize("segment_count", [1, 2])
def test_n1_n2_none_fusion_uses_provider_generated_output(
    tmp_path, monkeypatch, segment_count,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        segment_count,
        complete_generation=True,
        dialogue_mode="none",
    )


def test_n2_mixed_fusion_uses_provider_generated_output_without_source_overlay(
    tmp_path, monkeypatch,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        2,
        complete_generation=True,
        silent_segment_indices=(2,),
    )


@pytest.mark.parametrize(
    "postprocess_options",
    [
        {"remove_subtitle": True, "remove_brand": False,
         "optimize_image": False},
        {"remove_subtitle": False, "remove_brand": True,
         "optimize_image": False},
        {"remove_subtitle": True, "remove_brand": True,
         "optimize_image": False},
    ],
    ids=("subtitle-only", "logo-only", "subtitle-then-logo"),
)
def test_n1_mediakit_only_v4_uses_unified_fusion_not_legacy_short(
    tmp_path, monkeypatch, postprocess_options,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        1,
        postprocess_options=postprocess_options,
        forbid_legacy_short=True,
    )


def test_v4_manual_acceptance_rejects_stale_cas_and_generation_start(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid, _cdir, _originals = _completed_v4_project(settings, monkeypatch)
    initial = storage.load_meta(settings.data_dir, cid)
    stale = postprocess.image_acceptance_status(settings, cid, initial)[
        "expected_meta_sha256"
    ]
    storage.update_meta(settings.data_dir, cid, note="changed after review")
    with pytest.raises(postprocess.PostprocessError, match="image_acceptance_meta_changed"):
        postprocess.accept_images(settings, cid, {
            "confirm": True, "expected_meta_sha256": stale,
        })
    current = storage.load_meta(settings.data_dir, cid)
    expected = postprocess.image_acceptance_status(settings, cid, current)[
        "expected_meta_sha256"
    ]
    storage.update_meta(settings.data_dir, cid, generation={"status": "queued"})
    with pytest.raises(postprocess.PostprocessError, match="generation_in_progress"):
        postprocess.accept_images(settings, cid, {
            "confirm": True, "expected_meta_sha256": expected,
        })


@pytest.mark.parametrize("drift", ["raw", "output", "anchor_receipt", "cid"])
def test_v4_manual_acceptance_is_revoked_by_any_bound_byte_drift(
    tmp_path, monkeypatch, drift,
):
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid, cdir, originals = _completed_v4_project(settings, monkeypatch)
    meta = storage.load_meta(settings.data_dir, cid)
    status = postprocess.image_acceptance_status(settings, cid, meta)
    postprocess.accept_images(settings, cid, {
        "confirm": True,
        "expected_meta_sha256": status["expected_meta_sha256"],
    })
    if drift == "raw":
        originals[0].write_bytes(originals[0].read_bytes() + b"drift")
    elif drift == "output":
        output = cdir / "work" / "postprocessed" / originals[0].name
        output.write_bytes(output.read_bytes() + b"drift")
    elif drift == "anchor_receipt":
        receipt = next((
            cdir / "work" / ".postprocess-private" / "scene-anchors"
        ).glob("SCENE_*/*.json"))
        receipt.write_text("{}", encoding="utf-8")
    latest = storage.load_meta(settings.data_dir, cid)
    if drift == "cid":
        latest["id"] = "0" * 32
    assert postprocess.image_acceptance_status(settings, cid, latest)["accepted"] is False
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(cdir, latest, originals, settings=settings)


def test_v4_generation_keyframes_require_an_intact_verified_output_receipt(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    source = cdir / "work" / "keyframes" / "01.png"
    output_dir = cdir / "work" / "postprocessed"
    output_dir.mkdir()
    output = output_dir / "01.png"
    output.write_bytes(source.read_bytes())
    schedule = {"version": 4, "scenes": []}
    optimization = {
        "version": 4,
        "plan_sha256": "a" * 64,
        "continuity_sha256": "b" * 64,
        "scene_anchor_schedule": schedule,
    }
    monkeypatch.setattr(
        postprocess.image_optimization, "receipt", lambda _meta: optimization
    )
    monkeypatch.setattr(
        postprocess.image_optimization,
        "dual_target_plan_receipt",
        lambda _meta: {"version": 4, "person_plans": [], "scene_plans": []},
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        _postprocess_receipt={
            "version": 4,
            "options": {
                "remove_subtitle": False, "remove_brand": False,
                "optimize_image": True,
            },
        },
        postprocess={
            "status": "done", "options": {"remove_subtitle": False,
            "remove_brand": False, "optimize_image": True},
            "frames": ["01.png"], "segments": [], "error": None,
        },
    )
    meta = storage.load_meta(settings.data_dir, cid)
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(cdir, meta, [source])


def test_generation_keyframes_rejects_orphaned_manual_acceptance_authority(tmp_path):
    cdir = tmp_path / ("a" * 32)
    source = cdir / "work" / "keyframes" / "01.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(PNG)
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir,
            {"_image_user_acceptance": {"version": 1, "sha256": "a" * 64}},
            [source],
        )


def test_v4_postprocess_builds_board_before_existing_dag_without_quality_pack_gate(
    tmp_path, monkeypatch,
):
    """The generation DAG is global -> layout -> fanout, without verify_pack."""
    settings = make_settings(tmp_path)
    cdir = tmp_path / "session"
    cdir.mkdir()
    calls = []
    private = {
        "options": {"remove_subtitle": False, "remove_brand": False},
        "plan_sha256": "a" * 64,
        "continuity_sha256": "b" * 64,
    }
    metric = {
        "version": 1,
        "algorithm": postprocess._PALETTE_METRIC_ALGORITHM,
        "thresholds": postprocess._PALETTE_METRIC_THRESHOLDS,
        "frames": [],
    }
    metric["sha256"] = postprocess._receipt_sha256(metric)
    monkeypatch.setattr(postprocess, "_v4_frozen_plan", lambda *_args: {"segments": []})
    monkeypatch.setattr(postprocess, "_v4_frame_sources", lambda *_args: {})
    monkeypatch.setattr(postprocess, "_v4_preflight", lambda *_args: None)
    monkeypatch.setattr(postprocess, "_v4_palette_metrics", lambda *_args: metric)

    async def board(*_args, **_kwargs):
        calls.append("replacement-board")
        return cdir / "work" / ".postprocess-private" / "replacement-board" / "composite.png"

    async def bootstrap(*_args, **_kwargs):
        calls.append("global-anchor")
        return {}, []

    async def forbidden_pack(*_args, **_kwargs):
        pytest.fail("verify_pack must not be called by runtime")

    async def layout(*_args, **_kwargs):
        calls.append("layout")

    async def fanout(*_args, **_kwargs):
        calls.append("fanout")
        return []

    monkeypatch.setattr(postprocess, "_v4_bootstrap_scene_anchors", bootstrap)
    monkeypatch.setattr(postprocess, "_v4_generate_composite_replacement_board", board)
    monkeypatch.setattr(postprocess, "_v4_verify_bootstrap_packs", forbidden_pack)
    monkeypatch.setattr(postprocess, "_v4_generate_layout_anchors", layout)
    monkeypatch.setattr(postprocess, "_v4_fan_out", fanout)

    asyncio.run(postprocess._run_v4_task(
        settings, "cid", cdir, {}, private, {}, asyncio.Semaphore(1),
        skill_bytes=b"skill",
    ))
    assert calls == ["replacement-board", "global-anchor", "layout", "fanout"]


def test_v4_postprocess_preserves_the_failing_technical_phase(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cdir = tmp_path / "session"
    cdir.mkdir()
    private = {
        "options": {"remove_subtitle": False, "remove_brand": False},
        "plan_sha256": "a" * 64,
        "continuity_sha256": "b" * 64,
    }
    metric = {
        "version": 1,
        "algorithm": postprocess._PALETTE_METRIC_ALGORITHM,
        "thresholds": postprocess._PALETTE_METRIC_THRESHOLDS,
        "frames": [],
    }
    metric["sha256"] = postprocess._receipt_sha256(metric)
    monkeypatch.setattr(postprocess, "_v4_frozen_plan", lambda *_args: {"segments": []})
    monkeypatch.setattr(postprocess, "_v4_frame_sources", lambda *_args: {})
    monkeypatch.setattr(postprocess, "_v4_preflight", lambda *_args: None)
    monkeypatch.setattr(postprocess, "_v4_palette_metrics", lambda *_args: metric)

    async def board(*_args, **_kwargs):
        return None

    async def bootstrap(*_args, **_kwargs):
        return {}, []

    async def broken_layout(*_args, **_kwargs):
        raise RuntimeError("transport exploded")

    monkeypatch.setattr(postprocess, "_v4_generate_composite_replacement_board", board)
    monkeypatch.setattr(postprocess, "_v4_bootstrap_scene_anchors", bootstrap)
    monkeypatch.setattr(postprocess, "_v4_generate_layout_anchors", broken_layout)

    with pytest.raises(postprocess.PostprocessError) as raised:
        asyncio.run(postprocess._run_v4_task(
            settings, "cid", cdir, {}, private, {}, asyncio.Semaphore(1),
            skill_bytes=b"skill",
        ))
    assert raised.value.detail == "postprocess_layout_anchor_failed"


def test_v4_every_segment_anchor_reuses_the_same_composite_board_path(tmp_path):
    cdir = tmp_path / "session"
    board = (
        cdir / "work" / ".postprocess-private" / "replacement-board"
        / "composite.png"
    )
    board.parent.mkdir(parents=True)
    board.write_bytes(PNG)

    assert postprocess._v4_shared_references(cdir, []) == [
        ("composite_replacement_board", board)
    ]
    assert postprocess._v4_shared_references(
        cdir, [("segment_layout_anchor", cdir / "layout.png")]
    ) == [
        ("segment_layout_anchor", cdir / "layout.png"),
        ("composite_replacement_board", board),
    ]


def test_v4_element_evidence_is_locally_collapsed_to_one_numbered_sheet(tmp_path):
    sources = []
    for index, value in enumerate((31, 62, 93), 1):
        source = tmp_path / f"{index:02d}.png"
        ok, encoded = cv2.imencode(
            ".png", np.full((4, 6, 3), value, dtype=np.uint8)
        )
        assert ok
        source.write_bytes(encoded.tobytes())
        sources.append(source)
    sheet = tmp_path / "source-evidence.png"

    postprocess._compose_replacement_evidence_sheet(sources, sheet)

    decoded = cv2.imread(str(sheet), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (8, 12)


def test_v4_board_provider_call_uses_only_blank_canvas_and_one_evidence_sheet(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    cdir = tmp_path / "session"
    source = cdir / "work" / "keyframes" / "01.png"
    source.parent.mkdir(parents=True)
    ok, encoded = cv2.imencode(
        ".png", np.full((4, 6, 3), 63, dtype=np.uint8)
    )
    assert ok
    source.write_bytes(encoded.tobytes())
    plan = {}
    private = {
        "model": settings.seedream_model,
        "edit_mode": settings.seedream_edit_mode,
        "timeout_s": settings.seedream_timeout_s,
    }
    calls = []

    monkeypatch.setattr(postprocess, "_v4_project_revision", lambda *_args: 1)
    monkeypatch.setattr(
        image_optimization,
        "composite_replacement_board_spec",
        lambda _plan: {"tiles": [
            {
                "tile_id": f"TILE_{index:02d}",
                "stable_key": stable_key,
                "kind": kind,
                "replacement_description": description,
                "reference": {"segment_index": 0, "frame_index": 1},
            }
            for index, (stable_key, kind, description) in enumerate((
                ("hero", "person", "新人物"),
                ("prop", "entity", "新道具"),
                ("studio", "scene", "新场景"),
            ), 1)
        ]},
    )
    monkeypatch.setattr(
        image_optimization,
        "composite_replacement_board_prompt",
        lambda _plan: "TILE_01=hero；TILE_02=prop；TILE_03=studio",
    )
    monkeypatch.setattr(
        postprocess, "_v4_frame_sources", lambda *_args: {(0, 1): source, (0, 2): source}
    )

    async def edit(_settings, images, prompt, output, *, receipt_path):
        calls.append((len(images), prompt, receipt_path))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])

    monkeypatch.setattr(postprocess.seedream, "edit", edit)
    output = asyncio.run(postprocess._v4_generate_composite_replacement_board(
        settings,
        cdir,
        "cid",
        private,
        {0: [(source, cdir / "canonical" / "01.png")]},
        asyncio.Semaphore(1),
        plan,
    ))

    assert output == postprocess._replacement_board_path(cdir)
    assert output.is_file()
    assert calls and calls[0][0] == 2
    assert "TILE_01=hero" in calls[0][1]


def test_v4_startup_marks_ambiguous_typed_anchor_attempt_get_only(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running",
        "options": {"remove_subtitle": False, "remove_brand": False, "optimize_image": True},
        "frames": [], "error": None,
        "segments": [postprocess._segment_state(0, 2)],
    })
    attempt = cdir / "work" / ".postprocess-private" / "scene-anchors" / "SCENE_01" / "attempts" / "1-global-r1.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text(json.dumps({"status": "submission_unknown"}), encoding="utf-8")
    monkeypatch.setattr(postprocess, "_private_receipt", lambda _meta: {"version": 4})

    assert postprocess.recover_running(settings) == []
    latest = storage.load_meta(settings.data_dir, cid)
    assert latest["postprocess"]["status"] == "failed"
    assert latest["postprocess"]["error"] == "submission_unknown"


def test_v4_anchor_recovery_reads_only_frozen_typed_schedule_paths(tmp_path):
    cdir = tmp_path / "session"
    schedule = {
        "nodes": [{
            "scene_id": "SCENE_01", "label": "global",
            "anchor": {"order": 1},
        }],
    }
    private = {"scene_anchor_schedule": schedule}
    post = {"segments": [{"revision": 1}]}
    attempts = (
        cdir / "work" / ".postprocess-private" / "scene-anchors" / "SCENE_01"
        / "attempts"
    )
    attempts.mkdir(parents=True)
    # A non-scheduled filename is not a recoverable provider attempt and may
    # not make startup infer a second paid submission.
    (attempts / "99-unrelated-r1.json").write_text(
        json.dumps({"status": "submission_unknown"}), encoding="utf-8"
    )
    assert not postprocess._ambiguous_v4_anchor_attempts(cdir, private, post)

    (attempts / "0001-global-r1.json").write_text(
        json.dumps({"status": "submission_unknown"}), encoding="utf-8"
    )
    assert postprocess._ambiguous_v4_anchor_attempts(cdir, private, post)


def test_palette_metric_uses_area_weighted_lab_b_star_and_allows_local_change(tmp_path):
    yellow = tmp_path / "yellow.png"
    blue = tmp_path / "blue.png"
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _solid_png(yellow, (0, 255, 255))
    _solid_png(blue, (255, 0, 0))
    _solid_png(source, (127, 127, 127))
    _solid_png(output, (127, 127, 127), local_bgr=(0, 0, 255))

    warm = postprocess._area_weighted_palette_metric(yellow)
    cool = postprocess._area_weighted_palette_metric(blue)
    assert warm["warm_cool_family"] == "warm"
    assert cool["warm_cool_family"] == "cool"
    assert warm["mean_lab_b_star"] > 0 > cool["mean_lab_b_star"]

    plan = {"segments": [{
        "segment_index": 0,
        "frame_constraints": [{
            "frame_index": 1,
            "dominant_palette_contract": {
                "area_weighted_warm_cool_family": "balanced",
                "saturation_style": "muted",
            },
        }],
    }]}
    metrics = postprocess._v4_palette_metrics(
        plan, {(0, 1): source}, {(0, 1): output},
    )
    assert metrics["frames"][0]["source"]["warm_cool_family"] == "balanced"
    assert metrics["frames"][0]["output"]["warm_cool_family"] == "balanced"


class FakeEdit:
    """桩 mediakit.erase_image：记录场景并写 out；按 fail 名单抛 MediaKitError。"""

    def __init__(self, fail=(), *, receipts=False):
        self.calls = []
        self.fail = list(fail)
        self.receipts = receipts

    async def __call__(self, settings, cdir, image, out, confirm, scenes):
        self.calls.append({
            "image": image.name, "out": out, "confirm": confirm, "scenes": scenes,
        })
        if image.name in self.fail:
            raise mediakit.MediaKitError(502, "stub failure")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(PNG + b"edited")
        if self.receipts:
            receipt = out.parent / ".mediakit" / f"{out.name}.json"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({
                "version": mediakit.RECEIPT_VERSION,
                "state": "succeeded",
                "output": out.name,
                "scenes": list(scenes),
                "source": {
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                },
                "stages": [
                    {"scene": scene, "state": "succeeded"}
                    for scene in scenes
                ],
            }), encoding="utf-8")
        return out


def _post(c, cid, options, confirm=True):
    return c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH,
                  json={"options": options, "confirm": confirm})


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


# ---------- 门控矩阵 ----------

def test_upload_single_operation_continues_despite_nonblocking_quality_scores(
    tmp_path, video_1s, monkeypatch,
):
    settings = make_settings(tmp_path, enable_pipeline=True)
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    events = []

    def run_pipeline(_settings, cid, _runner, *, claimed_owner=None):
        assert claimed_owner is None
        events.append(("pipeline", cid))
        storage.update_meta(
            settings.data_dir, cid, status="done", error=None,
        )

    async def start_postprocess(_settings, cid, payload, _locks):
        events.append(("postprocess_start", cid, payload))
        storage.update_meta(
            settings.data_dir,
            cid,
            postprocess={
                "status": "running",
                "error": None,
                "options": payload["options"],
                "segments": [],
                "frames": [],
            },
        )

    async def run_postprocess(_settings, cid, *_args, **_kwargs):
        events.append(("postprocess_done", cid))
        current = storage.load_meta(settings.data_dir, cid)["postprocess"]
        storage.update_meta(
            settings.data_dir,
            cid,
            postprocess={**current, "status": "done"},
            _image_optimization={
                "quality_scores": {
                    "score": 0.0,
                    "issues": ["visual_continuity_low_confidence"],
                }
            },
        )

    def acceptance_status(_settings, cid, meta):
        return {
            "required": True,
            "accepted": meta.get("test_technical_acceptance") is True,
            "expected_meta_sha256": "a" * 64,
        }

    def accept(_settings, cid, payload):
        events.append(("receipt_commit", cid, payload))
        storage.update_meta(
            settings.data_dir, cid, test_technical_acceptance=True,
        )
        return acceptance_status(
            settings, cid, storage.load_meta(settings.data_dir, cid)
        )

    def generate(_settings, cid, _runner):
        diagnostics = storage.load_meta(settings.data_dir, cid)[
            "_image_optimization"
        ]["quality_scores"]
        assert diagnostics["score"] == 0.0
        events.append(("generation", cid))

    monkeypatch.setattr(pipeline, "run", run_pipeline)
    monkeypatch.setattr(postprocess, "start", start_postprocess)
    monkeypatch.setattr(postprocess, "run_task", run_postprocess)
    monkeypatch.setattr(
        postprocess, "image_acceptance_status", acceptance_status
    )
    monkeypatch.setattr(postprocess, "accept_images", accept)
    monkeypatch.setattr(
        long_generation, "plan_receipt", lambda *_args, **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        main_module, "_start_automatic_v4_generation", generate
    )

    with TestClient(create_app(settings)) as client:
        with video_1s.open("rb") as source:
            response = client.post(
                "/api/conversations",
                headers=AUTH,
                files={"file": ("clip.mp4", source, "video/mp4")},
            )

    assert response.status_code == 201, response.json()
    cid = response.json()["id"]
    assert [event[0] for event in events] == [
        "pipeline",
        "postprocess_start",
        "postprocess_done",
        "receipt_commit",
        "generation",
    ]
    assert all(event[1] == cid for event in events)
    assert events[1][2] == {
        "confirm": True,
        "options": {
            "remove_subtitle": False,
            "remove_brand": False,
            "optimize_image": True,
        },
    }


def test_requires_auth(client, video_1s):
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations", headers=AUTH,
                        files={"file": ("clip.mp4", f, "video/mp4")})
    cid = r.json()["id"]
    assert client.post(f"/api/conversations/{cid}/postprocess",
                       json={"options": OPTIONS_SUB, "confirm": True}).status_code == 401


def test_disabled_501(client):
    # 开关最优先（默认关）：不看 confirm、不看会话是否存在、不看选项
    r = client.post(f"/api/conversations/{'0' * 32}/postprocess", headers=AUTH,
                    json={"options": OPTIONS_SUB, "confirm": True})
    assert r.status_code == 501
    assert r.json() == {"detail": "MediaKit erase is disabled."}


def test_404_when_enabled(enabled):
    _, c = enabled
    r = _post(c, "0" * 32, OPTIONS_SUB)
    assert r.status_code == 404
    assert r.json() == {"detail": "not found"}


def test_confirm_required_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    monkeypatch.setattr(postprocess.mediakit, "erase_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    for body in ({"options": OPTIONS_SUB, "confirm": False},
                 {"options": OPTIONS_SUB, "confirm": "true"}, {"options": OPTIONS_SUB, "confirm": 1}):
        r = c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH, json=body)
        assert r.status_code == 409, body
        assert r.json() == {"detail": "confirmation required"}


def test_no_options_422(enabled):
    settings, c = enabled
    cid = _make_conv(settings)
    empty = {"remove_subtitle": False, "remove_brand": False}
    for body in ({"options": {}, "confirm": True},
                 {"options": empty, "confirm": True}):
        r = c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH, json=body)
        assert r.status_code == 422, body
        assert r.json() == {"detail": "at least one option required"}


def test_options_non_bool_422(enabled):
    settings, c = enabled
    cid = _make_conv(settings)
    r = c.post(f"/api/conversations/{cid}/postprocess", headers=AUTH,
               json={"options": {"remove_subtitle": "yes"}, "confirm": True})
    assert r.status_code == 422
    assert r.json() == {"detail": "options must be booleans"}


@pytest.mark.parametrize(
    ("legacy_key", "legacy_value"),
    [("change_bg", True), ("change_bg", False), ("face_hold", True), ("face_hold", False)],
)
def test_known_stale_postprocess_options_require_refresh_without_side_effects(
    enabled, monkeypatch, legacy_key, legacy_value
):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    before = _file_snapshot(cdir)
    calls = []
    monkeypatch.setattr(
        postprocess.mediakit, "erase_image",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    r = _post(c, cid, {**OPTIONS_SUB, legacy_key: legacy_value})

    assert r.status_code == 409
    assert r.json() == {"detail": "页面版本已更新，请刷新页面后重试。"}
    assert calls == []
    assert _file_snapshot(cdir) == before


@pytest.mark.parametrize(
    ("options", "detail"),
    [
        ({**OPTIONS_SUB, "future_option": True}, "unknown options: future_option"),
        (
            {**OPTIONS_SUB, "face_hold": True, "future_option": True},
            "unknown options: face_hold, future_option",
        ),
    ],
)
def test_other_unknown_postprocess_option_remains_fail_closed(
    enabled, options, detail
):
    settings, c = enabled
    cid = _make_conv(settings)
    r = _post(c, cid, options)
    assert r.status_code == 422
    assert r.json() == {"detail": detail}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"confirm": True},
        {"options": OPTIONS_SUB},
        {"confirm": True, "options": OPTIONS_SUB, "unexpected": True},
    ],
)
def test_invalid_postprocess_top_level_shape_is_rejected_before_write_or_provider(
    enabled, monkeypatch, body
):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    before = _file_snapshot(cdir)
    calls = []
    monkeypatch.setattr(
        postprocess.mediakit, "erase_image",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    r = c.post(
        f"/api/conversations/{cid}/postprocess",
        headers=AUTH,
        json=body,
    )

    assert r.status_code == 422
    assert r.json() == {"detail": "invalid_postprocess_request"}
    assert calls == []
    assert _file_snapshot(cdir) == before


def test_not_done_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, status="queued")
    monkeypatch.setattr(postprocess.mediakit, "erase_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json() == {"detail": "artifacts not ready"}


def test_already_running_409(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running", "options": OPTIONS_SUB, "frames": [], "error": None,
    })
    monkeypatch.setattr(postprocess.mediakit, "erase_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json() == {"detail": "already running"}


def test_artifacts_gone_409(enabled, monkeypatch):
    """status=done 但帧目录缺失 → 409 artifacts not ready，不写 meta.postprocess。"""
    settings, c = enabled
    cid = _make_conv(settings)
    (settings.data_dir / cid / "work" / "keyframes").rename(
        settings.data_dir / cid / "work" / "gone")
    monkeypatch.setattr(postprocess.mediakit, "erase_image",
                        lambda *a, **k: pytest.fail("edit must not be called"))
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json() == {"detail": "artifacts not ready"}
    assert storage.load_meta(settings.data_dir, cid).get("postprocess") is None


def test_postprocess_cannot_start_after_generation_input_is_frozen(enabled):
    settings, client = enabled
    cid = _make_conv(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            "status": "queued",
            "client_request_id": "already-frozen",
        },
    )

    response = _post(client, cid, OPTIONS_SUB)

    assert response.status_code == 409
    assert response.json() == {"detail": "generation_already_started"}
    assert storage.load_meta(settings.data_dir, cid).get("postprocess") is None


# ---------- 单段全链路 ----------

def test_single_segment_full_chain(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    options = {"remove_subtitle": True, "remove_brand": True}
    r = _post(c, cid, options)
    assert r.status_code == 200
    assert r.json() == {"status": "running", "frames": []}  # 受理即返回，进度走 detail 轮询

    # 严格阶段屏障：全段文字擦除完成后才开始全段图标擦除。
    assert [call["scenes"] for call in fake.calls] == [
        (mediakit.TEXT_SCENE,), (mediakit.TEXT_SCENE,),
        (mediakit.ICON_SCENE,), (mediakit.ICON_SCENE,),
    ]
    assert all(call["confirm"] is True for call in fake.calls)

    # 产出：work/postprocessed/<帧名>.png（与源帧同目录层级）
    assert (cdir / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / "postprocessed" / "02.png").is_file()

    # meta.postprocess done + frames；detail 中保留后处理状态
    meta = storage.load_meta(settings.data_dir, cid)
    pp = meta["postprocess"]
    assert pp["status"] == "done"
    assert pp["options"] == {**options, "optimize_image": False}
    assert pp["frames"] == ["01.png", "02.png"]
    assert pp["error"] is None

    d = c.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert d["postprocess"] == pp
    assert d["postprocess_enabled"] is True


# ---------- 多段全链路 ----------

def test_multi_segment_full_chain(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    r = _post(c, cid, {
        "remove_subtitle": True,
        "remove_brand": True,
    })
    assert r.status_code == 200

    assert sorted(call["image"] for call in fake.calls) == [
        "01.png", "01.png", "01.png", "01.png", "02.png", "02.png",
    ]
    assert [call["scenes"] for call in fake.calls] == [
        (mediakit.TEXT_SCENE,),
        (mediakit.TEXT_SCENE,),
        (mediakit.TEXT_SCENE,),
        (mediakit.ICON_SCENE,),
        (mediakit.ICON_SCENE,),
        (mediakit.ICON_SCENE,),
    ]
    assert (cdir / "work" / "segments" / "1" / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / "segments" / "2" / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / "segments" / "2" / "work" / "postprocessed" / "02.png").is_file()
    assert not (cdir / "work" / "postprocessed").exists()

    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "done"
    # frames 为全形路径（与 files 白名单同形），前端按 segments/N/work/postprocessed/ 前缀过滤展示
    assert pp["frames"] == [
        "segments/1/work/postprocessed/01.png",
        "segments/2/work/postprocessed/01.png",
        "segments/2/work/postprocessed/02.png",
    ]


# ---------- 擦除场景 ----------

def test_subtitle_option_maps_to_text_scene(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    assert all(call["scenes"] == (mediakit.TEXT_SCENE,) for call in fake.calls)


# ---------- 失败处理 ----------

def test_frame_failure_marks_failed_keeps_successes(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    monkeypatch.setattr(postprocess.mediakit, "erase_image", FakeEdit(fail=["02.png"]))

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200  # 受理成功；结果走 detail

    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "failed"
    assert pp["error"] == "segment_failed"
    assert "02.png" in pp["segments"][0]["error"]
    assert pp["frames"] == []  # 整段完整前不发布 canonical
    private = cdir / "work" / ".postprocess-private" / "0" / "text"
    assert (private / "01.png").is_file()
    assert not (cdir / "work" / "postprocessed").exists()

    d = c.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert d["postprocess"]["status"] == "failed"
    assert d["postprocess"]["error"] == pp["error"]


def test_rerun_skips_existing_outputs(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "postprocessed").mkdir()
    (cdir / "work" / "postprocessed" / "01.png").write_bytes(b"kept")
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json() == {"detail": "postprocess_canonical_conflict"}
    assert fake.calls == []
    assert (cdir / "work" / "postprocessed" / "01.png").read_bytes() == b"kept"


# ---------- 并发：每会话一把锁 ----------

def test_concurrent_start_single_runner(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings)
    locks = {}

    async def run_both():
        return await asyncio.gather(
            postprocess.start(settings, cid, {"options": OPTIONS_SUB, "confirm": True}, locks),
            postprocess.start(settings, cid, {"options": OPTIONS_SUB, "confirm": True}, locks),
            return_exceptions=True,
        )

    results = asyncio.run(run_both())
    oks = [r for r in results if r is None]
    errs = [r for r in results if isinstance(r, postprocess.PostprocessError)]
    assert len(oks) == 1 and len(errs) == 1
    assert errs[0].status == 409 and errs[0].detail == "already running"
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["status"] == "running"


def test_options_lock_is_rechecked_inside_lock_without_writing(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    initial = storage.load_meta(settings.data_dir, cid)
    locked = {
        **initial,
        "postprocess": {
            "status": "done",
            "options": OPTIONS_BRAND,
            "frames": [],
            "error": None,
        },
    }
    (cdir / "meta.json").write_text(json.dumps(locked), encoding="utf-8")
    before = _file_snapshot(cdir)

    with pytest.raises(postprocess.PostprocessError) as caught:
        asyncio.run(postprocess.start(
            settings,
            cid,
            {"confirm": True, "options": OPTIONS_SUB},
            {},
        ))

    assert caught.value.status == 409
    assert caught.value.detail == {
        "code": "postprocess_options_locked",
        "message": "后处理选项已锁定，请刷新页面后按原选项重试。",
    }
    assert _file_snapshot(cdir) == before


# ---------- files 接口（前端取图路径） ----------

def test_files_endpoint_serves_postprocessed(enabled):
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    (cdir / "work" / "postprocessed").mkdir()
    (cdir / "work" / "postprocessed" / "01.png").write_bytes(b"opt")

    r = c.get(f"/api/conversations/{cid}/files/postprocessed/01.png", headers=AUTH)
    assert r.status_code == 200 and r.content == b"opt"
    r = c.get(f"/api/conversations/{cid}/files/segments/2/work/keyframes/01.png", headers=AUTH)
    assert r.status_code == 200 and r.content == PNG
    r = c.get(f"/api/conversations/{cid}/files/segments/2/work/postprocessed/01.png", headers=AUTH)
    assert r.status_code == 404  # 磁盘上不存在
    # 穿越一律 404：%2F 编码斜杠会绕过 HTTP 客户端的路径归一化，直击服务端白名单
    for name in ("segments/2/work/postprocessed/..%2Fkeyframes/01.png",
                 "postprocessed/..%2Fmeta.json"):
        r = c.get(f"/api/conversations/{cid}/files/{name}", headers=AUTH)
        assert r.status_code == 404, name


def test_rerun_different_options_409(enabled, monkeypatch):
    """上次 done 的 options 与本次不同 → 409（防旧产物贴新标签）；同选项重跑照常跳过已有图。"""
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    assert _post(c, cid, OPTIONS_SUB).status_code == 200
    other = {"remove_subtitle": True, "remove_brand": True}
    r = _post(c, cid, other)
    assert r.status_code == 409
    assert r.json() == {"detail": {
        "code": "postprocess_options_locked",
        "message": "后处理选项已锁定，请刷新页面后按原选项重试。",
    }}
    assert len(fake.calls) == 2  # 未产生新编辑

    # 同选项重跑：跳过已有优化图，正常 200 done
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    assert len(fake.calls) == 2  # 全部帧已存在，无新编辑
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["status"] == "done"


def test_legacy_change_bg_in_meta_options_rerun_no_409(enabled, monkeypatch):
    """旧会话 meta 含已废弃 change_bg，新请求当前两键且共有键一致 →
    锁定比对只认当前 OPTION_KEYS 共有键，重跑不 409，正常覆盖为新两键 options。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    # 历史会话：meta 里存有 change_bg 时代的废弃键
    legacy = {"change_bg": True, "remove_subtitle": True, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "done", "options": legacy, "frames": ["01.png", "02.png"], "error": None,
    })
    (cdir / "work" / "postprocessed").mkdir(parents=True)
    for name in ("01.png", "02.png"):
        (cdir / "work" / "postprocessed" / name).write_bytes(PNG + b"legacy")

    # 新请求两键与旧状态中的当前键一致；change_bg 忽略，不比
    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 200
    assert len(fake.calls) == 0  # 已有优化图，全部跳过

    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "done"
    assert pp["options"] == {**OPTIONS_SUB, "optimize_image": False}

    # 共有键真变了仍然 409：兼容比对不放松锁定
    r = _post(c, cid, OPTIONS_BRAND)
    assert r.status_code == 409
    assert r.json() == {"detail": {
        "code": "postprocess_options_locked",
        "message": "后处理选项已锁定，请刷新页面后按原选项重试。",
    }}


def test_legacy_pure_change_bg_rerun_clears_artifacts_and_reedits(enabled, monkeypatch):
    """任何 failed 状态都只能走分段重试，包括旧 change_bg 状态。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    legacy = {"change_bg": True, "remove_subtitle": False, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "failed", "options": legacy, "frames": ["01.png", "02.png"], "error": "x",
    })
    (cdir / "work" / "postprocessed").mkdir(parents=True)
    for name in ("01.png", "02.png"):
        (cdir / "work" / "postprocessed" / name).write_bytes(PNG + b"legacy")

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "postprocess_segment_retry_required"
    assert len(fake.calls) == 0


def test_legacy_pure_change_bg_any_new_option_requires_segment_retry(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)
    legacy = {"change_bg": True, "remove_subtitle": False, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "failed", "options": legacy, "frames": [], "error": "x",
    })
    r = _post(c, cid, OPTIONS_BRAND)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "postprocess_segment_retry_required"
    assert fake.calls == []


def test_legacy_pure_change_bg_multi_segment_clears_artifacts(enabled, monkeypatch):
    """多段 failed 旧状态也不得由普通 start 重建。"""
    settings, c = enabled
    cid = _make_conv(settings, segments=True)
    cdir = settings.data_dir / cid
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)

    legacy = {"change_bg": True, "remove_subtitle": False, "remove_brand": False}
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "failed", "options": legacy, "frames": ["segments/1/work/postprocessed/01.png"], "error": "x",
    })
    for n in (1, 2):
        d = cdir / "work" / "segments" / str(n) / "work" / "postprocessed"
        d.mkdir(parents=True)
        (d / "01.png").write_bytes(PNG + b"legacy")

    r = _post(c, cid, OPTIONS_SUB)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "postprocess_segment_retry_required"
    for n in (1, 2):
        d = cdir / "work" / "segments" / str(n) / "work" / "postprocessed"
        assert (d / "01.png").read_bytes() == PNG + b"legacy"
    assert len(fake.calls) == 0


def test_failed_start_preserves_meta_revision_and_never_calls_provider(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)
    assert _post(c, cid, OPTIONS_SUB).status_code == 200
    post = storage.load_meta(settings.data_dir, cid)["postprocess"]
    post.update(status="failed", error="segment_failed")
    post["segments"][0].update(status="failed", revision=7, error="provider_rejected")
    storage.update_meta(settings.data_dir, cid, postprocess=post)
    before = storage.load_meta(settings.data_dir, cid)
    calls_before = len(fake.calls)

    response = _post(c, cid, OPTIONS_SUB)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "postprocess_segment_retry_required"
    assert storage.load_meta(settings.data_dir, cid) == before
    assert len(fake.calls) == calls_before


def test_corrupt_existing_canonical_is_not_reused(enabled, monkeypatch):
    settings, c = enabled
    cid = _make_conv(settings)
    fake = FakeEdit()
    monkeypatch.setattr(postprocess.mediakit, "erase_image", fake)
    assert _post(c, cid, OPTIONS_SUB).status_code == 200
    canonical = settings.data_dir / cid / "work" / "postprocessed" / "01.png"
    canonical.write_bytes(b"not-a-png")
    calls_before = len(fake.calls)

    response = _post(c, cid, OPTIONS_SUB)

    assert response.status_code == 409
    assert response.json() == {"detail": "postprocess_canonical_conflict"}
    assert len(fake.calls) == calls_before


def test_publish_fsyncs_staged_directory_before_parent(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    staged = tmp_path / "private" / "01.png"
    canonical = tmp_path / "postprocessed" / "01.png"
    source.write_bytes(PNG)
    staged.parent.mkdir()
    staged.write_bytes(PNG)
    synced = []
    monkeypatch.setattr(postprocess, "_fsync_dir", lambda path: synced.append(path))

    postprocess._publish_segment([staged], [(source, canonical)])

    assert synced == [
        tmp_path / ".postprocessed.publishing",
        tmp_path,
    ]
    assert canonical.read_bytes() == PNG


# ---------- 并行提交：进程级信号量限流与失败语义 ----------

class SlowEdit:
    """慢速桩：记录 MediaKit 帧级并发峰值；按 fail 名单抛错。"""

    def __init__(self, fail=(), delay=0.05):
        self.fail = list(fail)
        self.delay = delay  # float（统一延时）或 {帧名: 秒}（打乱完成顺序）
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, settings, cdir, image, out, confirm, scenes):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append({"image": image.name, "scenes": scenes})
        try:
            if isinstance(self.delay, dict):
                await asyncio.sleep(self.delay.get(image.name, 0.0))
            else:
                await asyncio.sleep(self.delay)
            if image.name in self.fail:
                raise mediakit.MediaKitError(502, "stub failure")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(PNG + b"edited")
            return out
        finally:
            self.active -= 1


def _add_frames(settings, cid, names):
    for name in names:
        (settings.data_dir / cid / "work" / "keyframes" / name).write_bytes(PNG)


def test_parallel_edits_respect_process_semaphore(tmp_path, monkeypatch):
    """5 帧、进程并发上限 2：编辑并行提交且活跃数峰值恰为 2（信号量限流）。"""
    settings = make_settings(tmp_path, enable_mediakit_erase=True, mediakit_concurrency=2)
    cid = _make_conv(settings)
    _add_frames(settings, cid, ["03.png", "04.png", "05.png"])
    slow = SlowEdit(delay=0.05)
    monkeypatch.setattr(postprocess.mediakit, "erase_image", slow)

    with TestClient(create_app(settings)) as c:
        assert _post(c, cid, OPTIONS_SUB).status_code == 200

    assert slow.max_active == 2
    assert slow.max_active <= settings.mediakit_concurrency
    # 帧到达顺序不定（线程池并发读尺寸），断言按集合：每帧恰一次
    assert sorted(call["image"] for call in slow.calls) == \
        ["01.png", "02.png", "03.png", "04.png", "05.png"]
    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "done"
    assert pp["frames"] == ["01.png", "02.png", "03.png", "04.png", "05.png"]


def test_parallel_frame_failure_waits_for_rest(enabled, monkeypatch):
    """并发下任一帧失败 → 整体 failed（error 指明帧名），其余帧照常跑完、成功帧全保留；
    frames 终序为目标顺序，与完成顺序无关。"""
    settings, c = enabled
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    _add_frames(settings, cid, ["03.png"])
    # 打乱完成顺序：01 最慢、03 次之、02 最快且失败
    slow = SlowEdit(fail=["02.png"], delay={"01.png": 0.08, "02.png": 0.01, "03.png": 0.04})
    monkeypatch.setattr(postprocess.mediakit, "erase_image", slow)

    assert _post(c, cid, OPTIONS_SUB).status_code == 200

    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "failed"
    assert pp["error"] == "segment_failed"
    assert "02.png" in pp["segments"][0]["error"]
    assert pp["frames"] == []  # 段内有失败帧时不发布任何 canonical
    private = cdir / "work" / ".postprocess-private" / "0" / "text"
    assert (private / "01.png").is_file()
    assert (private / "03.png").is_file()
    assert not (private / "02.png").exists()


# ---------- 取消：父任务取消写 failed 终态 ----------

def test_run_task_cancelled_writes_failed(tmp_path, monkeypatch):
    """父任务被取消（uvicorn graceful shutdown）：CancelledError 是 BaseException，run_task 须在
    继续传播前把 meta.postprocess 写成 failed——否则永久 running、start 永久 409 拒重跑。"""
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings)
    asyncio.run(postprocess.start(
        settings, cid, {"confirm": True, "options": OPTIONS_SUB}, {}
    ))

    async def hang(*a, **k):
        await asyncio.Event().wait()  # 被取消时才结束的挂起桩

    monkeypatch.setattr(postprocess, "_mediakit_stage", hang)
    sem = asyncio.Semaphore(10)

    async def drive():
        task = asyncio.create_task(postprocess.run_task(settings, cid, sem))
        await asyncio.sleep(0.05)  # 让出至进入 gather
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    pp = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert pp["status"] == "failed"
    assert "cancelled" in pp["error"]
    assert pp["options"] == {**OPTIONS_SUB, "optimize_image": False}


def test_parallel_segment_updates_use_atomic_storage_mutation(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _make_conv(settings, segments=True)
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running", "options": {**OPTIONS_SUB, "optimize_image": False},
        "frames": [], "error": None,
        "segments": [
            postprocess._segment_state(1, 1),
            postprocess._segment_state(2, 2),
        ],
    })
    public_load_calls = 0

    def forbidden_stale_load(*_args):
        nonlocal public_load_calls
        public_load_calls += 1
        raise AssertionError("_update_segment must not perform a lock-outside load")

    monkeypatch.setattr(postprocess.storage, "load_meta", forbidden_stale_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(
            lambda args: postprocess._update_segment(
                settings, cid, args[0], stage=args[1], completed_frames=args[2]
            ),
            ((1, "brand", 1), (2, "seedream", 2)),
        ))
    current = storage._load_meta_unlocked(settings.data_dir, cid)["postprocess"]
    by_index = {item["index"]: item for item in current["segments"]}
    assert public_load_calls == 0
    assert (by_index[1]["stage"], by_index[1]["completed_frames"]) == ("brand", 1)
    assert (by_index[2]["stage"], by_index[2]["completed_frames"]) == ("seedream", 2)


def test_v4_receipts_project_real_frame_progress_one_through_nine(tmp_path):
    settings = make_settings(tmp_path)
    cid = _make_conv(settings)
    cdir = settings.data_dir / cid
    nodes = [
        {
            "scene_id": "scene-1",
            "label": f"fanout-0001-{frame_index:04d}",
            "anchor": {
                "order": frame_index,
                "segment_index": 0,
                "frame_index": frame_index,
                "frame_name": f"{frame_index:02d}.png",
            },
        }
        for frame_index in range(1, 10)
    ]
    private = {
        "plan_sha256": "a" * 64,
        "continuity_sha256": "b" * 64,
        "scene_anchor_schedule": {"nodes": nodes},
    }
    storage.update_meta(settings.data_dir, cid, postprocess={
        "status": "running", "frames": [], "error": None,
        "segments": [{
            "index": 0, "status": "running", "stage": "seedream",
            "completed_frames": 0, "total_frames": 9, "revision": 1,
            "error": None,
        }],
    })

    observed = []
    for node in nodes:
        payload = {
            "plan_sha256": private["plan_sha256"],
            "continuity_sha256": private["continuity_sha256"],
            "scene_id": node["scene_id"],
            "label": node["label"],
            "anchor": node["anchor"],
        }
        receipt = {**payload, "sha256": postprocess._receipt_sha256(payload)}
        postprocess._write_json_receipt(
            postprocess._anchor_receipt_path(cdir, node["scene_id"], node["label"]),
            receipt,
        )
        postprocess._record_v4_completed_frames(settings, cid, cdir, private)
        current = storage.load_meta(settings.data_dir, cid)["postprocess"]
        observed.append(current["segments"][0]["completed_frames"])

    assert observed == list(range(1, 10))
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["frames"] == []


def test_public_state_maps_untrusted_status_stage_and_error_to_safe_closed_values():
    public = postprocess.public_state({
        "status": "provider-secret-status", "options": OPTIONS_SUB,
        "frames": [], "error": "request_id=req-secret-123",
        "segments": [{
            "index": 1, "status": "remote-running", "stage": "task_id=secret",
            "completed_frames": 0, "total_frames": 1, "revision": 1,
            "error": "provider task_id=secret-456",
        }, {
            "index": 2, "status": "failed", "stage": "brand",
            "completed_frames": 0, "total_frames": 1, "revision": 2,
            "error": "frame 02.png failed: provider request req-secret",
        }],
    })
    assert public["status"] == "failed"
    assert public["error"] == "postprocess_failed"
    assert public["segments"][0] == {
        "index": 1, "status": "failed", "stage": "unknown",
        "completed_frames": 0, "total_frames": 1, "revision": 1,
        "error": "postprocess_failed",
    }
    assert public["segments"][1]["error"] == "frame 02.png failed"
    assert "secret" not in json.dumps(public)


@pytest.mark.parametrize("segments", [
    [{"index": 0, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None},
     {"index": 1, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None}],
    [{"index": 1, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None},
     {"index": 3, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None}],
    [{"index": 0, "status": "running", "stage": "queued", "completed_frames": 2,
      "total_frames": 1, "revision": 1, "error": None}],
    [{"index": True, "status": "running", "stage": "queued", "completed_frames": 0,
      "total_frames": 1, "revision": 1, "error": None}],
])
def test_public_state_fails_closed_for_invalid_segment_collection(segments):
    public = postprocess.public_state({
        "status": "running", "options": {**OPTIONS_SUB, "optimize_image": False},
        "frames": ["01.png"], "error": None, "segments": segments,
    })
    assert public["status"] == "failed"
    assert public["error"] == "postprocess_receipt_invalid"
    assert public["segments"] == []
    assert public["frames"] == []
