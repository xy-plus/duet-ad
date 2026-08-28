"""后处理编排：HTTP 门控、MediaKit 场景映射、失败保留和并发限流。"""

import asyncio
import base64
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
    plan, prompts = image_optimization.generic_project_prompts(
        [{
            "index": segment_index,
            "chain_id": (
                "short-000" if segment_index == 0 else "chain-001"
            ),
            "join_mode": "hard_cut" if segment_index in {0, 1} else "continue",
            "keyframes_dir": segment_frames[0].parent,
            "transition_skeleton": skeletons[segment_index],
        } for segment_index, segment_frames in grouped_frames.items()],
        settings.seedream_edit_mode,
        session_dir=cdir,
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
    frozen_single_segment=False, forbid_legacy_short=False,
    complete_generation=False, dialogue_mode="auto",
    silent_segment_indices=(), context_semantic_recovery=False,
    web_output_validation=False,
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
        segments=segment_count == 2 or frozen_single_segment,
        postprocess_options=postprocess_options,
        segment_total=segment_count,
    )
    (cdir / "source.mp4").write_bytes(b"source-video")
    old_visual_prompt = "九张已验收图片中的人物保持静默，歌声来自画外。"
    (cdir / "work" / "visual_prompt.txt").write_text(
        old_visual_prompt, encoding="utf-8"
    )
    subprocess.run(
        [
            "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", "14.58", "-q:a", "9", "-y",
            str(cdir / "work" / "voice.mp3"),
        ],
        check=True,
    )
    common_changes = dict(
        duration_s=14.5,
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
        voice_line_provenance=[{
            "text": "تجربة غنائية",
            "start_s": 0.0,
            "end_s": 14.44,
            "classification": "sung",
            "provenance": "asr",
            "kept": True,
        }],
    )
    expected_receipt = None
    has_frozen_segment_plan = segment_count == 2 or frozen_single_segment
    if has_frozen_segment_plan:
        public_segments = []
        receipt_segments = []
        for index in range(1, segment_count + 1):
            start_s = 14.0 * (index - 1)
            end_s = 14.0 * index
            segdir = cdir / "work" / "segments" / str(index)
            work = segdir / "work"
            source = segdir / "source.mp4"
            source.write_bytes(f"segment-{index}".encode())
            visual_text = f"第{index}段九张优化图片视觉动作"
            visual = work / "visual_prompt.txt"
            visual.write_text(visual_text, encoding="utf-8")
            dialogue = ([] if index in silent_segment_indices else [{
                "text": f"画外歌声{index}",
                "start_s": 1.0,
                "end_s": 2.0,
                "classification": "sung",
            }])
            final_text = (
                "不要生成背景音乐\n"
                + prepared_input.compose_final_prompt(
                    long_video.compose_segment_visual_prompt(visual_text),
                    dialogue,
                )
            )
            final = work / "prompt.txt"
            final.write_text(final_text, encoding="utf-8")
            (work / "voice.mp3").write_bytes(
                (cdir / "work" / "voice.mp3").read_bytes()
            )
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
            duration_s=14.0 * segment_count,
            segments=receipt_segments,
            workflow=h3.H3_WORKFLOW,
        )
        expected_receipt = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        common_changes.update(
            duration_s=14.0 * segment_count,
            segments=public_segments,
            long_video_plan_receipt=receipt_path.name,
            voice_line_provenance=[],
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

    def skill(self, scope, _prompt):
        fusion_calls.append(scope)
        input_data = (scope / "work" / "multimodal_input.json").read_bytes()
        input_payload = json.loads(input_data.decode("utf-8"))
        output_segments = []
        for segment in input_payload["segments"]:
            final_prompt = (
                "只使用九张已验收优化图片中的人物、场景与对象。\n"
                f"<AUDIO_CONTENT_JSON>{segment['audio_content']['lines_json']}"
                "</AUDIO_CONTENT_JSON>"
            )
            fusion_prompts.append(final_prompt)
            output_segments.append({
                "index": segment["index"], "final_prompt": final_prompt,
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

    monkeypatch.setattr(CodexRunner, "run", skill)
    context_sources = []
    context_effective = {}
    context_requests = []
    injected_semantic_failure = False

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
            if context_semantic_recovery and not injected_semantic_failure:
                raise httpx.ReadTimeout(
                    "ambiguous Context query", request=request,
                )
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
        nonlocal injected_semantic_failure
        with httpx.Client(
            transport=httpx.MockTransport(context_gateway)
        ) as context_client:
            result = real_optimize(frozen, client=context_client)
        if (
            context_semantic_recovery
            and not injected_semantic_failure
            and result.status == "query_unknown"
        ):
            attempt_path = (
                frozen.workdir / ".context-ir" / "attempts"
                / result.attempt_id / "attempt.json"
            )
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            attempt["status"] = "failed"
            attempt["error"] = "context_ir_semantic_mismatch"
            attempt_path.write_text(
                json.dumps(
                    attempt, ensure_ascii=False, sort_keys=True, indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            injected_semantic_failure = True
            return replace(
                result,
                status="failed",
                error_code="context_ir_semantic_mismatch",
            )
        return result

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
    with TestClient(create_app(settings)) as client:
        if dialogue_mode != "none":
            missing_delivery = client.post(
                f"/api/conversations/{cid}/submit",
                headers=AUTH,
                json={key: value for key, value in payload.items()
                      if key != "dialogue_delivery"},
            )
            assert missing_delivery.status_code == 409
            assert missing_delivery.json() == {
                "detail": "dialogue_delivery_required"
            }
            assert h3_requests == []
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
        first = client.post(
            f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
        )
        assert first.status_code == 409, first.json()
        assert first.json() == {"detail": "prompt_fusion_refresh_required"}
        assert h3_requests == []
        second = client.post(
            f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
        )
        final_submit = second
        if context_semantic_recovery:
            assert second.status_code == 202, second.json()
            assert h3_requests == []
            failed_meta = storage.load_meta(settings.data_dir, cid)
            failed_generation = failed_meta["generation"]
            failed_segment = failed_generation["segments"][0]
            assert failed_generation["status"] == "failed"
            assert failed_segment["error"] == "context_ir_semantic_mismatch"
            assert failed_segment["h3_attempt_id"] is None
            assert failed_segment["context_ir"]["provider_task_id"] == (
                "context-task-1"
            )
            assert failed_segment["context_ir"]["receipt_path"] is None

            monkeypatch.setattr(
                postprocess,
                "_v4_user_acceptance_matches",
                lambda *_args, **_kwargs: False,
            )
            with pytest.raises(
                postprocess.PostprocessError,
                match="postprocess_artifacts_invalid",
            ):
                postprocess.generation_keyframes(
                    cdir,
                    storage.load_meta(settings.data_dir, cid),
                    originals,
                    settings=settings,
                )

            baseline_generation = storage.load_meta(
                settings.data_dir, cid,
            )["generation"]
            drifted_payload = {**payload, "resolution": "480p"}
            params_drift = client.post(
                f"/api/conversations/{cid}/submit",
                headers=AUTH,
                json=drifted_payload,
            )
            assert params_drift.status_code == 409
            assert params_drift.json() == {
                "detail": "resume_parameters_changed"
            }

            for mutation in ("task", "attempt"):
                drifted_generation = json.loads(json.dumps(baseline_generation))
                if mutation == "task":
                    drifted_generation["segments"][0]["context_ir"][
                        "provider_task_id"
                    ] = "another-context-task"
                else:
                    drifted_generation["segments"][0]["context_ir"][
                        "attempt_id"
                    ] = "000002"
                storage.update_meta(
                    settings.data_dir, cid, generation=drifted_generation,
                )
                rejected = client.post(
                    f"/api/conversations/{cid}/submit",
                    headers=AUTH,
                    json=payload,
                )
                assert rejected.status_code == 409
                assert rejected.json() == {
                    "detail": "submission_outcome_unknown"
                }
                assert storage.load_meta(
                    settings.data_dir, cid,
                )["generation"]["status"] == "submission_unknown"
                storage.update_meta(
                    settings.data_dir, cid, generation=baseline_generation,
                )

            unrelated = json.loads(json.dumps(baseline_generation))
            unrelated["segments"][0]["error"] = "context_ir_provider_failed"
            storage.update_meta(
                settings.data_dir, cid, generation=unrelated,
            )
            real_freeze_plan = long_generation.freeze_plan
            monkeypatch.setattr(
                long_generation,
                "freeze_plan",
                lambda *_args, **_kwargs: pytest.fail(
                    "ordinary failed replay reached freeze_plan"
                ),
            )
            unrelated_replay = client.post(
                f"/api/conversations/{cid}/submit",
                headers=AUTH,
                json=payload,
            )
            assert unrelated_replay.status_code == 409
            assert unrelated_replay.json() == {
                "detail": "new client_request_id required"
            }
            monkeypatch.setattr(
                long_generation, "freeze_plan", real_freeze_plan
            )
            storage.update_meta(
                settings.data_dir, cid, generation=baseline_generation,
            )

            context_receipt = (
                cdir / "work" / "segments" / "1" / ".context-ir"
                / "attempts" / "000001" / "receipt.json"
            )
            context_receipt.write_text("{}\n", encoding="utf-8")
            receipt_drift = client.post(
                f"/api/conversations/{cid}/submit",
                headers=AUTH,
                json=payload,
            )
            assert receipt_drift.status_code == 409
            assert receipt_drift.json() == {
                "detail": "submission_outcome_unknown"
            }
            context_receipt.unlink()
            storage.update_meta(
                settings.data_dir, cid, generation=baseline_generation,
            )

            h3_root = cdir / "work" / "segments" / "1" / ".h3"
            h3_root.mkdir()
            h3_drift = client.post(
                f"/api/conversations/{cid}/submit",
                headers=AUTH,
                json=payload,
            )
            assert h3_drift.status_code == 409
            assert h3_drift.json() == {
                "detail": "submission_outcome_unknown"
            }
            h3_root.rmdir()
            storage.update_meta(
                settings.data_dir, cid, generation=baseline_generation,
            )

            plan_path = cdir / long_video.PLAN_RECEIPT_FILENAME
            plan_bytes = plan_path.read_bytes()
            plan_path.write_bytes(plan_bytes + b" ")
            plan_drift = client.post(
                f"/api/conversations/{cid}/submit",
                headers=AUTH,
                json=payload,
            )
            assert plan_drift.status_code == 409
            assert plan_drift.json() == {
                "detail": "submission_outcome_unknown"
            }
            plan_path.write_bytes(plan_bytes)
            storage.update_meta(
                settings.data_dir, cid, generation=baseline_generation,
            )

            assert h3_requests == []
            assert sum(
                method == "POST" and path == "/v2/h3_context_ir"
                for method, path in context_requests
            ) == 1
            recovery = client.post(
                f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
            )
            assert recovery.status_code == 202, recovery.json()
            assert h3_requests == []
            context_ready = storage.load_meta(
                settings.data_dir, cid,
            )["generation"]
            assert context_ready["status"] == "resume_required"
            assert context_ready["segments"][0]["status"] == "resume_required"
            assert context_ready["segments"][0]["error"] == "context_ir_ready"
            assert context_ready["segments"][0]["context_ir"]["status"] == (
                "succeeded"
            )
            final_submit = client.post(
                f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
            )
            assert final_submit.status_code == 202, final_submit.json()
        if web_output_validation:
            assert complete_generation is True
            detail = client.get(
                f"/api/conversations/{cid}", headers=AUTH
            )
            assert detail.status_code == 200
            assert detail.json()["has_video"] is True
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

    assert final_submit.status_code == 202, final_submit.json()
    assert fusion_calls == [cdir]
    assert len(h3_requests) == (segment_count if complete_generation else 1)
    for bound_request in h3_requests:
        assert bound_request.context_ir_required is True
        assert bound_request.context_ir_receipt_path is not None
        assert bound_request.context_ir_receipt_sha256 is not None
        h3._require_context_ir_receipt(bound_request)
    request = h3_requests[0]
    assert context_sources == fusion_prompts[:len(context_sources)]
    assert request.prompt == f"EFFECTIVE::{fusion_prompts[0]}"
    assert "<AUDIO_CONTENT_JSON>" in request.prompt
    assert "[AUDIO_CONTENT_JSON]" not in request.prompt
    assert request.on_screen_dialogue == ()
    assert len(request.keyframes) == 9
    first_has_audio = dialogue_mode != "none" and 1 not in silent_segment_indices
    assert len(request.reference_audios) == (1 if first_has_audio else 0)
    assert request.workflow == (
        h3.H3_MULTIMODAL_WORKFLOW if first_has_audio else h3.H3_WORKFLOW
    )
    if first_has_audio:
        assert request.reference_audios[0].data == (
            cdir / "work" / (
                "voice.mp3"
                if not has_frozen_segment_plan else "segments/1/work/voice.mp3"
            )
        ).read_bytes()
    assert len(json.loads(
        (cdir / "work" / "multimodal_input.json").read_text(encoding="utf-8")
    )["segments"]) == segment_count
    assert all(old_visual_prompt not in prompt for prompt in context_sources)
    if complete_generation:
        assert [request.prompt for request in h3_requests] == [
            f"EFFECTIVE::{prompt}" for prompt in fusion_prompts
        ]
        assert len(stitch_calls) == 1
        native_indices = {
            index for index in range(1, segment_count + 1)
            if dialogue_mode != "none" and index not in silent_segment_indices
        }
        assert stitch_calls[0]["audio_mode"] == (
            "provider_generated" if native_indices else "mute"
        )
        validation_passes = 1 + int(web_output_validation)
        assert len(reuse_calls) == validation_passes
        assert all(call[1] == dialogue_mode for call in reuse_calls)
        if not native_indices:
            assert all(call[3] is None for call in reuse_calls)
        else:
            assert all(
                set(call[3]) == native_indices for call in reuse_calls
            )
        assert len(timeline_calls) == len(native_indices) * validation_passes
        completed = storage.load_meta(settings.data_dir, cid)
        assert completed["generation"]["status"] == "succeeded"
        for state in completed["generation"]["segments"]:
            assert ("h3_attempt_id" in state) == (
                state["index"] in native_indices
            )
    if context_semantic_recovery:
        assert sum(
            method == "POST" and path == "/v2/h3_context_ir"
            for method, path in context_requests
        ) == 1
        assert sum(
            method == "GET"
            and path == "/v2/query/video_generation/context-task-1"
            for method, path in context_requests
        ) == 2
        recovered_context = storage.load_meta(
            settings.data_dir, cid,
        )["generation"]["segments"][0]["context_ir"]
        assert recovered_context["status"] == "succeeded"
        assert recovered_context["provider_task_id"] == "context-task-1"
        assert recovered_context["receipt_path"] is not None
    for index in range(1, segment_count + 1):
        segment_work = cdir / "work" / "segments" / str(index) / "work"
        assert not (segment_work / "multimodal_input.json").exists()
        assert not (segment_work / "h3_prompt_plan.json").exists()
        assert not (segment_work / "h3_multimodal_source.json").exists()
    if segment_count == 1 and postprocess_options is None:
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


def test_n2_off_screen_fusion_completes_context_h3_and_native_stitch(
    tmp_path, monkeypatch,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path, monkeypatch, 2, complete_generation=True,
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


def test_same_submit_recovers_long_context_semantic_mismatch_get_only(
    tmp_path, monkeypatch,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        1,
        complete_generation=True,
        context_semantic_recovery=True,
    )


@pytest.mark.parametrize("segment_count", [1, 2])
def test_n1_n2_none_fusion_enters_context_h3_without_audio(
    tmp_path, monkeypatch, segment_count,
):
    _assert_off_screen_fusion_bootstraps_then_enters_context_h3(
        tmp_path,
        monkeypatch,
        segment_count,
        complete_generation=True,
        dialogue_mode="none",
    )


def test_n2_mixed_fusion_uses_native_audio_then_mute_without_source_overlay(
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


def test_v4_postprocess_never_calls_quality_pack_gate(
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
    monkeypatch.setattr(postprocess, "_v4_verify_bootstrap_packs", forbidden_pack)
    monkeypatch.setattr(postprocess, "_v4_generate_layout_anchors", layout)
    monkeypatch.setattr(postprocess, "_v4_fan_out", fanout)

    asyncio.run(postprocess._run_v4_task(
        settings, "cid", cdir, {}, private, {}, asyncio.Semaphore(1),
    ))
    assert calls == ["global-anchor", "layout", "fanout"]


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
