import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from app import (
    context_ir_bridge,
    h3,
    h3_project,
    long_generation,
    long_video,
    main,
    pipeline,
    storage,
)
from conftest import make_settings


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _current_v4_meta(*, segments: list[dict] | None) -> dict:
    meta = {
        "schema_version": 2,
        "duration_s": 14.5 if segments is None else 28.0,
        "status": "done",
        "_postprocess_receipt": {
            "version": 4,
            "options": {"optimize_image": True},
        },
        "_image_user_acceptance": {"version": 1, "sha256": "a" * 64},
    }
    if segments is not None:
        meta["segments"] = segments
        meta["long_video_plan_receipt"] = "long-video-plan.json"
    return meta


def test_current_v4_n1_and_n2_use_the_same_segment_coordinator() -> None:
    single = _current_v4_meta(segments=None)
    multiple = _current_v4_meta(
        segments=[
            {"index": 1, "start_s": 0.0, "end_s": 14.0},
            {"index": 2, "start_s": 14.0, "end_s": 28.0},
        ]
    )

    assert main._uses_segment_coordinator(single) is True
    assert main._uses_segment_coordinator(multiple) is True


def test_prompt_fusion_output_is_the_only_prompt_authority(tmp_path: Path) -> None:
    old_prompt = "OLD_VISUAL_PROMPT_MUST_NOT_REACH_CONTEXT_OR_H3"
    final_visual = "FUSED_PROMPT_FROM_OPTIMIZED_IMAGES"
    for order in range(1, 10):
        (tmp_path / f"{order:02d}.png").write_bytes(f"frame-{order}".encode())
    fusion_input = {
        "schema": "duet.video-prompt-fusion-input",
        "version": 2,
        "segments": [{
            "index": 1,
            "new_keyframes": [
                    {
                        "order": order,
                        "path": f"{order:02d}.png",
                        "sha256": hashlib.sha256(f"frame-{order}".encode()).hexdigest(),
                        "segment_time_s": float(order - 1),
                        "source_scene_id": "SCENE_01",
                        "transition": (
                            {"type": "start", "at_segment_s": 0.0}
                            if order == 1 else
                            {"type": "continuous", "at_segment_s": None}
                        ),
                    }
                for order in range(1, 10)
            ],
            "old_video_prompt": {
                "text": old_prompt,
                "sha256": hashlib.sha256(old_prompt.encode("utf-8")).hexdigest(),
            },
            "image_optimization_prompt": [{
                "order": order,
                "text": "replace person and scene",
                "sha256": hashlib.sha256(b"replace person and scene").hexdigest(),
            } for order in range(1, 10)],
            "audio_content": {
                "lines_json": "[]",
                "voice_references": [],
                "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
                "music_policy": "forbid",
            },
        }],
    }
    input_data = _canonical(fusion_input)
    fusion_output = {
        "schema": "duet.video-prompt-fusion-output",
        "version": 2,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [
            _fusion_v2_output(fusion_input["segments"][0], final_visual)
        ],
    }
    input_path = tmp_path / "multimodal_input.json"
    output_path = tmp_path / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(_canonical(fusion_output))

    frozen = long_generation.load_prompt_fusion(
        input_path=input_path,
        output_path=output_path,
        root=tmp_path,
    )

    assert frozen.final_prompts == (_fusion_v2_final_prompt(
        fusion_input["segments"][0], final_visual,
    ),)
    assert old_prompt not in frozen.final_prompts


def test_prompt_fusion_v2_binds_exact_source_timeline_and_hard_cut(
    tmp_path: Path,
) -> None:
    frames = []
    timeline = []
    source_times = [0.0, 0.75, 2.0, 2.5, 4.0, 6.0, 8.0, 11.0, 14.0]
    for order, segment_time_s in enumerate(source_times, 1):
        data = f"frame-{order}".encode()
        path = tmp_path / f"{order:02d}.png"
        path.write_bytes(data)
        transition = (
            {"type": "start", "at_segment_s": 0.0}
            if order == 1 else
            {"type": "hard_cut", "at_segment_s": 2.267}
            if order == 4 else
            {"type": "continuous", "at_segment_s": None}
        )
        source = {
            "order": order,
            "segment_time_s": segment_time_s,
            "source_scene_id": "SCENE_01" if order < 4 else "SCENE_02",
            "transition": transition,
        }
        timeline.append(source)
        frames.append({
            "order": order,
            "path": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            **{key: source[key] for key in (
                "segment_time_s", "source_scene_id", "transition",
            )},
        })
    old_prompt = "source action order"
    input_payload = {
        "schema": long_generation.PROMPT_FUSION_INPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "segments": [{
            "index": 1,
            "new_keyframes": frames,
            "old_video_prompt": {
                "text": old_prompt,
                "sha256": hashlib.sha256(old_prompt.encode()).hexdigest(),
            },
            "image_optimization_prompt": [{
                "order": order,
                "text": "replace person and scene",
                "sha256": hashlib.sha256(
                    b"replace person and scene"
                ).hexdigest(),
            } for order in range(1, 10)],
            "audio_content": {
                "lines_json": "[]",
                "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
                "voice_references": [],
                "music_policy": "forbid",
            },
        }],
    }
    input_data = _canonical(input_payload)
    final_prompt = _fusion_v2_final_prompt(
        input_payload["segments"][0], "hard cut at the frozen source boundary",
    )
    input_path = tmp_path / "multimodal_input.json"
    output_path = tmp_path / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [_fusion_v2_output(
            input_payload["segments"][0], "hard cut at the frozen source boundary",
        )],
    }))

    frozen = long_generation.load_prompt_fusion(
        input_path=input_path,
        output_path=output_path,
        root=tmp_path,
    )

    assert frozen.final_prompts == (final_prompt,)
    assert frozen.segments[0]["new_keyframes"][3]["transition"] == {
        "type": "hard_cut", "at_segment_s": 2.267,
    }

    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{"index": 1, "visual": []}],
    }))
    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_output_invalid",
    ):
        long_generation.load_prompt_fusion(
            input_path=input_path,
            output_path=output_path,
            root=tmp_path,
        )


def test_prompt_fusion_v2_ignores_visual_hard_cut_drift(tmp_path: Path) -> None:
    frames = []
    timeline = []
    for order in range(1, 10):
        data = f"frame-{order}".encode()
        path = tmp_path / f"{order:02d}.png"
        path.write_bytes(data)
        transition = (
            {"type": "start", "at_segment_s": 0.0}
            if order == 1 else
            {"type": "hard_cut", "at_segment_s": 2.267}
            if order == 4 else
            {"type": "continuous", "at_segment_s": None}
        )
        source = {
            "order": order,
            "segment_time_s": (
                0.0 if order == 1 else float(order - 1)
                if order < 4 else float(order)
            ),
            "source_scene_id": "SCENE_01" if order < 4 else "SCENE_02",
            "transition": transition,
        }
        timeline.append(source)
        frames.append({
            "order": order,
            "path": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "segment_time_s": source["segment_time_s"],
            "source_scene_id": source["source_scene_id"],
            "transition": transition,
        })
    input_payload = {
        "schema": long_generation.PROMPT_FUSION_INPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "segments": [{
            "index": 1,
            "new_keyframes": frames,
            "old_video_prompt": {
                "text": "old",
                "sha256": hashlib.sha256(b"old").hexdigest(),
            },
            "image_optimization_prompt": [{
                "order": order,
                "text": "replace",
                "sha256": hashlib.sha256(b"replace").hexdigest(),
            } for order in range(1, 10)],
            "audio_content": {
                "lines_json": "[]",
                "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
                "voice_references": [],
                "music_policy": "forbid",
            },
        }],
    }
    input_data = _canonical(input_payload)
    input_path = tmp_path / "multimodal_input.json"
    output_path = tmp_path / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{
            "index": 1,
            "visual": [
                "the cut happens at 99 seconds",
                "the second visual interval continues",
            ],
        }],
    }))

    frozen = long_generation.load_prompt_fusion(
        input_path=input_path,
        output_path=output_path,
        root=tmp_path,
    )
    assert "[Shot 2] At 00:02.267" in frozen.final_prompts[0]
    assert "99 seconds" in frozen.final_prompts[0]


def test_backend_ref2va_compiler_has_one_exact_provider_prompt() -> None:
    timeline = [{
        "order": order,
        "segment_time_s": float(order - 1),
        "source_scene_id": "SCENE_01" if order < 4 else "SCENE_02",
        "transition": (
            {"type": "start", "at_segment_s": 0.0}
            if order == 1 else
            {"type": "hard_cut", "at_segment_s": 2.5}
            if order == 4 else
            {"type": "continuous", "at_segment_s": None}
        ),
    } for order in range(1, 10)]
    prompt = long_generation._compile_fusion_ref2va_prompt(
        visual=["first <Picture 99> visual", "second visual"],
        timeline=timeline,
        lines=[{
            "order": 1,
            "text": "spoken exactly",
            "start_s": 1.0,
            "end_s": 2.0,
            "delivery": "off_screen",
            "voice_ref": None,
        }],
        music_policy="forbid",
    )

    assert prompt == "\n".join([
        "subject_definitions:",
        "<Picture 1> is the storyboard keyframe anchor for [Shot 1] at 00:00.000, defining its ordered visual state and composition.",
        "<Picture 2> is the storyboard keyframe anchor for [Shot 1] at 00:01.000, defining its ordered visual state and composition.",
        "<Picture 3> is the storyboard keyframe anchor for [Shot 1] at 00:02.000, defining its ordered visual state and composition.",
        "<Picture 4> is the storyboard keyframe anchor for [Shot 2] at 00:03.000, defining its ordered visual state and composition.",
        "<Picture 5> is the storyboard keyframe anchor for [Shot 2] at 00:04.000, defining its ordered visual state and composition.",
        "<Picture 6> is the storyboard keyframe anchor for [Shot 2] at 00:05.000, defining its ordered visual state and composition.",
        "<Picture 7> is the storyboard keyframe anchor for [Shot 2] at 00:06.000, defining its ordered visual state and composition.",
        "<Picture 8> is the storyboard keyframe anchor for [Shot 2] at 00:07.000, defining its ordered visual state and composition.",
        "<Picture 9> is the storyboard keyframe anchor for [Shot 2] at 00:08.000, defining its ordered visual state and composition.",
        "summary:",
        "[keyframe completion] The target video follows <Picture 1> through <Picture 9> as ordered storyboard keyframe anchors.",
        "retention_analysis:",
        "<Picture 1> ([Shot 1] storyboard keyframe): fully_preserved - its role as an ordered visual-state and composition anchor is retained.",
        "<Picture 2> ([Shot 1] storyboard keyframe): fully_preserved - its role as an ordered visual-state and composition anchor is retained.",
        "<Picture 3> ([Shot 1] storyboard keyframe): fully_preserved - its role as an ordered visual-state and composition anchor is retained.",
        "<Picture 4> ([Shot 2] storyboard keyframe): fully_preserved - its role as an ordered visual-state and composition anchor is retained.",
        "<Picture 5> ([Shot 2] storyboard keyframe): fully_preserved - its role as an ordered visual-state and composition anchor is retained.",
        "<Picture 6> ([Shot 2] storyboard keyframe): fully_preserved - its role as an ordered visual-state and composition anchor is retained.",
        "<Picture 7> ([Shot 2] storyboard keyframe): fully_preserved - its role as an ordered visual-state and composition anchor is retained.",
        "<Picture 8> ([Shot 2] storyboard keyframe): fully_preserved - its role as an ordered visual-state and composition anchor is retained.",
        "<Picture 9> ([Shot 2] storyboard keyframe): fully_preserved - its role as an ordered visual-state and composition anchor is retained.",
        "detailed_description:",
        "[Shot 1] The shot follows the ordered storyboard anchors <Picture 1>, <Picture 2>, and <Picture 3>. first ‹Picture 99› visual",
        "From 00:01.000 to 00:02.000, the off-screen narrator (S1) says in an off-screen voiceover: <d>[Undetermined]spoken exactly</d> while every visible person's lips remain completely closed.",
        "[Shot 2] At 00:02.500, the shot cuts to <Picture 4>. The shot then follows the ordered storyboard anchors <Picture 4>, <Picture 5>, <Picture 6>, <Picture 7>, <Picture 8>, and <Picture 9>. second visual",
        "overall_soundscape:",
        "The frozen spoken events described above are the only specified audible layer; no additional ambience, physical-action sounds, or non-verbal human sounds are added.",
        "non_diegetic_music:",
        "N/A",
    ])
    assert "<Audio " not in prompt
    assert not prompt.endswith("\n")


def test_prompt_fusion_builder_copies_receipt_bound_source_timeline(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "work" / "segments" / "1"
    keyframe_dir = workdir / "work" / "keyframes"
    keyframe_dir.mkdir(parents=True)
    keyframes = []
    sources = []
    optimization_frames = []
    for order, source_time_s in enumerate(
        [0.0, 0.75, 2.0, 2.5, 4.0, 6.0, 8.0, 11.0, 14.0], 1
    ):
        path = keyframe_dir / f"{order:02d}.png"
        data = f"frame-{order}".encode()
        path.write_bytes(data)
        keyframes.append((path, data))
        sources.append({
            "order": order,
            "source_time_s": source_time_s,
            "source_scene_id": "SCENE_01" if order < 4 else "SCENE_02",
            "transition": (
                {"type": "start", "at_s": 0.0}
                if order == 1 else
                {"type": "hard_cut", "at_s": 2.267}
                if order == 4 else
                {"type": "continuous", "at_s": None}
            ),
        })
        current = f"optimized frame {order}"
        optimization_frames.append({
            "segment_index": 1,
            "frame_index": order,
            "current": current,
            "sha256": hashlib.sha256(current.encode()).hexdigest(),
        })
    anchor = keyframe_dir / "01.png"
    segment = long_generation.FrozenSegment(
        index=1,
        start_s=0.0,
        end_s=14.5,
        chain_id="chain-001",
        join_mode="hard_cut",
        workdir=workdir,
        first_frame=anchor,
        first_frame_data=keyframes[0][1],
        last_frame=keyframe_dir / "09.png",
        last_frame_data=keyframes[-1][1],
        prompt="final",
        keyframes=tuple(keyframes),
        keyframe_sources=tuple(sources),
        dialogue=(),
        dialogue_sha256=hashlib.sha256(b"[]\n").hexdigest(),
    )
    plan = long_generation.FrozenPlan(
        root=tmp_path,
        source=tmp_path / "source.mp4",
        receipt="a" * 64,
        segments=(segment,),
        receipt_version=long_video.VISUAL_PLAN_RECEIPT_VERSION,
    )
    meta = {
        "segments": [{"index": 1, "visual_prompt": "source actions"}],
        "_image_optimization": {"frames": optimization_frames},
    }

    payload = json.loads(long_generation.build_prompt_fusion_input(
        root=tmp_path,
        meta=meta,
        plan=plan,
        dialogue_mode="none",
        dialogue_delivery="auto",
    ))

    assert payload["version"] == long_generation.VISUAL_PROMPT_FUSION_VERSION
    assert payload["segments"][0]["new_keyframes"][3] == {
        "order": 4,
        "path": "work/segments/1/work/keyframes/04.png",
        "sha256": hashlib.sha256(b"frame-4").hexdigest(),
        "segment_time_s": 2.5,
        "source_scene_id": "SCENE_02",
        "transition": {"type": "hard_cut", "at_segment_s": 2.267},
    }
    assert payload["segments"][0]["audio_content"]["music_policy"] == "forbid"

    legacy_plan = replace(
        plan,
        receipt_version=long_video.PLAN_RECEIPT_VERSION,
        segments=(replace(segment, keyframe_sources=()),),
    )
    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_refresh_required",
    ):
        long_generation.build_prompt_fusion_input(
            root=tmp_path,
            meta=meta,
            plan=legacy_plan,
            dialogue_mode="none",
            dialogue_delivery="auto",
        )


@pytest.mark.parametrize("segment_count", [1, 2])
def test_real_source_binding_reaches_fusion_v2_and_context_contract(
    tmp_path: Path, segment_count: int,
) -> None:
    root = tmp_path / "project"
    work = root / "work"
    root.mkdir()
    (root / "source.mp4").write_bytes(b"source-video")
    segments = []
    metas = []
    local_times = [float(value) for value in range(9)]
    for segment_index in range(1, segment_count + 1):
        start_s = float((segment_index - 1) * 10)
        segwork = work / "segments" / str(segment_index) / "work"
        selected_dir = segwork / "keyframes"
        frames_dir = segwork / "frames"
        selected_dir.mkdir(parents=True)
        frames_dir.mkdir()
        names = []
        manifest_frames = []
        for order, local_time in enumerate(local_times, 1):
            data = f"segment-{segment_index}-source-{order}".encode()
            raw_name = f"frames/{order:03d}.png"
            (segwork / raw_name).write_bytes(data)
            name = f"{order:02d}.png"
            (selected_dir / name).write_bytes(data)
            names.append(name)
            manifest_frames.append({
                "index": order,
                "time_seconds": local_time,
                "file": raw_name,
            })
        (segwork / "manifest.json").write_text(
            json.dumps({"frames": manifest_frames}), encoding="utf-8"
        )
        segment = {
            "index": segment_index,
            "start_s": start_s,
            "end_s": start_s + 10.0,
            "chain_id": "chain-001",
            "join_mode": "hard_cut" if segment_index == 1 else "continue",
        }
        segments.append(segment)
        metas.append({**segment, "keyframes": names})

    cut_at_s = 2.267 if segment_count == 1 else 12.267
    bound = pipeline._bind_keyframe_source_timeline(
        work,
        segments,
        metas,
        [
            {"index": 1, "start_s": 0.0, "end_s": cut_at_s},
            {
                "index": 2,
                "start_s": cut_at_s,
                "end_s": float(segment_count * 10),
            },
        ],
    )

    optimization_frames = []
    frozen_segments = []
    for segment, meta in zip(segments, bound):
        selected_dir = (
            work / "segments" / str(segment["index"]) / "work" / "keyframes"
        )
        keyframes = tuple(
            (selected_dir / name, (selected_dir / name).read_bytes())
            for name in meta["keyframes"]
        )
        for order in range(1, 10):
            prompt = f"optimized segment {segment['index']} frame {order}"
            optimization_frames.append({
                "segment_index": segment["index"],
                "frame_index": order,
                "current": prompt,
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
        frozen_segments.append(long_generation.FrozenSegment(
            index=segment["index"],
            start_s=segment["start_s"],
            end_s=segment["end_s"],
            chain_id=segment["chain_id"],
            join_mode=segment["join_mode"],
            workdir=work / "segments" / str(segment["index"]),
            first_frame=keyframes[0][0],
            first_frame_data=keyframes[0][1],
            last_frame=keyframes[-1][0],
            last_frame_data=keyframes[-1][1],
            prompt="frozen visual",
            keyframes=keyframes,
            keyframe_sources=tuple(meta["keyframe_sources"]),
            dialogue=(),
            dialogue_sha256=hashlib.sha256(b"[]\n").hexdigest(),
        ))
    plan = long_generation.FrozenPlan(
        root=root,
        source=root / "source.mp4",
        receipt="a" * 64,
        segments=tuple(frozen_segments),
        receipt_version=long_video.VISUAL_PLAN_RECEIPT_VERSION,
    )
    meta = {
        "segments": [{
            "index": segment["index"],
            "visual_prompt": f"source action {segment['index']}",
        } for segment in segments],
        "_image_optimization": {"frames": optimization_frames},
    }
    input_data = long_generation.build_prompt_fusion_input(
        root=root,
        meta=meta,
        plan=plan,
        dialogue_mode="none",
        dialogue_delivery="auto",
    )
    input_payload = json.loads(input_data)
    for compiled_segment in input_payload["segments"]:
        first = compiled_segment["new_keyframes"][0]
        assert first["segment_time_s"] == 0.0
        assert first["transition"] == {
            "type": "start", "at_segment_s": 0.0,
        }
    assert "source_time_s" not in input_data.decode("utf-8")
    assert '"at_s"' not in input_data.decode("utf-8")
    input_path = work / h3_project.SKILL_INPUT_FILENAME
    output_path = work / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [
            _fusion_v2_output(segment, f"fused visual {segment['index']}")
            for segment in input_payload["segments"]
        ],
    }))
    frozen = long_generation.load_prompt_fusion(
        input_path=input_path, output_path=output_path, root=root,
    )

    dialogue_sha256 = "d" * 64
    artifact_path = work / "prepared_input.json"
    artifact_data = json.dumps(
        {"dialogue": {"sha256": dialogue_sha256}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    artifact_path.write_bytes(artifact_data)

    class NoHttp:
        def get(self, *_args, **_kwargs):
            raise AssertionError("current Fusion Context must perform zero HTTP")

        def post(self, *_args, **_kwargs):
            raise AssertionError("current Fusion Context must perform zero HTTP")

    for segment, prompt in zip(frozen_segments, frozen.final_prompts):
        request = h3.H3Request(
            cid=f"segment-{segment.index}",
            workdir=segment.workdir,
            client_request_id=f"context-{segment.index}",
            prompt=prompt,
            keyframes=segment.keyframes,
            voice_texts=(),
            voice_receipt=h3.voice_texts_receipt(()),
            duration=10,
            autodl_token="autodl-secret",
            workflow=h3.H3_WORKFLOW,
            skill_plan_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            upstream_dialogue_receipt_sha256=dialogue_sha256,
            context_ir_required=True,
        )
        context = context_ir_bridge.freeze_context_ir_request(
            source_h3_request=request,
            upstream_dialogue_sha256=dialogue_sha256,
            upstream_artifact_path=artifact_path,
            upstream_artifact_sha256=hashlib.sha256(artifact_data).hexdigest(),
            upstream_dialogue_sha256_path=("dialogue", "sha256"),
            source_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            minimax_api_key="",
        )
        assert context.keyframe_timeline_json is None
        result = context_ir_bridge.optimize_h3_prompt(context, client=NoHttp())
        assert result.status == "succeeded"
        assert result.effective_prompt == prompt
        assert result.provider_task_id == (
            "local:identity:" + hashlib.sha256(prompt.encode()).hexdigest()
        )

    hard_cuts = [
        frame["transition"]["at_segment_s"]
        for segment in input_payload["segments"]
        for frame in segment["new_keyframes"]
        if frame["transition"]["type"] == "hard_cut"
    ]
    assert hard_cuts == [2.267]


def _audio_compile_fixture(
    root: Path, *, segment_count: int, classification: str,
) -> tuple[dict, long_generation.FrozenPlan]:
    (root / "source.mp4").write_bytes(b"source")
    raw_segments = []
    frozen_segments = []
    optimization_frames = []
    for index in range(1, segment_count + 1):
        segment_work = root / "work" / "segments" / str(index) / "work"
        frames = []
        sources = []
        for order in range(1, 10):
            path = segment_work / "postprocessed" / f"{order:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = f"segment-{index}-frame-{order}".encode()
            path.write_bytes(data)
            frames.append((path, data))
            prompt = f"optimization {index}-{order}"
            optimization_frames.append({
                "segment_index": 0 if segment_count == 1 else index,
                "frame_index": order,
                "current": prompt,
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
            source_time_s = float((index - 1) * 8 + order - 1)
            sources.append({
                "order": order,
                "source_time_s": source_time_s,
                "source_scene_id": "SCENE_01",
                "transition": (
                    {"type": "start", "at_s": 0.0}
                    if index == order == 1 else
                    {"type": "continuous", "at_s": None}
                ),
            })
        line = {
            "text": f"line {index}",
            "start_s": 0.5,
            "end_s": 1.5,
            "classification": classification,
        }
        raw_segments.append({"index": index, "visual_prompt": f"visual {index}"})
        frozen_segments.append(long_generation.FrozenSegment(
            index=index,
            start_s=float(index - 1) * 8.0,
            end_s=float(index) * 8.0,
            chain_id="chain-001",
            join_mode="hard_cut" if index == 1 else "continue",
            workdir=root / "work" / "segments" / str(index),
            first_frame=frames[0][0],
            first_frame_data=frames[0][1],
            last_frame=frames[-1][0],
            last_frame_data=frames[-1][1],
            prompt=f"prompt {index}",
            keyframes=tuple(frames),
            keyframe_sources=tuple(sources),
            dialogue=(line,),
            dialogue_sha256=hashlib.sha256(_canonical([line])).hexdigest(),
        ))
    voice = root / "work" / "voice.mp3"
    voice.parent.mkdir(parents=True, exist_ok=True)
    voice.write_bytes(b"whole-source-mix")
    provenance = [{
        "text": "spoken authority",
        "start_s": 0.0,
        "end_s": 1.0,
        "classification": "spoken",
        "provenance": "asr",
        "kept": True,
    }]
    evidence = long_generation.classification_evidence_sha256(
        audio_path="work/voice.mp3",
        audio_sha256=hashlib.sha256(b"whole-source-mix").hexdigest(),
        has_bgm=False,
        decisions=provenance,
    )
    provenance = [{
        **line,
        "analysis_audio_path": "work/voice.mp3",
        "analysis_audio_sha256": hashlib.sha256(
            b"whole-source-mix"
        ).hexdigest(),
        "analysis_has_bgm": False,
        "classification_evidence_sha256": evidence,
    } for line in provenance]
    return ({
        "segments": raw_segments,
        "_image_optimization": {"frames": optimization_frames},
        "has_bgm": False,
        "voice_line_provenance": provenance,
    }, long_generation.FrozenPlan(
        root=root,
        source=root / "source.mp4",
        receipt="a" * 64,
        segments=tuple(frozen_segments),
        receipt_version=long_video.VISUAL_PLAN_RECEIPT_VERSION,
    ))


@pytest.mark.parametrize("segment_count", [1, 2], ids=("n1", "n2"))
@pytest.mark.parametrize("has_bgm", [False, True, None], ids=("no-bgm", "bgm", "unknown"))
def test_auto_sung_compiles_empty_audio_with_forbid_music_policy(
    tmp_path: Path, segment_count: int, has_bgm: bool | None,
) -> None:
    meta, plan = _audio_compile_fixture(
        tmp_path, segment_count=segment_count, classification="sung",
    )
    meta["has_bgm"] = has_bgm

    payload = json.loads(long_generation.build_prompt_fusion_input(
        root=tmp_path,
        meta=meta,
        plan=plan,
        dialogue_mode="auto",
        dialogue_delivery="off_screen",
    ))

    assert payload["version"] == 2
    assert [segment["audio_content"] for segment in payload["segments"]] == [
        {
            "lines_json": "[]",
            "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
            "voice_references": [],
            "music_policy": "forbid",
        }
        for _ in range(segment_count)
    ]


@pytest.mark.parametrize(
    "drift",
    ("audio", "path", "audio_sha256", "has_bgm", "decision", "evidence"),
)
def test_analysis_provenance_drift_never_selects_a_generation_chain(
    tmp_path: Path, drift: str,
) -> None:
    meta, plan = _audio_compile_fixture(
        tmp_path, segment_count=1, classification="spoken",
    )
    line = meta["voice_line_provenance"][0]
    if drift == "audio":
        (tmp_path / "work" / "voice.mp3").write_bytes(b"replaced mix")
    elif drift == "path":
        line["analysis_audio_path"] = "work/other.mp3"
    elif drift == "audio_sha256":
        line["analysis_audio_sha256"] = "0" * 64
    elif drift == "has_bgm":
        line["analysis_has_bgm"] = True
    elif drift == "decision":
        line["classification"] = "sung"
    else:
        line["classification_evidence_sha256"] = "0" * 64

    payload = json.loads(long_generation.build_prompt_fusion_input(
        root=tmp_path,
        meta=meta,
        plan=plan,
        dialogue_mode="auto",
        dialogue_delivery="off_screen",
    ))
    audio = payload["segments"][0]["audio_content"]
    assert audio["voice_references"] == []
    assert json.loads(audio["lines_json"])[0]["voice_ref"] is None


@pytest.mark.parametrize("segment_count", [1, 2], ids=("n1", "n2"))
@pytest.mark.parametrize("has_bgm", [False, True, None])
def test_spoken_lines_never_turn_source_audio_into_a_reference(
    tmp_path: Path, segment_count: int, has_bgm: bool | None,
) -> None:
    meta, plan = _audio_compile_fixture(
        tmp_path, segment_count=segment_count, classification="spoken",
    )
    meta["has_bgm"] = has_bgm

    payload = json.loads(long_generation.build_prompt_fusion_input(
        root=tmp_path,
        meta=meta,
        plan=plan,
        dialogue_mode="auto",
        dialogue_delivery="off_screen",
    ))

    for segment in payload["segments"]:
        audio = segment["audio_content"]
        assert audio["music_policy"] == "forbid"
        assert json.loads(audio["lines_json"])[0]["voice_ref"] is None
        assert audio["voice_references"] == []


def test_current_fusion_request_has_one_workflow_and_zero_audio_references(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(tmp_path), autodl_art_token="test-token")
    root = settings.data_dir / "single-audio-chain"
    root.mkdir(parents=True)
    meta, base = _audio_compile_fixture(
        root, segment_count=1, classification="spoken",
    )
    fusion_input = json.loads(long_generation.build_prompt_fusion_input(
        root=root,
        meta=meta,
        plan=base,
        dialogue_mode="auto",
        dialogue_delivery="off_screen",
    ))
    segment = replace(
        base.segments[0],
        prompt=_fusion_v2_final_prompt(
            fusion_input["segments"][0], "fused visual",
        ),
    )
    fusion = long_generation.FrozenPromptFusion(
        version=long_generation.PROMPT_FUSION_VERSION,
        input_path=root / "work" / "multimodal_input.json",
        input_data=b"input",
        input_sha256=hashlib.sha256(b"input").hexdigest(),
        output_path=root / "work" / "h3_prompt_plan.json",
        output_data=b"output",
        output_sha256=hashlib.sha256(b"output").hexdigest(),
        segments=(fusion_input["segments"][0],),
        final_prompts=(segment.prompt,),
    )
    plan = long_generation.FrozenPlan(
        **{
            **base.__dict__,
            "segments": (segment,),
            "workflow": h3.H3_WORKFLOW,
            "prompt_fusion": fusion,
        }
    )

    request = long_generation._request(
        settings,
        "single-audio-chain",
        plan,
        segment,
        "parent-request",
        "none",
    )

    assert request.workflow == h3.H3_WORKFLOW
    assert request.reference_audios == ()
    assert request.audio_required is False
    assert request.voice_texts == ("line 1",)
    assert request.context_ir_required is True


@pytest.mark.parametrize("dialogue_mode", ["edit", "custom"])
def test_manual_dialogue_is_not_reclassified_when_clean_authority_exists(
    tmp_path: Path, dialogue_mode: str,
) -> None:
    meta, plan = _audio_compile_fixture(
        tmp_path, segment_count=1, classification="sung",
    )

    payload = json.loads(long_generation.build_prompt_fusion_input(
        root=tmp_path,
        meta=meta,
        plan=plan,
        dialogue_mode=dialogue_mode,
        dialogue_delivery="off_screen",
    ))

    audio = payload["segments"][0]["audio_content"]
    assert json.loads(audio["lines_json"])[0]["text"] == "line 1"
    assert json.loads(audio["lines_json"])[0]["voice_ref"] is None
    assert audio["voice_references"] == []


def test_explicit_none_discards_spoken_even_when_bgm_is_unknown(
    tmp_path: Path,
) -> None:
    meta, plan = _audio_compile_fixture(
        tmp_path, segment_count=1, classification="spoken",
    )
    meta["has_bgm"] = None

    payload = json.loads(long_generation.build_prompt_fusion_input(
        root=tmp_path,
        meta=meta,
        plan=plan,
        dialogue_mode="none",
        dialogue_delivery="auto",
    ))

    audio = payload["segments"][0]["audio_content"]
    assert audio["lines_json"] == "[]"
    assert audio["voice_references"] == []
    assert audio["music_policy"] == "forbid"


_CID_9533_LINES_JSON = (
    '[{"order":1,"text":"تول جمالا بها القادم وقتك يا لدى عوصة اللوري تاب '
    'البدري يبريزها بجمال يا شعوري","start_s":0.0,"end_s":14.44,'
    '"delivery":"off_screen","voice_ref":1}]'
)


def _write_9533_fusion_fixture(
    root: Path, *, audio_envelope: str, lines_json: str = _CID_9533_LINES_JSON,
) -> tuple[Path, Path, str]:
    (root / "work").mkdir(parents=True, exist_ok=True)
    input_payload = json.loads(_fusion_input(
        root, 1, version=long_generation.PROMPT_FUSION_LEGACY_VERSION,
    ).decode("utf-8"))
    voice = root / "work" / "segments" / "1" / "work" / "voice.mp3"
    voice.parent.mkdir(parents=True, exist_ok=True)
    voice.write_bytes(b"9533-frozen-normalized-voice")
    input_payload["segments"][0]["audio_content"] = {
        "lines_json": lines_json,
        "lines_sha256": hashlib.sha256(
            lines_json.encode("utf-8")
        ).hexdigest(),
        "voice_references": [{
            "voice_ref": 1,
            "path": "work/segments/1/work/voice.mp3",
            "sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
            "purpose": "voice",
        }],
    }
    input_data = _canonical(input_payload)
    input_path = root / "work" / "multimodal_input.json"
    output_path = root / "work" / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    final_prompt = f"<VISUAL>9533 fused visual</VISUAL>\n{audio_envelope}"
    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": input_payload["version"],
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{"index": 1, "final_prompt": final_prompt}],
    }))
    return input_path, output_path, final_prompt


def test_9533_lf_audio_envelope_is_canonicalized_without_rewriting_json(
    tmp_path: Path,
) -> None:
    envelope = (
        "<AUDIO_CONTENT_JSON>\n"
        f"{_CID_9533_LINES_JSON}\n"
        "</AUDIO_CONTENT_JSON>"
    )
    input_path, output_path, _raw_prompt = _write_9533_fusion_fixture(
        tmp_path, audio_envelope=envelope,
    )

    frozen = long_generation.load_prompt_fusion(
        input_path=input_path, output_path=output_path, root=tmp_path,
    )

    canonical_block = (
        f"<AUDIO_CONTENT_JSON>{_CID_9533_LINES_JSON}"
        "</AUDIO_CONTENT_JSON>"
    )
    assert frozen.final_prompts == (
        f"<VISUAL>9533 fused visual</VISUAL>\n{canonical_block}",
    )
    assert frozen.output_data == output_path.read_bytes()


@pytest.mark.parametrize(
    "text",
    [
        "say <AUDIO_CONTENT_JSON> literally",
        "say </AUDIO_CONTENT_JSON> literally",
        "say <AUDIO_CONTENT_JSON> and </AUDIO_CONTENT_JSON> literally",
    ],
    ids=("opening-literal", "closing-literal", "both-literals"),
)
def test_audio_line_text_may_contain_outer_tag_literals(
    tmp_path: Path, text: str,
) -> None:
    lines_json = json.dumps(
        [{
            "order": 1,
            "text": text,
            "start_s": 0.0,
            "end_s": 14.44,
            "delivery": "off_screen",
            "voice_ref": 1,
        }],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    canonical_block = (
        f"<AUDIO_CONTENT_JSON>{lines_json}</AUDIO_CONTENT_JSON>"
    )
    input_path, output_path, _raw_prompt = _write_9533_fusion_fixture(
        tmp_path,
        audio_envelope=canonical_block,
        lines_json=lines_json,
    )

    frozen = long_generation.load_prompt_fusion(
        input_path=input_path, output_path=output_path, root=tmp_path,
    )

    assert frozen.final_prompts == (
        f"<VISUAL>9533 fused visual</VISUAL>\n{canonical_block}",
    )


def test_prompt_fusion_rejects_outer_tag_in_visual_prefix(
    tmp_path: Path,
) -> None:
    canonical_block = (
        f"<AUDIO_CONTENT_JSON>{_CID_9533_LINES_JSON}"
        "</AUDIO_CONTENT_JSON>"
    )
    input_path, output_path, _raw_prompt = _write_9533_fusion_fixture(
        tmp_path,
        audio_envelope=(
            "<AUDIO_CONTENT_JSON>visual-prefix-injection</AUDIO_CONTENT_JSON>"
            + canonical_block
        ),
    )

    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_output_invalid",
    ):
        long_generation.load_prompt_fusion(
            input_path=input_path, output_path=output_path, root=tmp_path,
        )


@pytest.mark.parametrize(
    "audio_envelope",
    [
        (
            "<AUDIO_CONTENT_JSON> " + _CID_9533_LINES_JSON
            + " </AUDIO_CONTENT_JSON>"
        ),
        (
            "<AUDIO_CONTENT_JSON>\r\n" + _CID_9533_LINES_JSON
            + "\r\n</AUDIO_CONTENT_JSON>"
        ),
        (
            "<AUDIO_CONTENT_JSON>"
            + _CID_9533_LINES_JSON.replace("off_screen", "on_screen")
            + "</AUDIO_CONTENT_JSON>"
        ),
        (
            "<AUDIO_CONTENT_JSON>" + _CID_9533_LINES_JSON
            + "</AUDIO_CONTENT_JSON><AUDIO_CONTENT_JSON>"
            + _CID_9533_LINES_JSON + "</AUDIO_CONTENT_JSON>"
        ),
    ],
    ids=("space", "crlf", "json-rewrite", "duplicate-block"),
)
def test_prompt_fusion_rejects_noncanonical_audio_envelopes(
    tmp_path: Path, audio_envelope: str,
) -> None:
    input_path, output_path, _raw_prompt = _write_9533_fusion_fixture(
        tmp_path, audio_envelope=audio_envelope,
    )

    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_output_invalid",
    ):
        long_generation.load_prompt_fusion(
            input_path=input_path, output_path=output_path, root=tmp_path,
        )


def test_binding_skill_production_seams_are_absent() -> None:
    assert not hasattr(pipeline, "queue_multimodal_binding")
    assert not hasattr(pipeline, "produce_multimodal_binding")
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "queue_multimodal_binding" not in source
    assert "produce_multimodal_binding" not in source


def _fusion_input(
    root: Path, segment_count: int, *,
    version: int = long_generation.PROMPT_FUSION_VERSION,
) -> bytes:
    segments = []
    for index in range(1, segment_count + 1):
        frames = []
        prompts = []
        for order in range(1, 10):
            relative = f"work/segment-{index}-{order:02d}.png"
            data = f"segment-{index}-frame-{order}".encode()
            (root / relative).write_bytes(data)
            prompt = f"segment {index} optimized image {order}"
            segment_time_s = float(order - 1)
            frame = {
                "order": order,
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            if version == long_generation.PROMPT_FUSION_VERSION:
                frame.update({
                    "segment_time_s": segment_time_s,
                    "source_scene_id": "SCENE_01",
                    "transition": (
                    {"type": "start", "at_segment_s": 0.0}
                    if order == 1 else
                    {"type": "continuous", "at_segment_s": None}
                    ),
                })
            frames.append(frame)
            prompts.append({
                "order": order,
                "text": prompt,
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            })
        old_prompt = f"old visual prompt {index}"
        audio_content = {
            "lines_json": "[]",
            "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
            "voice_references": [],
        }
        if version == long_generation.PROMPT_FUSION_VERSION:
            audio_content["music_policy"] = "forbid"
        segments.append({
            "index": index,
            "new_keyframes": frames,
            "old_video_prompt": {
                "text": old_prompt,
                "sha256": hashlib.sha256(
                    old_prompt.encode("utf-8")
                ).hexdigest(),
            },
            "image_optimization_prompt": prompts,
            "audio_content": audio_content,
        })
    return _canonical({
        "schema": long_generation.PROMPT_FUSION_INPUT_SCHEMA,
        "version": version,
        "segments": segments,
    })


def _fusion_v1_input(root: Path, segment_count: int) -> bytes:
    """Build an exact historical artifact for load/recovery-only coverage."""
    payload = json.loads(_fusion_input(
        root,
        segment_count,
        version=long_generation.PROMPT_FUSION_LEGACY_VERSION,
    ))
    for segment in payload["segments"]:
        for frame in segment["new_keyframes"]:
            frame.pop("segment_time_s", None)
            frame.pop("source_scene_id", None)
            frame.pop("transition", None)
        segment["audio_content"].pop("music_policy", None)
    return _canonical(payload)


def _fusion_v2_visual(segment: dict, visual: str) -> list[str]:
    shot_count = 1 + sum(
        frame["transition"]["type"] == "hard_cut"
        for frame in segment["new_keyframes"][1:]
    )
    return [f"{visual} shot {index}" for index in range(1, shot_count + 1)]


def _fusion_v2_output(segment: dict, visual: str) -> dict:
    return {
        "index": segment["index"],
        "visual": _fusion_v2_visual(segment, visual),
    }


def _fusion_v2_final_prompt(segment: dict, visual: str) -> str:
    timeline = [{
        "order": frame["order"],
        "segment_time_s": frame["segment_time_s"],
        "source_scene_id": frame["source_scene_id"],
        "transition": {
            "type": frame["transition"]["type"],
            "at_segment_s": frame["transition"]["at_segment_s"],
        },
    } for frame in segment["new_keyframes"]]
    return long_generation._compile_fusion_ref2va_prompt(
        visual=_fusion_v2_visual(segment, visual),
        timeline=timeline,
        lines=json.loads(segment["audio_content"]["lines_json"]),
        music_policy=segment["audio_content"]["music_policy"],
    )


@pytest.mark.parametrize("segment_count", [1, 2])
def test_project_prompt_fusion_runs_once_and_publishes_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, segment_count: int,
) -> None:
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "fusion", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    acceptance_sha256 = "a" * 64
    storage.update_meta(
        settings.data_dir,
        cid,
        _image_user_acceptance={
            "version": 1,
            "sha256": acceptance_sha256,
        },
    )
    input_data = _fusion_input(root, segment_count)
    (root / "work" / "unrelated-project-secret.txt").write_text(
        "must stay outside the fusion stage", encoding="utf-8",
    )
    if segment_count == 1:
        payload = json.loads(input_data)
        lines_json = (
            '[{"order":1,"text":"spoken","start_s":0.0,"end_s":1.0,'
            '"delivery":"off_screen","voice_ref":null}]'
        )
        payload["segments"][0]["audio_content"] = {
            "lines_json": lines_json,
            "lines_sha256": hashlib.sha256(lines_json.encode()).hexdigest(),
            "voice_references": [],
            "music_policy": "forbid",
        }
        input_data = _canonical(payload)
    skill_path = tmp_path / "video-prompt-fusion-SKILL.md"
    skill_path.write_text("strict project prompt fusion", encoding="utf-8")
    monkeypatch.setattr(pipeline, "PROMPT_FUSION_SKILL_MD", skill_path)
    monkeypatch.setattr(
        long_generation, "PROMPT_FUSION_SKILL_SOURCE", skill_path
    )
    calls: list[Path] = []

    class Runner:
        def run(self, _cwd: Path, _prompt: str) -> None:
            raise AssertionError("prompt fusion must not see the project root")

        def run_isolated(
            self, cwd: Path, prompt: str, *, session_dir: Path,
            writable_paths: tuple[Path, ...],
        ) -> None:
            calls.append(session_dir)
            assert session_dir == root
            assert cwd != root
            assert "multimodal_input.json" in prompt
            output_path = cwd / "work" / "h3_prompt_plan.json"
            assert writable_paths == (output_path,)
            assert output_path.read_bytes() == b""
            frozen_input = (cwd / "work" / "multimodal_input.json").read_bytes()
            payload = json.loads(frozen_input.decode("utf-8"))
            expected_files = {
                "SKILL.md",
                "work/multimodal_input.json",
                "work/h3_prompt_plan.json",
                *(
                    frame["path"]
                    for segment in payload["segments"]
                    for frame in segment["new_keyframes"]
                ),
            }
            assert {
                path.relative_to(cwd).as_posix()
                for path in cwd.rglob("*")
                if path.is_file()
            } == expected_files
            output = {
                "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
                "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
                "input_sha256": hashlib.sha256(frozen_input).hexdigest(),
                "segments": [
                    _fusion_v2_output(item, f"fused prompt {item['index']}")
                    for item in payload["segments"]
                ],
            }
            output_path.write_bytes(_canonical(output))

    if segment_count == 1:
        legacy = json.loads(input_data)
        legacy["version"] = long_generation.PROMPT_FUSION_LEGACY_VERSION
        with pytest.raises(
            pipeline.PipelineError,
            match="prompt fusion input is invalid",
        ):
            pipeline.queue_prompt_fusion(
                settings,
                cid,
                input_data=_canonical(legacy),
                image_acceptance_sha256=acceptance_sha256,
            )

    assert pipeline.queue_prompt_fusion(
        settings,
        cid,
        input_data=input_data,
        image_acceptance_sha256=acceptance_sha256,
    ) == "queued"
    produced = pipeline.produce_prompt_fusion(settings, cid, Runner())
    assert produced == "done", storage.load_meta(
        settings.data_dir, cid
    )["_prompt_fusion"]
    assert calls == [root]
    frozen = long_generation.load_prompt_fusion_manifest(
        root=root,
        skill_source_path=skill_path,
    )
    assert frozen.final_prompts == tuple(
        _fusion_v2_final_prompt(
            json.loads(input_data)["segments"][index - 1],
            f"fused prompt {index}",
        )
        for index in range(1, segment_count + 1)
    )
    assert pipeline.queue_prompt_fusion(
        settings,
        cid,
        input_data=input_data,
        image_acceptance_sha256=acceptance_sha256,
    ) == "done"
    assert pipeline.produce_prompt_fusion(settings, cid, Runner()) == "done"
    assert calls == [root]
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta is not None
    assert meta["_prompt_fusion"]["raw_output_path"] == (
        "work/h3_prompt_plan.json"
    )
    assert meta["_prompt_fusion"]["raw_output_sha256"] == hashlib.sha256(
        (root / "work" / "h3_prompt_plan.json").read_bytes()
    ).hexdigest()
    assert main._public_prompt_fusion(meta, root) == {
        "status": "done",
        "error": None,
        "segments": [{
            "index": index,
            "status": "done",
            "final_prompt": _fusion_v2_final_prompt(
                json.loads(input_data)["segments"][index - 1],
                f"fused prompt {index}",
            ),
            "error": None,
        } for index in range(1, segment_count + 1)],
    }


def test_prompt_fusion_runner_refuses_historical_v1_queue(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "fusion-v1-runner", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    input_data = _fusion_v1_input(root, 1)
    input_path = root / "work" / h3_project.SKILL_INPUT_FILENAME
    input_path.write_bytes(input_data)
    storage.update_meta(
        settings.data_dir,
        cid,
        _prompt_fusion={
            "version": 1,
            "status": "queued",
            "error": None,
            "input_sha256": hashlib.sha256(input_data).hexdigest(),
            "image_acceptance_sha256": "a" * 64,
            "manifest_sha256": None,
        },
    )

    class Runner:
        def run(self, _cwd: Path, _prompt: str) -> None:
            raise AssertionError("v1 must not reach the prompt fusion runner")

    assert pipeline.produce_prompt_fusion(settings, cid, Runner()) == "failed"
    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"][
        "error"
    ] == "prompt fusion input is invalid"


def test_new_prompt_fusion_queue_rejects_legacy_v1_input(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "legacy-fusion-input", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    legacy = _fusion_input(
        root, 1, version=long_generation.PROMPT_FUSION_LEGACY_VERSION,
    )

    with pytest.raises(pipeline.PipelineError, match="input is invalid"):
        pipeline.queue_prompt_fusion(
            settings,
            cid,
            input_data=legacy,
            image_acceptance_sha256="a" * 64,
        )

    assert storage.load_meta(settings.data_dir, cid).get("_prompt_fusion") is None


def test_legacy_v1_fusion_cannot_build_a_new_h3_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(make_settings(tmp_path), autodl_art_token="test-token")
    root = settings.data_dir / "legacy-v1-paid-gate"
    workdir = root / "work" / "segments" / "1"
    frame = workdir / "work" / "frame.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"frame")
    source = root / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    input_path = root / "work" / "multimodal_input.json"
    output_path = root / "work" / "h3_prompt_plan.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"legacy-input")
    output_path.write_bytes(b"legacy-output")
    voice = root / "work" / "voice.wav"
    voice.write_bytes(b"receipt-bound-legacy-voice")
    fusion = long_generation.FrozenPromptFusion(
        version=long_generation.PROMPT_FUSION_LEGACY_VERSION,
        input_path=input_path,
        input_data=b"legacy-input",
        input_sha256=hashlib.sha256(b"legacy-input").hexdigest(),
        output_path=output_path,
        output_data=b"legacy-output",
        output_sha256=hashlib.sha256(b"legacy-output").hexdigest(),
        segments=({
            "audio_content": {
                "voice_references": [{
                    "voice_ref": 1,
                    "path": "work/voice.wav",
                    "sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
                    "purpose": "voice",
                }],
            },
        },),
        final_prompts=("legacy prompt",),
    )
    segment = long_generation.FrozenSegment(
        index=1,
        start_s=0.0,
        end_s=8.0,
        chain_id="chain-001",
        join_mode="hard_cut",
        workdir=workdir,
        first_frame=frame,
        first_frame_data=b"frame",
        last_frame=frame,
        last_frame_data=b"frame",
        prompt="legacy prompt",
        keyframes=((frame, b"frame"),),
        dialogue=(),
        dialogue_sha256=hashlib.sha256(b"[]").hexdigest(),
    )
    plan = long_generation.FrozenPlan(
        root=root,
        source=source,
        receipt="a" * 64,
        segments=(segment,),
        receipt_version=long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
        workflow=h3.H3_WORKFLOW,
        prompt_fusion=fusion,
    )

    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_v2_refresh_required",
    ):
        long_generation._request(
            settings,
            "legacy-v1-paid-gate",
            plan,
            segment,
            "legacy-parent-request",
            "none",
            context_ir_binding=None,
        )

    monkeypatch.setattr(h3.storage, "probe_audio", lambda _path: True)
    monkeypatch.setattr(h3.voice, "probe_audio_duration", lambda _path: 4.0)
    historical = long_generation._request(
        settings,
        "legacy-v1-paid-gate",
        plan,
        segment,
        "legacy-parent-request",
        "none",
        context_ir_binding=None,
        legacy_terminal_read=True,
    )

    assert historical.workflow == h3.H3_MULTIMODAL_WORKFLOW
    assert historical.multimodal_compiler_version == "video-prompt-fusion-v1"
    assert historical.audio_required is True
    assert len(historical.reference_audios) == 1
    assert historical.reference_audios[0].data == voice.read_bytes()


def test_completed_v1_fusion_remains_local_read_only_without_h3_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    cid = "f" * 32
    root = settings.data_dir / cid
    root.mkdir(parents=True)
    _meta, base = _audio_compile_fixture(
        root, segment_count=1, classification="sung",
    )
    fusion = long_generation.FrozenPromptFusion(
        version=long_generation.PROMPT_FUSION_LEGACY_VERSION,
        input_path=root / "work" / "multimodal_input.json",
        input_data=b"legacy-input",
        input_sha256=hashlib.sha256(b"legacy-input").hexdigest(),
        output_path=root / "work" / "h3_prompt_plan.json",
        output_data=b"legacy-output",
        output_sha256=hashlib.sha256(b"legacy-output").hexdigest(),
        segments=({},),
        final_prompts=(base.segments[0].prompt,),
    )
    plan = long_generation.FrozenPlan(
        **{
            **base.__dict__,
            "prompt_fusion": fusion,
            "receipt_version": long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
        }
    )
    meta = {
        "id": cid,
        "segments": [{"index": 1}],
        "frozen_plan_receipt": "a" * 64,
        "fit_mode": "none",
        "dialogue_mode": "none",
        "generation": {"status": "succeeded", "segments": []},
    }
    monkeypatch.setattr(main, "_uses_segment_coordinator", lambda _meta: True)
    monkeypatch.setattr(
        main, "_long_receipt_multimodal_intent", lambda *_args: True,
    )
    monkeypatch.setattr(
        long_generation, "generation_segments_are_valid", lambda *_args: True,
    )
    monkeypatch.setattr(long_generation, "freeze_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(
        long_generation,
        "bound_reusable_segment_indices",
        lambda *_args: pytest.fail("historical read must not inspect H3"),
    )
    monkeypatch.setattr(
        long_generation, "stitched_output_is_reusable", lambda *_a, **_k: True,
    )

    assert main._validate_generated_video_uncached(settings, meta) is True


def _failed_lf_prompt_fusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: int = long_generation.PROMPT_FUSION_LEGACY_VERSION,
) -> tuple[object, str, Path, Path, Path]:
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "fusion-receipt-recovery", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    acceptance_sha256 = "a" * 64
    storage.update_meta(
        settings.data_dir,
        cid,
        _image_user_acceptance={
            "version": 1,
            "sha256": acceptance_sha256,
        },
    )
    skill_path = tmp_path / "video-prompt-fusion-SKILL.md"
    skill_path.write_text("strict frozen prompt fusion", encoding="utf-8")
    monkeypatch.setattr(pipeline, "PROMPT_FUSION_SKILL_MD", skill_path)
    monkeypatch.setattr(
        long_generation, "PROMPT_FUSION_SKILL_SOURCE", skill_path
    )
    input_data = _fusion_input(
        root, 1, version=version,
    )
    input_path = root / "work" / h3_project.SKILL_INPUT_FILENAME
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(input_data)
    storage.update_meta(
        settings.data_dir,
        cid,
        _prompt_fusion={
            "version": version,
            "status": "queued",
            "error": None,
            "input_sha256": hashlib.sha256(input_data).hexdigest(),
            "image_acceptance_sha256": acceptance_sha256,
            "manifest_sha256": None,
        },
    )
    frozen_skill = root / "work" / pipeline.PROMPT_FUSION_FROZEN_SKILL_FILENAME
    frozen_skill.write_bytes(skill_path.read_bytes())
    output = root / "work" / "h3_prompt_plan.json"
    if version == long_generation.PROMPT_FUSION_VERSION:
        output_segment = _fusion_v2_output(
            json.loads(input_data)["segments"][0], "fused",
        )
    else:
        output_segment = {
            "index": 1,
            "final_prompt": (
                "<VISUAL>fused</VISUAL>\n"
                "<AUDIO_CONTENT_JSON>\n[]\n</AUDIO_CONTENT_JSON>"
            ),
        }
    output.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": version,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [output_segment],
    }))
    storage.update_meta(
        settings.data_dir,
        cid,
        _prompt_fusion={
            "version": version,
            "status": "failed",
            "error": "prompt_fusion_output_invalid",
            "input_sha256": hashlib.sha256(input_data).hexdigest(),
            "image_acceptance_sha256": acceptance_sha256,
            "manifest_sha256": None,
        },
    )
    return settings, cid, root, skill_path, output


def _commit_historical_v1_prompt_fusion(
    settings, cid: str, root: Path, output: Path,
) -> dict:
    """Represent a v1 done receipt written by an older accepted release."""
    meta = storage.load_meta(settings.data_dir, cid)
    state = meta["_prompt_fusion"]
    _frozen, manifest_data = pipeline._publish_prompt_fusion_manifest(
        root=root,
        meta=meta,
        state=state,
    )
    raw_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    committed_state = {
        **state,
        "status": "done",
        "error": None,
        "raw_output_path": "work/h3_prompt_plan.json",
        "raw_output_sha256": raw_output_sha256,
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
    }
    storage.update_meta(
        settings.data_dir,
        cid,
        _prompt_fusion=committed_state,
    )
    return storage.load_meta(settings.data_dir, cid)


def test_failed_v1_output_cannot_publish_a_new_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, root, _skill_path, output = _failed_lf_prompt_fusion(
        tmp_path, monkeypatch,
    )
    raw_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    with pytest.raises(
        pipeline.PipelineError,
        match="receipt finalization is not allowed",
    ):
        pipeline.finalize_prompt_fusion_receipt(
            settings,
            cid,
            expected_raw_output_sha256=raw_output_sha256,
        )

    assert not (root / "work" / h3_project.SOURCE_FILENAME).exists()
    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"][
        "status"
    ] == "failed"


def test_done_receipt_survives_current_skill_source_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, root, skill_path, output = _failed_lf_prompt_fusion(
        tmp_path, monkeypatch,
    )
    committed = _commit_historical_v1_prompt_fusion(
        settings, cid, root, output,
    )

    skill_path.write_text("upgraded current prompt fusion", encoding="utf-8")

    assert long_generation.load_bound_prompt_fusion_manifest(
        root=root,
        meta=committed,
    ).final_prompts == (
        "<VISUAL>fused</VISUAL>\n"
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>",
    )
    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_manifest_invalid",
    ):
        long_generation.load_prompt_fusion_manifest(
            root=root,
            skill_source_path=skill_path,
        )


def test_done_receipt_rejects_frozen_skill_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, root, _skill_path, output = _failed_lf_prompt_fusion(
        tmp_path, monkeypatch,
    )
    committed = _commit_historical_v1_prompt_fusion(
        settings, cid, root, output,
    )
    (root / "work" / pipeline.PROMPT_FUSION_FROZEN_SKILL_FILENAME).write_bytes(
        b"tampered frozen prompt fusion"
    )

    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_manifest_invalid",
    ):
        long_generation.load_bound_prompt_fusion_manifest(
            root=root,
            meta=committed,
        )


@pytest.mark.parametrize(
    "expected_raw_output_sha256",
    [None, "0" * 64],
    ids=("missing", "wrong"),
)
def test_legacy_receipt_finalization_requires_exact_operator_audited_raw_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_raw_output_sha256: str | None,
) -> None:
    settings, cid, root, _skill_path, _output = _failed_lf_prompt_fusion(
        tmp_path, monkeypatch,
    )

    with pytest.raises(pipeline.PipelineError):
        pipeline.finalize_prompt_fusion_receipt(
            settings,
            cid,
            expected_raw_output_sha256=expected_raw_output_sha256,
        )

    assert not (root / "work" / h3_project.SOURCE_FILENAME).exists()
    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"][
        "status"
    ] == "failed"


def test_producer_failure_binds_raw_sha_and_rejects_schema_valid_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "fusion-raw-binding", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    acceptance_sha256 = "a" * 64
    storage.update_meta(
        settings.data_dir,
        cid,
        _image_user_acceptance={"version": 1, "sha256": acceptance_sha256},
    )
    skill_path = tmp_path / "video-prompt-fusion-SKILL.md"
    skill_path.write_text("strict frozen prompt fusion", encoding="utf-8")
    monkeypatch.setattr(pipeline, "PROMPT_FUSION_SKILL_MD", skill_path)
    monkeypatch.setattr(
        long_generation, "PROMPT_FUSION_SKILL_SOURCE", skill_path
    )
    input_data = _fusion_input(root, 1)
    assert pipeline.queue_prompt_fusion(
        settings,
        cid,
        input_data=input_data,
        image_acceptance_sha256=acceptance_sha256,
        ) == "queued"

    class Runner:
        def run_isolated(
            self, cwd: Path, _prompt: str, *, session_dir: Path,
            writable_paths: tuple[Path, ...],
        ) -> None:
            assert session_dir == root
            assert writable_paths == (
                cwd / "work" / "h3_prompt_plan.json",
            )
            (cwd / "work" / "h3_prompt_plan.json").write_bytes(_canonical({
                "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
                "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
                "input_sha256": hashlib.sha256(input_data).hexdigest(),
                "segments": [{
                    "index": 1,
                    "final_prompt": (
                        "<VISUAL>original</VISUAL>"
                        "<AUDIO_CONTENT_JSON> [] </AUDIO_CONTENT_JSON>"
                    ),
                }],
            }))

    assert pipeline.produce_prompt_fusion(settings, cid, Runner()) == "failed"
    output = root / "work" / "h3_prompt_plan.json"
    original_raw_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    failed = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
    assert failed["status"] == "failed"
    assert failed["error"] == "prompt_fusion_output_invalid"
    assert failed["raw_output_path"] == "work/h3_prompt_plan.json"
    assert failed["raw_output_sha256"] == original_raw_sha256

    output.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{
            "index": 1,
            "final_prompt": (
                "<VISUAL>schema-valid replacement</VISUAL>"
                "<AUDIO_CONTENT_JSON>\n[]\n</AUDIO_CONTENT_JSON>"
            ),
        }],
    }))

    with pytest.raises(pipeline.PipelineError, match="raw output drifted"):
        pipeline.finalize_prompt_fusion_receipt(settings, cid)
    assert not (root / "work" / h3_project.SOURCE_FILENAME).exists()


def test_new_producer_rejects_current_skill_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "fusion-skill-drift", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    acceptance_sha256 = "a" * 64
    storage.update_meta(
        settings.data_dir,
        cid,
        _image_user_acceptance={"version": 1, "sha256": acceptance_sha256},
    )
    skill_path = tmp_path / "video-prompt-fusion-SKILL.md"
    skill_path.write_text("initial prompt fusion Skill", encoding="utf-8")
    monkeypatch.setattr(pipeline, "PROMPT_FUSION_SKILL_MD", skill_path)
    monkeypatch.setattr(
        long_generation, "PROMPT_FUSION_SKILL_SOURCE", skill_path
    )
    input_data = _fusion_input(root, 1)
    assert pipeline.queue_prompt_fusion(
        settings,
        cid,
        input_data=input_data,
        image_acceptance_sha256=acceptance_sha256,
        ) == "queued"

    class Runner:
        def run_isolated(
            self, cwd: Path, _prompt: str, *, session_dir: Path,
            writable_paths: tuple[Path, ...],
        ) -> None:
            assert session_dir == root
            assert writable_paths == (
                cwd / "work" / "h3_prompt_plan.json",
            )
            skill_path.write_text("drifted prompt fusion Skill", encoding="utf-8")
            segment = json.loads(input_data)["segments"][0]
            (cwd / "work" / "h3_prompt_plan.json").write_bytes(_canonical({
                "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
                "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
                "input_sha256": hashlib.sha256(input_data).hexdigest(),
                "segments": [_fusion_v2_output(segment, "valid")],
            }))

    assert pipeline.produce_prompt_fusion(settings, cid, Runner()) == "failed"
    failed = storage.load_meta(settings.data_dir, cid)["_prompt_fusion"]
    assert failed["error"] == "prompt fusion Skill drifted"
    assert failed["raw_output_sha256"] == hashlib.sha256(
        (root / "work" / "h3_prompt_plan.json").read_bytes()
    ).hexdigest()
    assert not (root / "work" / h3_project.SOURCE_FILENAME).exists()


def test_receipt_finalization_serializes_a_concurrent_generation_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, _root, _skill_path, output = _failed_lf_prompt_fusion(
        tmp_path,
        monkeypatch,
        version=long_generation.PROMPT_FUSION_VERSION,
    )
    expected_raw_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    original_publish = pipeline._publish_prompt_fusion_manifest
    writer_started = threading.Event()
    writer_finished = threading.Event()
    writer: list[threading.Thread] = []

    def publish(**kwargs):
        def write_generation() -> None:
            writer_started.set()
            storage.update_meta(
                settings.data_dir,
                cid,
                generation={"status": "not_started"},
            )
            writer_finished.set()

        thread = threading.Thread(target=write_generation)
        writer.append(thread)
        thread.start()
        assert writer_started.wait(1)
        result = original_publish(**kwargs)
        assert not writer_finished.wait(0.2)
        return result

    monkeypatch.setattr(pipeline, "_publish_prompt_fusion_manifest", publish)

    assert pipeline.finalize_prompt_fusion_receipt(
        settings,
        cid,
        expected_raw_output_sha256=expected_raw_output_sha256,
    ) == "done"
    writer[0].join(1)
    assert writer_finished.is_set()
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["_prompt_fusion"]["status"] == "done"
    assert meta["generation"] == {"status": "not_started"}


def test_receipt_finalization_rejects_publish_hook_cas_drift_without_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, root, _skill_path, output = _failed_lf_prompt_fusion(
        tmp_path,
        monkeypatch,
        version=long_generation.PROMPT_FUSION_VERSION,
    )
    expected_raw_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    original_publish = pipeline._publish_prompt_fusion_manifest

    def publish(**kwargs):
        result = original_publish(**kwargs)
        kwargs["meta"]["_prompt_fusion"] = {
            **kwargs["state"],
            "status": "running",
        }
        return result

    monkeypatch.setattr(pipeline, "_publish_prompt_fusion_manifest", publish)

    with pytest.raises(
        pipeline.PipelineError,
        match="state drifted during finalization",
    ):
        pipeline.finalize_prompt_fusion_receipt(
            settings,
            cid,
            expected_raw_output_sha256=expected_raw_output_sha256,
        )

    assert not (root / "work" / h3_project.SOURCE_FILENAME).exists()
    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"][
        "status"
    ] == "failed"


def test_receipt_finalization_cleans_exact_manifest_after_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, root, _skill_path, output = _failed_lf_prompt_fusion(
        tmp_path,
        monkeypatch,
        version=long_generation.PROMPT_FUSION_VERSION,
    )
    expected_raw_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    original_atomic = pipeline._atomic_bytes

    def write_then_fail(path: Path, data: bytes) -> None:
        original_atomic(path, data)
        raise OSError("simulated publish failure")

    monkeypatch.setattr(pipeline, "_atomic_bytes", write_then_fail)

    with pytest.raises(OSError, match="simulated publish failure"):
        pipeline.finalize_prompt_fusion_receipt(
            settings,
            cid,
            expected_raw_output_sha256=expected_raw_output_sha256,
        )

    assert not (root / "work" / h3_project.SOURCE_FILENAME).exists()
    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"][
        "status"
    ] == "failed"


def test_exact_crash_residue_is_unreadable_until_receipt_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, root, _skill_path, output = _failed_lf_prompt_fusion(
        tmp_path,
        monkeypatch,
        version=long_generation.PROMPT_FUSION_VERSION,
    )
    expected_raw_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    meta = storage.load_meta(settings.data_dir, cid)
    pipeline._publish_prompt_fusion_manifest(
        root=root,
        meta=meta,
        state=meta["_prompt_fusion"],
    )

    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_manifest_invalid",
    ):
        long_generation.load_bound_prompt_fusion_manifest(
            root=root,
            meta=meta,
        )

    assert pipeline.finalize_prompt_fusion_receipt(
        settings,
        cid,
        expected_raw_output_sha256=expected_raw_output_sha256,
    ) == "done"
    committed = storage.load_meta(settings.data_dir, cid)
    input_payload = json.loads(
        (root / "work" / h3_project.SKILL_INPUT_FILENAME).read_bytes()
    )
    assert long_generation.load_bound_prompt_fusion_manifest(
        root=root,
        meta=committed,
    ).final_prompts == (
        _fusion_v2_final_prompt(input_payload["segments"][0], "fused"),
    )


@pytest.mark.parametrize(
    "drift", ["input", "output", "skill", "acceptance", "generation", "h3"],
)
def test_receipt_only_finalization_rejects_any_frozen_authority_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    settings, cid, root, _skill_path, output = _failed_lf_prompt_fusion(
        tmp_path,
        monkeypatch,
        version=long_generation.PROMPT_FUSION_VERSION,
    )
    expected_raw_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    if drift == "input":
        (root / "work" / h3_project.SKILL_INPUT_FILENAME).write_bytes(b"{}")
    elif drift == "output":
        output.write_bytes(b"{}")
    elif drift == "skill":
        (root / "work" / pipeline.PROMPT_FUSION_FROZEN_SKILL_FILENAME).write_bytes(
            b"drifted skill"
        )
    elif drift == "acceptance":
        storage.update_meta(
            settings.data_dir,
            cid,
            _image_user_acceptance={"version": 1, "sha256": "b" * 64},
        )
    elif drift == "generation":
        storage.update_meta(
            settings.data_dir, cid, generation={"status": "not_started"},
        )
    else:
        (root / "work" / "segments" / "1" / ".h3").mkdir(
            parents=True, exist_ok=True,
        )

    with pytest.raises(pipeline.PipelineError):
        pipeline.finalize_prompt_fusion_receipt(
            settings,
            cid,
            expected_raw_output_sha256=expected_raw_output_sha256,
        )

    assert not (root / "work" / h3_project.SOURCE_FILENAME).exists()
    assert storage.load_meta(settings.data_dir, cid)["_prompt_fusion"][
        "status"
    ] == "failed"


def test_frozen_current_v4_projects_publish_a_stable_n1_fusion_shape(
    tmp_path: Path,
) -> None:
    meta = _current_v4_meta(segments=None)
    meta["_prompt_fusion"] = {"status": "running"}

    assert main._public_prompt_fusion(meta, tmp_path) == {
        "status": "running",
        "error": None,
        "segments": [{
            "index": 1,
            "status": "running",
            "final_prompt": None,
            "error": None,
        }],
    }


def test_validation_fingerprint_binds_immutable_fusion_artifact_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / ("a" * 32)
    work = root / "work"
    work.mkdir(parents=True)
    voice_path = work / "voice.mp3"
    voice_path.write_bytes(b"frozen-voice")
    input_data = _canonical({
        "segments": [{
            "audio_content": {
                "voice_references": [{
                    "path": "work/voice.mp3",
                    "sha256": hashlib.sha256(
                        voice_path.read_bytes()
                    ).hexdigest(),
                }],
            },
        }],
    })
    output_data = b'{"output":"fused"}\n'
    input_path = work / h3_project.SKILL_INPUT_FILENAME
    output_path = work / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(output_data)
    manifest = {
        "input": {
            "path": f"work/{h3_project.SKILL_INPUT_FILENAME}",
            "sha256": hashlib.sha256(input_data).hexdigest(),
        },
        "output": {
            "path": "work/h3_prompt_plan.json",
            "sha256": hashlib.sha256(output_data).hexdigest(),
        },
    }
    manifest_data = _canonical(manifest)
    manifest_path = work / h3_project.SOURCE_FILENAME
    manifest_path.write_bytes(manifest_data)
    plan = {
        "schema": "duet.long-video-plan",
        "version": long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
        "prompt_fusion": {
            "path": f"work/{h3_project.SOURCE_FILENAME}",
            "sha256": hashlib.sha256(manifest_data).hexdigest(),
        },
        "segments": [],
    }
    (root / long_video.PLAN_RECEIPT_FILENAME).write_bytes(_canonical(plan))
    meta = {
        "id": root.name,
        "duration_s": 28.0,
        "segments": [{"index": 1}],
        "long_video_plan_receipt": long_video.PLAN_RECEIPT_FILENAME,
        "_prompt_fusion": {"status": "done"},
    }

    paths = main._long_validation_paths(root, meta)
    assert {manifest_path, input_path, output_path, voice_path} <= paths
    before_entries = main._prompt_fusion_fingerprint_entries(root, meta)
    before = main._generated_video_validation_fingerprint(root, meta)
    del meta["_prompt_fusion"]
    assert main._prompt_fusion_fingerprint_entries(root, meta) == before_entries
    assert main._generated_video_validation_fingerprint(root, meta) == before

    output_path.write_bytes(b'{"output":"DRIFT"}\n')

    after_entries = main._prompt_fusion_fingerprint_entries(root, meta)
    after = main._generated_video_validation_fingerprint(root, meta)
    assert after_entries != before_entries
    assert after != before

    output_path.write_bytes(output_data)
    before_voice = main._generated_video_validation_fingerprint(root, meta)
    voice_path.write_bytes(b"drifted-voice")
    assert main._generated_video_validation_fingerprint(root, meta) != before_voice


def test_warm_validation_cache_invalidates_when_fusion_voice_bytes_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    cid = "c" * 32
    root = settings.data_dir / cid
    work = root / "work"
    work.mkdir(parents=True)
    voice = work / "voice.mp3"
    voice.write_bytes(b"voice-one")
    input_data = _canonical({
        "segments": [{"audio_content": {"voice_references": [{
            "path": "work/voice.mp3",
            "sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
        }]}}],
    })
    output_data = b'{"output":"fused"}\n'
    (work / h3_project.SKILL_INPUT_FILENAME).write_bytes(input_data)
    (work / "h3_prompt_plan.json").write_bytes(output_data)
    manifest_data = _canonical({
        "input": {
            "path": f"work/{h3_project.SKILL_INPUT_FILENAME}",
            "sha256": hashlib.sha256(input_data).hexdigest(),
        },
        "output": {
            "path": "work/h3_prompt_plan.json",
            "sha256": hashlib.sha256(output_data).hexdigest(),
        },
    })
    (work / h3_project.SOURCE_FILENAME).write_bytes(manifest_data)
    (root / long_video.PLAN_RECEIPT_FILENAME).write_bytes(_canonical({
        "schema": "duet.long-video-plan",
        "version": long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
        "prompt_fusion": {
            "path": f"work/{h3_project.SOURCE_FILENAME}",
            "sha256": hashlib.sha256(manifest_data).hexdigest(),
        },
        "segments": [],
    }))
    meta = {
        "id": cid,
        "duration_s": 28.0,
        "segments": [{"index": 1}],
        "long_video_plan_receipt": long_video.PLAN_RECEIPT_FILENAME,
        "generation": {"status": "succeeded"},
    }
    calls = 0

    def validate(_settings, _meta):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(main, "_validate_generated_video_uncached", validate)
    assert main._has_valid_generated_video(settings, meta) is True
    assert main._has_valid_generated_video(settings, meta) is True
    assert calls == 1
    voice.write_bytes(b"voice-two")
    assert main._has_valid_generated_video(settings, meta) is True
    assert calls == 2


@pytest.mark.parametrize("dialogue_mode", ["auto", "none"])
def test_legacy_v2_plan_does_not_bootstrap_prompt_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dialogue_mode: str,
) -> None:
    root = tmp_path / ("b" * 32)
    root.mkdir()
    receipt_data = _canonical({
        "schema": "duet.long-video-plan",
        "version": long_video.PLAN_RECEIPT_VERSION,
        "workflow": "legacy-read-only",
    })
    (root / long_video.PLAN_RECEIPT_FILENAME).write_bytes(receipt_data)
    monkeypatch.setattr(
        pipeline,
        "queue_prompt_fusion",
        lambda *_args, **_kwargs: pytest.fail(
            "legacy v2 may not enter current prompt fusion"
        ),
    )

    assert long_generation.finalize_multimodal_plan(
        root,
        {
            "id": root.name,
            "long_video_plan_receipt": long_video.PLAN_RECEIPT_FILENAME,
        },
        hashlib.sha256(receipt_data).hexdigest(),
        "none",
        dialogue_mode,
        aspect_ratio="9:16",
        resolution="768p",
        settings=make_settings(tmp_path),
    ) is None
