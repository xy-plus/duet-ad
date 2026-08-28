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
    final_prompt = (
        "FUSED_PROMPT_FROM_OPTIMIZED_IMAGES"
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>"
    )
    for order in range(1, 10):
        (tmp_path / f"{order:02d}.png").write_bytes(f"frame-{order}".encode())
    fusion_input = {
        "schema": "duet.video-prompt-fusion-input",
        "version": 1,
        "segments": [{
            "index": 1,
            "new_keyframes": [
                {
                    "order": order,
                    "path": f"{order:02d}.png",
                    "sha256": hashlib.sha256(f"frame-{order}".encode()).hexdigest(),
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
            },
        }],
    }
    input_data = _canonical(fusion_input)
    fusion_output = {
        "schema": "duet.video-prompt-fusion-output",
        "version": 1,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{"index": 1, "final_prompt": final_prompt}],
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

    assert frozen.final_prompts == (final_prompt,)
    assert old_prompt not in frozen.final_prompts


def test_prompt_fusion_v2_binds_exact_source_timeline_and_hard_cut(
    tmp_path: Path,
) -> None:
    frames = []
    timeline = []
    source_times = [0.0, 0.75, 2.0, 2.5, 4.0, 6.0, 8.0, 11.0, 14.0]
    for order, source_time_s in enumerate(source_times, 1):
        data = f"frame-{order}".encode()
        path = tmp_path / f"{order:02d}.png"
        path.write_bytes(data)
        transition = (
            {"type": "start", "at_s": 0.0}
            if order == 1 else
            {"type": "hard_cut", "at_s": 2.267}
            if order == 4 else
            {"type": "continuous", "at_s": None}
        )
        source = {
            "order": order,
            "source_time_s": source_time_s,
            "source_scene_id": "SCENE_01" if order < 4 else "SCENE_02",
            "transition": transition,
        }
        timeline.append(source)
        frames.append({
            "order": order,
            "path": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            **{key: source[key] for key in (
                "source_time_s", "source_scene_id", "transition",
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
    timeline_json = json.dumps(
        timeline,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    final_prompt = (
        "<VISUAL>\n"
        "hard cut at the frozen source boundary\n"
        "</VISUAL>\n"
        f"<KEYFRAME_TIMELINE_JSON>{timeline_json}"
        "</KEYFRAME_TIMELINE_JSON>\n"
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>\n"
        "<MUSIC_POLICY>forbid</MUSIC_POLICY>"
    )
    input_path = tmp_path / "multimodal_input.json"
    output_path = tmp_path / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{"index": 1, "final_prompt": final_prompt}],
    }))

    frozen = long_generation.load_prompt_fusion(
        input_path=input_path,
        output_path=output_path,
        root=tmp_path,
    )

    assert frozen.final_prompts == (final_prompt,)
    assert frozen.segments[0]["new_keyframes"][3]["transition"] == {
        "type": "hard_cut", "at_s": 2.267,
    }

    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{
            "index": 1,
            "final_prompt": final_prompt.replace("<VISUAL>\n", "<VISUAL>", 1),
        }],
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


def test_prompt_fusion_v2_rejects_output_hard_cut_time_drift(tmp_path: Path) -> None:
    frames = []
    timeline = []
    for order in range(1, 10):
        data = f"frame-{order}".encode()
        path = tmp_path / f"{order:02d}.png"
        path.write_bytes(data)
        transition = (
            {"type": "start", "at_s": 0.0}
            if order == 1 else
            {"type": "hard_cut", "at_s": 2.267}
            if order == 4 else
            {"type": "continuous", "at_s": None}
        )
        source = {
            "order": order,
            "source_time_s": (
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
            "source_time_s": source["source_time_s"],
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
    drifted = json.loads(json.dumps(timeline))
    drifted[3]["transition"]["at_s"] = 3.5
    drifted_json = json.dumps(
        drifted, separators=(",", ":"),
    )
    input_path = tmp_path / "multimodal_input.json"
    output_path = tmp_path / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{
            "index": 1,
            "final_prompt": (
                "<VISUAL>\n"
                "drifted hard cut\n"
                "</VISUAL>\n"
                f"<KEYFRAME_TIMELINE_JSON>{drifted_json}"
                "</KEYFRAME_TIMELINE_JSON>\n"
                "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>\n"
                "<MUSIC_POLICY>forbid</MUSIC_POLICY>"
            ),
        }],
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
        "source_time_s": 2.5,
        "source_scene_id": "SCENE_02",
        "transition": {"type": "hard_cut", "at_s": 2.267},
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
    input_path = work / h3_project.SKILL_INPUT_FILENAME
    output_path = work / "h3_prompt_plan.json"
    input_path.write_bytes(input_data)
    output_path.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{
            "index": segment["index"],
            "final_prompt": _fusion_v2_final_prompt(
                segment, f"fused visual {segment['index']}"
            ),
        } for segment in input_payload["segments"]],
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
            minimax_api_key="minimax-secret",
        )
        assert context.keyframe_timeline_json is not None
        assert len(json.loads(context.keyframe_timeline_json)) == 9

    hard_cuts = [
        frame["transition"]["at_s"]
        for segment in input_payload["segments"]
        for frame in segment["new_keyframes"]
        if frame["transition"]["type"] == "hard_cut"
    ]
    assert hard_cuts == [cut_at_s]


_CID_9533_LINES_JSON = (
    '[{"order":1,"text":"تول جمالا بها القادم وقتك يا لدى عوصة اللوري تاب '
    'البدري يبريزها بجمال يا شعوري","start_s":0.0,"end_s":14.44,'
    '"delivery":"off_screen","voice_ref":1}]'
)


def _write_9533_fusion_fixture(
    root: Path, *, audio_envelope: str, lines_json: str = _CID_9533_LINES_JSON,
) -> tuple[Path, Path, str]:
    (root / "work").mkdir(parents=True, exist_ok=True)
    input_payload = json.loads(_fusion_v1_input(root, 1).decode("utf-8"))
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
        "version": long_generation.PROMPT_FUSION_VERSION,
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


def _fusion_input(root: Path, segment_count: int) -> bytes:
    segments = []
    for index in range(1, segment_count + 1):
        frames = []
        prompts = []
        for order in range(1, 10):
            relative = f"work/segment-{index}-{order:02d}.png"
            data = f"segment-{index}-frame-{order}".encode()
            (root / relative).write_bytes(data)
            prompt = f"segment {index} optimized image {order}"
            source_time_s = float((index - 1) * 20 + order - 1)
            frames.append({
                "order": order,
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "source_time_s": source_time_s,
                "source_scene_id": "SCENE_01",
                "transition": (
                    {"type": "start", "at_s": source_time_s}
                    if index == order == 1 else
                    {"type": "continuous", "at_s": None}
                ),
            })
            prompts.append({
                "order": order,
                "text": prompt,
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            })
        old_prompt = f"old visual prompt {index}"
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
            "audio_content": {
                "lines_json": "[]",
                "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
                "voice_references": [],
                "music_policy": "forbid",
            },
        })
    return _canonical({
        "schema": long_generation.PROMPT_FUSION_INPUT_SCHEMA,
        "version": long_generation.VISUAL_PROMPT_FUSION_VERSION,
        "segments": segments,
    })


def _fusion_v1_input(root: Path, segment_count: int) -> bytes:
    """Build an exact historical artifact for load/recovery-only coverage."""
    payload = json.loads(_fusion_input(root, segment_count))
    payload["version"] = long_generation.PROMPT_FUSION_VERSION
    for segment in payload["segments"]:
        for frame in segment["new_keyframes"]:
            frame.pop("source_time_s")
            frame.pop("source_scene_id")
            frame.pop("transition")
        segment["audio_content"].pop("music_policy")
    return _canonical(payload)


def _fusion_v2_final_prompt(segment: dict, visual: str) -> str:
    timeline = [{
        "order": frame["order"],
        "source_time_s": frame["source_time_s"],
        "source_scene_id": frame["source_scene_id"],
        "transition": {
            "type": frame["transition"]["type"],
            "at_s": frame["transition"]["at_s"],
        },
    } for frame in segment["new_keyframes"]]
    timeline_json = json.dumps(
        timeline, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    )
    return (
        f"<VISUAL>\n{visual}\n</VISUAL>\n"
        f"<KEYFRAME_TIMELINE_JSON>{timeline_json}"
        "</KEYFRAME_TIMELINE_JSON>\n"
        "<AUDIO_CONTENT_JSON>"
        f"{segment['audio_content']['lines_json']}"
        "</AUDIO_CONTENT_JSON>\n"
        "<MUSIC_POLICY>forbid</MUSIC_POLICY>"
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
        voice = root / "work" / "segments" / "1" / "work" / "voice.mp3"
        voice.parent.mkdir(parents=True, exist_ok=True)
        voice.write_bytes(b"voice-reference-must-not-be-staged")
        payload = json.loads(input_data)
        lines_json = (
            '[{"order":1,"text":"spoken","start_s":0.0,"end_s":1.0,'
            '"delivery":"off_screen","voice_ref":1}]'
        )
        payload["segments"][0]["audio_content"] = {
            "lines_json": lines_json,
            "lines_sha256": hashlib.sha256(lines_json.encode()).hexdigest(),
            "voice_references": [{
                "voice_ref": 1,
                "path": "work/segments/1/work/voice.mp3",
                "sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
                "purpose": "voice",
            }],
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
                "segments": [{
                    "index": item["index"],
                    "final_prompt": _fusion_v2_final_prompt(
                        item, f"fused prompt {item['index']}"
                    ),
                } for item in payload["segments"]],
            }
            output_path.write_bytes(_canonical(output))

    if segment_count == 1:
        legacy = json.loads(input_data)
        legacy["version"] = long_generation.PROMPT_FUSION_VERSION
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
    current_input = json.loads(_fusion_input(root, 1))
    current_input["version"] = long_generation.PROMPT_FUSION_VERSION
    input_data = _canonical(current_input)
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


def _failed_lf_prompt_fusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    input_data = _fusion_v1_input(root, 1)
    input_path = root / "work" / h3_project.SKILL_INPUT_FILENAME
    input_path.write_bytes(input_data)
    frozen_skill = root / "work" / pipeline.PROMPT_FUSION_FROZEN_SKILL_FILENAME
    frozen_skill.write_bytes(skill_path.read_bytes())
    output = root / "work" / "h3_prompt_plan.json"
    output.write_bytes(_canonical({
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{
            "index": 1,
            "final_prompt": (
                "<VISUAL>fused</VISUAL>\n"
                "<AUDIO_CONTENT_JSON>\n[]\n</AUDIO_CONTENT_JSON>"
            ),
        }],
    }))
    storage.update_meta(
        settings.data_dir,
        cid,
        _prompt_fusion={
            "version": 1,
            "status": "failed",
            "error": "prompt_fusion_output_invalid",
            "input_sha256": hashlib.sha256(input_data).hexdigest(),
            "image_acceptance_sha256": acceptance_sha256,
            "manifest_sha256": None,
        },
    )
    return settings, cid, root, skill_path, output


def test_failed_lf_output_can_publish_receipt_without_rerunning_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, root, skill_path, output = _failed_lf_prompt_fusion(
        tmp_path, monkeypatch,
    )
    raw_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    assert pipeline.finalize_prompt_fusion_receipt(
        settings,
        cid,
        expected_raw_output_sha256=raw_output_sha256,
    ) == "done"

    assert hashlib.sha256(output.read_bytes()).hexdigest() == raw_output_sha256
    frozen = long_generation.load_prompt_fusion_manifest(
        root=root, skill_source_path=skill_path,
    )
    assert frozen.final_prompts == (
        "<VISUAL>fused</VISUAL>\n"
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>",
    )
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["_prompt_fusion"]["status"] == "done"
    assert meta["_prompt_fusion"]["error"] is None
    assert meta["_prompt_fusion"]["recovered_error"] == (
        "prompt_fusion_output_invalid"
    )
    assert meta["_prompt_fusion"]["raw_output_path"] == (
        "work/h3_prompt_plan.json"
    )
    assert meta["_prompt_fusion"]["raw_output_sha256"] == raw_output_sha256
    assert meta["_prompt_fusion"]["manifest_sha256"] == hashlib.sha256(
        (root / "work" / h3_project.SOURCE_FILENAME).read_bytes()
    ).hexdigest()
    manifest = json.loads(
        (root / "work" / h3_project.SOURCE_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["output"]["sha256"] == raw_output_sha256
    assert manifest["skill"]["sha256"] == hashlib.sha256(
        (root / "work" / pipeline.PROMPT_FUSION_FROZEN_SKILL_FILENAME).read_bytes()
    ).hexdigest()


def test_done_receipt_survives_current_skill_source_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, cid, root, skill_path, output = _failed_lf_prompt_fusion(
        tmp_path, monkeypatch,
    )
    assert pipeline.finalize_prompt_fusion_receipt(
        settings,
        cid,
        expected_raw_output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    ) == "done"
    committed = storage.load_meta(settings.data_dir, cid)

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
    assert pipeline.finalize_prompt_fusion_receipt(
        settings,
        cid,
        expected_raw_output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    ) == "done"
    committed = storage.load_meta(settings.data_dir, cid)
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
                "segments": [{
                    "index": 1,
                    "final_prompt": _fusion_v2_final_prompt(segment, "valid"),
                }],
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
        tmp_path, monkeypatch,
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
        tmp_path, monkeypatch,
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
        tmp_path, monkeypatch,
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
        tmp_path, monkeypatch,
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
    assert long_generation.load_bound_prompt_fusion_manifest(
        root=root,
        meta=committed,
    ).final_prompts == (
        "<VISUAL>fused</VISUAL>\n"
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>",
    )


@pytest.mark.parametrize(
    "drift", ["input", "output", "skill", "acceptance", "generation", "h3"],
)
def test_receipt_only_finalization_rejects_any_frozen_authority_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    settings, cid, root, _skill_path, output = _failed_lf_prompt_fusion(
        tmp_path, monkeypatch,
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
