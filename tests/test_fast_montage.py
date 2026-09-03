import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import context_ir_bridge, h3, h3_project, long_generation, long_video


def _png(value: int) -> bytes:
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _cuts() -> list[dict]:
    bounds = [round(index * 0.3, 6) for index in range(11)]
    bounds += [round(3.0 + index * (7.0 / 15), 6) for index in range(1, 16)]
    return [
        {
            "order": index,
            "start_segment_s": bounds[index - 1],
            "end_segment_s": bounds[index],
            "source_scene_id": f"SCENE_{index:02d}",
        }
        for index in range(1, 26)
    ]


def _timeline(cuts: list[dict]) -> list[dict]:
    cut_orders = [1, 13, 25]
    result = []
    for order, cut_order in enumerate(cut_orders, 1):
        cut = cuts[cut_order - 1]
        time_s = 0.0 if order == 1 else cut["start_segment_s"]
        result.append({
            "order": order,
            "segment_time_s": time_s,
            "source_scene_id": cut["source_scene_id"],
            "transition": {
                "type": "start" if order == 1 else "hard_cut",
                "at_segment_s": time_s,
            },
        })
    return result


def _occurrence(frame: dict, relation_id: str) -> dict:
    return {
        "relation_id": relation_id,
        "subject_key": "entity-a",
        "predicate": "holds",
        "object_key": "entity-b",
        "state": "directly visible",
        "geometry": "source geometry",
        "preserve": ["roles", "state"],
        "replace_together": False,
        "frame": {
            "order": frame["order"],
            "segment_time_s": frame["segment_time_s"],
            "source_scene_id": frame["source_scene_id"],
        },
    }


def test_twenty_five_cut_fusion_keeps_empty_intervals_and_splits_dialogue():
    cuts = _cuts()
    timeline = _timeline(cuts)
    occurrences = [
        _occurrence(timeline[0], "relation-first"),
        _occurrence(timeline[-1], "relation-last"),
    ]

    projected = long_generation._expected_fusion_relation_states(
        timeline, occurrences, cuts,
    )
    assert len(projected) == 3
    assert projected[0]["relations"][0]["relation_id"] == "relation-first"
    assert projected[-1]["relations"][0]["relation_id"] == "relation-last"
    assert all(not item["relations"] for item in projected[1:-1])

    prompt = long_generation._compile_fusion_ref2va_prompt(
        visual=[f"analysis visual {index}" for index in range(1, 4)],
        timeline=timeline,
        lines=[{
            "order": 1,
            "text": "ABCD",
            "start_s": 2.9,
            "end_s": 3.2,
            "delivery": "off_screen",
            "voice_ref": None,
        }],
        music_policy="forbid",
        relation_occurrences=occurrences,
        cut_timeline=cuts,
    )
    assert long_generation.CUT_TIMELINE_OPEN in prompt
    assert "00:02.900 to 00:03.000" in prompt
    assert "00:03.000 to 00:03.200" in prompt
    assert "".join(re.findall(r"<d>\[Undetermined\]([^<]*)</d>", prompt)) == "ABCD"
    encoded = prompt.split(
        long_generation.RELATION_STATES_OPEN, 1
    )[1].split(long_generation.RELATION_STATES_CLOSE, 1)[0]
    contract = json.loads(encoded)
    assert contract["v"] == 3
    assert len(long_generation._expand_h3_relation_contract(contract)) == 3
    encoded_cuts = prompt.split(
        long_generation.CUT_TIMELINE_OPEN, 1
    )[1].split(long_generation.CUT_TIMELINE_CLOSE, 1)[0]
    assert len(json.loads(encoded_cuts)["b"]) == 24


@pytest.mark.parametrize("damage", ["gap", "overlap"])
def test_cut_timeline_rejects_gap_or_overlap(damage):
    timeline = [
        {"order": 1, "start_s": 0.0, "end_s": 2.0,
         "source_scene_id": "SCENE_01"},
        {"order": 2, "start_s": 2.0, "end_s": 4.0,
         "source_scene_id": "SCENE_02"},
    ]
    timeline[1]["start_s"] = 2.1 if damage == "gap" else 1.9
    with pytest.raises(
        long_video.LongVideoError,
        match="long_video_plan_invalid_cut_timeline",
    ):
        long_video.freeze_source_cut_timeline(
            timeline, segment_start_s=0.0, segment_end_s=4.0,
        )


def test_build_and_load_fusion_consumes_all_twenty_five_cuts(tmp_path: Path):
    root = tmp_path.resolve()
    keyframe_dir = root / "work" / "segments" / "1" / "work" / "keyframes"
    keyframe_dir.mkdir(parents=True)
    cuts = _cuts()
    local_timeline = _timeline(cuts)
    frozen_frames = []
    source_timeline = []
    optimization_frames = []
    for frame in local_timeline:
        path = keyframe_dir / f"{frame['order']:02d}.png"
        data = _png(30 + frame["order"])
        path.write_bytes(data)
        frozen_frames.append((path, data))
        source_timeline.append({
            "order": frame["order"],
            "source_time_s": frame["segment_time_s"],
            "source_scene_id": frame["source_scene_id"],
            "transition": {
                "type": frame["transition"]["type"],
                "at_s": frame["transition"]["at_segment_s"],
            },
        })
        text = f"optimized frame {frame['order']}"
        optimization_frames.append({
            "segment_index": 1,
            "frame_index": frame["order"],
            "current": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
    global_cuts = [{
        "order": cut["order"],
        "start_s": cut["start_segment_s"],
        "end_s": cut["end_segment_s"],
        "source_scene_id": cut["source_scene_id"],
    } for cut in cuts]
    segment = long_generation.FrozenSegment(
        index=1, start_s=0.0, end_s=10.0,
        chain_id="chain-001", join_mode="hard_cut",
        workdir=root / "work" / "segments" / "1",
        first_frame=frozen_frames[0][0], first_frame_data=frozen_frames[0][1],
        last_frame=frozen_frames[-1][0], last_frame_data=frozen_frames[-1][1],
        prompt="frozen", keyframes=tuple(frozen_frames),
        keyframe_sources=tuple(source_timeline),
        source_cut_timeline=tuple(global_cuts),
        dialogue=({
            "text": "ABCD", "start_s": 2.9, "end_s": 3.2,
            "classification": "spoken",
        },),
    )
    plan = long_generation.FrozenPlan(
        root=root, source=root / "source.mp4", receipt="r" * 64,
        segments=(segment,),
    )
    meta = {
        "segments": [{"visual_prompt": "source visual"}],
        "_image_optimization": {"frames": optimization_frames},
    }
    input_data = long_generation.build_prompt_fusion_input(
        root=root, meta=meta, plan=plan,
        dialogue_mode="custom", dialogue_delivery="off_screen",
    )
    source = json.loads(input_data)
    assert len(source["segments"][0]["source_cut_timeline"]) == 25
    input_path = root / "work" / h3_project.SKILL_INPUT_FILENAME
    input_path.write_bytes(input_data)
    output = {
        "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
        "version": long_generation.PROMPT_FUSION_VERSION,
        "input_sha256": hashlib.sha256(input_data).hexdigest(),
        "segments": [{
            "index": 1,
            "visual": [f"visual {index}" for index in range(1, 4)],
        }],
    }
    output_path = root / "work" / "h3_prompt_plan.json"
    output_path.write_text(json.dumps(output), encoding="utf-8")

    frozen = long_generation.load_prompt_fusion(
        input_path=input_path, output_path=output_path, root=root,
    )
    prompt = frozen.final_prompts[0]
    assert long_generation.CUT_TIMELINE_OPEN in prompt
    assert "00:02.900 to 00:03.000" in prompt
    assert "00:03.000 to 00:03.200" in prompt


@pytest.mark.parametrize("cut_count", [25, 100])
def test_dense_cut_prompt_stays_within_budget_without_truncating_visual(cut_count):
    cuts = [{
        "order": index,
        "start_segment_s": round((index - 1) * 10 / cut_count, 6),
        "end_segment_s": round(index * 10 / cut_count, 6),
        "source_scene_id": f"SCENE_{index:03d}",
    } for index in range(1, cut_count + 1)]
    selected = [
        1 + round(position * (cut_count - 1) / 2)
        for position in range(3)
    ]
    timeline = []
    for order, cut_order in enumerate(selected, 1):
        cut = cuts[cut_order - 1]
        time_s = 0.0 if order == 1 else cut["start_segment_s"]
        timeline.append({
            "order": order,
            "segment_time_s": time_s,
            "source_scene_id": cut["source_scene_id"],
            "transition": {
                "type": "start" if order == 1 else "hard_cut",
                "at_segment_s": time_s,
            },
        })
    visuals = [f"V{index}-" + "语义" * 99 for index in range(1, 4)]
    prompt = long_generation._compile_fusion_ref2va_prompt(
        visual=visuals, timeline=timeline, lines=[], music_policy="forbid",
        relation_occurrences=[], cut_timeline=cuts,
    )
    assert len(prompt) <= long_generation._MAX_COMPILED_FUSION_CHARS
    assert all(visual in prompt for visual in visuals)
    if cut_count == 25:
        compact_prompt = long_generation._compile_fusion_ref2va_prompt(
            visual=[f"V{index}" for index in range(1, 4)],
            timeline=timeline, lines=[], music_policy="forbid",
            relation_occurrences=[], cut_timeline=cuts,
        )
        authority = next(
            line for line in compact_prompt.splitlines()
            if line.startswith(long_generation.CUT_TIMELINE_OPEN)
        )
        assert len(authority) < 500
        assert len(compact_prompt) < 3_000


def test_dense_cuts_and_unique_180_relations_share_budget_without_loss():
    cuts = _cuts()
    timeline = _timeline(cuts)
    occurrences = []
    for frame in timeline:
        for relation in range(1, 61):
            occurrence = _occurrence(frame, f"relation-{relation:02d}")
            occurrence["subject_key"] = f"entity-{relation:02d}"
            occurrence["state"] = (
                f"unique state frame {frame['order']} relation {relation} "
                "released and fully separated"
            )
            occurrence["geometry"] = (
                f"unique geometry frame {frame['order']} relation {relation} "
                "with a visible separation gap"
            )
            occurrences.append(occurrence)
    visuals = [f"V{index}-" + "视觉语义" * 75 for index in range(1, 4)]

    prompt = long_generation._compile_fusion_ref2va_prompt(
        visual=visuals, timeline=timeline, lines=[], music_policy="forbid",
        relation_occurrences=occurrences, cut_timeline=cuts,
    )

    relation_marker = prompt.split(
        long_generation.RELATION_STATES_OPEN, 1
    )[1].split(long_generation.RELATION_STATES_CLOSE, 1)[0]
    cut_marker = prompt.split(
        long_generation.CUT_TIMELINE_OPEN, 1
    )[1].split(long_generation.CUT_TIMELINE_CLOSE, 1)[0]
    relation_contract = json.loads(relation_marker)
    assert relation_contract["v"] == 3
    assert len(relation_marker) < long_generation._MAX_RELATION_MARKER_CHARS
    assert len(cut_marker) < 500
    assert len(prompt) <= long_generation._MAX_COMPILED_FUSION_CHARS
    assert all(visual in prompt for visual in visuals)

    request = SimpleNamespace(
        source_prompt=prompt,
        dialogue_tokens=(),
        source_h3_request=SimpleNamespace(
            mode="reference", workflow=h3.H3_WORKFLOW,
            context_ir_required=True, keyframes=tuple(range(3)),
            reference_audios=(),
        ),
    )
    provider_output = (
        "Context-expanded visual prose. " * 300
        + "<CUT_TIMELINE_JSON>{\"b\":[9.0],\"v\":1}"
        "</CUT_TIMELINE_JSON>"
    )
    effective = context_ir_bridge._compile_effective_prompt(
        request,
        provider_output,
    )
    assert effective == provider_output
