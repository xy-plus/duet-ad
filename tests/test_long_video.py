"""Long-video planning is deterministic and fails closed before H3 submission."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from app import long_video as long_video_module
from app.long_video import (
    LongVideoError,
    build_continuity_block,
    localize_dialogue,
    plan_segments,
    provider_duration_s,
    write_plan_receipt,
)


def test_short_video_is_the_single_segment_contract():
    assert plan_segments(15.0, [(0.0, 15.0)], []) == [{
        "index": 1,
        "start_s": 0.0,
        "end_s": 15.0,
        "chain_id": "chain-001",
        "join_mode": "hard_cut",
    }]


def test_provider_integer_duration_boundaries_do_not_hide_positive_overflow():
    assert provider_duration_s(37.52, 47.52) == 10
    assert provider_duration_s(0.0, 10.000001) == 11
    assert provider_duration_s(
        25.52, 37.52, receipt_version=1
    ) == 13

    segments = plan_segments(15.000001, [(0.0, 15.000001)], [])
    assert len(segments) == 2
    assert all(
        provider_duration_s(item["start_s"], item["end_s"]) <= 14
        for item in segments
    )

    with pytest.raises(LongVideoError, match="long_video_duration_exceeded"):
        plan_segments(math.nextafter(300.0, math.inf), [], [])


def test_long_scene_is_split_at_provider_safe_fourteen_seconds_and_continues_chain():
    segments = plan_segments(31.0, [(0.0, 31.0)], [])

    assert [(s["start_s"], s["end_s"]) for s in segments] == [
        (0.0, 14.0),
        (14.0, 28.0),
        (28.0, 31.0),
    ]
    assert [s["chain_id"] for s in segments] == ["chain-001"] * 3
    assert [s["join_mode"] for s in segments] == [
        "hard_cut",
        "continue",
        "continue",
    ]


@pytest.mark.parametrize(
    "duration,expected,expected_join_modes",
    [
        (15.0, [(0.0, 15.0)], ["hard_cut"]),
        (15.001, [(0.0, 14.0), (14.0, 15.001)], ["hard_cut", "continue"]),
        (
            30.0,
            [(0.0, 14.0), (14.0, 28.0), (28.0, 30.0)],
            ["hard_cut", "continue", "continue"],
        ),
    ],
)
def test_fifteen_second_gate_and_fourteen_second_segment_examples(
    duration, expected, expected_join_modes
):
    segments = plan_segments(duration, [(0.0, duration)], [])
    assert [(segment["start_s"], segment["end_s"]) for segment in segments] == expected
    assert [segment["join_mode"] for segment in segments] == expected_join_modes


def test_scene_cut_starts_a_new_chain_and_coverage_is_exact():
    segments = plan_segments(28.0, [(0.0, 13.0), (13.0, 28.0)], [])

    assert [(s["start_s"], s["end_s"]) for s in segments] == [
        (0.0, 13.0),
        (13.0, 27.0),
        (27.0, 28.0),
    ]
    assert [s["chain_id"] for s in segments] == [
        "chain-001", "chain-002", "chain-002"
    ]
    assert [s["join_mode"] for s in segments] == [
        "hard_cut", "hard_cut", "continue"
    ]
    assert all(1.0 <= s["end_s"] - s["start_s"] for s in segments)
    assert all(provider_duration_s(s["start_s"], s["end_s"]) <= 14 for s in segments)


def test_real_night_market_scene_bounds_never_plan_over_fourteen_seconds():
    duration = 50.64
    ends = [
        8.08, 8.92, 10.84, 18.68, 19.92, 20.84,
        21.72, 22.64, 24.88, 25.52, 37.52, 50.64,
    ]
    starts = [0.0, *ends[:-1]]
    scenes = list(zip(starts, ends))

    segments = plan_segments(duration, scenes, [])

    assert segments[0]["end_s"] == 10.84
    assert all(
        provider_duration_s(item["start_s"], item["end_s"]) <= 14
        for item in segments
    )
    assert segments[0]["start_s"] == 0.0
    assert segments[-1]["end_s"] == duration
    assert all(
        left["end_s"] == right["start_s"]
        for left, right in zip(segments, segments[1:])
    )


def test_reserved_one_second_tail_uses_frozen_boundary_precision():
    duration = 128.842639
    scenes = [
        (0.0, 33.97548),
        (33.97548, 78.404549),
        (78.404549, duration),
    ]

    segments = plan_segments(duration, scenes, [])

    assert segments[-1]["start_s"] == 120.404549
    assert segments[-1]["end_s"] == duration
    assert round(segments[-1]["end_s"] - segments[-1]["start_s"], 6) == 8.43809
    assert all(
        provider_duration_s(item["start_s"], item["end_s"]) <= 14
        for item in segments
    )


@pytest.mark.parametrize(
    "duration",
    [16.123457, 32.000001, 64.654321, 256.123457],
)
def test_six_decimal_plans_keep_frozen_segment_lengths_in_range(duration):
    hard_cut = round(duration - 10.43809, 6)
    segments = plan_segments(
        duration,
        [(0.0, hard_cut), (hard_cut, duration)],
        [],
    )

    assert segments[0]["start_s"] == 0.0
    assert segments[-1]["end_s"] == duration
    assert all(
        left["end_s"] == right["start_s"]
        for left, right in zip(segments, segments[1:])
    )
    assert all(
        1.0
        <= long_video_module.segment_duration_s(item["start_s"], item["end_s"])
        <= 14.0
        for item in segments
    )


@pytest.mark.parametrize(
    ("start_s", "end_s", "expected"),
    [
        (127.842639, 128.842639, 1.0),
        (37.52, 47.52, 10.0),
        (0.0, 0.999999, 0.999999),
    ],
)
def test_segment_duration_uses_frozen_six_decimal_boundaries(
    start_s, end_s, expected
):
    assert long_video_module.segment_duration_s(start_s, end_s) == expected


def test_dialogue_boundary_moves_split_without_cutting_a_sentence():
    lines = [{"text": "one sentence", "start_s": 13.0, "end_s": 15.0}]
    segments = plan_segments(28.0, [(0.0, 28.0)], lines)

    assert segments[0]["end_s"] == 13.0
    assert segments[1]["start_s"] == 13.0
    for boundary in [s["end_s"] for s in segments[:-1]]:
        assert not any(line["start_s"] < boundary < line["end_s"] for line in lines)


def test_no_safe_dialogue_boundary_has_stable_error():
    with pytest.raises(LongVideoError) as exc:
        plan_segments(
            20.0,
            [(0.0, 20.0)],
            [{"text": "too long", "start_s": 0.0, "end_s": 16.0}],
        )
    assert exc.value.code == "long_video_no_safe_dialogue_boundary"
    assert str(exc.value) == "long_video_no_safe_dialogue_boundary"


def test_over_three_hundred_seconds_has_stable_error():
    with pytest.raises(LongVideoError) as exc:
        plan_segments(300.001, [(0.0, 300.001)], [])
    assert exc.value.code == "long_video_duration_exceeded"


def test_local_dialogue_is_shifted_and_strictly_inside_segment():
    lines = [
        {"text": "a", "start_s": 10.0, "end_s": 11.25},
        {"text": "b", "start_s": 14.5, "end_s": 15.0},
    ]
    assert localize_dialogue(lines, {"start_s": 10.0, "end_s": 15.0}) == [
        {"text": "a", "start_s": 0.0, "end_s": 1.25},
        {"text": "b", "start_s": 4.5, "end_s": 5.0},
    ]

    with pytest.raises(LongVideoError) as exc:
        localize_dialogue(
            [{"text": "crosses", "start_s": 9.5, "end_s": 10.5}],
            {"start_s": 10.0, "end_s": 15.0},
        )
    assert exc.value.code == "long_video_no_safe_dialogue_boundary"


def test_continuity_block_is_identical_and_does_not_promote_ocr():
    block = build_continuity_block()
    assert block == build_continuity_block()
    assert "OCR" in block
    assert "台词" in block
    assert "局部动作" in block


def test_plan_receipt_binds_every_segment_artifact_deterministically(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"whole source")
    segment_dir = tmp_path / "work" / "segments" / "1"
    segment_work = segment_dir / "work"
    keyframes = segment_work / "keyframes"
    keyframes.mkdir(parents=True)
    segment_source = segment_dir / "source.mp4"
    segment_source.write_bytes(b"segment source")
    frame = keyframes / "01.png"
    frame.write_bytes(b"frame")
    anchors = segment_work / "anchors"
    anchors.mkdir()
    first_anchor = anchors / "first.png"
    last_anchor = anchors / "last.png"
    first_anchor.write_bytes(b"source first")
    last_anchor.write_bytes(b"source last")
    visual = segment_work / "visual_prompt.txt"
    visual.write_text("local action", encoding="utf-8")
    final = segment_work / "prompt.txt"
    final.write_text("global continuity\nlocal action", encoding="utf-8")
    dialogue = [{"text": "hello", "start_s": 0.0, "end_s": 1.0}]
    duration = 15.000001
    first_segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 14.0,
        "chain_id": "chain-001",
        "join_mode": "hard_cut",
        "source_path": segment_source,
        "keyframe_paths": [frame],
        "first_frame_path": first_anchor,
        "last_frame_path": last_anchor,
        "visual_prompt_path": visual,
        "final_prompt_path": final,
        "dialogue": dialogue,
    }
    segments = [first_segment, {
        **first_segment,
        "index": 2,
        "start_s": 14.0,
        "end_s": duration,
        "join_mode": "continue",
    }]

    first = write_plan_receipt(
        tmp_path,
        source=source,
        duration_s=duration,
        segments=segments,
        workflow="minimax_h3_lightx2v",
    )
    first_bytes = first.read_bytes()
    second = write_plan_receipt(
        tmp_path,
        source=source,
        duration_s=duration,
        segments=segments,
        workflow="minimax_h3_lightx2v",
    )
    assert second.read_bytes() == first_bytes

    receipt = json.loads(first_bytes)
    assert receipt["schema"] == "duet.long-video-plan"
    assert receipt["version"] == 2
    assert receipt["source"]["sha256"] == hashlib.sha256(b"whole source").hexdigest()
    assert receipt["video"]["duration_s"] == duration
    assert receipt["workflow"] == "minimax_h3_lightx2v"
    planned = receipt["segments"][0]
    assert (planned["start_s"], planned["end_s"]) == (0.0, 14.0)
    assert (planned["chain_id"], planned["join_mode"]) == ("chain-001", "hard_cut")
    assert planned["source"]["sha256"] == hashlib.sha256(b"segment source").hexdigest()
    assert planned["keyframes"][0]["sha256"] == hashlib.sha256(b"frame").hexdigest()
    assert planned["anchors"] == [
        {
            "role": "first",
            "path": "work/segments/1/work/anchors/first.png",
            "sha256": hashlib.sha256(b"source first").hexdigest(),
        },
        {
            "role": "end",
            "path": "work/segments/1/work/anchors/last.png",
            "sha256": hashlib.sha256(b"source last").hexdigest(),
        },
    ]
    assert planned["visual_prompt"]["sha256"] == hashlib.sha256(b"local action").hexdigest()
    assert planned["final_prompt"]["sha256"] == hashlib.sha256(
        b"global continuity\nlocal action"
    ).hexdigest()
    assert planned["dialogue"]["count"] == 1


def test_visual_plan_receipt_binds_keyframe_source_timeline(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"whole source")
    segment_dir = tmp_path / "work" / "segments" / "1"
    segment_work = segment_dir / "work"
    keyframes = segment_work / "keyframes"
    anchors = segment_work / "anchors"
    keyframes.mkdir(parents=True)
    anchors.mkdir()
    segment_source = segment_dir / "source.mp4"
    segment_source.write_bytes(b"segment source")
    frame_paths = []
    keyframe_sources = []
    for order, source_time_s in enumerate(
        [0.0, 0.75, 2.0, 2.5, 4.0, 6.0, 8.0, 11.0, 14.0], 1
    ):
        frame = keyframes / f"{order:02d}.png"
        frame.write_bytes(f"frame-{order}".encode())
        frame_paths.append(frame)
        keyframe_sources.append({
            "order": order,
            "source_time_s": source_time_s,
            "source_scene_id": "SCENE_01" if order < 4 else "SCENE_02",
            "transition": (
                {"type": "start", "at_s": 0.0}
                if order == 1 else
                {"type": "hard_cut", "at_s": 2.267}
                if order == 4 else
                {"type": "same_camera", "at_s": None}
            ),
        })
    first = anchors / "first.png"
    last = anchors / "last.png"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    visual = segment_work / "visual_prompt.txt"
    final = segment_work / "prompt.txt"
    visual.write_text("visual", encoding="utf-8")
    final.write_text("final", encoding="utf-8")

    path = write_plan_receipt(
        tmp_path,
        source=source,
        duration_s=14.5,
        segments=[{
            "index": 1,
            "start_s": 0.0,
            "end_s": 14.5,
            "chain_id": "chain-001",
            "join_mode": "hard_cut",
            "source_path": segment_source,
            "keyframe_paths": frame_paths,
            "keyframe_sources": keyframe_sources,
            "first_frame_path": first,
            "last_frame_path": last,
            "visual_prompt_path": visual,
            "final_prompt_path": final,
            "dialogue": [],
        }],
        workflow="minimax_h3_lightx2v",
    )

    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["version"] == long_video_module.VISUAL_PLAN_RECEIPT_VERSION
    assert receipt["segments"][0]["keyframe_sources"] == keyframe_sources


@pytest.mark.parametrize("duration", [30.0])
def test_exact_planner_receipts_bind_source_anchors_not_codex_keyframes(tmp_path, duration):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    planned = plan_segments(duration, [(0.0, duration)], [])
    receipt_input = []
    for segment in planned:
        index = segment["index"]
        segdir = tmp_path / "work" / "segments" / str(index)
        segwork = segdir / "work"
        keyframes = segwork / "keyframes"
        anchors = segwork / "anchors"
        keyframes.mkdir(parents=True)
        anchors.mkdir()
        segment_source = segdir / "source.mp4"
        segment_source.write_bytes(f"segment-{index}".encode())
        selected = keyframes / "01.png"
        selected.write_bytes(f"codex-selected-{index}".encode())
        first = anchors / "first.png"
        last = anchors / "last.png"
        first.write_bytes(f"source-first-{index}".encode())
        last.write_bytes(f"source-last-{index}".encode())
        visual = segwork / "visual_prompt.txt"
        final = segwork / "prompt.txt"
        visual.write_text(f"visual-{index}", encoding="utf-8")
        final.write_text(f"final-{index}", encoding="utf-8")
        receipt_input.append(
            {
                **segment,
                "source_path": segment_source,
                "keyframe_paths": [selected],
                "first_frame_path": first,
                "last_frame_path": last,
                "visual_prompt_path": visual,
                "final_prompt_path": final,
                "dialogue": [],
            }
        )

    path = write_plan_receipt(
        tmp_path,
        source=source,
        duration_s=duration,
        segments=receipt_input,
        workflow="minimax_h3_lightx2v",
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert len(receipt["segments"]) == len(planned)
    for segment in receipt["segments"]:
        assert [anchor["role"] for anchor in segment["anchors"]] == ["first", "end"]
        anchor_hashes = {anchor["sha256"] for anchor in segment["anchors"]}
        assert segment["keyframes"][0]["sha256"] not in anchor_hashes
