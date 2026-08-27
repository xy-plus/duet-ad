import hashlib
import json
from types import SimpleNamespace

import pytest

from app import dialogue_timing, h3, h3_project, stitch


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _speaker_timing(*, start_pts: int = 750, end_pts: int = 4000) -> dict:
    timeline = {
        "time_base": {"numerator": 1, "denominator": 1000},
        "keyframes": [
            {"order": 1, "sha256": "1" * 64, "pts": 0},
            {"order": 2, "sha256": "2" * 64, "pts": 750},
            {"order": 3, "sha256": "3" * 64, "pts": 4000},
        ],
    }
    return {
        "schema": "duet.speaker-timing",
        "version": 1,
        "source_sha256": "a" * 64,
        "timeline": timeline,
        "timeline_sha256": _sha(timeline),
        "speakers": [
            {
                "subject_id": "S1",
                "windows": [
                    {
                        "kind": "lip_verifiable",
                        "status": "verified",
                        "start_pts": start_pts,
                        "end_pts": end_pts,
                        "evidence_keyframes": [2, 3],
                    }
                ],
            }
        ],
    }


def test_authoritative_dialogue_must_be_inside_verified_lip_window():
    frozen = dialogue_timing.freeze_speaker_timing(
        _speaker_timing(),
        source_sha256="a" * 64,
        keyframe_sha256s=("1" * 64, "2" * 64, "3" * 64),
    )

    dialogue_timing.require_authoritative_window(
        frozen,
        subject_id="S1",
        start_s=0.75,
        end_s=3.6,
    )
    with pytest.raises(
        dialogue_timing.DialogueTimingError,
        match="dialogue_before_speaker_lip_window",
    ):
        dialogue_timing.require_authoritative_window(
            frozen,
            subject_id="S1",
            start_s=0.4,
            end_s=3.6,
        )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update(source_sha256="b" * 64), "speaker_timing_source_mismatch"),
        (
            lambda value: value["timeline"]["keyframes"][1].update(sha256="9" * 64),
            "speaker_timing_timeline_mismatch",
        ),
        (
            lambda value: value["speakers"][0]["windows"][0].update(status="estimated"),
            "speaker_timing_unverified",
        ),
    ],
)
def test_speaker_timing_never_accepts_unbound_or_inferred_evidence(mutation, code):
    artifact = _speaker_timing()
    mutation(artifact)

    with pytest.raises(dialogue_timing.DialogueTimingError, match=code):
        dialogue_timing.freeze_speaker_timing(
            artifact,
            source_sha256="a" * 64,
            keyframe_sha256s=("1" * 64, "2" * 64, "3" * 64),
        )


def _acceptance(
    *,
    asr_start_pts: int = 760,
    asr_end_pts: int = 3580,
    lip_start_pts: int = 750,
    lip_end_pts: int = 3600,
) -> dict:
    return {
        "schema": "duet.dialogue-av-acceptance",
        "version": 1,
        "output": {
            "sha256": "f" * 64,
            "size": 1234,
            "media_timeline_sha256": "e" * 64,
        },
        "authority": {
            "dialogue_sha256": "d" * 64,
            "speaker_timing_sha256": "c" * 64,
        },
        "max_asr_boundary_drift_ms": 250,
        "asr": {
            "engine": "whisper.cpp",
            "model_sha256": "b" * 64,
            "transcript_sha256": "9" * 64,
            "unmatched_speech_count": 0,
        },
        "lip": {
            "engine": "local-lip-verifier",
            "model_sha256": "8" * 64,
            "analysis_sha256": "7" * 64,
        },
        "lines": [
            {
                "line_index": 1,
                "subject_id": "S1",
                "text_sha256": hashlib.sha256(
                    "准备好了，现在让它飞起来。".encode("utf-8")
                ).hexdigest(),
                "time_base": {"numerator": 1, "denominator": 1000},
                "asr_start_pts": asr_start_pts,
                "asr_end_pts": asr_end_pts,
                "lip_start_pts": lip_start_pts,
                "lip_end_pts": lip_end_pts,
                "lip_status": "verified",
            }
        ],
    }


def _authoritative_dialogue() -> tuple[dict, ...]:
    return (
        {
            "text": "准备好了，现在让它飞起来。",
            "start_s": 0.75,
            "end_s": 3.6,
        },
    )


def test_final_acceptance_binds_output_authority_and_allows_small_asr_drift():
    frozen = dialogue_timing.validate_final_acceptance(
        _acceptance(),
        dialogue=_authoritative_dialogue(),
        subjects=("S1",),
        output_sha256="f" * 64,
        output_size=1234,
        media_timeline_sha256="e" * 64,
        dialogue_sha256="d" * 64,
        speaker_timing_sha256="c" * 64,
    )

    assert frozen.sha256 == _sha(_acceptance())


@pytest.mark.parametrize(
    ("artifact", "code"),
    [
        (_acceptance(asr_start_pts=0, asr_end_pts=2440), "asr_authority_window_mismatch"),
        (_acceptance(lip_start_pts=800), "final_lip_window_mismatch"),
        (_acceptance(), "final_output_binding_mismatch"),
    ],
)
def test_final_acceptance_fails_closed_on_timing_or_output_drift(artifact, code):
    kwargs = {
        "dialogue": _authoritative_dialogue(),
        "subjects": ("S1",),
        "output_sha256": "f" * 64,
        "output_size": 1234,
        "media_timeline_sha256": "e" * 64,
        "dialogue_sha256": "d" * 64,
        "speaker_timing_sha256": "c" * 64,
    }
    if code == "final_output_binding_mismatch":
        kwargs["output_sha256"] = "0" * 64

    with pytest.raises(dialogue_timing.DialogueTimingError, match=code):
        dialogue_timing.validate_final_acceptance(artifact, **kwargs)


def test_final_acceptance_rejects_missing_lines_extra_speech_and_unverified_lips():
    for mutate, code in (
        (lambda value: value.update(lines=[]), "final_dialogue_evidence_missing"),
        (
            lambda value: value["asr"].update(unmatched_speech_count=1),
            "final_unmatched_speech",
        ),
        (
            lambda value: value["lines"][0].update(lip_status="unknown"),
            "final_lip_unverified",
        ),
    ):
        artifact = _acceptance()
        mutate(artifact)
        with pytest.raises(dialogue_timing.DialogueTimingError, match=code):
            dialogue_timing.validate_final_acceptance(
                artifact,
                dialogue=_authoritative_dialogue(),
                subjects=("S1",),
                output_sha256="f" * 64,
                output_size=1234,
                media_timeline_sha256="e" * 64,
                dialogue_sha256="d" * 64,
                speaker_timing_sha256="c" * 64,
            )


def test_h3_project_native_gate_requires_and_binds_exact_acceptance_file(tmp_path):
    output = tmp_path / "generated.mp4"
    output.write_bytes(b"exact-h3-output")
    timeline = {
        "schema": "duet.h3.media_timeline",
        "version": 1,
        "decode_complete": True,
        "video": {"frame_end_s": 4.0},
        "audio": {"frame_end_s": 4.0, "decoded_sha256": "6" * 64},
        "av_delta_s": {"start": 0.0, "end": 0.0},
        "container": {"duration_s": 4.0},
    }
    artifact = _acceptance()
    artifact["output"] = {
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "size": output.stat().st_size,
        "media_timeline_sha256": h3.canonical_json_sha256(timeline),
    }
    path = tmp_path / h3_project.FINAL_ACCEPTANCE_FILENAME
    path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    request = SimpleNamespace(
        workdir=tmp_path,
        on_screen_dialogue=(
            {
                "text": "准备好了，现在让它飞起来。",
                "start_s": 0.75,
                "end_s": 3.6,
                "subject_id": "S1",
            },
        ),
        upstream_dialogue_receipt_sha256="d" * 64,
        speaker_timing_sha256="c" * 64,
    )

    digest = h3_project.validate_dialogue_acceptance(
        request=request,
        output=output,
        timeline=timeline,
    )
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    segment = stitch.StitchSegment(
        output,
        4.0,
        "hard_cut",
        "000001",
        timeline,
        digest,
    )
    assert stitch._provider_binding(segment, 1)[
        "dialogue_acceptance_sha256"
    ] == digest

    path.unlink()
    with pytest.raises(
        h3_project.ProjectMultimodalError,
        match="final_dialogue_evidence_missing",
    ):
        h3_project.validate_dialogue_acceptance(
            request=request,
            output=output,
            timeline=timeline,
        )
