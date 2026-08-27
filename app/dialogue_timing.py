"""Strict evidence gates for on-screen dialogue timing.

This module never derives speaker identity or visibility from ASR, prompts, or
picture references.  Callers must supply a frozen, byte-bound authoritative
speaker timeline before H3 and an independently produced output audit after H3.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence


SPEAKER_TIMING_SCHEMA = "duet.speaker-timing"
FINAL_ACCEPTANCE_SCHEMA = "duet.dialogue-av-acceptance"
SCHEMA_VERSION = 1
MAX_ASR_BOUNDARY_DRIFT_MS = 250


class DialogueTimingError(RuntimeError):
    """Stable fail-closed evidence error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise DialogueTimingError(code)


def canonical_sha256(value: object) -> str:
    try:
        data = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("dialogue_timing_artifact_invalid")
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _object(value: object, keys: set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(code)
    return value


def _time_base(value: object, code: str) -> Fraction:
    raw = _object(value, {"numerator", "denominator"}, code)
    numerator, denominator = raw["numerator"], raw["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator < 1
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 1
    ):
        _fail(code)
    return Fraction(numerator, denominator)


def _pts(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(code)
    return value


def _second(value: object, code: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code)
    number = float(value)
    if not math.isfinite(number) or number < 0:
        _fail(code)
    return Fraction(str(number))


@dataclass(frozen=True, slots=True)
class FrozenLipWindow:
    start: Fraction
    end: Fraction


@dataclass(frozen=True, slots=True)
class FrozenSpeakerTiming:
    sha256: str
    source_sha256: str
    duration: Fraction
    windows: Mapping[str, tuple[FrozenLipWindow, ...]]


@dataclass(frozen=True, slots=True)
class FrozenFinalAcceptance:
    sha256: str


def freeze_speaker_timing(
    artifact: object,
    *,
    source_sha256: str,
    keyframe_sha256s: Sequence[str],
    source_duration_s: float,
) -> FrozenSpeakerTiming:
    """Validate an authoritative source/keyframe/PTS-bound speaker artifact."""
    raw = _object(
        artifact,
        {
            "schema", "version", "source_sha256", "timeline",
            "timeline_sha256", "speakers",
        },
        "speaker_timing_artifact_invalid",
    )
    if raw["schema"] != SPEAKER_TIMING_SCHEMA or raw["version"] != SCHEMA_VERSION:
        _fail("speaker_timing_artifact_invalid")
    if not _is_sha256(source_sha256) or raw["source_sha256"] != source_sha256:
        _fail("speaker_timing_source_mismatch")
    timeline = _object(
        raw["timeline"], {"time_base", "duration_pts", "keyframes"},
        "speaker_timing_timeline_mismatch",
    )
    base = _time_base(timeline["time_base"], "speaker_timing_timeline_mismatch")
    duration_pts = _pts(
        timeline["duration_pts"], "speaker_timing_timeline_mismatch"
    )
    duration = _second(
        source_duration_s, "speaker_timing_duration_mismatch"
    )
    if duration_pts < 1 or duration_pts * base != duration:
        _fail("speaker_timing_duration_mismatch")
    keyframes = timeline["keyframes"]
    expected_hashes = tuple(keyframe_sha256s)
    if (
        not isinstance(keyframes, list)
        or not keyframes
        or len(keyframes) != len(expected_hashes)
        or any(not _is_sha256(value) for value in expected_hashes)
        or raw["timeline_sha256"] != canonical_sha256(timeline)
    ):
        _fail("speaker_timing_timeline_mismatch")
    keyframe_pts: dict[int, int] = {}
    previous_pts = -1
    for order, (item, expected_hash) in enumerate(
        zip(keyframes, expected_hashes, strict=True), 1
    ):
        bound = _object(
            item, {"order", "sha256", "pts"},
            "speaker_timing_timeline_mismatch",
        )
        pts = _pts(bound["pts"], "speaker_timing_timeline_mismatch")
        if (
            bound["order"] != order
            or bound["sha256"] != expected_hash
            or pts <= previous_pts
            or pts >= duration_pts
        ):
            _fail("speaker_timing_pts_out_of_range")
        keyframe_pts[order] = pts
        previous_pts = pts

    speakers = raw["speakers"]
    if not isinstance(speakers, list) or not speakers:
        _fail("speaker_timing_artifact_invalid")
    frozen: dict[str, tuple[FrozenLipWindow, ...]] = {}
    for speaker in speakers:
        item = _object(
            speaker, {"subject_id", "windows"},
            "speaker_timing_artifact_invalid",
        )
        subject_id, windows = item["subject_id"], item["windows"]
        if (
            not isinstance(subject_id, str)
            or not subject_id
            or subject_id in frozen
            or not isinstance(windows, list)
            or not windows
        ):
            _fail("speaker_timing_artifact_invalid")
        values: list[FrozenLipWindow] = []
        previous_end = Fraction(-1)
        for window in windows:
            bound = _object(
                window,
                {
                    "kind", "status", "start_pts", "end_pts",
                    "evidence_keyframes",
                },
                "speaker_timing_artifact_invalid",
            )
            if bound["kind"] != "lip_verifiable" or bound["status"] != "verified":
                _fail("speaker_timing_unverified")
            start_pts = _pts(bound["start_pts"], "speaker_timing_artifact_invalid")
            end_pts = _pts(bound["end_pts"], "speaker_timing_artifact_invalid")
            evidence = bound["evidence_keyframes"]
            if (
                start_pts >= end_pts
                or end_pts > duration_pts
                or not isinstance(evidence, list)
                or not evidence
                or any(
                    isinstance(order, bool)
                    or not isinstance(order, int)
                    or order not in keyframe_pts
                    or not start_pts <= keyframe_pts[order] <= end_pts
                    for order in evidence
                )
                or evidence != sorted(set(evidence))
            ):
                _fail("speaker_timing_pts_out_of_range")
            start, end = start_pts * base, end_pts * base
            if start < previous_end:
                _fail("speaker_timing_artifact_invalid")
            values.append(FrozenLipWindow(start, end))
            previous_end = end
        frozen[subject_id] = tuple(values)
    return FrozenSpeakerTiming(
        canonical_sha256(raw), str(source_sha256), duration, frozen
    )


def require_authoritative_window(
    frozen: FrozenSpeakerTiming,
    *,
    subject_id: str,
    start_s: float,
    end_s: float,
) -> None:
    """Require the full authoritative dialogue interval to be lip-verifiable."""
    if not isinstance(frozen, FrozenSpeakerTiming):
        _fail("speaker_timing_evidence_missing")
    start = _second(start_s, "authoritative_dialogue_window_invalid")
    end = _second(end_s, "authoritative_dialogue_window_invalid")
    if start >= end:
        _fail("authoritative_dialogue_window_invalid")
    windows = frozen.windows.get(subject_id)
    if not windows:
        _fail("speaker_timing_subject_missing")
    if any(window.start <= start and end <= window.end for window in windows):
        return
    if all(start < window.start for window in windows):
        _fail("dialogue_before_speaker_lip_window")
    _fail("dialogue_outside_speaker_lip_window")


def _analyzer(value: object, *, analysis_field: str) -> Mapping[str, Any]:
    keys = {"engine", "model_sha256", analysis_field}
    if analysis_field == "transcript_sha256":
        keys.add("unmatched_speech_count")
    raw = _object(value, keys, "final_dialogue_evidence_invalid")
    if (
        not isinstance(raw["engine"], str)
        or not raw["engine"].strip()
        or not _is_sha256(raw["model_sha256"])
        or not _is_sha256(raw[analysis_field])
    ):
        _fail("final_dialogue_evidence_invalid")
    return raw


def validate_final_acceptance(
    artifact: object,
    *,
    dialogue: Sequence[Mapping[str, Any]],
    subjects: Sequence[str],
    output_sha256: str,
    output_size: int,
    media_timeline_sha256: str,
    dialogue_sha256: str,
    speaker_timing_sha256: str,
) -> FrozenFinalAcceptance:
    """Validate output-bound ASR and lip evidence against authoritative lines."""
    raw = _object(
        artifact,
        {
            "schema", "version", "output", "authority",
            "max_asr_boundary_drift_ms", "asr", "lip", "lines",
        },
        "final_dialogue_evidence_invalid",
    )
    if raw["schema"] != FINAL_ACCEPTANCE_SCHEMA or raw["version"] != SCHEMA_VERSION:
        _fail("final_dialogue_evidence_invalid")
    output = _object(
        raw["output"], {"sha256", "size", "media_timeline_sha256"},
        "final_output_binding_mismatch",
    )
    if output != {
        "sha256": output_sha256,
        "size": output_size,
        "media_timeline_sha256": media_timeline_sha256,
    }:
        _fail("final_output_binding_mismatch")
    authority = _object(
        raw["authority"], {"dialogue_sha256", "speaker_timing_sha256"},
        "final_authority_binding_mismatch",
    )
    if authority != {
        "dialogue_sha256": dialogue_sha256,
        "speaker_timing_sha256": speaker_timing_sha256,
    }:
        _fail("final_authority_binding_mismatch")
    if raw["max_asr_boundary_drift_ms"] != MAX_ASR_BOUNDARY_DRIFT_MS:
        _fail("final_dialogue_evidence_invalid")
    asr = _analyzer(raw["asr"], analysis_field="transcript_sha256")
    _analyzer(raw["lip"], analysis_field="analysis_sha256")
    if asr["unmatched_speech_count"] != 0:
        _fail("final_unmatched_speech")
    lines = raw["lines"]
    expected_dialogue = tuple(dialogue)
    expected_subjects = tuple(subjects)
    if (
        not isinstance(lines, list)
        or len(lines) != len(expected_dialogue)
        or len(expected_subjects) != len(expected_dialogue)
    ):
        _fail("final_dialogue_evidence_missing")
    tolerance = Fraction(MAX_ASR_BOUNDARY_DRIFT_MS, 1000)
    for index, (evidence, expected, expected_subject) in enumerate(
        zip(lines, expected_dialogue, expected_subjects, strict=True), 1
    ):
        item = _object(
            evidence,
            {
                "line_index", "subject_id", "text_sha256", "time_base",
                "asr_start_pts", "asr_end_pts", "lip_start_pts",
                "lip_end_pts", "lip_status",
            },
            "final_dialogue_evidence_invalid",
        )
        if (
            item["line_index"] != index
            or item["subject_id"] != expected_subject
            or not isinstance(expected.get("text"), str)
            or item["text_sha256"]
            != hashlib.sha256(expected["text"].encode("utf-8")).hexdigest()
        ):
            _fail("final_dialogue_evidence_invalid")
        base = _time_base(item["time_base"], "final_dialogue_evidence_invalid")
        asr_start = _pts(item["asr_start_pts"], "final_dialogue_evidence_invalid") * base
        asr_end = _pts(item["asr_end_pts"], "final_dialogue_evidence_invalid") * base
        lip_start = _pts(item["lip_start_pts"], "final_dialogue_evidence_invalid") * base
        lip_end = _pts(item["lip_end_pts"], "final_dialogue_evidence_invalid") * base
        start = _second(expected.get("start_s"), "final_dialogue_evidence_invalid")
        end = _second(expected.get("end_s"), "final_dialogue_evidence_invalid")
        if asr_start >= asr_end or abs(asr_start - start) > tolerance or abs(asr_end - end) > tolerance:
            _fail("asr_authority_window_mismatch")
        if item["lip_status"] != "verified":
            _fail("final_lip_unverified")
        if lip_start > start or lip_end < end or lip_start >= lip_end:
            _fail("final_lip_window_mismatch")
    return FrozenFinalAcceptance(canonical_sha256(raw))
