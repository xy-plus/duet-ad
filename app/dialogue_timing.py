"""Strict evidence gates for on-screen dialogue timing.

This module never derives speaker identity or visibility from ASR, prompts, or
picture references.  Callers must supply a frozen, byte-bound authoritative
speaker timeline before H3 and an independently produced output audit after H3.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence


SPEAKER_TIMING_SCHEMA = "duet.speaker-timing"
FINAL_ACCEPTANCE_SCHEMA = "duet.dialogue-av-acceptance"
SCHEMA_VERSION = 1
MAX_ASR_BOUNDARY_DRIFT_MS = 250
SPEAKER_VISIBILITY_INPUT_SCHEMA = "duet.speaker-visibility-input"
SPEAKER_VISIBILITY_OUTPUT_SCHEMA = "duet.speaker-visibility-output"
SPEAKER_TIMING_PRODUCTION_SCHEMA = "duet.speaker-timing-production"
SPEAKER_VISIBILITY_ALGORITHM = "decoded_pts_nearest_v1"
SPEAKER_VISIBILITY_CADENCE_FPS = 8
MAX_SPEAKER_VISIBILITY_SAMPLES = 15 * SPEAKER_VISIBILITY_CADENCE_FPS
_PERSON_ID = re.compile(r"PERSON_[0-9]{2}")


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


@dataclass(frozen=True, slots=True)
class FrozenSpeakerTimingProduction:
    speaker_timing: Mapping[str, Any]
    receipt: Mapping[str, Any]


def _json_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError):
        _fail("speaker_visibility_artifact_invalid")


def _raw_json_object(data: bytes, code: str) -> Mapping[str, Any]:
    if not isinstance(data, bytes) or not data:
        _fail(code)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def freeze_speaker_visibility(
    *,
    producer_input_data: bytes,
    skill_output_data: bytes,
    source_data: bytes,
    frame_data: Mapping[str, bytes],
    skill_data: bytes,
) -> FrozenSpeakerTimingProduction:
    """Project sampled Skill evidence into the existing strict timing schema.

    The Skill classifies frozen real-frame samples and maps semantic subjects to
    authoritative PERSON ids.  It never receives dialogue intervals.  This
    backend projection only joins adjacent verified samples, breaks at cuts or
    unknown gaps, and removes one observed sample interval from both ends.
    """
    raw_input = _object(
        _raw_json_object(
            producer_input_data, "speaker_visibility_input_invalid"
        ),
        {
            "schema", "version", "phase", "source", "sampling", "frames",
            "decoded_frame_pts", "cut_pts", "cut_source", "contact_sheets",
            "persons", "on_screen_subjects",
        },
        "speaker_visibility_input_invalid",
    )
    if (
        raw_input["schema"] != SPEAKER_VISIBILITY_INPUT_SCHEMA
        or raw_input["version"] != SCHEMA_VERSION
        or raw_input["phase"] != "speaker_visibility"
        or not isinstance(source_data, bytes)
        or not source_data
        or not isinstance(skill_data, bytes)
        or not skill_data
        or not isinstance(frame_data, Mapping)
    ):
        _fail("speaker_visibility_input_invalid")
    source = _object(
        raw_input["source"], {"sha256", "duration_pts", "time_base"},
        "speaker_visibility_input_invalid",
    )
    duration_pts = _pts(
        source["duration_pts"], "speaker_visibility_input_invalid"
    )
    base = _time_base(source["time_base"], "speaker_visibility_input_invalid")
    if (
        duration_pts < 1
        or duration_pts * base > 15
        or not _is_sha256(source["sha256"])
        or source["sha256"] != hashlib.sha256(source_data).hexdigest()
    ):
        _fail("speaker_visibility_source_mismatch")
    sampling = _object(
        raw_input["sampling"],
        {
            "algorithm", "cadence_fps", "max_unobserved_gap_pts",
            "endpoint_shrink_intervals",
        },
        "speaker_visibility_sampling_invalid",
    )
    max_gap = _pts(
        sampling["max_unobserved_gap_pts"],
        "speaker_visibility_sampling_invalid",
    )
    if (
        sampling["algorithm"] != SPEAKER_VISIBILITY_ALGORITHM
        or sampling["cadence_fps"] != SPEAKER_VISIBILITY_CADENCE_FPS
        or sampling["endpoint_shrink_intervals"] != 1
        or max_gap < 1
    ):
        _fail("speaker_visibility_sampling_invalid")

    dense_pts = raw_input["decoded_frame_pts"]
    if (
        not isinstance(dense_pts, list)
        or not dense_pts
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= duration_pts
            for value in dense_pts
        )
        or dense_pts != sorted(set(dense_pts))
    ):
        _fail("speaker_visibility_dense_inventory_invalid")
    nominal_gap = math.ceil(Fraction(1, SPEAKER_VISIBILITY_CADENCE_FPS) / base)
    if max_gap != nominal_gap:
        _fail("speaker_visibility_sampling_invalid")
    cut_pts = raw_input["cut_pts"]
    if (
        not isinstance(cut_pts, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value < duration_pts
            for value in cut_pts
        )
        or cut_pts != sorted(set(cut_pts))
    ):
        _fail("speaker_visibility_cut_inventory_invalid")
    cut_source = _object(
        raw_input["cut_source"], {"path", "sha256"},
        "speaker_visibility_cut_inventory_invalid",
    )
    if (
        cut_source["path"] != "scenes.json"
        or not _is_sha256(cut_source["sha256"])
        or not isinstance(frame_data.get("scenes.json"), bytes)
        or hashlib.sha256(frame_data["scenes.json"]).hexdigest()
        != cut_source["sha256"]
    ):
        _fail("speaker_visibility_cut_inventory_invalid")

    expected_sample_pts: list[int] = []
    sample_index = 0
    while Fraction(sample_index, SPEAKER_VISIBILITY_CADENCE_FPS) < duration_pts * base:
        target = Fraction(sample_index, SPEAKER_VISIBILITY_CADENCE_FPS) / base
        nearest = min(dense_pts, key=lambda value: (abs(Fraction(value) - target), value))
        if not expected_sample_pts or nearest != expected_sample_pts[-1]:
            expected_sample_pts.append(nearest)
        sample_index += 1

    frames = raw_input["frames"]
    if (
        not isinstance(frames, list)
        or not 4 <= len(frames) <= MAX_SPEAKER_VISIBILITY_SAMPLES
    ):
        _fail("speaker_visibility_sample_inventory_invalid")
    frozen_frames: list[Mapping[str, Any]] = []
    frame_paths: set[str] = set()
    previous_pts = -1
    for expected_order, value in enumerate(frames, 1):
        frame = _object(
            value, {"order", "path", "sha256", "pts", "cut_before"},
            "speaker_visibility_sample_inventory_invalid",
        )
        path, pts = frame["path"], _pts(
            frame["pts"], "speaker_visibility_sample_inventory_invalid"
        )
        if (
            frame["order"] != expected_order
            or not isinstance(path, str)
            or not path.startswith("speaker-visibility-frames/")
            or path in frame_paths
            or not isinstance(frame["cut_before"], bool)
            or pts <= previous_pts
            or pts >= duration_pts
            or not _is_sha256(frame["sha256"])
            or not isinstance(frame_data.get(path), bytes)
            or hashlib.sha256(frame_data[path]).hexdigest() != frame["sha256"]
        ):
            _fail("speaker_visibility_sample_inventory_invalid")
        frozen_frames.append(frame)
        frame_paths.add(path)
        previous_pts = pts
    if [frame["pts"] for frame in frozen_frames] != expected_sample_pts:
        _fail("speaker_visibility_sample_inventory_invalid")
    for index, frame in enumerate(frozen_frames):
        previous = frozen_frames[index - 1]["pts"] if index else -1
        expected_cut = any(previous < value <= frame["pts"] for value in cut_pts)
        if frame["cut_before"] != expected_cut:
            _fail("speaker_visibility_cut_inventory_invalid")

    sheets = raw_input["contact_sheets"]
    if not isinstance(sheets, list) or not 1 <= len(sheets) <= 8:
        _fail("speaker_visibility_contact_sheet_invalid")
    covered_orders: list[int] = []
    sheet_paths: set[str] = set()
    for expected_order, value in enumerate(sheets, 1):
        sheet = _object(
            value, {"order", "path", "sha256", "frame_orders"},
            "speaker_visibility_contact_sheet_invalid",
        )
        path, orders = sheet["path"], sheet["frame_orders"]
        if (
            sheet["order"] != expected_order
            or not isinstance(path, str)
            or not path.startswith("speaker-visibility-contact-sheets/")
            or path in sheet_paths
            or not _is_sha256(sheet["sha256"])
            or not isinstance(frame_data.get(path), bytes)
            or hashlib.sha256(frame_data[path]).hexdigest() != sheet["sha256"]
            or not isinstance(orders, list)
            or not 1 <= len(orders) <= 16
            or any(
                isinstance(order, bool)
                or not isinstance(order, int)
                or not 1 <= order <= len(frames)
                for order in orders
            )
        ):
            _fail("speaker_visibility_contact_sheet_invalid")
        covered_orders.extend(orders)
        sheet_paths.add(path)
    if covered_orders != list(range(1, len(frames) + 1)):
        _fail("speaker_visibility_contact_sheet_invalid")
    persons = raw_input["persons"]
    if not isinstance(persons, list) or not 1 <= len(persons) <= 3:
        _fail("speaker_visibility_person_inventory_invalid")
    person_ids: set[str] = set()
    identity_paths: set[str] = set()
    for value in persons:
        person = _object(
            value, {"person_id", "identity_refs"},
            "speaker_visibility_person_inventory_invalid",
        )
        person_id, identity_refs = person["person_id"], person["identity_refs"]
        if (
            not isinstance(person_id, str)
            or _PERSON_ID.fullmatch(person_id) is None
            or person_id in person_ids
            or not isinstance(identity_refs, list)
            or not identity_refs
        ):
            _fail("speaker_visibility_person_inventory_invalid")
        for reference_value in identity_refs:
            reference = _object(
                reference_value, {"path", "sha256"},
                "speaker_visibility_person_inventory_invalid",
            )
            path = reference["path"]
            if (
                not isinstance(path, str)
                or not path.startswith("keyframes/")
                or path in identity_paths
                or not _is_sha256(reference["sha256"])
                or not isinstance(frame_data.get(path), bytes)
                or hashlib.sha256(frame_data[path]).hexdigest()
                != reference["sha256"]
            ):
                _fail("speaker_visibility_person_inventory_invalid")
            identity_paths.add(path)
        person_ids.add(person_id)
    if set(frame_data) != frame_paths | sheet_paths | identity_paths | {"scenes.json"}:
        _fail("speaker_visibility_sample_inventory_invalid")
    subjects = raw_input["on_screen_subjects"]
    if (
        not isinstance(subjects, list)
        or not 1 <= len(subjects) <= 3
        or any(not isinstance(item, str) or not item for item in subjects)
        or subjects != sorted(set(subjects))
    ):
        _fail("speaker_visibility_subject_inventory_invalid")

    raw_output = _object(
        _raw_json_object(
            skill_output_data, "speaker_visibility_output_invalid"
        ),
        {
            "schema", "version", "phase", "input_sha256",
            "subject_person_mapping", "frames",
        },
        "speaker_visibility_output_invalid",
    )
    if (
        raw_output["schema"] != SPEAKER_VISIBILITY_OUTPUT_SCHEMA
        or raw_output["version"] != SCHEMA_VERSION
        or raw_output["phase"] != "speaker_visibility"
        or raw_output["input_sha256"]
        != hashlib.sha256(producer_input_data).hexdigest()
    ):
        _fail("speaker_visibility_output_invalid")
    mappings = raw_output["subject_person_mapping"]
    if not isinstance(mappings, list) or len(mappings) != len(subjects):
        _fail("speaker_visibility_mapping_invalid")
    subject_to_person: dict[str, str] = {}
    mapped_persons: set[str] = set()
    for expected_subject, value in zip(subjects, mappings, strict=True):
        mapping = _object(
            value, {"subject_id", "person_id"},
            "speaker_visibility_mapping_invalid",
        )
        subject_id, person_id = mapping["subject_id"], mapping["person_id"]
        if (
            subject_id != expected_subject
            or person_id not in person_ids
            or person_id in mapped_persons
        ):
            _fail("speaker_visibility_mapping_invalid")
        subject_to_person[subject_id] = person_id
        mapped_persons.add(person_id)

    output_frames = raw_output["frames"]
    if not isinstance(output_frames, list) or len(output_frames) != len(frames):
        _fail("speaker_visibility_output_invalid")
    evidence_by_order: dict[int, tuple[set[str], set[str]]] = {}
    for expected_order, value in enumerate(output_frames, 1):
        evidence = _object(
            value, {"order", "visible_person_ids", "lip_verifiable_person_ids"},
            "speaker_visibility_output_invalid",
        )
        visible, lips = evidence["visible_person_ids"], evidence[
            "lip_verifiable_person_ids"
        ]
        if (
            evidence["order"] != expected_order
            or not isinstance(visible, list)
            or not isinstance(lips, list)
            or visible != sorted(set(visible))
            or lips != sorted(set(lips))
            or not set(visible) <= person_ids
            or not set(lips) <= set(visible)
        ):
            _fail("speaker_visibility_output_invalid")
        evidence_by_order[expected_order] = (set(visible), set(lips))

    speaker_entries: list[dict[str, Any]] = []
    for subject_id in subjects:
        person_id = subject_to_person[subject_id]
        runs: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        for frame in frozen_frames:
            visible, lips = evidence_by_order[frame["order"]]
            verified = person_id in visible and person_id in lips
            contiguous = (
                bool(current)
                and not frame["cut_before"]
                and frame["pts"] - current[-1]["pts"] <= max_gap
            )
            if not verified:
                if current:
                    runs.append(current)
                    current = []
                continue
            if current and not contiguous:
                runs.append(current)
                current = []
            current.append(frame)
        if current:
            runs.append(current)
        windows: list[dict[str, Any]] = []
        for run in runs:
            if len(run) < 4:
                continue
            start_pts, end_pts = run[1]["pts"], run[-2]["pts"]
            if start_pts >= end_pts:
                continue
            windows.append({
                "kind": "lip_verifiable",
                "status": "verified",
                "start_pts": start_pts,
                "end_pts": end_pts,
                "evidence_keyframes": [
                    frame["order"] for frame in run[1:-1]
                ],
            })
        if not windows:
            _fail("speaker_visibility_subject_unverified")
        speaker_entries.append({"subject_id": subject_id, "windows": windows})

    timeline = {
        "time_base": {
            "numerator": base.numerator,
            "denominator": base.denominator,
        },
        "duration_pts": duration_pts,
        "keyframes": [
            {
                "order": frame["order"],
                "sha256": frame["sha256"],
                "pts": frame["pts"],
            }
            for frame in frozen_frames
        ],
    }
    speaker_timing = {
        "schema": SPEAKER_TIMING_SCHEMA,
        "version": SCHEMA_VERSION,
        "source_sha256": source["sha256"],
        "timeline": timeline,
        "timeline_sha256": canonical_sha256(timeline),
        "speakers": speaker_entries,
    }
    timing_data = _json_bytes(speaker_timing)
    receipt = {
        "schema": SPEAKER_TIMING_PRODUCTION_SCHEMA,
        "version": SCHEMA_VERSION,
        "source_sha256": source["sha256"],
        "source_duration_pts": duration_pts,
        "source_time_base": {
            "numerator": base.numerator,
            "denominator": base.denominator,
        },
        "sampling": dict(sampling),
        "dense_frame_inventory_sha256": canonical_sha256(dense_pts),
        "cut_inventory_sha256": canonical_sha256(cut_pts),
        "cut_source_sha256": cut_source["sha256"],
        "sample_inventory_sha256": canonical_sha256(frames),
        "contact_sheet_inventory_sha256": canonical_sha256(sheets),
        "subject_mapping_sha256": canonical_sha256(mappings),
        "artifacts": {
            "producer_input": {
                "path": "speaker_visibility_input.json",
                "sha256": hashlib.sha256(producer_input_data).hexdigest(),
            },
            "raw_output": {
                "path": "speaker_visibility_output.json",
                "sha256": hashlib.sha256(skill_output_data).hexdigest(),
            },
            "skill": {
                "path": "speaker_visibility_skill.md",
                "sha256": hashlib.sha256(skill_data).hexdigest(),
            },
            "speaker_timing": {
                "path": "speaker_timing.json",
                "sha256": hashlib.sha256(timing_data).hexdigest(),
                "canonical_sha256": canonical_sha256(speaker_timing),
            },
        },
    }
    return FrozenSpeakerTimingProduction(speaker_timing, receipt)


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
