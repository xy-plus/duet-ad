"""Offline evidence inventory and rubric recorder for real fusion artifacts.

The input and artifact are produced by an actual Skill run.  This helper only
freezes byte/order evidence and records human semantic ratings; it does not
score SKILL.md text and has no runtime or pass/fail gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "material_sha_order_binding",
    "story_continuity",
    "cross_segment_character_consistency",
    "action_phase_direction_consistency",
    "environment_consistency",
    "hard_cut_non_projection",
    "dialogue_audio_timing_alignment",
    "stable_key_tile_consumption",
)

SCORE_ANCHORS = {
    "0": "Missing or contradicts frozen evidence.",
    "1": "Most relevant claims are unbound to frozen evidence.",
    "2": "Partly bound, with material omissions or conflicts.",
    "3": "Main evidence is bound correctly, with localized gaps.",
    "4": "Relevant claims are traceably bound with no observed overreach.",
}

_BINDING_RE = re.compile(
    r"全项目共享替换参考板绑定：\s*([^\n；;]+?)\s*->\s*"
    r"(TILE_[A-Za-z0-9_-]+)\s*->\s*([^\n]+)"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return raw, value


def _extract_bindings(text: str) -> list[dict[str, str]]:
    return [
        {
            "stable_key": match.group(1).strip(),
            "tile_id": match.group(2).strip(),
            "replacement_description": match.group(3).strip(),
        }
        for match in _BINDING_RE.finditer(text)
    ]


def _intervals(segment: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[dict[str, str]]]]:
    intervals: list[dict[str, Any]] = []
    interval_bindings: list[list[dict[str, str]]] = []
    current: dict[str, Any] | None = None
    current_bindings: list[dict[str, str]] = []
    for frame, prompt in zip(
        segment["new_keyframes"], segment["image_optimization_prompt"], strict=True
    ):
        if current is None or frame["transition"]["type"] == "hard_cut":
            if current is not None:
                interval_bindings.append(current_bindings)
            current = {
                "ordinal": len(intervals) + 1,
                "start_order": frame["order"],
                "start_segment_time_s": frame["segment_time_s"],
                "hard_cut_at_segment_s": frame["transition"]["at_segment_s"],
                "source_scene_ids": [],
            }
            intervals.append(current)
            current_bindings = []
        scene_id = frame["source_scene_id"]
        if scene_id not in current["source_scene_ids"]:
            current["source_scene_ids"].append(scene_id)
        current["end_order"] = frame["order"]
        current["end_segment_time_s"] = frame["segment_time_s"]
        current_bindings.extend(_extract_bindings(prompt["text"]))
    if current is not None:
        interval_bindings.append(current_bindings)
    return intervals, interval_bindings


def build_inventory(input_path: Path, artifact_path: Path) -> dict[str, Any]:
    input_raw, frozen = _load_json(input_path)
    artifact_raw, artifact = _load_json(artifact_path)
    project_root = input_path.parent.parent.resolve()
    output_by_index = {
        segment.get("index"): segment
        for segment in artifact.get("segments", [])
        if isinstance(segment, dict)
    }
    segments: list[dict[str, Any]] = []

    for source_segment in frozen["segments"]:
        intervals, interval_bindings = _intervals(source_segment)
        media: list[dict[str, Any]] = []
        frame_bindings: list[dict[str, Any]] = []
        for frame, prompt in zip(
            source_segment["new_keyframes"],
            source_segment["image_optimization_prompt"],
            strict=True,
        ):
            image_path = (project_root / frame["path"]).resolve()
            actual_sha = (
                _sha256_bytes(image_path.read_bytes()) if image_path.is_file() else None
            )
            media.append(
                {
                    "order": frame["order"],
                    "path": frame["path"],
                    "declared_sha256": frame["sha256"],
                    "actual_sha256": actual_sha,
                    "sha_matches": actual_sha == frame["sha256"],
                    "segment_time_s": frame["segment_time_s"],
                    "source_scene_id": frame["source_scene_id"],
                    "transition": frame["transition"],
                }
            )
            frame_bindings.append(
                {
                    "order": prompt["order"],
                    "prompt_sha256": prompt["sha256"],
                    "bindings": _extract_bindings(prompt["text"]),
                }
            )

        audio_lines = json.loads(source_segment["audio_content"]["lines_json"])
        output_segment = output_by_index.get(source_segment["index"], {})
        visuals = output_segment.get("visual", [])
        all_bindings = [item for frame in frame_bindings for item in frame["bindings"]]
        known_tokens = sorted(
            {
                token
                for item in all_bindings
                for token in (item["stable_key"], item["tile_id"])
            }
        )
        visual_text = "\n".join(value for value in visuals if isinstance(value, str))
        segments.append(
            {
                "index": source_segment["index"],
                "media": media,
                "hard_cut_intervals": intervals,
                "interval_bindings": interval_bindings,
                "expected_visual_count": len(intervals),
                "actual_visual_count": len(visuals),
                "visual": visuals,
                "visual_binding_token_leaks": [
                    token for token in known_tokens if token in visual_text
                ],
                "old_video_prompt_sha256": source_segment["old_video_prompt"]["sha256"],
                "frame_prompt_bindings": frame_bindings,
                "audio_lines_sha256": source_segment["audio_content"]["lines_sha256"],
                "audio_lines": audio_lines,
                "audio_text_leaks": [
                    line["text"]
                    for line in audio_lines
                    if isinstance(line, dict)
                    and isinstance(line.get("text"), str)
                    and line["text"]
                    and line["text"] in visual_text
                ],
            }
        )

    return {
        "rubric": {"dimensions": list(DIMENSIONS), "score_anchors": SCORE_ANCHORS},
        "input": {
            "path": str(input_path.resolve()),
            "sha256": _sha256_bytes(input_raw),
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": _sha256_bytes(artifact_raw),
            "declared_input_sha256": artifact.get("input_sha256"),
            "schema": artifact.get("schema"),
            "version": artifact.get("version"),
        },
        "segments": segments,
    }


def attach_ratings(inventory: dict[str, Any], ratings_path: Path) -> None:
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    if set(ratings) != set(DIMENSIONS):
        raise ValueError("ratings must contain exactly the eight rubric dimensions")
    for dimension, rating in ratings.items():
        if set(rating) != {"score", "evidence"}:
            raise ValueError(f"{dimension}: expected score and evidence")
        if not isinstance(rating["score"], int) or rating["score"] not in range(5):
            raise ValueError(f"{dimension}: score must be an integer from 0 through 4")
        if not isinstance(rating["evidence"], list) or not rating["evidence"]:
            raise ValueError(f"{dimension}: evidence must be a non-empty list")
    inventory["ratings"] = ratings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    inventory = build_inventory(args.input, args.artifact)
    if args.ratings:
        attach_ratings(inventory, args.ratings)
    args.report.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
