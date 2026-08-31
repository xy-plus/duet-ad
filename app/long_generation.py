"""Fail-closed orchestration for paid long-video H3 segment generation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from app import (
    context_ir_bridge,
    dialogue_delivery as dialogue_delivery_contract,
    error_trace,
    frame_fit,
    h3,
    h3_project,
    image_optimization,
    long_video,
    postprocess,
    prepared_input,
    stitch,
    storage,
    vocal,
)
from app.config import Settings

WORKFLOW = h3.H3_WORKFLOW
_PLAN_WORKFLOWS = h3.H3_REFERENCE_WORKFLOWS | {h3.H3_BOUNDARY_WORKFLOW}
_LEGACY_PIPELINE_NO_BGM = "不要生成背景音乐"
_EPS = 1e-6
FIT_LAYOUT_LEGACY = "legacy-v0"
FIT_LAYOUT_ASPECT = "aspect-v1"
_FIT_LAYOUTS = frozenset({FIT_LAYOUT_LEGACY, FIT_LAYOUT_ASPECT})
_FAST_MODE_WORKERS = 8
# Context IR mechanically adds this immutable policy before H3 submission.
# Reserve it here so every newly compiled relation prompt has a deterministic
# source fallback that still fits the conservative 7000-character transport.
_MAX_COMPILED_FUSION_CHARS = (
    context_ir_bridge.MAX_SOURCE_PROMPT_CHARS
    - len(context_ir_bridge._DIALOGUE_POLICY)
    - 1
)
_LOGGER = logging.getLogger(__name__)
AUDIO_ROUTE_SCHEMA = "duet.long-generation.audio-route"
AUDIO_ROUTE_VERSION = 1
H3_NATIVE_AUDIO_ROUTE = {
    "schema": AUDIO_ROUTE_SCHEMA,
    "version": AUDIO_ROUTE_VERSION,
    "mode": "h3_native",
}
PROMPT_FUSION_INPUT_SCHEMA = "duet.video-prompt-fusion-input"
PROMPT_FUSION_OUTPUT_SCHEMA = "duet.video-prompt-fusion-output"
PROMPT_FUSION_LEGACY_VERSION = 1
PROMPT_FUSION_VERSION = 2
# Compatibility name retained for the visual-v2 callers and receipts.  Audio
# policy and the source timeline are one indivisible v2 contract.
VISUAL_PROMPT_FUSION_VERSION = PROMPT_FUSION_VERSION
PROMPT_FUSION_MANIFEST_SCHEMA = "duet.video-prompt-fusion-production"
PROMPT_FUSION_MANIFEST_VERSION = 1
PROMPT_FUSION_PROXY_DIR = "prompt-fusion-proxies"
RELATION_STATES_OPEN = "<RELATION_STATES_JSON>"
RELATION_STATES_CLOSE = "</RELATION_STATES_JSON>"
CUT_TIMELINE_OPEN = "<CUT_TIMELINE_JSON>"
CUT_TIMELINE_CLOSE = "</CUT_TIMELINE_JSON>"
_RELATION_CONTRACT_LEGEND = (
    "d[R]=[S,P,O,preserve[],replace];i=[a,b,C,H,[R,runs][]];"
    "run=[a,b,state,geometry];H:0=start/1=hard_cut;direction:S->O;"
    "L2:d+=[mask,s0,sN,g0,gN],i=no-runs"
)
_MAX_RELATION_MARKER_CHARS = 2_000
_RELATION_OCCURRENCE_KEYS = {
    "relation_id", "subject_key", "predicate", "object_key", "state",
    "geometry", "preserve", "replace_together", "frame",
}
PROMPT_FUSION_SKILL_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "skills" / "video-prompt-fusion" / "SKILL.md"
)
class LongGenerationError(RuntimeError):
    def __init__(self, code: str, status: int = 409) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


@dataclass(frozen=True)
class FrozenPromptFusion:
    version: int
    input_path: Path
    input_data: bytes
    input_sha256: str
    output_path: Path
    output_data: bytes
    output_sha256: str
    segments: tuple[Mapping, ...]
    final_prompts: tuple[str, ...]


def _fusion_json(path: Path, code: str) -> tuple[bytes, dict]:
    if path.is_symlink():
        raise LongGenerationError(code)
    try:
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise LongGenerationError(code) from None
    if not data or not isinstance(value, dict):
        raise LongGenerationError(code)
    return data, value


def _fusion_text_binding(value: object) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"text", "sha256"}
        and isinstance(value.get("text"), str)
        and value["text"].strip()
        and value.get("sha256")
        == hashlib.sha256(value["text"].encode("utf-8")).hexdigest()
    )


def _fusion_audio_block(lines_json: str) -> str:
    return (
        f"<AUDIO_CONTENT_JSON>{lines_json}</AUDIO_CONTENT_JSON>"
    )


def _freeze_local_keyframe_sources(value: object) -> list[dict]:
    """Validate one provider-segment-local nine-frame timeline."""
    if not isinstance(value, list) or len(value) != 9:
        raise LongGenerationError("prompt_fusion_input_invalid")
    frozen: list[dict] = []
    previous: dict | None = None
    for order, item in enumerate(value, 1):
        if (
            not isinstance(item, Mapping)
            or set(item) != {
                "order", "segment_time_s", "source_scene_id", "transition",
            }
            or item.get("order") != order
            or isinstance(item.get("segment_time_s"), bool)
            or not isinstance(item.get("segment_time_s"), (int, float))
            or not math.isfinite(float(item["segment_time_s"]))
            or float(item["segment_time_s"]) < 0
            or not isinstance(item.get("source_scene_id"), str)
            or not item["source_scene_id"].strip()
            or not isinstance(item.get("transition"), Mapping)
            or set(item["transition"]) != {"type", "at_segment_s"}
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        segment_time_s = float(item["segment_time_s"])
        transition_type = item["transition"].get("type")
        at_segment_s = item["transition"].get("at_segment_s")
        if previous is None:
            if (
                segment_time_s != 0.0
                or transition_type != "start"
                or at_segment_s != 0.0
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
        else:
            previous_time_s = float(previous["segment_time_s"])
            if segment_time_s <= previous_time_s or transition_type == "start":
                raise LongGenerationError("prompt_fusion_input_invalid")
            if transition_type == "hard_cut":
                if (
                    isinstance(at_segment_s, bool)
                    or not isinstance(at_segment_s, (int, float))
                    or not math.isfinite(float(at_segment_s))
                    or not previous_time_s < float(at_segment_s) <= segment_time_s
                    or item["source_scene_id"] == previous["source_scene_id"]
                ):
                    raise LongGenerationError("prompt_fusion_input_invalid")
            elif (
                transition_type != "continuous"
                or at_segment_s is not None
                or item["source_scene_id"] != previous["source_scene_id"]
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
        normalized = {
            "order": order,
            "segment_time_s": segment_time_s,
            "source_scene_id": item["source_scene_id"],
            "transition": {
                "type": transition_type,
                "at_segment_s": at_segment_s,
            },
        }
        frozen.append(normalized)
        previous = normalized
    return frozen


def _freeze_local_cut_timeline(value: object, duration_s: float) -> list[dict]:
    """Validate the complete source-cut topology independently of 9 frames."""
    if (
        not isinstance(value, list) or not value
        or isinstance(duration_s, bool)
        or not isinstance(duration_s, (int, float))
        or not math.isfinite(float(duration_s))
        or float(duration_s) <= 0
    ):
        raise LongGenerationError("prompt_fusion_input_invalid")
    frozen = []
    previous_end = 0.0
    previous_scene_id = None
    for order, item in enumerate(value, 1):
        if (
            not isinstance(item, Mapping)
            or set(item) != {
                "order", "start_segment_s", "end_segment_s",
                "source_scene_id",
            }
            or item.get("order") != order
            or isinstance(item.get("start_segment_s"), bool)
            or isinstance(item.get("end_segment_s"), bool)
            or not isinstance(item.get("start_segment_s"), (int, float))
            or not isinstance(item.get("end_segment_s"), (int, float))
            or not math.isfinite(float(item["start_segment_s"]))
            or not math.isfinite(float(item["end_segment_s"]))
            or not isinstance(item.get("source_scene_id"), str)
            or not item["source_scene_id"].strip()
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        start = round(float(item["start_segment_s"]), 6)
        end = round(float(item["end_segment_s"]), 6)
        if (
            start != round(previous_end, 6)
            or end <= start
            or item["source_scene_id"] == previous_scene_id
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        frozen.append({
            "order": order,
            "start_segment_s": start,
            "end_segment_s": end,
            "source_scene_id": item["source_scene_id"],
        })
        previous_end = end
        previous_scene_id = item["source_scene_id"]
    if round(previous_end, 6) != round(float(duration_s), 6):
        raise LongGenerationError("prompt_fusion_input_invalid")
    return frozen


def _localize_source_cut_timeline(
    value: object, *, segment_start_s: float, segment_end_s: float,
) -> list[dict]:
    if not isinstance(value, list):
        raise LongGenerationError("prompt_fusion_input_invalid")
    localized = []
    for order, item in enumerate(value, 1):
        if not isinstance(item, Mapping):
            raise LongGenerationError("prompt_fusion_input_invalid")
        try:
            start = round(float(item["start_s"]) - segment_start_s, 6)
            end = round(float(item["end_s"]) - segment_start_s, 6)
            scene_id = item["source_scene_id"]
        except (KeyError, TypeError, ValueError):
            raise LongGenerationError("prompt_fusion_input_invalid") from None
        localized.append({
            "order": order,
            "start_segment_s": start,
            "end_segment_s": end,
            "source_scene_id": scene_id,
        })
    return _freeze_local_cut_timeline(
        localized, round(segment_end_s - segment_start_s, 6)
    )


def _bounded_relation_text(value: object, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongGenerationError("prompt_fusion_input_invalid")
    text = value.strip()
    if len(text.encode("utf-8")) > maximum:
        raise LongGenerationError("prompt_fusion_input_invalid")
    return text


def _freeze_fusion_relation_occurrences(
    value: object, timeline: list[dict],
) -> list[dict]:
    """Freeze exact direct evidence for each frame; never infer across cuts."""
    if not isinstance(value, list) or len(value) > 540:
        raise LongGenerationError("prompt_fusion_input_invalid")
    timeline_by_order = {item["order"]: item for item in timeline}
    frozen = []
    seen = set()
    definitions: dict[str, tuple] = {}
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _RELATION_OCCURRENCE_KEYS:
            raise LongGenerationError("prompt_fusion_input_invalid")
        frame = item.get("frame")
        if not isinstance(frame, Mapping) or set(frame) != {
            "order", "segment_time_s", "source_scene_id",
        }:
            raise LongGenerationError("prompt_fusion_input_invalid")
        order = frame.get("order")
        expected_frame = timeline_by_order.get(order)
        if (
            expected_frame is None
            or frame.get("segment_time_s") != expected_frame["segment_time_s"]
            or frame.get("source_scene_id") != expected_frame["source_scene_id"]
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        relation_id = _bounded_relation_text(item.get("relation_id"), maximum=128)
        subject_key = _bounded_relation_text(item.get("subject_key"), maximum=128)
        predicate = _bounded_relation_text(item.get("predicate"), maximum=128)
        object_key = _bounded_relation_text(item.get("object_key"), maximum=128)
        state = _bounded_relation_text(item.get("state"), maximum=512)
        geometry = _bounded_relation_text(item.get("geometry"), maximum=512)
        preserve_value = item.get("preserve")
        if (
            subject_key == object_key
            or not isinstance(preserve_value, list)
            or len(preserve_value) > 30
            or not isinstance(item.get("replace_together"), bool)
            or (order, relation_id) in seen
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        preserve = [
            _bounded_relation_text(text, maximum=512)
            for text in preserve_value
        ]
        definition = (
            subject_key, predicate, object_key, tuple(preserve),
            item["replace_together"],
        )
        prior_definition = definitions.setdefault(relation_id, definition)
        if prior_definition != definition:
            raise LongGenerationError("prompt_fusion_input_invalid")
        seen.add((order, relation_id))
        frozen.append({
            "relation_id": relation_id,
            "subject_key": subject_key,
            "predicate": predicate,
            "object_key": object_key,
            "state": state,
            "geometry": geometry,
            "preserve": preserve,
            "replace_together": item["replace_together"],
            "frame": {
                "order": order,
                "segment_time_s": expected_frame["segment_time_s"],
                "source_scene_id": expected_frame["source_scene_id"],
            },
        })
    expected_order = sorted(
        frozen, key=lambda item: (item["frame"]["order"], item["relation_id"])
    )
    if frozen != expected_order:
        raise LongGenerationError("prompt_fusion_input_invalid")
    return frozen


def _expected_fusion_relation_states(
    timeline: list[dict], occurrences: list[dict],
    cut_timeline: list[dict] | None = None,
) -> list[dict]:
    """Project frame evidence into hard-cut-local intervals mechanically."""
    if cut_timeline is not None:
        duration = float(cut_timeline[-1].get("end_segment_s", 0)) \
            if cut_timeline else 0.0
        cuts = _freeze_local_cut_timeline(cut_timeline, duration)
        frames_by_cut: list[list[dict]] = [[] for _item in cuts]
        for frame in timeline:
            matches = [
                index for index, cut in enumerate(cuts)
                if cut["source_scene_id"] == frame["source_scene_id"]
                and cut["start_segment_s"] <= frame["segment_time_s"]
                and (
                    frame["segment_time_s"] < cut["end_segment_s"]
                    or index == len(cuts) - 1
                    and frame["segment_time_s"] == cut["end_segment_s"]
                )
            ]
            if len(matches) != 1:
                raise LongGenerationError("prompt_fusion_input_invalid")
            frames_by_cut[matches[0]].append(frame)
        projected = []
        for cut, frames in zip(cuts, frames_by_cut, strict=True):
            if not frames:
                continue
            orders = {frame["order"] for frame in frames}
            members: dict[tuple, dict] = {}
            for occurrence in occurrences:
                if occurrence["frame"]["order"] not in orders:
                    continue
                identity = tuple(occurrence[key] for key in (
                    "relation_id", "subject_key", "predicate", "object_key",
                ))
                base = members.setdefault(identity, {
                    "relation_id": occurrence["relation_id"],
                    "subject_key": occurrence["subject_key"],
                    "predicate": occurrence["predicate"],
                    "object_key": occurrence["object_key"],
                    "preserve": occurrence["preserve"],
                    "replace_together": occurrence["replace_together"],
                    "states": [],
                })
                if (
                    base["preserve"] != occurrence["preserve"]
                    or base["replace_together"] != occurrence["replace_together"]
                ):
                    raise LongGenerationError("prompt_fusion_input_invalid")
                base["states"].append({
                    "frame_order": occurrence["frame"]["order"],
                    "state": occurrence["state"],
                    "geometry": occurrence["geometry"],
                })
            projected.append({
                "interval": {
                    "start_frame_order": frames[0]["order"],
                    "end_frame_order": frames[-1]["order"],
                    "source_scene_id": cut["source_scene_id"],
                },
                "relations": [members[key] for key in sorted(members)],
            })
        return projected
    intervals: list[list[dict]] = []
    for frame in timeline:
        if not intervals or frame["transition"]["type"] == "hard_cut":
            intervals.append([])
        intervals[-1].append(frame)
    projected = []
    for frames in intervals:
        orders = {frame["order"] for frame in frames}
        members: dict[tuple, dict] = {}
        for occurrence in occurrences:
            if occurrence["frame"]["order"] not in orders:
                continue
            identity = tuple(
                occurrence[key] for key in (
                    "relation_id", "subject_key", "predicate", "object_key",
                )
            )
            base = members.get(identity)
            if base is None:
                base = {
                    "relation_id": occurrence["relation_id"],
                    "subject_key": occurrence["subject_key"],
                    "predicate": occurrence["predicate"],
                    "object_key": occurrence["object_key"],
                    "preserve": occurrence["preserve"],
                    "replace_together": occurrence["replace_together"],
                    "states": [],
                }
                members[identity] = base
            elif (
                base["preserve"] != occurrence["preserve"]
                or base["replace_together"] != occurrence["replace_together"]
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            base["states"].append({
                "frame_order": occurrence["frame"]["order"],
                "state": occurrence["state"],
                "geometry": occurrence["geometry"],
            })
        projected.append({
            "interval": {
                "start_frame_order": frames[0]["order"],
                "end_frame_order": frames[-1]["order"],
                "source_scene_id": frames[0]["source_scene_id"],
            },
            "relations": [members[key] for key in sorted(members)],
        })
    return projected


def _relation_state_category(value: str) -> str:
    lowered = value.casefold()
    categories = (
        ("released/separated", ("release", "separat", "detach", "释放", "分离", "脱离", "离开")),
        ("attached/installed", ("attach", "install", "mount", "connect", "assembl", "装配", "安装", "接合", "连接", "固定")),
        ("held/carried", ("hold", "held", "grip", "carry", "持有", "握", "携带", "托住")),
        ("operated", ("operat", "press", "turn", "操作", "按压", "转动", "准备")),
        ("supported/contacting", ("support", "contact", "touch", "支撑", "接触", "贴地")),
        ("moving/spinning", ("move", "spin", "rotat", "运动", "移动", "旋转")),
        ("displayed", ("display", "show", "展示", "举")),
    )
    return next(
        (name for name, words in categories if any(word in lowered for word in words)),
        "active/as-shown",
    )


def _relation_predicate_category(value: str) -> str:
    category = _relation_state_category(value)
    return "related-directed" if category == "active/as-shown" else category


def _relation_geometry_category(value: str) -> str:
    lowered = value.casefold()
    categories = (
        ("separated", ("separat", "gap", "分离", "间隔", "脱离", "不再接触")),
        ("aligned/interface", ("align", "coax", "interface", "同轴", "对齐", "接口", "重合")),
        ("contacting", ("contact", "touch", "接触", "贴", "压")),
        ("grounded/below", ("floor", "ground", "below", "地板", "地面", "下方", "底部")),
        ("above/top", ("above", "top", "上方", "顶部", "上端")),
        ("relative-position", ("left", "right", "front", "behind", "左", "右", "前", "后", "附近")),
    )
    return next(
        (name for name, words in categories if any(word in lowered for word in words)),
        "as-shown",
    )


def _compact_h3_relation_contract(relation_states: list[dict]) -> dict:
    """Encode exact relation facts with deterministic aliases and state RLE.

    ``r/e/q/p/s/g/c`` are dictionaries for relation ids, endpoints,
    predicates, preserve facts, states, geometry facts and scenes.  Definition
    position is the runtime relation alias, so neither long stable ids nor
    immutable facts are repeated.  A run is extended only across consecutive
    evidenced frames with the exact same state and geometry; absence remains
    absence and hard-cut intervals are never joined.
    """
    definitions_by_id: dict[str, tuple] = {}
    for item in relation_states:
        for relation in item["relations"]:
            relation_id = relation["relation_id"]
            definition = (
                relation["subject_key"], relation["predicate"],
                relation["object_key"], tuple(relation["preserve"]),
                relation["replace_together"],
            )
            prior = definitions_by_id.setdefault(relation_id, definition)
            if prior != definition:
                raise LongGenerationError("prompt_fusion_output_invalid")
    def canonical_alias_order(values: set[str], prefix: str) -> list[str]:
        expected = {f"{prefix}{index}" for index in range(1, len(values) + 1)}
        if values == expected:
            return sorted(values, key=lambda value: int(value[len(prefix):]))
        return sorted(values)

    relation_ids = canonical_alias_order(set(definitions_by_id), "R")
    endpoints = canonical_alias_order({
        endpoint
        for definition in definitions_by_id.values()
        for endpoint in (definition[0], definition[2])
    }, "E")
    predicates = sorted({definition[1] for definition in definitions_by_id.values()})
    preserve_facts = sorted({
        fact
        for definition in definitions_by_id.values()
        for fact in definition[3]
    })
    states = sorted({
        state["state"]
        for item in relation_states
        for relation in item["relations"]
        for state in relation["states"]
    })
    geometries = sorted({
        state["geometry"]
        for item in relation_states
        for relation in item["relations"]
        for state in relation["states"]
    })
    scenes = canonical_alias_order({
        item["interval"]["source_scene_id"] for item in relation_states
    }, "C")

    def aliases(values: list[str]) -> dict[str, int]:
        return {value: index for index, value in enumerate(values)}

    relation_alias = aliases(relation_ids)
    endpoint_alias = aliases(endpoints)
    predicate_alias = aliases(predicates)
    preserve_alias = aliases(preserve_facts)
    state_alias = aliases(states)
    geometry_alias = aliases(geometries)
    scene_alias = aliases(scenes)
    definitions = []
    for relation_id in relation_ids:
        subject, predicate, object_key, preserve, replace_together = (
            definitions_by_id[relation_id]
        )
        definitions.append([
            endpoint_alias[subject], predicate_alias[predicate],
            endpoint_alias[object_key],
            [preserve_alias[fact] for fact in preserve],
            int(replace_together),
        ])

    intervals = []
    for interval_index, item in enumerate(relation_states):
        interval = item["interval"]
        encoded_relations = []
        for relation in item["relations"]:
            runs: list[list[int]] = []
            for state in relation["states"]:
                frame_order = state["frame_order"]
                encoded_state = state_alias[state["state"]]
                encoded_geometry = geometry_alias[state["geometry"]]
                if (
                    runs
                    and frame_order == runs[-1][1] + 1
                    and encoded_state == runs[-1][2]
                    and encoded_geometry == runs[-1][3]
                ):
                    runs[-1][1] = frame_order
                else:
                    runs.append([
                        frame_order, frame_order,
                        encoded_state, encoded_geometry,
                    ])
            encoded_relations.append([
                relation_alias[relation["relation_id"]], runs,
            ])
        intervals.append([
            interval["start_frame_order"], interval["end_frame_order"],
            scene_alias[interval["source_scene_id"]],
            0 if interval_index == 0 else 1,
            encoded_relations,
        ])
    contract = {
        "v": 3,
        "m": [0, "exact"],
        "l": _RELATION_CONTRACT_LEGEND,
        "r": relation_ids,
        "e": endpoints,
        "q": predicates,
        "p": preserve_facts,
        "s": states,
        "g": geometries,
        "c": scenes,
        "d": definitions,
        "i": intervals,
    }
    if _expand_h3_relation_contract(contract) != relation_states:
        raise LongGenerationError("prompt_fusion_output_invalid")

    def marker_chars(value: dict) -> int:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        return len(RELATION_STATES_OPEN) + len(encoded) + len(RELATION_STATES_CLOSE)

    if marker_chars(contract) <= _MAX_RELATION_MARKER_CHARS:
        return contract

    # Level 1 keeps exact endpoints, predicates, state and geometry. Stable
    # relation ids and preserve prose remain losslessly frozen in the backend
    # input/receipt and no longer consume provider prompt space.
    compact = json.loads(json.dumps(contract, ensure_ascii=False))
    compact["m"] = [1, "ids+preserve@receipt"]
    compact["r"] = []
    compact["p"] = []
    for definition in compact["d"]:
        definition[3] = []
    if marker_chars(compact) <= _MAX_RELATION_MARKER_CHARS:
        _expand_h3_relation_contract(compact)
        return compact

    # Level 2 is the deterministic maximum-density projection. Every directed
    # edge, predicate, replace flag, hard-cut-local active run and lifecycle
    # category remains. Long verbatim ids/state/geometry/preserve prose stays in
    # the backend receipt; finite backend categories prevent valid 540-item
    # inputs from crowding visual action/composition out of H3's prompt.
    compact["m"] = [2, "aliases+categories;verbatim@receipt"]
    compact["e"] = []
    compact["c"] = []
    categorized_predicates = sorted({
        _relation_predicate_category(definition[1])
        for definition in definitions_by_id.values()
    })
    compact["q"] = categorized_predicates
    categorized_predicate_alias = aliases(categorized_predicates)
    for relation_index, relation_id in enumerate(relation_ids):
        compact["d"][relation_index][1] = categorized_predicate_alias[
            _relation_predicate_category(definitions_by_id[relation_id][1])
        ]
        del compact["d"][relation_index][3]
    states_by_relation: dict[str, list[dict]] = {
        relation_id: [] for relation_id in relation_ids
    }
    for source_interval in relation_states:
        for relation in source_interval["relations"]:
            states_by_relation[relation["relation_id"]].extend(relation["states"])
    retained_boundaries = [
        boundary
        for relation_id in relation_ids
        for boundary in (
            min(states_by_relation[relation_id], key=lambda state: state["frame_order"]),
            max(states_by_relation[relation_id], key=lambda state: state["frame_order"]),
        )
    ]
    categorized_states = sorted({
        _relation_state_category(state["state"])
        for state in retained_boundaries
    } | {"active/as-shown"})
    categorized_geometries = sorted({
        _relation_geometry_category(state["geometry"])
        for state in retained_boundaries
    } | {"as-shown"})
    compact["s"] = categorized_states
    compact["g"] = categorized_geometries
    categorized_state_alias = aliases(categorized_states)
    categorized_geometry_alias = aliases(categorized_geometries)
    compact_definitions = []
    for relation_id in relation_ids:
        subject, predicate, object_key, _preserve, replace_together = (
            definitions_by_id[relation_id]
        )
        relation_history = sorted(
            states_by_relation[relation_id], key=lambda state: state["frame_order"]
        )
        active_mask = sum(
            1 << (state["frame_order"] - 1) for state in relation_history
        )
        first_state = relation_history[0]
        current_state = relation_history[-1]
        compact_definitions.append([
            endpoint_alias[subject],
            categorized_predicate_alias[_relation_predicate_category(predicate)],
            endpoint_alias[object_key], int(replace_together), active_mask,
            categorized_state_alias[_relation_state_category(first_state["state"])],
            categorized_state_alias[_relation_state_category(current_state["state"])],
            categorized_geometry_alias[
                _relation_geometry_category(first_state["geometry"])
            ],
            categorized_geometry_alias[
                _relation_geometry_category(current_state["geometry"])
            ],
        ])
    compact["d"] = compact_definitions
    compact["i"] = [encoded_interval[:4] for encoded_interval in compact["i"]]
    if marker_chars(compact) > _MAX_RELATION_MARKER_CHARS:
        # Sixty distinct directed relations with nine frames fit this schema;
        # reaching here means the finite predicate vocabulary itself exceeded
        # the advertised transport contract, not verbose state/geometry prose.
        raise LongGenerationError("prompt_fusion_input_invalid")
    _expand_h3_relation_contract(compact)
    return compact


def _expand_h3_relation_contract(contract: object) -> list[dict]:
    """Expand and fully validate the model-readable v3 wire form."""
    if (
        not isinstance(contract, Mapping)
        or set(contract) != {
            "v", "m", "l", "r", "e", "q", "p", "s", "g", "c", "d", "i",
        }
        or contract.get("v") != 3
        or not isinstance(contract.get("m"), list)
        or len(contract["m"]) != 2
        or contract["m"][0] not in {0, 1, 2}
        or not isinstance(contract["m"][1], str)
        or contract.get("l") != _RELATION_CONTRACT_LEGEND
        or any(
            not isinstance(contract.get(key), list)
            for key in ("r", "e", "q", "p", "s", "g", "c", "d", "i")
        )
    ):
        raise LongGenerationError("prompt_fusion_output_invalid")
    compact_level = contract["m"][0]
    expected_metadata = {
        0: "exact",
        1: "ids+preserve@receipt",
        2: "aliases+categories;verbatim@receipt",
    }
    if contract["m"][1] != expected_metadata[compact_level]:
        raise LongGenerationError("prompt_fusion_output_invalid")
    relation_ids = contract["r"]
    endpoints = contract["e"]
    predicates = contract["q"]
    preserve_facts = contract["p"]
    states = contract["s"]
    geometries = contract["g"]
    scenes = contract["c"]
    definitions = contract["d"]
    for dictionary in (predicates, preserve_facts, states, geometries):
        if (
            any(not isinstance(value, str) or not value for value in dictionary)
            or dictionary != sorted(set(dictionary))
        ):
            raise LongGenerationError("prompt_fusion_output_invalid")
    for dictionary, prefix in (
        (relation_ids, "R"), (endpoints, "E"), (scenes, "C"),
    ):
        runtime_order = [
            f"{prefix}{index}" for index in range(1, len(dictionary) + 1)
        ]
        if (
            any(not isinstance(value, str) or not value for value in dictionary)
            or dictionary not in (sorted(set(dictionary)), runtime_order)
        ):
            raise LongGenerationError("prompt_fusion_output_invalid")
    if (
        (compact_level == 0 and len(relation_ids) != len(definitions))
        or (compact_level > 0 and relation_ids)
    ):
        raise LongGenerationError("prompt_fusion_output_invalid")

    def lookup(values: list, index: object) -> object:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(values)
        ):
            raise LongGenerationError("prompt_fusion_output_invalid")
        return values[index]

    def runtime_alias(index: object, prefix: str) -> str:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise LongGenerationError("prompt_fusion_output_invalid")
        return f"{prefix}{index + 1}"

    if compact_level == 2:
        endpoint_domain = set()
        for definition in definitions:
            if not isinstance(definition, list) or len(definition) != 9:
                raise LongGenerationError("prompt_fusion_output_invalid")
            for endpoint_index in (definition[0], definition[2]):
                runtime_alias(endpoint_index, "E")
                endpoint_domain.add(endpoint_index)
        if sorted(endpoint_domain) != list(range(len(endpoint_domain))):
            raise LongGenerationError("prompt_fusion_output_invalid")
        scene_domain = set()
        for encoded_interval in contract["i"]:
            if not isinstance(encoded_interval, list) or len(encoded_interval) != 4:
                raise LongGenerationError("prompt_fusion_output_invalid")
            runtime_alias(encoded_interval[2], "C")
            scene_domain.add(encoded_interval[2])
        if sorted(scene_domain) != list(range(len(scene_domain))):
            raise LongGenerationError("prompt_fusion_output_invalid")

    used_definitions: set[int] = set()
    used_endpoints: set[int] = set()
    used_predicates: set[int] = set()
    used_preserve: set[int] = set()
    used_states: set[int] = set()
    used_geometries: set[int] = set()
    used_scenes: set[int] = set()
    expanded = []
    previous_interval_end = 0
    if not contract["i"]:
        raise LongGenerationError("prompt_fusion_output_invalid")
    for interval_index, encoded_interval in enumerate(contract["i"]):
        expected_interval_size = 4 if compact_level == 2 else 5
        if (
            not isinstance(encoded_interval, list)
            or len(encoded_interval) != expected_interval_size
        ):
            raise LongGenerationError("prompt_fusion_output_invalid")
        first, last, scene_index, boundary = encoded_interval[:4]
        encoded_relations = (
            encoded_interval[4] if compact_level != 2 else []
        )
        if (
            isinstance(first, bool) or not isinstance(first, int)
            or isinstance(last, bool) or not isinstance(last, int)
            or not 1 <= first <= last <= 9
            or first != previous_interval_end + 1
            or boundary != (0 if interval_index == 0 else 1)
            or not isinstance(encoded_relations, list)
        ):
            raise LongGenerationError("prompt_fusion_output_invalid")
        previous_interval_end = last
        if scenes:
            lookup(scenes, scene_index)
            used_scenes.add(scene_index)
        relations = []
        if compact_level == 2:
            active_state_index = states.index("active/as-shown")
            active_geometry_index = geometries.index("as-shown")
            for relation_index, definition in enumerate(definitions):
                if not isinstance(definition, list) or len(definition) != 9:
                    raise LongGenerationError("prompt_fusion_output_invalid")
                active_mask = definition[4]
                if (
                    isinstance(active_mask, bool)
                    or not isinstance(active_mask, int)
                    or not 0 < active_mask < (1 << 9)
                ):
                    raise LongGenerationError("prompt_fusion_output_invalid")
                active_frames = [
                    frame for frame in range(1, 10)
                    if active_mask & (1 << (frame - 1))
                ]
                interval_frames = [
                    frame for frame in active_frames if first <= frame <= last
                ]
                if not interval_frames:
                    continue
                runs: list[list[int]] = []
                for frame in interval_frames:
                    state_index = (
                        definition[5] if frame == active_frames[0]
                        else definition[6] if frame == active_frames[-1]
                        else active_state_index
                    )
                    geometry_index = (
                        definition[7] if frame == active_frames[0]
                        else definition[8] if frame == active_frames[-1]
                        else active_geometry_index
                    )
                    if (
                        runs and frame == runs[-1][1] + 1
                        and state_index == runs[-1][2]
                        and geometry_index == runs[-1][3]
                    ):
                        runs[-1][1] = frame
                    else:
                        runs.append([
                            frame, frame, state_index, geometry_index,
                        ])
                encoded_relations.append([relation_index, runs])
        previous_relation_index = -1
        for encoded_relation in encoded_relations:
            if not isinstance(encoded_relation, list) or len(encoded_relation) != 2:
                raise LongGenerationError("prompt_fusion_output_invalid")
            relation_index, encoded_runs = encoded_relation
            if (
                isinstance(relation_index, bool)
                or not isinstance(relation_index, int)
                or not 0 <= relation_index < len(definitions)
                or relation_index <= previous_relation_index
            ):
                raise LongGenerationError("prompt_fusion_output_invalid")
            previous_relation_index = relation_index
            used_definitions.add(relation_index)
            relation_id = (
                lookup(relation_ids, relation_index)
                if compact_level == 0 else runtime_alias(relation_index, "R")
            )
            definition = lookup(definitions, relation_index)
            expected_definition_size = 9 if compact_level == 2 else 5
            if (
                not isinstance(definition, list)
                or len(definition) != expected_definition_size
            ):
                raise LongGenerationError("prompt_fusion_output_invalid")
            if compact_level == 2:
                subject, predicate, object_key, replace_together = definition[:4]
                preserve = []
            else:
                subject, predicate, object_key, preserve, replace_together = definition
            if (
                not isinstance(preserve, list)
                or replace_together not in {0, 1}
                or not isinstance(encoded_runs, list)
            ):
                raise LongGenerationError("prompt_fusion_output_invalid")
            if endpoints:
                lookup(endpoints, subject)
                lookup(endpoints, object_key)
                used_endpoints.update((subject, object_key))
            lookup(predicates, predicate)
            used_predicates.add(predicate)
            for preserve_index in preserve:
                lookup(preserve_facts, preserve_index)
                used_preserve.add(preserve_index)
            expanded_states = []
            previous_frame = 0
            for run in encoded_runs:
                if not isinstance(run, list) or len(run) != 4:
                    raise LongGenerationError("prompt_fusion_output_invalid")
                run_first, run_last, state_index, geometry_index = run
                if (
                    isinstance(run_first, bool) or not isinstance(run_first, int)
                    or isinstance(run_last, bool) or not isinstance(run_last, int)
                    or not first <= run_first <= run_last <= last
                    or run_first <= previous_frame
                    or (
                        expanded_states
                        and run_first == previous_frame + 1
                        and expanded_states[-1]["state"]
                        == lookup(states, state_index)
                        and expanded_states[-1]["geometry"]
                        == lookup(geometries, geometry_index)
                    )
                ):
                    raise LongGenerationError("prompt_fusion_output_invalid")
                lookup(states, state_index)
                lookup(geometries, geometry_index)
                used_states.add(state_index)
                used_geometries.add(geometry_index)
                for frame_order in range(run_first, run_last + 1):
                    expanded_states.append({
                        "frame_order": frame_order,
                        "state": lookup(states, state_index),
                        "geometry": lookup(geometries, geometry_index),
                    })
                previous_frame = run_last
            relations.append({
                "relation_id": relation_id,
                "subject_key": (
                    lookup(endpoints, subject)
                    if endpoints else runtime_alias(subject, "E")
                ),
                "predicate": lookup(predicates, predicate),
                "object_key": (
                    lookup(endpoints, object_key)
                    if endpoints else runtime_alias(object_key, "E")
                ),
                "preserve": [lookup(preserve_facts, item) for item in preserve],
                "replace_together": bool(replace_together),
                "states": expanded_states,
            })
        expanded.append({
            "interval": {
                "start_frame_order": first,
                "end_frame_order": last,
                "source_scene_id": (
                    lookup(scenes, scene_index)
                    if scenes else runtime_alias(scene_index, "C")
                ),
            },
            "relations": relations,
        })
    if previous_interval_end != 9:
        raise LongGenerationError("prompt_fusion_output_invalid")
    if used_definitions != set(range(len(definitions))):
        raise LongGenerationError("prompt_fusion_output_invalid")
    exact_domains = (
        (endpoints, used_endpoints),
        (predicates, used_predicates),
        (preserve_facts, used_preserve),
        (scenes, used_scenes),
    )
    if any(used != set(range(len(values))) for values, used in exact_domains):
        raise LongGenerationError("prompt_fusion_output_invalid")
    allowed_unused_states = set()
    allowed_unused_geometries = set()
    if compact_level == 2:
        allowed_unused_states.add(states.index("active/as-shown"))
        allowed_unused_geometries.add(geometries.index("as-shown"))
    if (
        set(range(len(states))) - used_states - allowed_unused_states
        or set(range(len(geometries))) - used_geometries
        - allowed_unused_geometries
    ):
        raise LongGenerationError("prompt_fusion_output_invalid")
    return expanded


def _h3_timecode(value: object) -> str:
    """Render one frozen segment-local second value as official MM:SS.mmm."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise LongGenerationError("prompt_fusion_input_invalid")
    milliseconds = int(float(value) * 1000 + 0.5)
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _h3_picture_list(orders: list[int]) -> str:
    labels = [f"<Picture {order}>" for order in orders]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return " and ".join(labels)
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _provider_neutral_visual(
    value: object, *, max_chars: int | None = None,
) -> str:
    """Keep visual prose usable while reserving all H3 markup for backend."""
    if not isinstance(value, str) or not value.strip():
        raise LongGenerationError("prompt_fusion_output_invalid")
    visual = value.replace("\r\n", " ").replace("\r", " ")
    visual = visual.replace("\n", " ").replace("\t", " ")
    visual = visual.replace("<", "‹").replace(">", "›").strip()
    if not visual or (max_chars is not None and len(visual) > max_chars):
        raise LongGenerationError("prompt_fusion_output_invalid")
    return visual


def _compile_fusion_ref2va_prompt(
    *, visual: object, timeline: object, lines: object, music_policy: object,
    relation_occurrences: object = None, relation_states: object = None,
    cut_timeline: object = None,
) -> str:
    """Compile the only provider-sendable prompt from backend authorities."""
    if music_policy != "forbid" or not isinstance(lines, list):
        raise LongGenerationError("prompt_fusion_input_invalid")
    frozen_timeline = _freeze_local_keyframe_sources(timeline)
    shots: list[list[dict]] = []
    for frame in frozen_timeline:
        transition_type = frame["transition"]["type"]
        if not shots or transition_type == "hard_cut":
            shots.append([])
        shots[-1].append(frame)
    if not isinstance(visual, list) or len(visual) != len(shots):
        raise LongGenerationError("prompt_fusion_output_invalid")
    relation_contract_enabled = relation_occurrences is not None
    visual_by_shot = [
        _provider_neutral_visual(item)
        for item in visual
    ]
    frozen_occurrences = _freeze_fusion_relation_occurrences(
        [] if relation_occurrences is None else relation_occurrences,
        frozen_timeline,
    )
    frozen_cuts = None
    if cut_timeline is not None:
        if not isinstance(cut_timeline, list) or not cut_timeline:
            raise LongGenerationError("prompt_fusion_input_invalid")
        frozen_cuts = _freeze_local_cut_timeline(
            cut_timeline, float(cut_timeline[-1].get("end_segment_s", 0)),
        )
    expected_relation_states = _expected_fusion_relation_states(
        frozen_timeline, frozen_occurrences, frozen_cuts,
    )
    # Model-authored relation_states is intentionally non-authoritative.  Keep
    # the argument for callers that still echo it, but compile only the exact
    # backend projection from frozen occurrences.
    _ = relation_states

    subject_definitions = ["subject_definitions:"]
    retention = ["retention_analysis:"]
    details: list[list[str]] = []
    cut_times: list[float] = []
    for shot_index, frames in enumerate(shots, 1):
        if shot_index > 1:
            cut_times.append(float(frames[0]["transition"]["at_segment_s"]))
        for frame in frames:
            order = int(frame["order"])
            if relation_contract_enabled:
                subject_definitions.append(
                    f"<Picture {order}>=[Shot {shot_index}]@"
                    f"{_h3_timecode(frame['segment_time_s'])}."
                )
            else:
                subject_definitions.append(
                    f"<Picture {order}> is the storyboard keyframe anchor for "
                    f"[Shot {shot_index}] at "
                    f"{_h3_timecode(frame['segment_time_s'])}, defining its "
                    "ordered visual state and composition."
                )
                retention.append(
                    f"<Picture {order}> ([Shot {shot_index}] storyboard "
                    "keyframe): fully_preserved - its role as an ordered "
                    "visual-state and composition anchor is retained."
                )
        orders = [int(frame["order"]) for frame in frames]
        anchors = _h3_picture_list(orders)
        if relation_contract_enabled and frozen_cuts is not None:
            opening = f"[Shot {shot_index}] {anchors}."
        elif shot_index == 1:
            opening = f"[Shot 1] The shot follows the ordered storyboard anchors {anchors}."
        else:
            cut = frames[0]["transition"]["at_segment_s"]
            opening = (
                f"[Shot {shot_index}] At {_h3_timecode(cut)}, the shot cuts "
                f"to <Picture {orders[0]}>. The shot then follows the ordered "
                f"storyboard anchors {anchors}."
            )
        details.append([
            f"{opening} {visual_by_shot[shot_index - 1]}"
        ])

    cut_dialogue_lines: list[str] = []
    for expected_order, line in enumerate(lines, 1):
        if (
            not isinstance(line, Mapping)
            or set(line) != {
                "order", "text", "start_s", "end_s", "delivery", "voice_ref",
            }
            or line.get("order") != expected_order
            or not isinstance(line.get("text"), str)
            or not line["text"].strip()
            or line.get("delivery") != "off_screen"
            or line.get("voice_ref") is not None
            or isinstance(line.get("start_s"), bool)
            or isinstance(line.get("end_s"), bool)
            or not isinstance(line.get("start_s"), (int, float))
            or not isinstance(line.get("end_s"), (int, float))
            or not math.isfinite(float(line["start_s"]))
            or not math.isfinite(float(line["end_s"]))
            or not 0 <= float(line["start_s"]) < float(line["end_s"])
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        if frozen_cuts is not None:
            cut_segments = [{
                "index": cut["order"],
                "start_s": cut["start_segment_s"],
                "end_s": cut["end_segment_s"],
            } for cut in frozen_cuts]
            for cut, cut_segment in zip(
                frozen_cuts, cut_segments, strict=True
            ):
                fragments = long_video.localize_dialogue(
                    [line], cut_segment, segments=cut_segments,
                )
                for fragment in fragments:
                    absolute_start = round(
                        cut["start_segment_s"] + fragment["start_s"], 6
                    )
                    absolute_end = round(
                        cut["start_segment_s"] + fragment["end_s"], 6
                    )
                    cut_dialogue_lines.append(
                        f"[Source Cut {cut['order']}] From "
                        f"{_h3_timecode(absolute_start)} to "
                        f"{_h3_timecode(absolute_end)}, S1 off-screen: "
                        f"<d>[Undetermined]{fragment['text']}</d> while every "
                        "visible person's lips remain completely closed."
                    )
        else:
            shot_index = 1
            for index, cut in enumerate(cut_times, 2):
                if float(line["start_s"]) + _EPS >= cut:
                    shot_index = index
            details[shot_index - 1].append(
                f"From {_h3_timecode(line['start_s'])} to "
                f"{_h3_timecode(line['end_s'])}, the off-screen narrator (S1) "
                "says in an off-screen voiceover: "
                f"<d>[Undetermined]{line['text']}</d> while every visible "
                "person's lips remain completely closed."
            )

    cut_authority = []
    if frozen_cuts is not None:
        cut_contract = json.dumps(
            {
                "v": 1,
                "b": [
                    cut["start_segment_s"] for cut in frozen_cuts[1:]
                ],
            },
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        cut_authority = [
            f"{CUT_TIMELINE_OPEN}{cut_contract}{CUT_TIMELINE_CLOSE}",
            context_ir_bridge._CUT_TIMELINE_POLICY,
            *cut_dialogue_lines,
        ]
    if relation_contract_enabled:
        prompt_lines = subject_definitions + [
            "summary:",
            "Use <Picture 1> through <Picture 9> as exact ordered anchors.",
            "retention_analysis:",
            "Preserve all nine anchors and the backend relation contract.",
        ] + cut_authority + ["detailed_description:"]
    else:
        prompt_lines = subject_definitions + [
            "summary:",
            "[keyframe completion] The target video follows <Picture 1> through "
            "<Picture 9> as ordered storyboard keyframe anchors.",
    ] + retention + cut_authority + ["detailed_description:"]
    for shot in details:
        prompt_lines.extend(shot)
    prompt_lines.extend([
        "overall_soundscape:",
        (
            "The frozen spoken events described above are the only specified "
            "audible layer; no additional ambience, physical-action sounds, "
            "or non-verbal human sounds are added."
            if lines else
            "No dialogue, ambience, physical-action sounds, or non-verbal "
            "human sounds are specified."
        ),
        "non_diegetic_music:",
        "N/A",
    ])
    if relation_contract_enabled:
        encoded_relations = json.dumps(
            _compact_h3_relation_contract(expected_relation_states),
            ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        prompt_lines.append(
            f"{RELATION_STATES_OPEN}{encoded_relations}{RELATION_STATES_CLOSE}"
        )
    compiled = "\n".join(prompt_lines)
    if relation_contract_enabled and len(compiled) > _MAX_COMPILED_FUSION_CHARS:
        raise LongGenerationError("prompt_fusion_output_invalid")
    return compiled


def _canonical_fusion_prompt(
    prompt: str,
    lines_json: str,
    *,
    version: int,
) -> str:
    """Read the historical v1 prompt envelope; v2 is backend-compiled."""
    opening = "<AUDIO_CONTENT_JSON>"
    closing = "</AUDIO_CONTENT_JSON>"
    canonical = _fusion_audio_block(lines_json)
    if version == PROMPT_FUSION_LEGACY_VERSION:
        lf_envelope = f"{opening}\n{lines_json}\n{closing}"
        if prompt.endswith(canonical):
            prefix = prompt[:-len(canonical)]
        elif prompt.endswith(lf_envelope):
            prefix = prompt[:-len(lf_envelope)]
        else:
            raise LongGenerationError("prompt_fusion_output_invalid")
        canonical_prompt = prefix + canonical
    else:
        raise LongGenerationError("prompt_fusion_output_invalid")
    if opening in prefix or closing in prefix:
        raise LongGenerationError("prompt_fusion_output_invalid")
    return canonical_prompt


def load_prompt_fusion(
    *, input_path: Path, output_path: Path, root: Path | None = None,
) -> FrozenPromptFusion:
    """Load one project-level fusion result and verify all ordered inputs."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_data, source = _fusion_json(input_path, "prompt_fusion_input_invalid")
    output_data, output = _fusion_json(output_path, "prompt_fusion_output_invalid")
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    project_root = (
        Path(root).resolve()
        if root is not None else input_path.parent.parent.resolve()
    )
    try:
        input_path.relative_to(project_root)
        output_path.relative_to(project_root)
    except ValueError:
        raise LongGenerationError("prompt_fusion_input_invalid") from None
    segments = source.get("segments")
    source_version = source.get("version")
    if (
        input_path.name != h3_project.SKILL_INPUT_FILENAME
        or output_path.name != "h3_prompt_plan.json"
        or set(source) != {"schema", "version", "segments"}
        or source.get("schema") != PROMPT_FUSION_INPUT_SCHEMA
        or source_version not in {
            PROMPT_FUSION_LEGACY_VERSION, PROMPT_FUSION_VERSION,
        }
        or not isinstance(segments, list)
        or not segments
    ):
        raise LongGenerationError("prompt_fusion_input_invalid")
    timelines: list[list[dict] | None] = []
    cut_timelines: list[list[dict] | None] = []
    relation_occurrences_by_segment: list[list[dict] | None] = []
    for index, segment in enumerate(segments, 1):
        base_segment_keys = {
            "index", "new_keyframes", "old_video_prompt",
            "image_optimization_prompt", "audio_content",
        }
        allowed_segment_keys = {
            frozenset(base_segment_keys),
            frozenset(base_segment_keys | {"relation_occurrences"}),
        }
        if source_version == PROMPT_FUSION_VERSION:
            allowed_segment_keys.update({
                frozenset(base_segment_keys | {"source_cut_timeline"}),
                frozenset(base_segment_keys | {
                    "relation_occurrences", "source_cut_timeline",
                }),
            })
        if (
            not isinstance(segment, Mapping)
            or frozenset(segment) not in allowed_segment_keys
            or segment.get("index") != index
            or (
                source_version == PROMPT_FUSION_LEGACY_VERSION
                and "relation_occurrences" in segment
            )
            or not _fusion_text_binding(segment.get("old_video_prompt"))
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        frames = segment.get("new_keyframes")
        prompts = segment.get("image_optimization_prompt")
        if (
            not isinstance(frames, list)
            or len(frames) != 9
            or not isinstance(prompts, list)
            or len(prompts) != 9
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        for order, (frame, prompt) in enumerate(zip(frames, prompts), 1):
            expected_frame_keys = (
                {"order", "path", "sha256"}
                if source_version == PROMPT_FUSION_LEGACY_VERSION
                else {
                    "order", "path", "sha256", "segment_time_s",
                    "source_scene_id", "transition",
                }
            )
            if (
                not isinstance(frame, Mapping)
                or set(frame) != expected_frame_keys
                or frame.get("order") != order
                or not isinstance(frame.get("path"), str)
                or not frame["path"]
                or not isinstance(frame.get("sha256"), str)
                or len(frame["sha256"]) != 64
                or not isinstance(prompt, Mapping)
                or set(prompt) != {"order", "text", "sha256"}
                or prompt.get("order") != order
                or not _fusion_text_binding({
                    "text": prompt.get("text"), "sha256": prompt.get("sha256")
                })
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            frame_candidate = project_root / frame["path"]
            if frame_candidate.is_symlink():
                raise LongGenerationError("prompt_fusion_input_invalid")
            frame_path = frame_candidate.resolve()
            try:
                frame_path.relative_to(project_root)
                frame_data = frame_path.read_bytes()
            except (OSError, ValueError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if (
                not frame_data
                or hashlib.sha256(frame_data).hexdigest() != frame["sha256"]
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
        if source_version == PROMPT_FUSION_VERSION:
            try:
                frozen_sources = _freeze_local_keyframe_sources([{
                    key: frame[key]
                    for key in (
                        "order", "segment_time_s", "source_scene_id",
                        "transition",
                    )
                } for frame in frames])
                timelines.append(frozen_sources)
                relation_occurrences_by_segment.append(
                    _freeze_fusion_relation_occurrences(
                        segment.get("relation_occurrences", []), frozen_sources,
                    ) if "relation_occurrences" in segment else None
                )
                raw_cuts = segment.get("source_cut_timeline")
                cut_timelines.append(
                    _freeze_local_cut_timeline(
                        raw_cuts,
                        float(raw_cuts[-1].get("end_segment_s", 0)),
                    )
                    if isinstance(raw_cuts, list) and raw_cuts
                    else None
                )
            except (KeyError, TypeError, ValueError, long_video.LongVideoError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
        else:
            timelines.append(None)
            cut_timelines.append(None)
            relation_occurrences_by_segment.append(None)
        audio = segment.get("audio_content")
        audio_keys = {
            "lines_json", "lines_sha256", "voice_references",
        }
        if source_version == PROMPT_FUSION_VERSION:
            audio_keys.add("music_policy")
        if (
            not isinstance(audio, Mapping)
            or set(audio) != audio_keys
            or not isinstance(audio.get("lines_json"), str)
            or not isinstance(audio.get("voice_references"), list)
            or audio.get("lines_sha256")
            != hashlib.sha256(audio["lines_json"].encode("utf-8")).hexdigest()
            or (
                source_version == PROMPT_FUSION_VERSION
                and audio.get("music_policy") != "forbid"
            )
        ):
            raise LongGenerationError("prompt_fusion_input_invalid")
        try:
            lines = json.loads(audio["lines_json"])
        except json.JSONDecodeError:
            raise LongGenerationError("prompt_fusion_input_invalid") from None
        if not isinstance(lines, list):
            raise LongGenerationError("prompt_fusion_input_invalid")
        used_voice_refs: set[int] = set()
        for order, line in enumerate(lines, 1):
            if (
                not isinstance(line, Mapping)
                or set(line) != {
                    "order", "text", "start_s", "end_s", "delivery", "voice_ref",
                }
                or line.get("order") != order
                or not isinstance(line.get("text"), str)
                or not line["text"].strip()
                or line.get("delivery") not in {"on_screen", "off_screen"}
                or (
                    line.get("voice_ref") is not None
                    if source_version == PROMPT_FUSION_VERSION
                    else line.get("voice_ref") not in {None, 1}
                )
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            if line["voice_ref"] is not None:
                used_voice_refs.add(line["voice_ref"])
        references = audio["voice_references"]
        if len(references) != len(used_voice_refs):
            raise LongGenerationError("prompt_fusion_input_invalid")
        for voice_ref, reference in enumerate(references, 1):
            if (
                not isinstance(reference, Mapping)
                or set(reference) != {"voice_ref", "path", "sha256", "purpose"}
                or reference.get("voice_ref") != voice_ref
                or reference.get("purpose") != "voice"
                or voice_ref not in used_voice_refs
                or not isinstance(reference.get("path"), str)
                or not reference["path"]
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            reference_candidate = project_root / reference["path"]
            if reference_candidate.is_symlink():
                raise LongGenerationError("prompt_fusion_input_invalid")
            reference_path = reference_candidate.resolve()
            try:
                reference_path.relative_to(project_root)
                reference_data = reference_path.read_bytes()
            except (OSError, ValueError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if (
                not reference_data
                or hashlib.sha256(reference_data).hexdigest()
                != reference.get("sha256")
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
    frozen_input_sha256 = hashlib.sha256(input_data).hexdigest()
    output_segments = output.get("segments")
    if (
        set(output) != {"schema", "version", "input_sha256", "segments"}
        or output.get("schema") != PROMPT_FUSION_OUTPUT_SCHEMA
        or output.get("version") != source_version
        or output.get("input_sha256") != frozen_input_sha256
        or not isinstance(output_segments, list)
        or len(output_segments) != len(segments)
    ):
        raise LongGenerationError("prompt_fusion_output_invalid")
    final_prompts: list[str] = []
    for index, segment in enumerate(output_segments, 1):
        if not isinstance(segment, Mapping) or segment.get("index") != index:
            raise LongGenerationError("prompt_fusion_output_invalid")
        audio = segments[index - 1]["audio_content"]
        if source_version == PROMPT_FUSION_VERSION:
            relation_occurrences = relation_occurrences_by_segment[index - 1]
            allowed_output_keys = (
                ({"index", "visual"}, {"index", "visual", "relation_states"})
                if relation_occurrences is not None
                else ({"index", "visual"},)
            )
            if set(segment) not in allowed_output_keys:
                raise LongGenerationError("prompt_fusion_output_invalid")
            timeline = timelines[index - 1]
            if timeline is None:
                raise LongGenerationError("prompt_fusion_input_invalid")
            final_prompts.append(_compile_fusion_ref2va_prompt(
                visual=segment.get("visual"),
                timeline=timeline,
                lines=json.loads(audio["lines_json"]),
                music_policy=audio["music_policy"],
                relation_occurrences=relation_occurrences,
                cut_timeline=cut_timelines[index - 1],
            ))
        else:
            if (
                set(segment) != {"index", "final_prompt"}
                or not isinstance(segment.get("final_prompt"), str)
                or not segment["final_prompt"].strip()
            ):
                raise LongGenerationError("prompt_fusion_output_invalid")
            final_prompts.append(_canonical_fusion_prompt(
                segment["final_prompt"],
                audio["lines_json"],
                version=source_version,
            ))
    return FrozenPromptFusion(
        version=source_version,
        input_path=input_path,
        input_data=input_data,
        input_sha256=frozen_input_sha256,
        output_path=output_path,
        output_data=output_data,
        output_sha256=hashlib.sha256(output_data).hexdigest(),
        segments=tuple(segments),
        final_prompts=tuple(final_prompts),
    )


def _load_prompt_fusion_manifest(
    *, root: Path, skill_source_path: Path | None,
) -> FrozenPromptFusion:
    """Revalidate frozen production; optionally prove current Skill equality."""
    root = Path(root).resolve()
    manifest_path = root / "work" / h3_project.SOURCE_FILENAME
    manifest_data, manifest = _fusion_json(
        manifest_path, "prompt_fusion_manifest_invalid"
    )
    if (
        set(manifest) != {
            "schema", "version", "image_acceptance_sha256", "input",
            "output", "skill", "segments",
        }
        or manifest.get("schema") != PROMPT_FUSION_MANIFEST_SCHEMA
        or manifest.get("version") != PROMPT_FUSION_MANIFEST_VERSION
        or not isinstance(manifest.get("image_acceptance_sha256"), str)
        or len(manifest["image_acceptance_sha256"]) != 64
    ):
        raise LongGenerationError("prompt_fusion_manifest_invalid")

    def artifact(value: object, expected_path: str) -> Path:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256"}
            or value.get("path") != expected_path
            or not isinstance(value.get("sha256"), str)
        ):
            raise LongGenerationError("prompt_fusion_manifest_invalid")
        candidate = root / expected_path
        if candidate.is_symlink():
            raise LongGenerationError("prompt_fusion_manifest_invalid")
        try:
            data = candidate.read_bytes()
        except OSError:
            raise LongGenerationError("prompt_fusion_manifest_invalid") from None
        if not data or hashlib.sha256(data).hexdigest() != value["sha256"]:
            raise LongGenerationError("prompt_fusion_manifest_invalid")
        return candidate

    input_path = artifact(
        manifest["input"], f"work/{h3_project.SKILL_INPUT_FILENAME}"
    )
    output_path = artifact(manifest["output"], "work/h3_prompt_plan.json")
    skill = manifest["skill"]
    if (
        not isinstance(skill, Mapping)
        or set(skill) != {"source_path", "frozen_path", "sha256"}
        or skill.get("source_path") != "skills/video-prompt-fusion/SKILL.md"
        or skill.get("frozen_path") != "work/video_prompt_fusion_skill.md"
    ):
        raise LongGenerationError("prompt_fusion_manifest_invalid")
    frozen_skill = root / skill["frozen_path"]
    try:
        frozen_skill_data = frozen_skill.read_bytes()
    except OSError:
        raise LongGenerationError("prompt_fusion_manifest_invalid") from None
    if (
        frozen_skill.is_symlink()
        or not frozen_skill_data
        or hashlib.sha256(frozen_skill_data).hexdigest() != skill.get("sha256")
    ):
        raise LongGenerationError("prompt_fusion_manifest_invalid")
    if skill_source_path is not None:
        try:
            source_skill_data = Path(skill_source_path).read_bytes()
        except OSError:
            raise LongGenerationError("prompt_fusion_manifest_invalid") from None
        if not source_skill_data or source_skill_data != frozen_skill_data:
            raise LongGenerationError("prompt_fusion_manifest_invalid")
    frozen = load_prompt_fusion(
        input_path=input_path, output_path=output_path, root=root,
    )
    segments = manifest.get("segments")
    if not isinstance(segments, list) or len(segments) != len(frozen.final_prompts):
        raise LongGenerationError("prompt_fusion_manifest_invalid")
    for index, (binding, prompt) in enumerate(
        zip(segments, frozen.final_prompts), 1
    ):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"index", "final_prompt_sha256"}
            or binding.get("index") != index
            or binding.get("final_prompt_sha256")
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        ):
            raise LongGenerationError("prompt_fusion_manifest_invalid")
    return frozen


def load_prompt_fusion_manifest(
    *, root: Path, skill_source_path: Path,
) -> FrozenPromptFusion:
    """Strict creation/recovery gate binding current and frozen Skill bytes."""
    return _load_prompt_fusion_manifest(
        root=root,
        skill_source_path=skill_source_path,
    )


def load_bound_prompt_fusion_manifest(
    *, root: Path, meta: Mapping,
) -> FrozenPromptFusion:
    """Load fusion only when durable project state binds this manifest."""
    root = Path(root).resolve()
    manifest_path = root / "work" / h3_project.SOURCE_FILENAME
    try:
        manifest_data = manifest_path.read_bytes()
        manifest = json.loads(manifest_data.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise LongGenerationError("prompt_fusion_manifest_invalid") from None
    state = meta.get("_prompt_fusion")
    if (
        not isinstance(state, Mapping)
        or state.get("version") not in {
            PROMPT_FUSION_LEGACY_VERSION, PROMPT_FUSION_VERSION,
        }
        or state.get("status") != "done"
        or state.get("error") is not None
        or state.get("manifest_sha256")
        != hashlib.sha256(manifest_data).hexdigest()
        or not isinstance(manifest, Mapping)
        or state.get("raw_output_path") != "work/h3_prompt_plan.json"
        or not isinstance(manifest.get("output"), Mapping)
        or state.get("raw_output_sha256")
        != manifest["output"].get("sha256")
        or state.get("image_acceptance_sha256")
        != prompt_fusion_image_authority_sha256(meta)
        or manifest.get("image_acceptance_sha256")
        != state.get("image_acceptance_sha256")
    ):
        raise LongGenerationError("prompt_fusion_manifest_invalid")
    frozen = _load_prompt_fusion_manifest(
        root=root,
        skill_source_path=None,
    )
    if (
        frozen.version != state.get("version")
        or frozen.input_sha256 != state.get("input_sha256")
    ):
        raise LongGenerationError("prompt_fusion_manifest_invalid")
    return frozen


@dataclass(frozen=True)
class FrozenSegment:
    index: int
    start_s: float
    end_s: float
    chain_id: str
    join_mode: str
    workdir: Path
    first_frame: Path
    first_frame_data: bytes
    last_frame: Path
    last_frame_data: bytes
    prompt: str
    keyframes: tuple[h3.FrozenFrame, ...] = ()
    keyframe_sources: tuple[Mapping, ...] = ()
    source_cut_timeline: tuple[Mapping, ...] = ()
    multimodal: h3_project.FrozenProjectMultimodal | None = None
    dialogue: tuple[dict, ...] = ()
    dialogue_sha256: str | None = None


@dataclass(frozen=True)
class FrozenPlan:
    root: Path
    source: Path
    receipt: str
    segments: tuple[FrozenSegment, ...]
    receipt_version: int = long_video.PLAN_RECEIPT_VERSION
    aspect_ratio: str = h3.H3_DEFAULT_ASPECT_RATIO
    resolution: str = h3.H3_DEFAULT_RESOLUTION
    legacy_layout: bool = False
    workflow: str = h3.H3_WORKFLOW
    prompt_fusion: FrozenPromptFusion | None = None


def _segment_duration_s(plan: FrozenPlan, segment: FrozenSegment) -> float:
    """Interpret a segment boundary with its frozen plan receipt version."""
    try:
        return long_video.segment_duration_s(
            segment.start_s,
            segment.end_s,
            receipt_version=plan.receipt_version,
        )
    except long_video.LongVideoError:
        raise LongGenerationError("long_video_plan_invalid") from None


def _is_h3_multimodal_plan(plan: FrozenPlan) -> bool:
    workflows = getattr(h3, "H3_MULTIMODAL_WORKFLOWS", frozenset())
    return isinstance(workflows, (set, frozenset)) and plan.workflow in workflows


def _requires_context_ir(plan: FrozenPlan) -> bool:
    """Current fusion prompts and historical multimodal prompts use Context IR."""
    return plan.prompt_fusion is not None or _is_h3_multimodal_plan(plan)


def _segment_uses_h3_native_audio(
    plan: FrozenPlan, segment: FrozenSegment,
) -> bool:
    if plan.prompt_fusion is not None:
        return True
    return _is_h3_multimodal_plan(plan)


def _native_audio_segment_indices(plan: FrozenPlan) -> frozenset[int]:
    return frozenset(
        segment.index for segment in plan.segments
        if _segment_uses_h3_native_audio(plan, segment)
    )


def _generation_uses_h3_native_audio(plan: FrozenPlan, generation: Mapping) -> bool:
    expected = (
        H3_NATIVE_AUDIO_ROUTE
        if _native_audio_segment_indices(plan)
        else None
    )
    actual = generation.get("audio_route")
    if actual != expected:
        raise LongGenerationError("long_video_audio_route_invalid")
    return expected is not None


def _stitch_segments(
    plan: FrozenPlan,
    provider_media: Mapping[int, tuple[str, Mapping[str, object]]] | None = None,
) -> list[stitch.StitchSegment]:
    media = provider_media or {}
    return [
        stitch.StitchSegment(
            item.workdir / "generated.mp4",
            _segment_duration_s(plan, item),
            item.join_mode,
            media.get(item.index, (None, None))[0],
            media.get(item.index, (None, None))[1],
        )
        for item in plan.segments
    ]


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise LongGenerationError("long_video_plan_invalid") from None


def _canonical_digest(value: object) -> str:
    try:
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode()
    except (TypeError, ValueError):
        raise LongGenerationError("long_video_plan_invalid") from None
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise LongGenerationError("prompt_fusion_input_invalid") from None


_ANALYSIS_PROVENANCE_FIELDS = frozenset({
    "analysis_audio_path",
    "analysis_audio_sha256",
    "analysis_has_bgm",
    "classification_evidence_sha256",
})


def classification_evidence_sha256(
    *, audio_path: str, audio_sha256: str, has_bgm: bool | None,
    decisions: list[Mapping],
) -> str:
    """Hash the existing classifier version and its complete decision summary."""
    stripped = [
        {
            key: value for key, value in decision.items()
            if key not in _ANALYSIS_PROVENANCE_FIELDS
        }
        for decision in decisions
    ]
    return hashlib.sha256(_canonical_json_bytes({
        "schema": "duet.yamnet-classification-evidence",
        "version": 1,
        "analysis_audio_path": audio_path,
        "analysis_audio_sha256": audio_sha256,
        "yamnet_sha256": vocal.YAMNET_SHA256,
        "has_bgm": has_bgm,
        "decisions": stripped,
    })).hexdigest()


def _compiled_dialogue(
    dialogue_mode: str, dialogue: tuple[dict, ...],
) -> tuple[dict, ...]:
    """Project current dialogue without reclassifying upstream evidence."""
    if dialogue_mode == "none":
        return ()
    if dialogue_mode == "auto":
        return tuple(
            line for line in dialogue
            if isinstance(line, Mapping) and line.get("classification") == "spoken"
        )
    if dialogue_mode in {"edit", "custom"}:
        return tuple(dialogue)
    raise LongGenerationError("invalid_dialogue_mode", 422)


def prompt_fusion_image_authority_sha256(meta: Mapping) -> str:
    """Bind manual acceptance, or the exact v4 MediaKit-only receipt."""
    acceptance = meta.get("_image_user_acceptance")
    if (
        isinstance(acceptance, Mapping)
        and isinstance(acceptance.get("sha256"), str)
        and len(acceptance["sha256"]) == 64
    ):
        return acceptance["sha256"]
    private = meta.get("_postprocess_receipt")
    post = meta.get("postprocess")
    options = private.get("options") if isinstance(private, Mapping) else None
    if (
        isinstance(private, Mapping)
        and private.get("version") == 4
        and isinstance(options, Mapping)
        and options.get("optimize_image") is False
        and isinstance(post, Mapping)
        and post.get("status") == "done"
    ):
        return h3.canonical_json_sha256(dict(private))
    raise LongGenerationError("image_acceptance_required")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_prompt_fusion_proxy(root: Path, data: bytes) -> tuple[Path, str]:
    """Publish one immutable content-addressed Fusion analysis image."""
    digest = hashlib.sha256(data).hexdigest()
    work = root / "work"
    directory = work / PROMPT_FUSION_PROXY_DIR
    if work.is_symlink():
        raise LongGenerationError("prompt_fusion_input_invalid")
    try:
        directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        if directory.is_symlink() or directory.resolve() != directory:
            raise OSError
    except OSError:
        raise LongGenerationError("prompt_fusion_input_invalid") from None
    path = directory / f"{digest}.png"
    if path.exists() or path.is_symlink():
        try:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise OSError
        except OSError:
            raise LongGenerationError("prompt_fusion_input_invalid") from None
        return path, digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{digest}.", suffix=".tmp", dir=directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise OSError
        descriptor = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise LongGenerationError("prompt_fusion_input_invalid") from None
    finally:
        temporary.unlink(missing_ok=True)
    return path, digest


def build_prompt_fusion_input(
    *, root: Path, meta: Mapping, plan: FrozenPlan,
    dialogue_mode: str, dialogue_delivery: str,
) -> bytes:
    """Compile the four frozen fusion inputs for all ordered segments."""
    root = Path(root).resolve()
    if plan.root != root or not plan.segments:
        raise LongGenerationError("prompt_fusion_input_invalid")
    raw_segments = meta.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) != len(plan.segments):
        raise LongGenerationError("prompt_fusion_input_invalid")
    optimization = meta.get("_image_optimization")
    optimization_frames = (
        optimization.get("frames") if isinstance(optimization, Mapping) else None
    )
    if not isinstance(optimization_frames, list):
        raise LongGenerationError("prompt_fusion_input_invalid")
    try:
        requested_delivery = dialogue_delivery_contract.parse(dialogue_delivery)
        authoritative_dialogue = tuple(
            line
            for segment in plan.segments
            for line in _compiled_dialogue(dialogue_mode, segment.dialogue)
        )
        resolved_delivery = dialogue_delivery_contract.resolve(
            requested_delivery, authoritative_dialogue
        ).value
    except ValueError:
        raise LongGenerationError("invalid_dialogue_delivery", 422) from None
    if dialogue_mode != "none" and authoritative_dialogue \
            and resolved_delivery == "on_screen":
        raise LongGenerationError("on_screen_authority_unavailable")
    has_visual_timeline = [
        len(segment.keyframe_sources) == len(segment.keyframes) == 9
        for segment in plan.segments
    ]
    if any(has_visual_timeline) and not all(has_visual_timeline):
        raise LongGenerationError("prompt_fusion_input_invalid")
    if not all(has_visual_timeline):
        raise LongGenerationError("prompt_fusion_refresh_required")
    fusion_version = PROMPT_FUSION_VERSION
    compiled_segments: list[dict] = []
    optimization_indices = {
        item.get("segment_index")
        for item in optimization_frames if isinstance(item, Mapping)
    }
    top_level_short = len(plan.segments) == 1 and optimization_indices == {0}
    if not top_level_short and optimization_indices != set(
        range(1, len(plan.segments) + 1)
    ):
        raise LongGenerationError("image_optimization_prompt_invalid")
    for index, (raw, segment) in enumerate(
        zip(raw_segments, plan.segments), 1
    ):
        if not isinstance(raw, Mapping) or segment.index != index:
            raise LongGenerationError("prompt_fusion_input_invalid")
        old_prompt = raw.get("visual_prompt")
        if not isinstance(old_prompt, str) or not old_prompt.strip():
            raise LongGenerationError("prompt_fusion_input_invalid")
        if len(segment.keyframes) != 9:
            raise LongGenerationError("postprocess_artifacts_invalid")
        keyframes: list[dict] = []
        for order, (path, data) in enumerate(segment.keyframes, 1):
            try:
                relative = path.resolve().relative_to(root).as_posix()
            except ValueError:
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if path.is_symlink():
                raise LongGenerationError("prompt_fusion_input_invalid")
            try:
                current_data = path.read_bytes()
            except OSError:
                raise LongGenerationError(
                    "prompt_fusion_input_invalid"
                ) from None
            if (
                not data
                or current_data != data
                or hashlib.sha256(current_data).hexdigest()
                != hashlib.sha256(data).hexdigest()
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            try:
                proxy_data = image_optimization.half_resolution_png(data)
            except ValueError:
                raise LongGenerationError(
                    "prompt_fusion_input_invalid"
                ) from None
            proxy_path, proxy_sha256 = _publish_prompt_fusion_proxy(
                root, proxy_data,
            )
            binding = {
                "order": order,
                "path": proxy_path.relative_to(root).as_posix(),
                "sha256": proxy_sha256,
            }
            if fusion_version == PROMPT_FUSION_VERSION:
                source = segment.keyframe_sources[order - 1]
                binding.update({
                    key: source[key]
                    for key in (
                        "source_time_s", "source_scene_id", "transition",
                    )
                })
            keyframes.append(binding)
        try:
            keyframes, _timeline_diagnostics = long_video.localize_keyframe_sources(
                keyframes,
                segment_start_s=segment.start_s,
                segment_end_s=segment.end_s,
                provider_duration_s=long_video.provider_duration_s(
                    segment.start_s,
                    segment.end_s,
                    receipt_version=plan.receipt_version,
                ),
            )
        except long_video.LongVideoError:
            raise LongGenerationError("prompt_fusion_input_invalid") from None

        source_index = 0 if top_level_short else index
        prompts = [
            item for item in optimization_frames
            if isinstance(item, Mapping)
            and item.get("segment_index") == source_index
        ]
        if len(prompts) != 9:
            raise LongGenerationError("image_optimization_prompt_invalid")
        prompt_bindings: list[dict] = []
        for order, item in enumerate(prompts, 1):
            text = item.get("current")
            digest = item.get("sha256")
            if (
                not isinstance(text, str)
                or not text.strip()
                or digest != hashlib.sha256(text.encode("utf-8")).hexdigest()
            ):
                raise LongGenerationError("image_optimization_prompt_invalid")
            prompt_bindings.append({
                "order": order, "text": text, "sha256": digest,
            })

        relation_occurrences: list[dict] | None = None
        continuity = meta.get("_image_continuity")
        continuity_segments = (
            continuity.get("segments")
            if isinstance(continuity, Mapping) else None
        )
        if isinstance(continuity_segments, list):
            image_segment = next((
                item for item in continuity_segments
                if isinstance(item, Mapping)
                and item.get("segment_index") == source_index
            ), None)
            frame_constraints = (
                image_segment.get("frame_constraints")
                if isinstance(image_segment, Mapping) else None
            )
            if (
                isinstance(frame_constraints, list)
                and len(frame_constraints) == 9
                and all(
                    isinstance(item, Mapping)
                    and "relation_occurrences" in item
                    for item in frame_constraints
                )
            ):
                relation_occurrences = []
                for order, (constraint, keyframe) in enumerate(
                    zip(frame_constraints, keyframes, strict=True), 1
                ):
                    if constraint.get("frame_index") != order:
                        raise LongGenerationError(
                            "image_optimization_prompt_invalid"
                        )
                    raw_occurrences = constraint.get("relation_occurrences")
                    if not isinstance(raw_occurrences, list):
                        raise LongGenerationError(
                            "image_optimization_prompt_invalid"
                        )
                    for occurrence in raw_occurrences:
                        if not isinstance(occurrence, Mapping):
                            raise LongGenerationError(
                                "image_optimization_prompt_invalid"
                            )
                        relation_occurrences.append({
                            **dict(occurrence),
                            "frame": {
                                "order": order,
                                "segment_time_s": keyframe["segment_time_s"],
                                "source_scene_id": keyframe["source_scene_id"],
                            },
                        })
                relation_occurrences = _freeze_fusion_relation_occurrences(
                    relation_occurrences,
                    _freeze_local_keyframe_sources([{
                        key: frame[key]
                        for key in (
                            "order", "segment_time_s", "source_scene_id",
                            "transition",
                        )
                    } for frame in keyframes]),
                )

        dialogue = _compiled_dialogue(dialogue_mode, segment.dialogue)
        lines: list[dict] = []
        for order, line in enumerate(dialogue, 1):
            try:
                text = line["text"]
                start_s = line["start_s"]
                end_s = line["end_s"]
            except (KeyError, TypeError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if not isinstance(text, str) or not text.strip():
                raise LongGenerationError("prompt_fusion_input_invalid")
            lines.append({
                "order": order,
                "text": text,
                "start_s": start_s,
                "end_s": end_s,
                "delivery": resolved_delivery,
                "voice_ref": None,
            })
        voice_references: list[dict] = []
        lines_json = json.dumps(
            lines, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        )
        compiled_segment = {
            "index": index,
            "new_keyframes": keyframes,
            "old_video_prompt": {
                "text": old_prompt,
                "sha256": hashlib.sha256(old_prompt.encode("utf-8")).hexdigest(),
            },
            "image_optimization_prompt": prompt_bindings,
            "audio_content": {
                "lines_json": lines_json,
                "lines_sha256": hashlib.sha256(
                    lines_json.encode("utf-8")
                ).hexdigest(),
                "voice_references": voice_references,
                "music_policy": "forbid",
            },
        }
        # The existing keyframe transition contract is already complete when
        # every cut has an analysis image.  Add the independent authority only
        # for dense montages, preserving the normal-video Fusion bytes.
        if (
            segment.source_cut_timeline
            and len(segment.source_cut_timeline) > len(segment.keyframe_sources)
        ):
            compiled_segment["source_cut_timeline"] = (
                _localize_source_cut_timeline(
                    list(segment.source_cut_timeline),
                    segment_start_s=segment.start_s,
                    segment_end_s=segment.end_s,
                )
            )
        if relation_occurrences is not None:
            compiled_segment["relation_occurrences"] = relation_occurrences
        compiled_segments.append(compiled_segment)
    return _canonical_json_bytes({
        "schema": PROMPT_FUSION_INPUT_SCHEMA,
        "version": fusion_version,
        "segments": compiled_segments,
    })


def _publish_fusion_h3_segments(
    *, root: Path, meta: Mapping, base: FrozenPlan,
    fusion: FrozenPromptFusion, dialogue_mode: str,
) -> tuple[list[dict], list[dict]]:
    """Freeze exact fusion prompts; Context IR is the sole later transformer."""
    receipt_segments: list[dict] = []
    updated_segments: list[dict] = []
    raw_segments = meta.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) != len(base.segments):
        raise LongGenerationError("prompt_fusion_input_invalid")
    for source, segment, fusion_source, final_prompt in zip(
        raw_segments, base.segments, fusion.segments, fusion.final_prompts
    ):
        if not isinstance(source, Mapping):
            raise LongGenerationError("prompt_fusion_input_invalid")
        work = segment.workdir / "work"
        visual_path = work / "fusion_prompt.txt"
        _atomic_bytes(visual_path, final_prompt.encode("utf-8"))

        for item in fusion_source["new_keyframes"]:
            source_candidate = root / item["path"]
            if source_candidate.is_symlink():
                raise LongGenerationError("prompt_fusion_input_invalid")
            source_path = source_candidate.resolve()
            try:
                source_path.relative_to(root)
                source_data = source_path.read_bytes()
            except (OSError, ValueError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if (
                not source_data
                or hashlib.sha256(source_data).hexdigest() != item["sha256"]
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")

        audio = fusion_source["audio_content"]
        if audio.get("music_policy") != "forbid":
            raise LongGenerationError("prompt_fusion_input_invalid")
        lines = json.loads(audio["lines_json"])
        if audio["voice_references"]:
            raise LongGenerationError("prompt_fusion_input_invalid")
        dialogue = _compiled_dialogue(dialogue_mode, segment.dialogue)
        if len(lines) != len(dialogue):
            raise LongGenerationError("prompt_fusion_input_invalid")
        for order, (compiled, authoritative) in enumerate(
            zip(lines, dialogue), 1
        ):
            if (
                compiled["order"] != order
                or compiled["text"] != authoritative.get("text")
                or compiled["start_s"] != authoritative.get("start_s")
                or compiled["end_s"] != authoritative.get("end_s")
                or compiled["delivery"] != "off_screen"
                or compiled["voice_ref"] is not None
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
        updated = {
            **dict(source),
            "prompt": final_prompt,
            "dialogue": list(dialogue),
            "lines": [line["text"] for line in dialogue],
        }
        updated_segments.append(updated)
        receipt_segments.append({
            **updated,
            "source_path": root / "work" / str(source["source"]),
            "keyframe_paths": [
                root / "work" / str(path) for path in source["keyframe_paths"]
            ],
            "first_frame_path": root / "work" / str(source["first_frame_path"]),
            "last_frame_path": root / "work" / str(source["last_frame_path"]),
            "visual_prompt_path": work / "visual_prompt.txt",
            "final_prompt_path": visual_path,
            "dialogue": list(dialogue),
        })
    return receipt_segments, updated_segments


def plan_receipt(root: Path, meta: Mapping) -> str | None:
    name = meta.get("long_video_plan_receipt")
    if name != long_video.PLAN_RECEIPT_FILENAME:
        return None
    path = Path(root) / name
    if not path.is_file():
        return None
    try:
        return _digest(path)
    except LongGenerationError:
        return None


def _derive_normalized_n1_keyframe_sources(
    *,
    authority_work: Path,
    root: Path,
    source: Path,
    source_data: bytes,
    originals: list[Path],
    private: Mapping,
    duration: float,
) -> list[dict]:
    """Bind legacy root selections to measured source time and scene facts."""
    manifest_path = authority_work / "manifest.json"
    scenes_path = authority_work / "scenes.json"
    if manifest_path.is_symlink() or scenes_path.is_symlink():
        raise LongGenerationError("long_video_plan_invalid")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scene_authority = json.loads(scenes_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise LongGenerationError("long_video_plan_invalid") from None
    manifest_frames = manifest.get("frames") if isinstance(manifest, Mapping) else None
    raw_scenes = (
        scene_authority.get("scenes")
        if isinstance(scene_authority, Mapping) else None
    )
    try:
        manifest_source = Path(manifest["source"]).resolve()
        manifest_size = manifest["file_size_bytes"]
        manifest_duration = float(manifest["duration_seconds"])
        scene_duration = float(scene_authority["duration_s"])
    except (KeyError, TypeError, ValueError, OSError):
        raise LongGenerationError("long_video_plan_invalid") from None
    if (
        manifest_source != source
        or isinstance(manifest_size, bool)
        or not isinstance(manifest_size, int)
        or manifest_size != len(source_data)
        or not math.isfinite(manifest_duration)
        or not math.isfinite(scene_duration)
        or abs(manifest_duration - duration) > _EPS
        or abs(scene_duration - duration) > 0.001 + _EPS
        or not isinstance(manifest_frames, list)
        or not manifest_frames
        or not isinstance(raw_scenes, list)
        or not raw_scenes
    ):
        raise LongGenerationError("long_video_plan_invalid")

    work_root = authority_work.resolve()
    candidates: list[dict] = []
    candidate_names: set[str] = set()
    previous_time = -1.0
    for position, raw in enumerate(manifest_frames, 1):
        if not isinstance(raw, Mapping):
            raise LongGenerationError("long_video_plan_invalid")
        try:
            name = raw["file"]
            time_s = round(
                float(raw["time_seconds"]), long_video.BOUNDARY_PRECISION
            )
        except (KeyError, TypeError, ValueError):
            raise LongGenerationError("long_video_plan_invalid") from None
        if (
            raw.get("index") != position
            or not isinstance(name, str)
            or not name
            or name in candidate_names
            or not math.isfinite(time_s)
            or time_s < 0
            or time_s > duration + _EPS
            or time_s <= previous_time
        ):
            raise LongGenerationError("long_video_plan_invalid")
        candidate = (authority_work / name)
        if candidate.is_symlink():
            raise LongGenerationError("long_video_plan_invalid")
        try:
            resolved = candidate.resolve()
            resolved.relative_to(work_root)
            data = resolved.read_bytes()
        except (OSError, ValueError):
            raise LongGenerationError("long_video_plan_invalid") from None
        if not data:
            raise LongGenerationError("long_video_plan_invalid")
        candidates.append({
            "name": name,
            "time_s": time_s,
            "sha256": hashlib.sha256(data).hexdigest(),
        })
        candidate_names.add(name)
        previous_time = time_s

    scenes: list[dict] = []
    prior_end = 0.0
    classified_names: set[str] = set()
    for position, raw in enumerate(raw_scenes, 1):
        if not isinstance(raw, Mapping):
            raise LongGenerationError("long_video_plan_invalid")
        try:
            start_s = round(
                float(raw["start_s"]), long_video.BOUNDARY_PRECISION
            )
            end_s = round(float(raw["end_s"]), long_video.BOUNDARY_PRECISION)
        except (KeyError, TypeError, ValueError):
            raise LongGenerationError("long_video_plan_invalid") from None
        names = raw.get("frames")
        if (
            raw.get("index") != position
            or not math.isfinite(start_s)
            or not math.isfinite(end_s)
            or start_s >= end_s
            or abs(start_s - prior_end) > 0.001 + _EPS
            or not isinstance(names, list)
            or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
            or classified_names.intersection(names)
            or any(name not in candidate_names for name in names)
        ):
            raise LongGenerationError("long_video_plan_invalid")
        scene = {
            "id": f"SCENE_{position:02d}",
            "start_s": start_s,
            "end_s": end_s,
            "frames": set(names),
        }
        for candidate in candidates:
            if candidate["name"] in scene["frames"] and not (
                start_s <= candidate["time_s"] < end_s
                or (
                    position == len(raw_scenes)
                    and abs(candidate["time_s"] - end_s) <= 0.001 + _EPS
                )
            ):
                raise LongGenerationError("long_video_plan_invalid")
        scenes.append(scene)
        classified_names.update(names)
        prior_end = end_s
    if (
        abs(prior_end - duration) > 0.001 + _EPS
        or classified_names != candidate_names
    ):
        raise LongGenerationError("long_video_plan_invalid")

    receipt_frames = private.get("frames")
    if not isinstance(receipt_frames, list) or len(receipt_frames) != len(originals):
        raise LongGenerationError("postprocess_receipt_invalid")
    selected: list[dict] = []
    used_candidates: set[int] = set()
    selected_time = -1.0
    for order, (path, frozen) in enumerate(zip(originals, receipt_frames), 1):
        if not isinstance(frozen, Mapping) or path.is_symlink():
            raise LongGenerationError("postprocess_receipt_invalid")
        try:
            selected_data = path.read_bytes()
            selected_relative = path.resolve().relative_to(root / "work").as_posix()
        except (OSError, ValueError):
            raise LongGenerationError("postprocess_receipt_invalid") from None
        selected_sha256 = hashlib.sha256(selected_data).hexdigest()
        if (
            not selected_data
            or frozen.get("segment_index") != 0
            or frozen.get("frame_name") != path.name
            or frozen.get("source_sha256") != selected_sha256
        ):
            raise LongGenerationError("postprocess_receipt_invalid")
        exact = [
            index for index, candidate in enumerate(candidates)
            if index not in used_candidates
            and candidate["sha256"] == selected_sha256
            and candidate["time_s"] > selected_time
            and candidate["name"] in {selected_relative, path.name}
        ]
        matches = exact or [
            index for index, candidate in enumerate(candidates)
            if index not in used_candidates
            and candidate["sha256"] == selected_sha256
            and candidate["time_s"] > selected_time
        ]
        if not matches:
            raise LongGenerationError("postprocess_artifacts_invalid")
        candidate_index = matches[0]
        candidate = candidates[candidate_index]
        scene_matches = [
            scene for scene in scenes if candidate["name"] in scene["frames"]
        ]
        if len(scene_matches) != 1:
            raise LongGenerationError("long_video_plan_invalid")
        scene = scene_matches[0]
        transition = (
            {"type": "start", "at_s": candidate["time_s"]}
            if not selected else (
                {
                    "type": "hard_cut",
                    "at_s": scene["start_s"],
                }
                if scene["id"] != selected[-1]["source_scene_id"]
                else {"type": "continuous", "at_s": None}
            )
        )
        selected.append({
            "order": order,
            "source_time_s": candidate["time_s"],
            "source_scene_id": scene["id"],
            "transition": transition,
        })
        used_candidates.add(candidate_index)
        selected_time = candidate["time_s"]
    try:
        frozen, _last = long_video.freeze_keyframe_sources(
            selected, expected_count=9
        )
    except long_video.LongVideoError:
        raise LongGenerationError("long_video_plan_invalid") from None
    return frozen


def _regenerate_normalized_n1_keyframe_sources(
    *,
    root: Path,
    source: Path,
    source_data: bytes,
    originals: list[Path],
    private: Mapping,
    duration: float,
) -> list[dict]:
    """Rebuild missing/stale root authority from the exact uploaded source."""
    repository = Path(__file__).resolve().parents[1]
    extract_script = repository / "skills" / "video-maker" / "scripts" / (
        "extract_keyframes.py"
    )
    scenes_script = repository / "app" / "scenes.py"
    try:
        with tempfile.TemporaryDirectory(
            prefix="duet-n1-timeline-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            subprocess.run(
                [
                    sys.executable, str(extract_script), str(source),
                    "--out-dir", str(stage), "--fps", "4",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess.run(
                [
                    sys.executable, str(scenes_script), str(source),
                    "--work-dir", str(stage),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return _derive_normalized_n1_keyframe_sources(
                authority_work=stage,
                root=root,
                source=source,
                source_data=source_data,
                originals=originals,
                private=private,
                duration=duration,
            )
    except (
        OSError, subprocess.SubprocessError, LongGenerationError,
    ):
        raise LongGenerationError("long_video_plan_invalid") from None


def normalize_single_segment_project(
    settings: Settings, cid: str, meta: Mapping,
) -> dict:
    """Adapt a frozen pre-unification v4 N=1 project to segments[1]."""
    root = (settings.data_dir / cid).resolve()
    if (
        meta.get("schema_version") != 2
        or isinstance(meta.get("segments"), list)
        or meta.get("long_video_plan_receipt") is not None
    ):
        return dict(meta)
    private = meta.get("_postprocess_receipt")
    if (
        not isinstance(private, Mapping)
        or private.get("version") != 4
        or not isinstance(private.get("options"), Mapping)
    ):
        return dict(meta)
    try:
        duration = float(meta["duration_s"])
    except (KeyError, TypeError, ValueError):
        raise LongGenerationError("long_video_plan_invalid") from None
    if not 0 < duration <= long_video.LEGACY_PROVIDER_MAX_DURATION_S:
        raise LongGenerationError("long_video_plan_invalid")
    names = meta.get("keyframes")
    if not isinstance(names, list) or len(names) != 9:
        raise LongGenerationError("postprocess_artifacts_invalid")
    originals = [root / "work" / "keyframes" / str(name) for name in names]
    try:
        selected = postprocess.generation_keyframes(
            root, dict(meta), originals, settings=settings,
        )
    except postprocess.PostprocessError as exc:
        detail = exc.detail if isinstance(exc.detail, str) else exc.detail["code"]
        raise LongGenerationError(detail, exc.status) from None
    if len(selected) != 9:
        raise LongGenerationError("postprocess_artifacts_invalid")
    sources = sorted(root.glob("source.*"))
    if len(sources) != 1:
        raise LongGenerationError("long_video_plan_invalid")
    segment_dir = root / "work" / "segments" / "1"
    segment_work = segment_dir / "work"
    if sources[0].is_symlink():
        raise LongGenerationError("long_video_plan_invalid")
    try:
        source_data = sources[0].read_bytes()
    except OSError:
        raise LongGenerationError("long_video_plan_invalid") from None
    if not source_data:
        raise LongGenerationError("long_video_plan_invalid")
    try:
        keyframe_sources = _derive_normalized_n1_keyframe_sources(
            authority_work=root / "work",
            root=root,
            source=sources[0].resolve(),
            source_data=source_data,
            originals=originals,
            private=private,
            duration=duration,
        )
    except LongGenerationError:
        keyframe_sources = _regenerate_normalized_n1_keyframe_sources(
            root=root,
            source=sources[0].resolve(),
            source_data=source_data,
            originals=originals,
            private=private,
            duration=duration,
        )
    _atomic_bytes(segment_dir / "source.mp4", source_data)
    segment_work.mkdir(parents=True, exist_ok=True)
    visual_path = root / "work" / "visual_prompt.txt"
    try:
        visual = visual_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise LongGenerationError("long_video_plan_invalid") from None
    frozen_visual = segment_work / "visual_prompt.txt"
    _atomic_bytes(frozen_visual, visual.encode("utf-8"))
    dialogue = []
    raw_lines = meta.get("voice_line_provenance")
    if isinstance(raw_lines, list):
        for raw in raw_lines:
            if not isinstance(raw, Mapping) or raw.get("kept") is not True:
                continue
            line = {
                key: raw[key] for key in ("text", "start_s", "end_s")
                if key in raw
            }
            for optional in ("classification", "language"):
                if optional in raw:
                    line[optional] = raw[optional]
            dialogue.append(line)
    try:
        final = (
            f"{_LEGACY_PIPELINE_NO_BGM}\n"
            + prepared_input.compose_final_prompt(
                long_video.compose_segment_visual_prompt(visual), dialogue,
            )
        )
    except (prepared_input.PreparedInputError, long_video.LongVideoError):
        raise LongGenerationError("long_video_plan_invalid") from None
    final_path = segment_work / "prompt.txt"
    _atomic_bytes(final_path, final.encode("utf-8"))
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": round(duration, long_video.BOUNDARY_PRECISION),
        "chain_id": "chain-001",
        "join_mode": "hard_cut",
        "source": "segments/1/source.mp4",
        "keyframes": list(names),
        "keyframe_paths": [
            path.resolve().relative_to(root / "work").as_posix()
            for path in originals
        ],
        "keyframe_sources": keyframe_sources,
        "first_frame_path": originals[0].resolve().relative_to(
            root / "work"
        ).as_posix(),
        "last_frame_path": originals[-1].resolve().relative_to(
            root / "work"
        ).as_posix(),
        "visual_prompt": visual,
        "prompt": final,
        "dialogue": dialogue,
        "lines": [line["text"] for line in dialogue],
    }
    receipt_writer = (
        long_video._write_frozen_v4_n1_plan_receipt
        if duration > long_video.SHORT_VIDEO_MAX_S
        else long_video.write_plan_receipt
    )
    receipt_path = receipt_writer(
        root,
        source=sources[0],
        duration_s=duration,
        segments=[{
            **segment,
            "source_path": segment_dir / "source.mp4",
            "keyframe_paths": originals,
            "first_frame_path": originals[0],
            "last_frame_path": originals[-1],
            "visual_prompt_path": frozen_visual,
            "final_prompt_path": final_path,
        }],
        workflow=h3.H3_WORKFLOW,
    )
    updated = storage.update_meta(
        settings.data_dir,
        cid,
        segments=[segment],
        long_video_plan_receipt=receipt_path.name,
    )
    if updated is None:
        raise LongGenerationError("long_video_plan_invalid")
    return updated


def _promote_legacy_segment_multimodal_intent(
    *, root: Path, meta: Mapping, expected_receipt: str,
    previous_bytes: bytes, fit_mode: str, dialogue_mode: str,
    dialogue_delivery: str, aspect_ratio: str, resolution: str,
    settings: Settings | None,
) -> str | None:
    """Preserve the exact historical v2 per-segment multimodal promotion."""
    current_segments = meta.get("segments")
    if not isinstance(current_segments, list) or not current_segments:
        return None
    intent: list[bool] = []
    complete: list[bool] = []
    receipt_segments: list[dict] = []
    authoritative_dialogue: list[dict] = []
    for expected_index, current in enumerate(current_segments, 1):
        if not isinstance(current, Mapping) or current.get("index") != expected_index:
            raise LongGenerationError("long_video_plan_invalid")
        segdir = root / "work" / "segments" / str(expected_index)
        segwork = segdir / "work"
        manifest = segwork / h3_project.SOURCE_FILENAME
        intent.append(any(
            (segwork / name).exists()
            for name in (
                h3_project.SKILL_INPUT_FILENAME,
                "h3_prompt_plan.json",
                h3_project.SOURCE_FILENAME,
            )
        ))
        complete.append(manifest.is_file())
        raw_dialogue = current.get("dialogue")
        if not isinstance(raw_dialogue, list):
            raise LongGenerationError("long_video_plan_invalid")
        authoritative_dialogue.extend(
            dict(line) for line in raw_dialogue if isinstance(line, Mapping)
        )
        try:
            receipt_segments.append({
                **dict(current),
                "source_path": root / "work" / str(current["source"]),
                "keyframe_paths": [
                    root / "work" / str(path)
                    for path in current["keyframe_paths"]
                ],
                "first_frame_path": root / "work" / str(
                    current["first_frame_path"]
                ),
                "last_frame_path": root / "work" / str(
                    current["last_frame_path"]
                ),
                "visual_prompt_path": segwork / "visual_prompt.txt",
                "final_prompt_path": segwork / "prompt.txt",
                "dialogue": [] if dialogue_mode == "none" else raw_dialogue,
                "multimodal_manifest_path": manifest,
            })
        except (KeyError, TypeError):
            raise LongGenerationError("long_video_plan_invalid") from None
    if not any(intent):
        return None
    if not all(complete):
        raise LongGenerationError("long_video_multimodal_incomplete")
    try:
        resolved_delivery = dialogue_delivery_contract.resolve(
            dialogue_delivery_contract.parse(dialogue_delivery),
            tuple(authoritative_dialogue),
        ).value
    except ValueError:
        raise LongGenerationError("invalid_dialogue_delivery", 422) from None
    base = freeze_plan(
        root, meta, expected_receipt, fit_mode, dialogue_mode,
        dialogue_delivery=dialogue_delivery,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        prepare_fit=True,
        settings=settings,
    )
    receipt_path = root / long_video.PLAN_RECEIPT_FILENAME
    try:
        long_video.write_plan_receipt(
            root,
            source=base.source,
            duration_s=float(meta["duration_s"]),
            segments=receipt_segments,
            workflow=h3.H3_MULTIMODAL_WORKFLOW,
            dialogue_mode=dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            resolved_dialogue_delivery=resolved_delivery,
        )
        promoted = _digest(receipt_path)
        freeze_plan(
            root, meta, promoted, fit_mode, dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            prepare_fit=True,
            settings=settings,
        )
        return promoted
    except (
        OSError, KeyError, TypeError, ValueError,
        long_video.LongVideoError, LongGenerationError,
    ) as exc:
        _atomic_bytes(receipt_path, previous_bytes)
        raise LongGenerationError(
            getattr(exc, "code", "long_video_multimodal_invalid")
        ) from None


def finalize_multimodal_plan(
    root: Path,
    meta: Mapping,
    expected_receipt: str,
    fit_mode: str,
    dialogue_mode: str,
    *,
    aspect_ratio: str,
    resolution: str,
    dialogue_delivery: str = "auto",
    settings: Settings | None = None,
) -> str | None:
    """Freeze/consume the one project-level prompt fusion before H3."""
    root = Path(root).resolve()
    if isinstance(meta.get("generation"), Mapping):
        return None
    receipt_path = root / long_video.PLAN_RECEIPT_FILENAME
    try:
        previous_bytes = receipt_path.read_bytes()
        payload = json.loads(previous_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise LongGenerationError("long_video_plan_invalid") from None
    if hashlib.sha256(previous_bytes).hexdigest() != expected_receipt:
        raise LongGenerationError("long_video_plan_changed")
    receipt_version = payload.get("version") if isinstance(payload, Mapping) else None
    if receipt_version in {
        long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
        long_video.VISUAL_MULTIMODAL_PLAN_RECEIPT_VERSION,
    }:
        if isinstance(payload, Mapping) and payload.get("prompt_fusion") is not None:
            if settings is None:
                raise LongGenerationError("prompt_fusion_manifest_invalid")
            frozen = load_bound_prompt_fusion_manifest(
                root=root,
                meta=meta,
            )
            if frozen.version == PROMPT_FUSION_LEGACY_VERSION:
                raise LongGenerationError("prompt_fusion_v2_refresh_required")
        return None
    if receipt_version not in {
        long_video.PLAN_RECEIPT_VERSION,
        long_video.VISUAL_PLAN_RECEIPT_VERSION,
    }:
        return None
    private_postprocess = meta.get("_postprocess_receipt")
    if (
        not isinstance(private_postprocess, Mapping)
        or private_postprocess.get("version") != 4
    ):
        return _promote_legacy_segment_multimodal_intent(
            root=root,
            meta=meta,
            expected_receipt=expected_receipt,
            previous_bytes=previous_bytes,
            fit_mode=fit_mode,
            dialogue_mode=dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            settings=settings,
        )
    if settings is None:
        raise LongGenerationError("prompt_fusion_refresh_required")

    base = freeze_plan(
        root,
        meta,
        expected_receipt,
        fit_mode,
        dialogue_mode,
        dialogue_delivery=dialogue_delivery,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        prepare_fit=True,
        settings=settings,
    )
    input_data = build_prompt_fusion_input(
        root=root,
        meta=meta,
        plan=base,
        dialogue_mode=dialogue_mode,
        dialogue_delivery=dialogue_delivery,
    )
    image_authority_sha256 = prompt_fusion_image_authority_sha256(meta)
    from app import pipeline  # local import: pipeline owns the Codex runner seam

    queued = pipeline.queue_prompt_fusion(
        settings,
        str(meta.get("id")),
        input_data=input_data,
        image_acceptance_sha256=image_authority_sha256,
    )
    if queued == "failed":
        raise LongGenerationError("prompt_fusion_failed")
    if queued != "done":
        raise LongGenerationError("prompt_fusion_refresh_required")
    fusion = load_bound_prompt_fusion_manifest(
        root=root,
        meta=meta,
    )
    if fusion.input_data != input_data:
        raise LongGenerationError("prompt_fusion_input_invalid")
    try:
        resolved_delivery = dialogue_delivery_contract.resolve(
            dialogue_delivery_contract.parse(dialogue_delivery),
            tuple(
                line
                for segment in base.segments
                for line in _compiled_dialogue(dialogue_mode, segment.dialogue)
            ),
        ).value
    except ValueError:
        raise LongGenerationError("invalid_dialogue_delivery", 422) from None
    receipt_segments, updated_segments = _publish_fusion_h3_segments(
        root=root,
        meta=meta,
        base=base,
        fusion=fusion,
        dialogue_mode=dialogue_mode,
    )
    try:
        fusion_has_audio = any(
            bool(segment["audio_content"]["voice_references"])
            for segment in fusion.segments
        )
    except (KeyError, TypeError):
        raise LongGenerationError("prompt_fusion_input_invalid") from None
    if fusion_has_audio:
        raise LongGenerationError("prompt_fusion_input_invalid")
    promoted_workflow = h3.H3_WORKFLOW
    try:
        duration = float(meta["duration_s"])
    except (KeyError, TypeError, ValueError):
        raise LongGenerationError("long_video_plan_invalid") from None
    historical_v4_n1 = (
        receipt_version == long_video.VISUAL_PLAN_RECEIPT_VERSION
        and payload.get("workflow") == h3.H3_WORKFLOW
        and isinstance(private_postprocess.get("options"), Mapping)
        and len(base.segments) == 1
        and long_video.SHORT_VIDEO_MAX_S
        < duration
        <= long_video.LEGACY_PROVIDER_MAX_DURATION_S
    )
    receipt_writer = (
        long_video._write_frozen_v4_n1_plan_receipt
        if historical_v4_n1
        else long_video.write_plan_receipt
    )
    try:
        receipt_writer(
            root,
            source=base.source,
            duration_s=duration,
            segments=receipt_segments,
            workflow=promoted_workflow,
            dialogue_mode=dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            resolved_dialogue_delivery=resolved_delivery,
            prompt_fusion_manifest_path=root / "work" / h3_project.SOURCE_FILENAME,
        )
        promoted = _digest(receipt_path)
        promoted_meta = {**dict(meta), "segments": updated_segments}
        freeze_plan(
            root,
            promoted_meta,
            promoted,
            fit_mode,
            dialogue_mode,
            dialogue_delivery=dialogue_delivery,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            prepare_fit=True,
            settings=settings,
        )
        storage.update_meta(
            settings.data_dir,
            str(meta.get("id")),
            segments=updated_segments,
            long_video_plan_receipt=long_video.PLAN_RECEIPT_FILENAME,
        )
        return promoted
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        long_video.LongVideoError,
        LongGenerationError,
    ) as exc:
        _atomic_bytes(receipt_path, previous_bytes)
        code = getattr(exc, "code", "long_video_multimodal_invalid")
        raise LongGenerationError(code) from None



def _bound_path(root: Path, artifact: object) -> Path:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise LongGenerationError("long_video_plan_invalid")
    relative, expected = artifact.get("path"), artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise LongGenerationError("long_video_plan_invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise LongGenerationError("long_video_plan_invalid") from None
    if not path.is_file() or _digest(path) != expected:
        raise LongGenerationError("long_video_plan_invalid")
    return path


def _bound_bytes(root: Path, artifact: object) -> tuple[Path, bytes]:
    """Read one receipt-bound artifact once and retain the verified bytes."""
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise LongGenerationError("long_video_plan_invalid")
    relative, expected = artifact.get("path"), artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise LongGenerationError("long_video_plan_invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
        data = path.read_bytes()
    except (ValueError, OSError):
        raise LongGenerationError("long_video_plan_invalid") from None
    if hashlib.sha256(data).hexdigest() != expected:
        raise LongGenerationError("long_video_plan_invalid")
    return path, data


def _relative_to_work(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root / "work").as_posix()
    except ValueError:
        raise LongGenerationError("long_video_plan_invalid") from None


def _fit_anchor(
    path: Path, data: bytes, output: Path, fit_mode: str, aspect_ratio: str,
    *, prepare: bool,
) -> tuple[Path, bytes]:
    if fit_mode == "none":
        return path, data
    try:
        fitted = frame_fit.fit_frame_bytes(
            data, fit_mode, aspect_ratio, label=path.name
        )
    except frame_fit.FrameFitError:
        raise LongGenerationError("frame_fit_failed") from None
    target = output / (path.stem + ".png")
    if prepare:
        output.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            temporary.write_bytes(fitted)
            temporary.replace(target)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise LongGenerationError("frame_fit_failed") from None
    else:
        try:
            if target.read_bytes() != fitted:
                raise LongGenerationError("frame_fit_failed")
        except OSError:
            raise LongGenerationError("frame_fit_failed") from None
    return target, fitted


def _persisted_fit_layout(meta: Mapping) -> str | None:
    generation = meta.get("generation")
    if not isinstance(generation, Mapping) or "fit_layout" not in generation:
        return None
    layout = generation.get("fit_layout")
    if layout not in _FIT_LAYOUTS:
        raise LongGenerationError("frame_fit_failed")
    return str(layout)


def _fit_outputs_complete(paths: tuple[Path, Path], aspect_ratio: str) -> bool:
    if not all(path.is_file() for path in paths):
        return False
    try:
        return not frame_fit.frames_require_fit(paths, aspect_ratio)
    except frame_fit.FrameFitError:
        return False


def _validate_frozen_dialogue_delivery(
    payload: Mapping,
    meta: Mapping,
    authoritative_lines: Sequence[Mapping[str, object]],
) -> None:
    """Validate frozen delivery facts without replaying today's auto policy."""
    requested = payload.get("dialogue_delivery")
    resolved = payload.get("resolved_dialogue_delivery")
    try:
        parsed = dialogue_delivery_contract.parse(requested)
    except ValueError:
        raise LongGenerationError("long_video_plan_invalid") from None
    if resolved not in {"on_screen", "off_screen"}:
        raise LongGenerationError("long_video_plan_invalid")
    has_meta_requested = "dialogue_delivery" in meta
    has_meta_resolved = "resolved_dialogue_delivery" in meta
    if has_meta_requested or has_meta_resolved:
        if (
            not (has_meta_requested and has_meta_resolved)
            or meta.get("dialogue_delivery") != requested
            or meta.get("resolved_dialogue_delivery") != resolved
        ):
            raise LongGenerationError("long_video_plan_invalid")
    else:
        current = dialogue_delivery_contract.resolve(
            parsed, authoritative_lines
        ).value
        if resolved != current:
            raise LongGenerationError("long_video_plan_invalid")
    if (
        parsed is not dialogue_delivery_contract.DialogueDelivery.AUTO
        and parsed.value != resolved
    ):
        raise LongGenerationError("long_video_plan_invalid")


def freeze_plan(root: Path, meta: Mapping, expected_receipt: str, fit_mode: str,
                dialogue_mode: str, *, aspect_ratio: str | None = None,
                resolution: str | None = None,
                dialogue_delivery: str = "auto",
                prepare_fit: bool = True,
                settings: Settings | None = None) -> FrozenPlan:
    """Validate every immutable plan fact and pre-fit every source anchor."""
    root = Path(root).resolve()
    if (aspect_ratio is None) != (resolution is None):
        raise LongGenerationError("invalid_generation_parameters", 422)
    parameters_explicit = aspect_ratio is not None
    persisted_layout = _persisted_fit_layout(meta)
    detect_existing_layout = (
        persisted_layout is None and not prepare_fit and fit_mode != "none"
    )
    if persisted_layout is not None:
        legacy_layout: bool | None = persisted_layout == FIT_LAYOUT_LEGACY
    elif prepare_fit or fit_mode == "none":
        # Current callers explicitly supply semantic parameters and always use
        # the versioned path.  Calls without them are the historical contract.
        # initial_generation persists this choice before any provider POST.
        legacy_layout = not parameters_explicit
    else:
        # Pre-marker attempts are recovered from their complete, decodable
        # frozen files.  Never let fields added after the POST select a path.
        legacy_layout = None
    aspect_ratio = (
        meta.get("aspect_ratio", h3.H3_DEFAULT_ASPECT_RATIO)
        if aspect_ratio is None else aspect_ratio
    )
    resolution = (
        meta.get("resolution", h3.H3_DEFAULT_RESOLUTION)
        if resolution is None else resolution
    )
    if aspect_ratio not in h3.H3_ASPECT_RATIOS:
        raise LongGenerationError("invalid_aspect_ratio", 422)
    if resolution not in h3.H3_RESOLUTIONS:
        raise LongGenerationError("invalid_resolution", 422)
    if fit_mode not in {"none", "crop", "pad"}:
        raise LongGenerationError("frame_fit_failed")
    name = meta.get("long_video_plan_receipt")
    if name != long_video.PLAN_RECEIPT_FILENAME:
        raise LongGenerationError("long_video_plan_invalid")
    receipt_path = root / name
    try:
        receipt_data = receipt_path.read_bytes()
    except OSError:
        raise LongGenerationError("long_video_plan_invalid") from None
    receipt = hashlib.sha256(receipt_data).hexdigest()
    if expected_receipt != receipt:
        raise LongGenerationError("long_video_plan_changed", 409)
    try:
        payload = json.loads(receipt_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LongGenerationError("long_video_plan_invalid") from None
    receipt_version = payload.get("version") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "duet.long-video-plan"
        or isinstance(receipt_version, bool)
        or not isinstance(receipt_version, int)
        or receipt_version not in {
            long_video.LEGACY_PLAN_RECEIPT_VERSION,
            long_video.PLAN_RECEIPT_VERSION,
            long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
            long_video.VISUAL_PLAN_RECEIPT_VERSION,
            long_video.VISUAL_MULTIMODAL_PLAN_RECEIPT_VERSION,
        }
        or payload.get("workflow") not in _PLAN_WORKFLOWS
    ):
        raise LongGenerationError("long_video_plan_invalid")
    source = _bound_path(root, payload.get("source"))
    receipt_workflow = payload["workflow"]
    is_multimodal_receipt = receipt_version in {
        long_video.MULTIMODAL_PLAN_RECEIPT_VERSION,
        long_video.VISUAL_MULTIMODAL_PLAN_RECEIPT_VERSION,
    }
    has_visual_timeline = receipt_version in {
        long_video.VISUAL_PLAN_RECEIPT_VERSION,
        long_video.VISUAL_MULTIMODAL_PLAN_RECEIPT_VERSION,
    }
    if is_multimodal_receipt and (
        ("dialogue_delivery" in payload)
        != ("resolved_dialogue_delivery" in payload)
    ):
        raise LongGenerationError("long_video_plan_invalid")
    if (
        is_multimodal_receipt
        and receipt_workflow not in h3.H3_REFERENCE_WORKFLOWS
    ) or (
        not is_multimodal_receipt
        and receipt_workflow in h3.H3_MULTIMODAL_WORKFLOWS
    ):
        raise LongGenerationError("long_video_plan_invalid")
    if is_multimodal_receipt and payload.get("dialogue_mode") != dialogue_mode:
        raise LongGenerationError(
            "long_video_multimodal_dialogue_refresh_required", 409
        )
    if is_multimodal_receipt and "dialogue_delivery" in payload:
        if payload.get("dialogue_delivery") != dialogue_delivery:
            raise LongGenerationError(
                "long_video_multimodal_dialogue_refresh_required", 409
            )
    frozen_fusion = None
    fusion_binding = payload.get("prompt_fusion")
    if fusion_binding is not None:
        if (
            not is_multimodal_receipt
            or not isinstance(fusion_binding, Mapping)
            or set(fusion_binding) != {"path", "sha256"}
            or fusion_binding.get("path") != f"work/{h3_project.SOURCE_FILENAME}"
        ):
            raise LongGenerationError("prompt_fusion_manifest_invalid")
        manifest_path = _bound_path(root, fusion_binding)
        frozen_fusion = load_bound_prompt_fusion_manifest(
            root=root,
            meta=meta,
        )
        if manifest_path != root / "work" / h3_project.SOURCE_FILENAME \
                or len(frozen_fusion.segments) != len(payload.get("segments", [])):
            raise LongGenerationError("prompt_fusion_manifest_invalid")
    generation = meta.get("generation")
    if (
        frozen_fusion is not None
        and frozen_fusion.version == PROMPT_FUSION_LEGACY_VERSION
        and (
            not isinstance(generation, Mapping)
            or generation.get("status") != "succeeded"
        )
    ):
        raise LongGenerationError("prompt_fusion_v2_refresh_required")
    persisted_workflow = (
        generation.get("workflow") if isinstance(generation, Mapping) else None
    )
    if persisted_workflow is not None and persisted_workflow not in _PLAN_WORKFLOWS:
        raise LongGenerationError("long_video_plan_invalid")
    # Existing attempts without a marker predate reference-mode long video and
    # must resume their boundary receipts GET-only.  Unsubmitted v2 plans are
    # safe to promote because every provider segment is within the current limit.
    workflow = (
        persisted_workflow
        or (receipt_workflow if isinstance(generation, Mapping) else None)
        or (
            h3.H3_WORKFLOW
            if receipt_version in {
                long_video.PLAN_RECEIPT_VERSION,
                long_video.VISUAL_PLAN_RECEIPT_VERSION,
            }
            else receipt_workflow
        )
    )
    try:
        duration = float(payload["video"]["duration_s"])
        meta_duration = float(meta["duration_s"])
    except (KeyError, TypeError, ValueError):
        raise LongGenerationError("long_video_plan_invalid") from None
    raw_segments, meta_segments = payload.get("segments"), meta.get("segments")
    minimum_plan_duration = (
        long_video.PREVIOUS_SHORT_VIDEO_MAX_S
        if receipt_version == long_video.LEGACY_PLAN_RECEIPT_VERSION
        else 0.0
    )
    if (
        not math.isfinite(duration)
        or duration <= minimum_plan_duration
        or duration > long_video.LONG_VIDEO_MAX_S
        or abs(duration - meta_duration) > _EPS
        or not isinstance(raw_segments, list)
        or not raw_segments
        or not isinstance(meta_segments, list)
        or len(raw_segments) != len(meta_segments)
    ):
        raise LongGenerationError("long_video_plan_invalid")

    project_selected_paths: dict[int, tuple[Path, ...]] = {}
    frozen_fusion_proxy_data: dict[int, tuple[bytes, ...]] = {}
    if workflow in h3.H3_REFERENCE_WORKFLOWS:
        if frozen_fusion is not None:
            for index, fusion_segment in enumerate(frozen_fusion.segments, 1):
                frames = fusion_segment.get("new_keyframes")
                if not isinstance(frames, list) or len(frames) != 9:
                    raise LongGenerationError("prompt_fusion_input_invalid")
                proxies: list[bytes] = []
                for order, frame in enumerate(frames, 1):
                    expected_keys = (
                        {"order", "path", "sha256"}
                        if frozen_fusion.version == PROMPT_FUSION_LEGACY_VERSION
                        else {
                            "order", "path", "sha256", "segment_time_s",
                            "source_scene_id", "transition",
                        }
                    )
                    if (
                        not isinstance(frame, Mapping)
                        or set(frame) != expected_keys
                        or frame.get("order") != order
                    ):
                        raise LongGenerationError("prompt_fusion_input_invalid")
                    try:
                        resolved, data = _bound_bytes(
                            root,
                            {key: frame[key] for key in ("path", "sha256")},
                        )
                    except LongGenerationError:
                        raise LongGenerationError(
                            "prompt_fusion_input_invalid"
                        ) from None
                    expected_proxy = (
                        root / "work" / PROMPT_FUSION_PROXY_DIR
                        / f"{frame['sha256']}.png"
                    )
                    if resolved != expected_proxy:
                        raise LongGenerationError(
                            "prompt_fusion_input_invalid"
                        )
                    proxies.append(data)
                frozen_fusion_proxy_data[index] = tuple(proxies)
        project_originals: list[Path] = []
        project_counts: list[int] = []
        for raw in raw_segments:
            keys = raw.get("keyframes") if isinstance(raw, Mapping) else None
            if not isinstance(keys, list) or not 1 <= len(keys) <= 9:
                raise LongGenerationError("long_video_plan_invalid")
            originals = [_bound_path(root, artifact) for artifact in keys]
            project_originals.extend(originals)
            project_counts.append(len(originals))
        try:
            selected_project = postprocess.generation_keyframes(
                root, dict(meta), project_originals, settings=settings,
            )
        except postprocess.PostprocessError as exc:
            detail = (
                exc.detail
                if isinstance(exc.detail, str)
                else exc.detail["code"]
            )
            raise LongGenerationError(detail, exc.status) from None
        if len(selected_project) != len(project_originals):
            raise LongGenerationError("postprocess_artifacts_invalid")
        cursor = 0
        for index, count in enumerate(project_counts, 1):
            selected = tuple(selected_project[cursor:cursor + count])
            cursor += count
            project_selected_paths[index] = selected
            if frozen_fusion is not None:
                proxies = frozen_fusion_proxy_data.get(index)
                if proxies is None or len(proxies) != len(selected):
                    raise LongGenerationError("prompt_fusion_input_invalid")
                try:
                    rebuilt = tuple(
                        image_optimization.half_resolution_png(path.read_bytes())
                        for path in selected
                    )
                except (OSError, ValueError):
                    raise LongGenerationError(
                        "prompt_fusion_input_invalid"
                    ) from None
                if rebuilt != proxies:
                    raise LongGenerationError("prompt_fusion_input_invalid")

    frozen: list[FrozenSegment] = []
    authoritative_dialogue_for_delivery: list[dict] = []
    # New receipts are written against the current production limit.  Frozen
    # receipts remain readable up to the unchanged H3 capability so an already
    # paid or confirmed project cannot be invalidated by a later planner limit.
    max_provider_duration = long_video.LEGACY_PROVIDER_MAX_DURATION_S
    previous_end = 0.0
    previous_chain = None
    previous_keyframe_source: dict | None = None
    for position, (raw, current) in enumerate(zip(raw_segments, meta_segments), 1):
        if not isinstance(raw, dict) or not isinstance(current, dict):
            raise LongGenerationError("long_video_plan_invalid")
        try:
            index = raw["index"]
            start_s, end_s = float(raw["start_s"]), float(raw["end_s"])
            chain_id, join_mode = raw["chain_id"], raw["join_mode"]
        except (KeyError, TypeError, ValueError):
            raise LongGenerationError("long_video_plan_invalid") from None
        try:
            frozen_duration = long_video.segment_duration_s(
                start_s, end_s, receipt_version=receipt_version
            )
        except long_video.LongVideoError:
            raise LongGenerationError("long_video_plan_invalid") from None
        comparable = ("index", "start_s", "end_s", "chain_id", "join_mode")
        if (
            index != position
            or any(current.get(key) != raw.get(key) for key in comparable)
            or not math.isfinite(start_s)
            or not math.isfinite(end_s)
            or abs(start_s - previous_end) > _EPS
            or frozen_duration < long_video.RECEIPT_COMPAT_SEGMENT_MIN_S
            or long_video.provider_duration_s(
                start_s, end_s, receipt_version=receipt_version
            )
            > max_provider_duration
            or not isinstance(chain_id, str)
            or not chain_id
            or join_mode not in {"hard_cut", "continue"}
            or (position == 1 and join_mode != "hard_cut")
            or (join_mode == "continue" and chain_id != previous_chain)
            or (position > 1 and join_mode == "hard_cut" and chain_id == previous_chain)
        ):
            raise LongGenerationError("long_video_plan_invalid")
        if raw.get("source") is None:
            raise LongGenerationError("long_video_plan_invalid")
        segment_source = _bound_path(root, raw["source"])
        keys = raw.get("keyframes")
        if not isinstance(keys, list) or not 1 <= len(keys) <= 9:
            raise LongGenerationError("long_video_plan_invalid")
        bound_keyframes = [_bound_bytes(root, artifact) for artifact in keys]
        keyframe_paths = [path for path, _data in bound_keyframes]
        frozen_keyframe_sources: tuple[Mapping, ...] = ()
        if has_visual_timeline:
            try:
                source_values, previous_keyframe_source = (
                    long_video.freeze_keyframe_sources(
                        raw.get("keyframe_sources"),
                        expected_count=len(keys),
                        previous=previous_keyframe_source,
                    )
                )
            except long_video.LongVideoError:
                raise LongGenerationError("long_video_plan_invalid") from None
            if current.get("keyframe_sources") != source_values:
                raise LongGenerationError("long_video_plan_invalid")
            frozen_keyframe_sources = tuple(source_values)
        elif raw.get("keyframe_sources") is not None:
            raise LongGenerationError("long_video_plan_invalid")
        frozen_source_cuts: tuple[Mapping, ...] = ()
        if raw.get("source_cut_timeline") is not None:
            try:
                source_cuts = long_video.freeze_source_cut_timeline(
                    raw["source_cut_timeline"],
                    segment_start_s=start_s,
                    segment_end_s=end_s,
                )
            except long_video.LongVideoError:
                raise LongGenerationError("long_video_plan_invalid") from None
            if current.get("source_cut_timeline") != source_cuts:
                raise LongGenerationError("long_video_plan_invalid")
            frozen_source_cuts = tuple(source_cuts)
        anchors = raw.get("anchors")
        if (
            not isinstance(anchors, list)
            or len(anchors) != 2
            or [item.get("role") if isinstance(item, dict) else None for item in anchors]
            != ["first", "end"]
        ):
            raise LongGenerationError("long_video_plan_invalid")
        first_source, first_source_data = _bound_bytes(
            root, {k: v for k, v in anchors[0].items() if k != "role"}
        )
        last_source, last_source_data = _bound_bytes(
            root, {k: v for k, v in anchors[1].items() if k != "role"}
        )
        expected_prefix = f"segments/{index}/"
        if (
            current.get("source") != expected_prefix + "source.mp4"
            or segment_source != root / "work" / current["source"]
            or current.get("keyframe_paths") != [
                _relative_to_work(root, path) for path in keyframe_paths
            ]
            or current.get("first_frame_path")
            != _relative_to_work(root, first_source)
            or current.get("last_frame_path")
            != _relative_to_work(root, last_source)
        ):
            raise LongGenerationError("long_video_plan_invalid")
        visual_path, visual_data = _bound_bytes(root, raw.get("visual_prompt"))
        final_path, final_data = _bound_bytes(root, raw.get("final_prompt"))
        try:
            visual = visual_data.decode("utf-8")
            final = final_data.decode("utf-8")
        except UnicodeDecodeError:
            raise LongGenerationError("long_video_plan_invalid") from None
        source_dialogue = current.get("dialogue")
        if isinstance(source_dialogue, list):
            authoritative_dialogue_for_delivery.extend(
                dict(line) for line in source_dialogue
                if isinstance(line, Mapping)
            )
        dialogue = (
            []
            if is_multimodal_receipt and dialogue_mode == "none"
            else source_dialogue
        )
        dialogue_binding = raw.get("dialogue")
        if (
            not isinstance(source_dialogue, list)
            or not isinstance(dialogue, list)
            or not isinstance(dialogue_binding, dict)
            or set(dialogue_binding) != {"count", "sha256"}
            or dialogue_binding.get("count") != len(dialogue)
            or dialogue_binding.get("sha256") != _canonical_digest(dialogue)
            or current.get("visual_prompt") != visual
            or (
                not is_multimodal_receipt
                and current.get("prompt") != final
            )
        ):
            raise LongGenerationError("long_video_plan_invalid")
        if is_multimodal_receipt:
            prompt = final
        else:
            try:
                rebuilt_visual = long_video.compose_segment_visual_prompt(visual)
                auto_prompt = (
                    f"{_LEGACY_PIPELINE_NO_BGM}\n"
                    + prepared_input.compose_final_prompt(
                        rebuilt_visual, source_dialogue
                    )
                )
            except (prepared_input.PreparedInputError, long_video.LongVideoError):
                raise LongGenerationError("long_video_plan_invalid") from None
            if final != auto_prompt:
                raise LongGenerationError("long_video_plan_invalid")
            if dialogue_mode == "none":
                try:
                    rebuilt = prepared_input.compose_final_prompt(
                        long_video.compose_segment_visual_prompt(visual), ()
                    )
                except (prepared_input.PreparedInputError, long_video.LongVideoError):
                    raise LongGenerationError("long_video_plan_invalid") from None
                prompt = f"{_LEGACY_PIPELINE_NO_BGM}\n{rebuilt}"
            else:
                prompt = final
        segdir = root / "work" / "segments" / str(index)
        fit_base = segdir / "work" / "h3_frames"
        legacy_root = fit_base / fit_mode
        aspect_root = fit_base / aspect_ratio.replace(":", "x") / fit_mode
        if detect_existing_layout:
            legacy_paths = (
                legacy_root / "first" / first_source.name,
                legacy_root / "end" / last_source.name,
            )
            aspect_paths = (
                aspect_root / "first" / first_source.name,
                aspect_root / "end" / last_source.name,
            )
            has_legacy = _fit_outputs_complete(legacy_paths, aspect_ratio)
            has_aspect = _fit_outputs_complete(aspect_paths, aspect_ratio)
            if has_legacy == has_aspect:
                raise LongGenerationError("frame_fit_failed")
            if legacy_layout is None:
                legacy_layout = has_legacy
            elif legacy_layout != has_legacy:
                raise LongGenerationError("frame_fit_failed")
        fit_root = legacy_root if legacy_layout else aspect_root
        # Complete all static transformations before the caller can make a POST.
        first, first_data = _fit_anchor(
            first_source, first_source_data, fit_root / "first", fit_mode,
            aspect_ratio,
            prepare=prepare_fit,
        )
        last, last_data = _fit_anchor(
            last_source, last_source_data, fit_root / "end", fit_mode,
            aspect_ratio,
            prepare=prepare_fit,
        )
        frozen_keyframes: tuple[h3.FrozenFrame, ...] = ()
        if workflow in h3.H3_REFERENCE_WORKFLOWS:
            selected_paths = project_selected_paths[index]
            selected: list[h3.FrozenFrame] = []
            original_data = {path.resolve(): data for path, data in bound_keyframes}
            for selected_path in selected_paths:
                resolved = selected_path.resolve()
                data = original_data.get(resolved)
                if data is None:
                    try:
                        data = resolved.read_bytes()
                    except OSError:
                        raise LongGenerationError("postprocess_artifacts_invalid") from None
                    if not data:
                        raise LongGenerationError("postprocess_artifacts_invalid")
                selected.append(
                    _fit_anchor(
                        resolved,
                        data,
                        fit_root / "keyframes",
                        fit_mode,
                        aspect_ratio,
                        prepare=prepare_fit,
                    )
                )
            frozen_keyframes = tuple(selected)
        frozen_multimodal = None
        if frozen_fusion is not None:
            if raw.get("multimodal") is not None:
                raise LongGenerationError("long_video_plan_invalid")
            fusion_segment = frozen_fusion.segments[position - 1]
            if final != frozen_fusion.final_prompts[position - 1]:
                raise LongGenerationError("prompt_fusion_output_invalid")
            if frozen_fusion.version == PROMPT_FUSION_VERSION:
                fusion_sources = [{
                    key: frame[key]
                    for key in (
                        "order", "segment_time_s", "source_scene_id", "transition",
                    )
                } for frame in fusion_segment["new_keyframes"]]
                localization_input = [{
                    "order": order,
                    "path": _relative_to_work(root, path),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    **dict(frozen_keyframe_sources[order - 1]),
                } for order, (path, data) in enumerate(bound_keyframes, 1)]
                try:
                    localized_sources, _timeline_diagnostics = (
                        long_video.localize_keyframe_sources(
                            localization_input,
                            segment_start_s=start_s,
                            segment_end_s=end_s,
                            provider_duration_s=long_video.provider_duration_s(
                                start_s,
                                end_s,
                                receipt_version=receipt_version,
                            ),
                        )
                    )
                except long_video.LongVideoError:
                    raise LongGenerationError(
                        "prompt_fusion_input_invalid"
                    ) from None
                expected_sources = [{
                    key: frame[key]
                    for key in (
                        "order", "segment_time_s", "source_scene_id", "transition",
                    )
                } for frame in localized_sources]
                if fusion_sources != expected_sources:
                    raise LongGenerationError("prompt_fusion_input_invalid")
            try:
                fusion_lines = json.loads(
                    fusion_segment["audio_content"]["lines_json"]
                )
                voice_references = fusion_segment["audio_content"][
                    "voice_references"
                ]
            except (KeyError, TypeError, json.JSONDecodeError):
                raise LongGenerationError("prompt_fusion_input_invalid") from None
            if not isinstance(fusion_lines, list) or len(fusion_lines) != len(dialogue):
                raise LongGenerationError("prompt_fusion_input_invalid")
            if (
                frozen_fusion.version == PROMPT_FUSION_VERSION
                and voice_references
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            for line_order, (compiled, authoritative) in enumerate(
                zip(fusion_lines, dialogue), 1
            ):
                if (
                    compiled.get("order") != line_order
                    or compiled.get("text") != authoritative.get("text")
                    or compiled.get("start_s") != authoritative.get("start_s")
                    or compiled.get("end_s") != authoritative.get("end_s")
                    or compiled.get("delivery") != payload.get(
                        "resolved_dialogue_delivery"
                    )
                    or (
                        frozen_fusion.version == PROMPT_FUSION_VERSION
                        and compiled.get("voice_ref") is not None
                    )
                ):
                    raise LongGenerationError("prompt_fusion_input_invalid")
        elif is_multimodal_receipt:
            try:
                frozen_multimodal = h3_project.load_bound(
                    root, raw.get("multimodal")
                )
            except h3_project.ProjectMultimodalError as exc:
                raise LongGenerationError(exc.code) from None
            expected_workflow = {
                "multimodal": h3.H3_MULTIMODAL_WORKFLOW,
                "multimodal_hd": h3.H3_MULTIMODAL_HD_WORKFLOW,
            }.get(frozen_multimodal.mode)
            if expected_workflow != workflow:
                raise LongGenerationError("long_video_plan_invalid")
            try:
                # Validate the exact Skill semantics before any caller can
                # prepare or claim a paid H3 attempt.
                h3_project.build_request_from_parts(
                    multimodal=frozen_multimodal,
                    visual_prompt=frozen_multimodal.skill_plan.get(
                        "visual_prompt"
                    ),
                    keyframes=frozen_keyframes,
                    upstream_dialogue=tuple(dict(line) for line in dialogue),
                    upstream_dialogue_receipt_sha256=dialogue_binding["sha256"],
                    source_sha256=_digest(segdir / "source.mp4"),
                    source_duration_s=end_s - start_s,
                    cid=f"freeze-segment-{index}",
                    workdir=segdir / "work" / "h3-native",
                    client_request_id=f"freeze-segment-{index}",
                    duration=long_video.provider_duration_s(
                        start_s, end_s, receipt_version=receipt_version
                    ),
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    autodl_token="freeze-validation-only",
                )
            except (h3.H3Error, h3_project.ProjectMultimodalError):
                raise LongGenerationError("long_video_multimodal_invalid") from None
        elif raw.get("multimodal") is not None:
            raise LongGenerationError("long_video_plan_invalid")
        frozen.append(FrozenSegment(
            index=index,
            start_s=start_s,
            end_s=end_s,
            chain_id=chain_id,
            join_mode=join_mode,
            workdir=segdir,
            first_frame=first,
            first_frame_data=first_data,
            last_frame=last,
            last_frame_data=last_data,
            prompt=prompt,
            keyframes=frozen_keyframes,
            keyframe_sources=frozen_keyframe_sources,
            source_cut_timeline=frozen_source_cuts,
            multimodal=frozen_multimodal,
            dialogue=tuple(dict(line) for line in dialogue),
            dialogue_sha256=dialogue_binding["sha256"],
        ))
        previous_end, previous_chain = end_s, chain_id
    if abs(previous_end - duration) > _EPS:
        raise LongGenerationError("long_video_plan_invalid")
    if is_multimodal_receipt and "dialogue_delivery" in payload:
        _validate_frozen_dialogue_delivery(
            payload, meta, authoritative_dialogue_for_delivery
        )
    assert legacy_layout is not None
    return FrozenPlan(
        root=root,
        source=source,
        receipt=receipt,
        segments=tuple(frozen),
        receipt_version=receipt_version,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        legacy_layout=legacy_layout,
        workflow=workflow,
        prompt_fusion=frozen_fusion,
    )


def child_request_id(parent_id: str, receipt: str, index: int) -> str:
    digest = hashlib.sha256(f"{parent_id}\0{receipt}\0{index}".encode()).hexdigest()
    return f"long-{digest[:59]}"  # 64 bytes, deterministic, provider-safe.


def _extract_last_frame(video: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-sseof", "-1", "-i", str(video),
         "-vf", "reverse", "-frames:v", "1", str(temporary)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise LongGenerationError("long_video_tail_frame_failed")
    temporary.replace(output)
    return output


def _bind_h3_operational_roots(
    settings,
    plan: FrozenPlan,
    request: h3.H3Request,
) -> h3.H3Request:
    if not h3.is_multimodal_request(request):
        return request
    return replace(
        request,
        gateway_storage_root=settings.h3_gateway_storage_root,
        speaker_timing_authority_root=(
            plan.root
            if request.speaker_timing_authority_version in {0, 1}
            else None
        ),
    )


def _request(settings, cid: str, plan: FrozenPlan, segment: FrozenSegment,
             parent_id: str, fit_mode: str, *, frozen_child_id: str | None = None,
             prepare_inputs: bool = True, fast_mode: bool = False,
             context_ir_binding: object = None,
             legacy_terminal_read: bool = False) -> h3.H3Request:
    if plan.prompt_fusion is not None:
        if (
            plan.prompt_fusion.version == PROMPT_FUSION_LEGACY_VERSION
            and not legacy_terminal_read
        ):
            raise LongGenerationError("prompt_fusion_v2_refresh_required")
        if plan.prompt_fusion.version == PROMPT_FUSION_LEGACY_VERSION:
            try:
                frozen_audio = plan.prompt_fusion.segments[
                    segment.index - 1
                ]["audio_content"]["voice_references"]
                reference_audios = h3.freeze_reference_audios(tuple(
                    (plan.root / reference["path"], "voice")
                    for reference in frozen_audio
                ))
                native_audio = bool(reference_audios)
                source_request = h3.H3Request(
                    cid=f"{cid}-segment-{segment.index}",
                    workdir=segment.workdir,
                    client_request_id=(
                        frozen_child_id
                        or child_request_id(parent_id, plan.receipt, segment.index)
                    ),
                    prompt=segment.prompt,
                    keyframes=segment.keyframes,
                    voice_texts=(),
                    voice_receipt=h3.voice_texts_receipt(()),
                    duration=long_video.provider_duration_s(
                        segment.start_s,
                        segment.end_s,
                        receipt_version=plan.receipt_version,
                    ),
                    autodl_token=settings.autodl_art_token,
                    timeouts=h3.Timeouts(
                        request_s=settings.h3_request_timeout_s,
                        h3_poll_s=settings.h3_poll_timeout_s,
                        download_s=settings.h3_download_timeout_s,
                        poll_interval_s=settings.h3_poll_interval_s,
                        retry_count=settings.retry_count,
                        retry_interval_s=settings.retry_interval_s,
                    ),
                    mode="reference",
                    aspect_ratio=plan.aspect_ratio,
                    resolution=plan.resolution,
                    workflow=(
                        h3.H3_MULTIMODAL_WORKFLOW
                        if native_audio else h3.H3_WORKFLOW
                    ),
                    reference_audios=reference_audios,
                    skill_plan_sha256=hashlib.sha256(
                        segment.prompt.encode("utf-8")
                    ).hexdigest(),
                    upstream_dialogue_receipt_sha256=segment.dialogue_sha256,
                    context_ir_required=True,
                    **(
                        {
                            "multimodal_compiler_version": (
                                "video-prompt-fusion-v1"
                            ),
                            "audio_required": True,
                        }
                        if native_audio else {}
                    ),
                )
                if context_ir_binding is None:
                    return source_request
                context = _freeze_segment_context_ir(
                    settings, plan, segment, source_request
                )
                return _bind_h3_operational_roots(
                    settings,
                    plan,
                    h3_project.apply_bound_context_ir(
                        context, context_ir_binding
                    ),
                )
            except (
                KeyError,
                IndexError,
                TypeError,
                h3.H3Error,
                h3_project.ProjectMultimodalError,
            ) as exc:
                raise LongGenerationError(
                    getattr(exc, "code", "prompt_fusion_input_invalid")
                ) from None
        try:
            if (
                len(segment.keyframes) != 9
                or len(segment.keyframe_sources) != 9
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            localization_input = [{
                "order": order,
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                **dict(segment.keyframe_sources[order - 1]),
            } for order, (path, data) in enumerate(segment.keyframes, 1)]
            localized_sources, _timeline_diagnostics = (
                long_video.localize_keyframe_sources(
                    localization_input,
                    segment_start_s=segment.start_s,
                    segment_end_s=segment.end_s,
                    provider_duration_s=long_video.provider_duration_s(
                        segment.start_s,
                        segment.end_s,
                        receipt_version=plan.receipt_version,
                    ),
                )
            )
            frozen_input_sources = _freeze_local_keyframe_sources([{
                key: frame[key]
                for key in (
                    "order", "segment_time_s", "source_scene_id", "transition",
                )
            } for frame in localized_sources])
            frozen_fusion_sources = _freeze_local_keyframe_sources([{
                key: frame[key]
                for key in (
                    "order", "segment_time_s", "source_scene_id", "transition",
                )
            } for frame in plan.prompt_fusion.segments[segment.index - 1][
                "new_keyframes"
            ]])
            if (
                frozen_input_sources != frozen_fusion_sources
                or segment.prompt
                != plan.prompt_fusion.final_prompts[segment.index - 1]
            ):
                raise LongGenerationError("prompt_fusion_input_invalid")
            fusion_prompt_sha256 = hashlib.sha256(
                segment.prompt.encode("utf-8")
            ).hexdigest()
            voice_texts = tuple(
                str(line["text"])
                for line in segment.dialogue
                if isinstance(line, Mapping)
                and isinstance(line.get("text"), str)
                and line["text"]
            )
            if len(voice_texts) != len(segment.dialogue):
                raise LongGenerationError("prompt_fusion_input_invalid")
            source_request = h3.H3Request(
                cid=f"{cid}-segment-{segment.index}",
                workdir=segment.workdir,
                client_request_id=(
                    frozen_child_id
                    or child_request_id(parent_id, plan.receipt, segment.index)
                ),
                prompt=segment.prompt,
                keyframes=segment.keyframes,
                voice_texts=voice_texts,
                voice_receipt=h3.voice_texts_receipt(voice_texts),
                duration=long_video.provider_duration_s(
                    segment.start_s,
                    segment.end_s,
                    receipt_version=plan.receipt_version,
                ),
                autodl_token=settings.autodl_art_token,
                timeouts=h3.Timeouts(
                    request_s=settings.h3_request_timeout_s,
                    h3_poll_s=settings.h3_poll_timeout_s,
                    download_s=settings.h3_download_timeout_s,
                    poll_interval_s=settings.h3_poll_interval_s,
                    retry_count=settings.retry_count,
                    retry_interval_s=settings.retry_interval_s,
                ),
                mode="reference",
                aspect_ratio=plan.aspect_ratio,
                resolution=plan.resolution,
                workflow=h3.H3_WORKFLOW,
                reference_audios=(),
                skill_plan_sha256=fusion_prompt_sha256,
                upstream_dialogue_receipt_sha256=segment.dialogue_sha256,
                context_ir_required=True,
            )
            if context_ir_binding is None:
                return source_request
            context = _freeze_segment_context_ir(
                settings, plan, segment, source_request
            )
            return _bind_h3_operational_roots(
                settings,
                plan,
                h3_project.apply_bound_context_ir(context, context_ir_binding),
            )
        except (
            h3.H3Error,
            h3_project.ProjectMultimodalError,
            long_video.LongVideoError,
        ) as exc:
            raise LongGenerationError(
                getattr(exc, "code", "long_video_multimodal_invalid")
            ) from None
    if plan.workflow in h3.H3_MULTIMODAL_WORKFLOWS:
        if segment.multimodal is None:
            raise LongGenerationError("long_video_multimodal_invalid")
        try:
            source_request = h3_project.build_request_from_parts(
                multimodal=segment.multimodal,
                visual_prompt=segment.multimodal.skill_plan.get("visual_prompt"),
                keyframes=segment.keyframes,
                upstream_dialogue=segment.dialogue,
                upstream_dialogue_receipt_sha256=(
                    segment.dialogue_sha256 or ""
                ),
                source_sha256=_digest(segment.workdir / "source.mp4"),
                source_duration_s=segment.end_s - segment.start_s,
                cid=f"{cid}-segment-{segment.index}",
                workdir=segment.workdir,
                client_request_id=(
                    frozen_child_id
                    or child_request_id(parent_id, plan.receipt, segment.index)
                ),
                duration=long_video.provider_duration_s(
                    segment.start_s,
                    segment.end_s,
                    receipt_version=plan.receipt_version,
                ),
                resolution=plan.resolution,
                aspect_ratio=plan.aspect_ratio,
                autodl_token=settings.autodl_art_token,
                timeouts=h3.Timeouts(
                    request_s=settings.h3_request_timeout_s,
                    h3_poll_s=settings.h3_poll_timeout_s,
                    download_s=settings.h3_download_timeout_s,
                    poll_interval_s=settings.h3_poll_interval_s,
                    retry_count=settings.retry_count,
                    retry_interval_s=settings.retry_interval_s,
                ),
            )
            if context_ir_binding is None:
                return source_request
            context = _freeze_segment_context_ir(
                settings, plan, segment, source_request
            )
            return _bind_h3_operational_roots(
                settings,
                plan,
                h3_project.apply_bound_context_ir(context, context_ir_binding),
            )
        except (h3.H3Error, h3_project.ProjectMultimodalError) as exc:
            raise LongGenerationError(
                getattr(exc, "code", "long_video_multimodal_invalid")
            ) from None
    if plan.workflow in h3.H3_REFERENCE_WORKFLOWS:
        return h3.H3Request(
            cid=f"{cid}-segment-{segment.index}",
            workdir=segment.workdir,
            client_request_id=(
                frozen_child_id
                or child_request_id(parent_id, plan.receipt, segment.index)
            ),
            prompt=segment.prompt,
            keyframes=segment.keyframes,
            voice_texts=(),
            voice_receipt=h3.voice_texts_receipt(()),
            duration=long_video.provider_duration_s(
                segment.start_s,
                segment.end_s,
                receipt_version=plan.receipt_version,
            ),
            autodl_token=settings.autodl_art_token,
            timeouts=h3.Timeouts(
                request_s=settings.h3_request_timeout_s,
                h3_poll_s=settings.h3_poll_timeout_s,
                download_s=settings.h3_download_timeout_s,
                poll_interval_s=settings.h3_poll_interval_s,
                retry_count=settings.retry_count,
                retry_interval_s=settings.retry_interval_s,
            ),
            mode="reference",
            aspect_ratio=plan.aspect_ratio,
            resolution=plan.resolution,
            workflow=plan.workflow,
        )
    first, first_data = segment.first_frame, segment.first_frame_data
    if segment.join_mode == "continue":
        upstream = plan.segments[segment.index - 2]
        if fast_mode:
            # The upstream end anchor is already receipt-bound and fitted by
            # freeze_plan.  Reuse those exact immutable bytes; no generated
            # output is read and no duplicate anchor file is created.
            first, first_data = upstream.last_frame, upstream.last_frame_data
        else:
            tail = upstream.workdir / "work" / "generated_last.png"
            # A paid start belongs to the current parent attempt and must refresh
            # the dependency; resume may only reuse the already-frozen tail.
            if prepare_inputs:
                tail = _extract_last_frame(upstream.workdir / "generated.mp4", tail)
            elif not tail.is_file():
                raise LongGenerationError("long_video_tail_frame_missing")
            try:
                tail_data = tail.read_bytes()
            except OSError:
                raise LongGenerationError("long_video_tail_frame_missing") from None
            continued = segment.workdir / "work" / "h3_frames"
            if not plan.legacy_layout:
                continued = continued / plan.aspect_ratio.replace(":", "x")
            continued = continued / fit_mode / "continued"
            first, first_data = _fit_anchor(
                tail, tail_data, continued, fit_mode, plan.aspect_ratio,
                prepare=prepare_inputs,
            )
    return h3.H3Request(
        cid=f"{cid}-segment-{segment.index}",
        workdir=segment.workdir,
        client_request_id=(
            frozen_child_id
            or child_request_id(parent_id, plan.receipt, segment.index)
        ),
        prompt=segment.prompt,
        keyframes=(),
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
        duration=long_video.provider_duration_s(
            segment.start_s,
            segment.end_s,
            receipt_version=plan.receipt_version,
        ),
        autodl_token=settings.autodl_art_token,
        timeouts=h3.Timeouts(
            request_s=settings.h3_request_timeout_s,
            h3_poll_s=settings.h3_poll_timeout_s,
            download_s=settings.h3_download_timeout_s,
            poll_interval_s=settings.h3_poll_interval_s,
            retry_count=settings.retry_count,
            retry_interval_s=settings.retry_interval_s,
        ),
        mode="boundary",
        first_frame=(first, first_data),
        last_frame=(segment.last_frame, segment.last_frame_data),
        aspect_ratio=plan.aspect_ratio,
        resolution=plan.resolution,
        workflow=h3.H3_BOUNDARY_WORKFLOW,
    )


def _revalidate_speaker_authority(
    plan: FrozenPlan, segment: FrozenSegment, request: h3.H3Request,
) -> None:
    if (
        plan.workflow not in h3.H3_MULTIMODAL_WORKFLOWS
        or plan.prompt_fusion is not None
    ):
        return
    try:
        h3_project.revalidate_production_authority(
            plan.root,
            segment.workdir / "work",
            request,
            expected_production_sha256=(
                hashlib.sha256(
                    segment.multimodal.speaker_timing_production_data
                ).hexdigest()
                if segment.multimodal is not None
                and segment.multimodal.speaker_timing_production_data is not None
                else None
            ),
        )
    except h3_project.ProjectMultimodalError as exc:
        raise LongGenerationError(exc.code) from exc


def _freeze_segment_context_ir(
    settings,
    plan: FrozenPlan,
    segment: FrozenSegment,
    source_request: h3.H3Request,
) -> context_ir_bridge.FrozenContextIrRequest:
    if not getattr(settings, "minimax_api_key", ""):
        raise LongGenerationError("context_ir_credential_missing", 503)
    try:
        return h3_project.freeze_context_ir(
            source_request=source_request,
            upstream_dialogue_sha256=segment.dialogue_sha256 or "",
            upstream_artifact_path=(
                plan.root / long_video.PLAN_RECEIPT_FILENAME
            ),
            upstream_artifact_sha256=plan.receipt,
            upstream_dialogue_sha256_path=(
                "segments", segment.index - 1, "dialogue", "sha256"
            ),
            minimax_api_key=settings.minimax_api_key,
            request_timeout_s=settings.h3_request_timeout_s,
            poll_timeout_s=settings.h3_poll_timeout_s,
            poll_interval_s=settings.h3_poll_interval_s,
        )
    except h3_project.ProjectMultimodalError as exc:
        raise LongGenerationError(exc.code) from exc


def _optimize_segment_context_ir(
    settings,
    plan: FrozenPlan,
    segment: FrozenSegment,
    source_request: h3.H3Request,
) -> tuple[h3.H3Request | None, dict[str, object], str, str | None]:
    context = _freeze_segment_context_ir(settings, plan, segment, source_request)
    try:
        result = context_ir_bridge.optimize_h3_prompt(context)
        binding = h3_project.context_ir_binding(result)
        if result.status == "succeeded":
            return (
                _bind_h3_operational_roots(
                    settings,
                    plan,
                    h3_project.apply_bound_context_ir(context, binding),
                ),
                binding,
                "succeeded",
                None,
            )
    except (context_ir_bridge.ContextIrError,
            h3_project.ProjectMultimodalError) as exc:
        raise LongGenerationError(
            getattr(exc, "code", "context_ir_invalid")
        ) from exc
    if result.status == "submission_unknown":
        return None, binding, "submission_unknown", "submission_unknown"
    if result.status in {"running", "query_unknown"}:
        return (
            None,
            binding,
            "resume_required",
            result.error_code or "context_ir_query_unknown",
        )
    return None, binding, "failed", result.error_code or "context_ir_failed"


def _context_ir_may_progress(
    state: Mapping,
    source_request: h3.H3Request,
    *,
    allow_create: bool = False,
    frozen_context: context_ir_bridge.FrozenContextIrRequest | None = None,
) -> bool:
    binding = state.get("context_ir")
    if binding is None:
        return (
            allow_create
            and state.get("child_request_id") is None
            and state.get("h3_attempt_id") is None
        )
    if not isinstance(binding, Mapping):
        return False
    return h3_project.context_ir_progress_binding_matches(
        source_request, binding, frozen_context=frozen_context
    )


def public_segments(generation: Mapping) -> list[dict]:
    result = []
    for item in generation.get("segments", []):
        if isinstance(item, dict):
            result.append({key: item.get(key) for key in
                           ("index", "chain_id", "join_mode", "status", "attempt", "error")})
    return result


def _result_status(result: h3.H3Result) -> tuple[str, str | None]:
    if result.status == "succeeded":
        return "succeeded", None
    if result.status in {"submission_unknown", "h3_submitting"}:
        return "submission_unknown", "submission_unknown"
    if result.status == "h3_running" or result.error_code in {
        "h3_query_failed", "h3_timeout", "download_failed", "download_dns_failed",
        "download_peer_unverified", "output_write_failed", "output_probe_failed",
    }:
        return "resume_required", result.error_code or result.status
    return "failed", result.error_code or "h3_failed"


def _exact_h3_attempt_id(value: object, fallback: object = None) -> str | None:
    for candidate in (value, fallback):
        if (
            isinstance(candidate, str)
            and len(candidate) == 6
            and candidate.isdigit()
        ):
            return candidate
    return None


def _result_state(
    result: h3.H3Result,
    fallback_attempt_id: object = None,
) -> tuple[str, str | None, str | None]:
    status, error = _result_status(result)
    return status, error, _exact_h3_attempt_id(
        result.attempt_id, fallback_attempt_id
    )


def generation_segments_are_valid(
    expected_segments: object,
    generation: Mapping,
) -> bool:
    """Validate the complete ordered persisted coordinator state."""
    raw = generation.get("segments")
    if "fast_mode" in generation and not isinstance(generation.get("fast_mode"), bool):
        return False
    if not isinstance(expected_segments, (list, tuple)) or not isinstance(raw, list):
        return False
    if len(raw) != len(expected_segments) or not raw:
        return False
    legacy_keys = {
        "index", "chain_id", "join_mode", "status", "attempt", "error",
        "child_request_id",
    }
    native_keys = legacy_keys | {"h3_attempt_id", "context_ir"}
    context_keys = legacy_keys | {"context_ir"}
    audio_route = generation.get("audio_route")
    native_required = audio_route == H3_NATIVE_AUDIO_ROUTE
    if audio_route is None:
        allowed_keys = (legacy_keys, context_keys)
    elif native_required:
        allowed_keys = (native_keys, context_keys)
    else:
        return False
    statuses = {
        "not_started", "queued", "running", "resume_required", "succeeded",
        "failed", "submission_unknown",
    }
    native_items = 0
    for position, (expected, item) in enumerate(zip(expected_segments, raw), 1):
        if (
            not isinstance(item, dict)
            or set(item) not in allowed_keys
        ):
            return False
        if isinstance(expected, FrozenSegment):
            expected_index = expected.index
            expected_chain = expected.chain_id
            expected_join = expected.join_mode
        elif isinstance(expected, Mapping):
            expected_index = expected.get("index")
            expected_chain = expected.get("chain_id")
            expected_join = expected.get("join_mode")
        else:
            return False
        attempt = item.get("attempt")
        child_id = item.get("child_request_id")
        h3_attempt_id = item.get("h3_attempt_id")
        context_ir = item.get("context_ir")
        context_required = "context_ir" in item
        item_native = "h3_attempt_id" in item
        native_items += int(item_native)
        if (
            expected_index != position
            or item.get("index") != expected_index
            or item.get("chain_id") != expected_chain
            or item.get("join_mode") != expected_join
            or item.get("status") not in statuses
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 0
            or (
                item.get("error") is not None
                and not isinstance(item.get("error"), str)
            )
            or (
                child_id is not None
                and (not isinstance(child_id, str) or not child_id)
            )
            or (
                h3_attempt_id is not None
                and (
                    not isinstance(h3_attempt_id, str)
                    or len(h3_attempt_id) != 6
                    or not h3_attempt_id.isdigit()
                )
            )
            or (
                context_required
                and item.get("status") == "succeeded"
                and (
                    (item_native and h3_attempt_id is None)
                    or not isinstance(context_ir, Mapping)
                    or context_ir.get("status") != "succeeded"
                )
            )
            or (
                context_required
                and context_ir is not None
                and not isinstance(context_ir, Mapping)
            )
        ):
            return False
    return bool(native_items) == native_required


def has_context_ir_semantic_failure_intent(generation: object) -> bool:
    """Classify the persisted coordinator state without reading artifacts."""
    if (
        not isinstance(generation, Mapping)
        or generation.get("status") != "failed"
        or generation.get("error") != "long_video_segment_failed"
        or generation.get("stage") != "h3"
        or not isinstance(generation.get("segments"), list)
    ):
        return False
    failed = [
        item for item in generation["segments"]
        if isinstance(item, Mapping) and item.get("status") == "failed"
    ]
    return (
        len(failed) == 1
        and failed[0].get("error") == "context_ir_semantic_mismatch"
        and all(
            isinstance(item, Mapping)
            and item.get("status") in {"succeeded", "failed"}
            for item in generation["segments"]
        )
    )


def context_ir_semantic_failure_is_recoverable(
    settings,
    cid: str,
    plan: FrozenPlan,
    generation: Mapping,
    fit_mode: str,
) -> bool:
    """Prove one current Fusion Context failure can resume its exact GET."""
    if (
        plan.prompt_fusion is None
        or fit_mode not in {"none", "crop", "pad"}
        or generation.get("status") != "failed"
        or generation.get("error") != "long_video_segment_failed"
        or generation.get("stage") != "h3"
        or generation.get("fast_mode", False) is not False
        or generation.get("workflow") != plan.workflow
        or not isinstance(generation.get("client_request_id"), str)
        or not generation_segments_are_valid(plan.segments, generation)
    ):
        return False
    try:
        _generation_uses_h3_native_audio(plan, generation)
    except LongGenerationError:
        return False
    states = generation.get("segments")
    failed = [
        item for item in states
        if isinstance(item, Mapping) and item.get("status") == "failed"
    ]
    if (
        len(failed) != 1
        or any(
            item.get("status") not in {"succeeded", "failed"}
            for item in states
        )
    ):
        return False
    state = failed[0]
    index = state.get("index")
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 1
        or index > len(plan.segments)
        or state.get("error") != "context_ir_semantic_mismatch"
        or state.get("child_request_id") is not None
        or state.get("h3_attempt_id") is not None
        or not isinstance(state.get("attempt"), int)
        or isinstance(state.get("attempt"), bool)
        or state.get("attempt") <= 0
    ):
        return False
    binding = state.get("context_ir")
    if (
        not isinstance(binding, Mapping)
        or binding.get("status") != "failed"
        or not isinstance(binding.get("provider_task_id"), str)
        or not binding["provider_task_id"]
        or binding.get("effective_prompt_sha256") is not None
        or binding.get("receipt_path") is not None
        or binding.get("receipt_sha256") is not None
    ):
        return False
    segment = plan.segments[index - 1]
    if (segment.workdir / ".h3").exists():
        return False
    attempt_id = binding.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 6
        or not attempt_id.isdigit()
    ):
        return False
    attempt_dir = segment.workdir / ".context-ir" / "attempts" / attempt_id
    if (attempt_dir / "receipt.json").exists():
        return False
    try:
        attempt_state = json.loads(
            (attempt_dir / "attempt.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(attempt_state, Mapping)
            or attempt_state.get("status") != "failed"
            or attempt_state.get("error") != "context_ir_semantic_mismatch"
            or attempt_state.get("provider_task_id")
            != binding.get("provider_task_id")
            or attempt_state.get("receipt") is not None
        ):
            return False
        source_request = _request(
            settings,
            cid,
            plan,
            segment,
            generation["client_request_id"],
            fit_mode,
            prepare_inputs=False,
            fast_mode=False,
        )
        frozen_context = _freeze_segment_context_ir(
            settings, plan, segment, source_request
        )
        return h3_project.context_ir_progress_binding_matches(
            source_request,
            binding,
            frozen_context=frozen_context,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, LongGenerationError):
        return False


def bound_reusable_segment_indices(
    settings,
    cid: str,
    plan: FrozenPlan,
    generation: Mapping,
) -> frozenset[int]:
    """Single source of truth for paid-count and execution reuse decisions."""
    segments = generation.get("segments")
    expected = tuple(item.index for item in plan.segments)
    if not generation_segments_are_valid(plan.segments, generation):
        return frozenset()
    meta = storage.load_meta(settings.data_dir, cid)
    fit_mode = meta.get("fit_mode") if isinstance(meta, dict) else None
    if fit_mode not in {"none", "crop", "pad"}:
        return frozenset()
    fast_mode = generation.get("fast_mode", False)
    by_index = {
        item.get("index"): item for item in segments or [] if isinstance(item, dict)
    }
    reusable: set[int] = set()

    def valid(index: int) -> bool:
        item = by_index.get(index)
        segment = plan.segments[index - 1]
        if not isinstance(item, dict):
            return False
        attempt = item.get("attempt")
        child_id = item.get("child_request_id")
        status_can_revalidate = item.get("status") == "succeeded" or (
            item.get("status") == "failed"
            and item.get("error") == "long_video_segment_output_invalid"
        )
        if (
            not status_can_revalidate
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt <= 0
            or not isinstance(child_id, str)
            or not child_id
            or (
                not fast_mode
                and
                segment.join_mode == "continue"
                and segment.index - 1 not in reusable
            )
        ):
            return False
        try:
            request = _request(
                settings, cid, plan, segment,
                str(generation.get("client_request_id") or "frozen-parent"),
                fit_mode,
                frozen_child_id=child_id,
                prepare_inputs=False,
                fast_mode=fast_mode,
                context_ir_binding=item.get("context_ir"),
            )
            if _segment_uses_h3_native_audio(plan, segment):
                inspected = h3.inspect(request)
                if (
                    inspected.status != "succeeded"
                    or inspected.attempt_id != item.get("h3_attempt_id")
                ):
                    return False
            return h3.output_is_reusable(
                request,
                expected_duration_s=_segment_duration_s(plan, segment),
                allow_provider_duration_ceiling=True,
            )
        except (OSError, h3.H3Error, LongGenerationError, ValueError):
            return False

    for index in expected:
        if valid(index):
            reusable.add(index)
    return frozenset(reusable)


def bound_h3_native_media(
    settings,
    cid: str,
    plan: FrozenPlan,
    generation: Mapping,
    *,
    legacy_terminal_read: bool = False,
) -> dict[int, tuple[str, Mapping[str, object]]]:
    """Bind each native-audio segment to its exact successful H3 attempt."""
    if not _generation_uses_h3_native_audio(plan, generation):
        raise LongGenerationError("long_video_audio_route_invalid")
    if not generation_segments_are_valid(plan.segments, generation):
        raise LongGenerationError("long_video_h3_native_audio_invalid")
    meta = storage.load_meta(settings.data_dir, cid)
    fit_mode = meta.get("fit_mode") if isinstance(meta, Mapping) else None
    parent_id = generation.get("client_request_id")
    fast_mode = generation.get("fast_mode", False)
    if fit_mode not in {"none", "crop", "pad"} or not isinstance(parent_id, str):
        raise LongGenerationError("long_video_h3_native_audio_invalid")
    states = {
        item.get("index"): item
        for item in generation.get("segments", [])
        if isinstance(item, Mapping)
    }
    result: dict[int, tuple[str, Mapping[str, object]]] = {}
    for segment in plan.segments:
        state = states.get(segment.index)
        if not isinstance(state, Mapping) or state.get("status") != "succeeded":
            raise LongGenerationError("long_video_h3_native_audio_invalid")
        if not _segment_uses_h3_native_audio(plan, segment):
            if state.get("h3_attempt_id") is not None:
                raise LongGenerationError("long_video_h3_native_audio_invalid")
            continue
        attempt_id = state.get("h3_attempt_id")
        child_id = state.get("child_request_id")
        if (
            _exact_h3_attempt_id(attempt_id) is None
            or not isinstance(child_id, str)
            or not child_id
        ):
            raise LongGenerationError("long_video_h3_native_audio_invalid")
        try:
            request = _request(
                settings,
                cid,
                plan,
                segment,
                parent_id,
                fit_mode,
                frozen_child_id=child_id,
                prepare_inputs=False,
                fast_mode=fast_mode,
                context_ir_binding=state.get("context_ir"),
                legacy_terminal_read=legacy_terminal_read,
            )
            timeline = h3.load_media_timeline_receipt(request, attempt_id)
        except (OSError, TypeError, ValueError, h3.H3Error, LongGenerationError):
            raise LongGenerationError("long_video_h3_native_audio_invalid") from None
        result[segment.index] = (attempt_id, timeline)
    return result


def initial_generation(settings, cid: str, plan: FrozenPlan, parent_id: str, attempt: int,
                       old: Mapping | None = None, *, fast_mode: bool = False) -> dict:
    if not isinstance(fast_mode, bool):
        raise LongGenerationError("invalid_fast_mode", 422)
    raw_old_segments = (old or {}).get("segments", [])
    old_segments = raw_old_segments if isinstance(raw_old_segments, list) else []
    reusable = bound_reusable_segment_indices(
        settings, cid, plan, old or {"segments": []}
    )
    old_by_index = {
        item.get("index"): item for item in old_segments
        if isinstance(item, dict)
    }
    items = []
    native_audio_indices = _native_audio_segment_indices(plan)
    context_ir_required = _requires_context_ir(plan)
    for segment in plan.segments:
        prior = old_by_index.get(segment.index, {})
        succeeded = segment.index in reusable
        item = {
            "index": segment.index,
            "chain_id": segment.chain_id,
            "join_mode": segment.join_mode,
            "status": "succeeded" if succeeded else "not_started",
            "attempt": prior.get("attempt", 0) if succeeded else int(prior.get("attempt", 0) or 0),
            "error": None,
            "child_request_id": prior.get("child_request_id") if succeeded else None,
        }
        if segment.index in native_audio_indices:
            item["h3_attempt_id"] = (
                prior.get("h3_attempt_id") if succeeded else None
            )
        if context_ir_required:
            item["context_ir"] = (
                prior.get("context_ir") if succeeded else None
            )
        items.append(item)
    generation = {
        "status": "queued",
        "error": None,
        "attempt": attempt,
        "client_request_id": parent_id,
        "stage": "h3",
        "fit_layout": (
            FIT_LAYOUT_LEGACY if plan.legacy_layout else FIT_LAYOUT_ASPECT
        ),
        "fast_mode": fast_mode,
        "workflow": plan.workflow,
        "segments": items,
    }
    if native_audio_indices:
        generation["audio_route"] = dict(H3_NATIVE_AUDIO_ROUTE)
    return generation


def _stitch(
    settings,
    cid: str,
    plan: FrozenPlan,
    dialogue_mode: str,
    *,
    generation: Mapping | None = None,
) -> None:
    provider_media = None
    if _native_audio_segment_indices(plan):
        if generation is None:
            raise LongGenerationError("long_video_h3_native_audio_invalid")
        provider_media = bound_h3_native_media(settings, cid, plan, generation)
        audio_mode: stitch.AudioMode = "provider_generated"
    else:
        if generation is not None:
            _generation_uses_h3_native_audio(plan, generation)
        if dialogue_mode not in {"auto", "none"}:
            raise LongGenerationError("invalid_dialogue_mode", 422)
        audio_mode = "mute"
    stitch.stitch_video(
        segments=_stitch_segments(plan, provider_media),
        source_video=plan.source,
        output=plan.root / "generated.mp4",
        audio_mode=audio_mode,
    )
    reusable = (
        stitched_output_is_reusable(
            plan,
            dialogue_mode,
            generation=generation,
            provider_media=provider_media,
        )
        if provider_media is not None
        else stitched_output_is_reusable(plan, dialogue_mode)
    )
    if not reusable:
        raise LongGenerationError("long_video_stitch_output_invalid")


def stitched_output_is_reusable(
    plan: FrozenPlan,
    dialogue_mode: str,
    *,
    generation: Mapping | None = None,
    provider_media: Mapping[int, tuple[str, Mapping[str, object]]] | None = None,
) -> bool:
    """Validate legacy output or native output with exact attempt evidence."""
    if dialogue_mode not in {"auto", "edit", "custom", "none"}:
        return False
    output = plan.root / "generated.mp4"
    receipt_path = plan.root / stitch.RECEIPT_FILENAME
    native_audio = bool(_native_audio_segment_indices(plan))
    if native_audio:
        native_indices = _native_audio_segment_indices(plan)
        if (
            not isinstance(generation, Mapping)
            or not generation
            or not isinstance(provider_media, Mapping)
            or not provider_media
            or not generation_segments_are_valid(plan.segments, generation)
        ):
            return False
        try:
            if not _generation_uses_h3_native_audio(plan, generation):
                return False
        except LongGenerationError:
            return False
        states = {
            item.get("index"): item
            for item in generation.get("segments", [])
            if isinstance(item, Mapping)
        }
        if set(provider_media) != set(native_indices):
            return False
        for segment in plan.segments:
            state = states.get(segment.index)
            media = provider_media.get(segment.index)
            if not isinstance(state, Mapping) or state.get("status") != "succeeded":
                return False
            if segment.index not in native_indices:
                if media is not None or state.get("h3_attempt_id") is not None:
                    return False
                continue
            if (
                not isinstance(media, tuple)
                or len(media) != 2
                or media[0] != state.get("h3_attempt_id")
                or not isinstance(media[1], Mapping)
            ):
                return False
    else:
        if provider_media is not None:
            return False
        if generation is not None:
            try:
                if _generation_uses_h3_native_audio(plan, generation):
                    return False
            except LongGenerationError:
                return False
    audio_mode: stitch.AudioMode = (
        "provider_generated" if native_audio else "mute"
    )
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema", "version", "segments", "audio", "output"}
            or payload.get("schema") != "duet.stitch"
            or payload.get("version") != (2 if native_audio else 1)
        ):
            return False
        stitch_segments = _stitch_segments(plan, provider_media)
        budgets = stitch._frame_budgets(stitch_segments)
        expected_segments = [
            {
                "index": index,
                "path": str((item.workdir / "generated.mp4").resolve()),
                "sha256": stitch._sha256(item.workdir / "generated.mp4"),
                "target_duration_s": stitch_segments[index - 1].target_duration_s,
                "output_frames": budgets[index - 1],
                "join_mode": item.join_mode,
            }
            for index, item in enumerate(plan.segments, 1)
        ]
        if payload.get("segments") != expected_segments:
            return False
        source_info = stitch._probe(plan.source)
        expected_audio = {
            "mode": audio_mode,
            "source": str(plan.source.resolve()),
            "source_sha256": stitch._sha256(plan.source),
            "source_has_audio": source_info.has_audio,
        }
        if native_audio:
            actual_audio = payload.get("audio")
            if not isinstance(actual_audio, dict):
                return False
            expected_audio["provider_segments"] = [
                stitch._segment_audio_binding(segment, index)
                for index, segment in enumerate(stitch_segments, 1)
            ]
            expected_audio["edl"] = {
                "schema": "duet.av-edl",
                "version": 1,
                "fps": stitch.FPS,
                "interval": "integer-half-open",
            }
        if payload.get("audio") != expected_audio:
            return False
        output_receipt = payload.get("output")
        stat = output.stat()
        if (
            not output.is_file()
            or stat.st_size <= 0
            or not isinstance(output_receipt, dict)
            or set(output_receipt)
            != {"name", "sha256", "size", "duration_s", "fps"}
            or output_receipt.get("name") != "generated.mp4"
            or output_receipt.get("size") != stat.st_size
            or output_receipt.get("sha256") != stitch._sha256(output)
            or output_receipt.get("fps") != stitch.FPS
        ):
            return False
        expected_duration = sum(
            item.target_duration_s for item in stitch_segments
        )
        output_info = stitch._validate_output(
            output, expected_duration, audio_mode, source_info.has_audio
        )
        receipt_duration = float(output_receipt.get("duration_s"))
        return (
            math.isfinite(receipt_duration)
            and abs(receipt_duration - output_info.duration_s) <= _EPS
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        LongGenerationError,
        stitch.StitchError,
    ):
        return False


def _run_unchecked(
    settings, cid: str, plan: FrozenPlan, *, startup: bool = False
) -> None:
    """Drive frozen serial chains or fast fan-out; only this coordinator writes meta."""
    meta = storage.load_meta(settings.data_dir, cid)
    if not meta or not isinstance(meta.get("generation"), dict):
        return
    generation = meta["generation"]
    if not generation_segments_are_valid(plan.segments, generation):
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={
                **generation,
                "status": "submission_unknown",
                "error": "submission_unknown",
            },
        )
        return
    try:
        _generation_uses_h3_native_audio(plan, generation)
    except LongGenerationError:
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={
                **generation,
                "status": "submission_unknown",
                "error": "submission_unknown",
            },
        )
        return
    context_ir_required = _requires_context_ir(plan)
    parent_id = generation.get("client_request_id")
    fast_mode = generation.get("fast_mode", False)
    fit_mode = meta.get("fit_mode")
    dialogue_mode = meta.get("dialogue_mode")
    states = {item["index"]: dict(item) for item in generation.get("segments", [])}
    native_audio_indices = _native_audio_segment_indices(plan)

    def record_error(
        phase: str,
        exc: BaseException,
        *,
        segment_index: int | None = None,
        action: str | None = None,
    ) -> None:
        parts = ["pipeline", "long_generation", phase]
        filename_parts = ["long-generation", phase]
        if segment_index is not None:
            parts.append(f"segment:{segment_index}")
            filename_parts.append(f"segment-{segment_index}")
        if action is not None:
            parts.append(f"action:{action}")
            filename_parts.append(action)
        error_trace.record(
            settings.data_dir
            / cid
            / "work"
            / "errors"
            / ("-".join(filename_parts) + ".json"),
            call_path=parts,
            error=exc,
            logger=_LOGGER,
            secrets=(settings.autodl_art_token, settings.minimax_api_key),
        )
    if any(
        ("context_ir" in state) != context_ir_required
        for state in states.values()
    ) or any(
        ("h3_attempt_id" in state) != (index in native_audio_indices)
        for index, state in states.items()
    ):
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={
                **generation,
                "status": "submission_unknown",
                "error": "submission_unknown",
            },
        )
        return
    if (
        not isinstance(parent_id, str)
        or fit_mode not in {"none", "crop", "pad"}
        or meta.get("aspect_ratio", h3.H3_DEFAULT_ASPECT_RATIO)
        != plan.aspect_ratio
        or meta.get("resolution", h3.H3_DEFAULT_RESOLUTION)
        != plan.resolution
    ):
        reason = {
            "code": "long_generation_parameter_mismatch",
            "parent_id_valid": isinstance(parent_id, str) and bool(parent_id),
            "fit_mode": {"expected": "none|crop|pad", "actual": fit_mode},
            "aspect_ratio": {
                "expected": plan.aspect_ratio,
                "actual": meta.get("aspect_ratio", h3.H3_DEFAULT_ASPECT_RATIO),
            },
            "resolution": {
                "expected": plan.resolution,
                "actual": meta.get("resolution", h3.H3_DEFAULT_RESOLUTION),
            },
        }
        error_trace.record(
            settings.data_dir / cid / "work" / "errors"
            / "long-generation-parameter-mismatch.json",
            call_path=["pipeline", "long_generation", "parameters"],
            reason=reason,
            logger=_LOGGER,
        )
        closed_segments = [
            {
                **item,
                "status": (
                    item.get("status")
                    if item.get("status") in {"succeeded", "failed"}
                    else "submission_unknown"
                ),
                "error": (
                    item.get("error")
                    if item.get("status") in {"succeeded", "failed"}
                    else "submission_unknown"
                ),
            }
            for item in generation.get("segments", [])
            if isinstance(item, dict)
        ]
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={
                **generation,
                "segments": closed_segments,
                "status": "submission_unknown",
                "error": "submission_unknown",
            },
        )
        return

    def persist(status: str | None = None, error: str | None = None, stage: str = "h3") -> None:
        nonlocal generation
        ordered = [states[item.index] for item in plan.segments]
        generation = {**generation, "segments": ordered,
                      "status": status or generation.get("status"), "error": error, "stage": stage}
        storage.update_meta(settings.data_dir, cid, generation=generation)

    def exact_generation() -> dict:
        return {
            **generation,
            "segments": [states[item.index] for item in plan.segments],
        }

    # New work must never turn a short source interval into an invented
    # provider-length interval.  Historical running/succeeded attempts remain
    # readable and GET-resumable, but no Context IR or H3 task is created for
    # an unpaid segment below the current provider minimum.
    short_unsubmitted = [
        segment
        for segment in plan.segments
        if _segment_duration_s(plan, segment) < long_video.SEGMENT_SOURCE_MIN_S
        and (
            states[segment.index].get("status") in {"not_started", "queued"}
            or (
                states[segment.index].get("status") == "failed"
                and states[segment.index].get("error") == "h3_provider_failed"
            )
            or (
                states[segment.index].get("status") == "resume_required"
                and states[segment.index].get("child_request_id") is None
                and states[segment.index].get("h3_attempt_id") is None
            )
        )
    ]
    if short_unsubmitted:
        code = "long_video_segment_below_provider_minimum"
        for segment in short_unsubmitted:
            states[segment.index].update(status="failed", error=code)
            record_error(
                "provider-duration-preflight",
                LongGenerationError(code),
                segment_index=segment.index,
            )
        persist("failed", code)
        return

    def parallel_update(segments, operation) -> None:
        with ThreadPoolExecutor(
            max_workers=min(_FAST_MODE_WORKERS, len(segments))
        ) as pool:
            futures = {pool.submit(operation, segment): segment for segment in segments}
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    segment = futures.pop(future)
                    status, error, attempt_id = future.result()
                    changes = {"status": status, "error": error}
                    if segment.index in native_audio_indices:
                        changes["h3_attempt_id"] = attempt_id
                    states[segment.index].update(changes)
                    persist("running", None)

    reusable = bound_reusable_segment_indices(settings, cid, plan, generation)
    for index, state in states.items():
        if (
            index in reusable
            and state.get("status") == "failed"
            and state.get("error") == "long_video_segment_output_invalid"
        ):
            state.update(status="succeeded", error=None)
        elif state.get("status") == "succeeded" and index not in reusable:
            state.update(
                status="failed",
                error="long_video_segment_output_invalid",
            )

    if startup:
        recoverable = []
        for segment in plan.segments:
            state = states[segment.index]
            if state.get("status") not in {"queued", "running", "resume_required"}:
                continue
            recoverable.append(segment)

        def recover(segment: FrozenSegment):
            state = states[segment.index]
            try:
                context_binding = state.get("context_ir")
                request = _request(
                    settings, cid, plan, segment, parent_id, fit_mode,
                    prepare_inputs=False,
                    fast_mode=fast_mode,
                    frozen_child_id=state.get("child_request_id"),
                    context_ir_binding=(
                        context_binding
                        if isinstance(context_binding, Mapping)
                        and context_binding.get("status") == "succeeded"
                        else None
                    ),
                )
                if context_ir_required and not (
                    isinstance(context_binding, Mapping)
                    and context_binding.get("status") == "succeeded"
                ):
                    if not isinstance(context_binding, Mapping):
                        if (
                            state.get("child_request_id") is not None
                            or state.get("h3_attempt_id") is not None
                        ):
                            return (
                                "submission_unknown",
                                "submission_unknown",
                                _exact_h3_attempt_id(state.get("h3_attempt_id")),
                            )
                        return (
                            "resume_required",
                            "context_ir_resume_required",
                            _exact_h3_attempt_id(state.get("h3_attempt_id")),
                        )
                    frozen_context = (
                        _freeze_segment_context_ir(
                            settings, plan, segment, request
                        )
                        if context_binding.get("status") == "failed"
                        else None
                    )
                    if not _context_ir_may_progress(
                        state, request, frozen_context=frozen_context
                    ):
                        return (
                            "submission_unknown",
                            "submission_unknown",
                            _exact_h3_attempt_id(state.get("h3_attempt_id")),
                        )
                    final_request, updated_binding, status, error = (
                        _optimize_segment_context_ir(
                            settings, plan, segment, request
                        )
                    )
                    state["context_ir"] = updated_binding
                    if status == "succeeded" and final_request is not None:
                        return (
                            "resume_required",
                            "context_ir_ready",
                            _exact_h3_attempt_id(state.get("h3_attempt_id")),
                        )
                    return (
                        status,
                        error,
                        _exact_h3_attempt_id(state.get("h3_attempt_id")),
                    )
                _revalidate_speaker_authority(plan, segment, request)
                result = h3.resume(request)
                if result.status == "not_started":
                    return (
                        (
                            "queued",
                            None,
                            _exact_h3_attempt_id(
                                result.attempt_id, state.get("h3_attempt_id")
                            ),
                        )
                        if (
                            fast_mode
                            and state.get("status") == "queued"
                            and isinstance(result.attempt_id, str)
                            and bool(result.attempt_id)
                        )
                        else (
                            "submission_unknown",
                            "submission_unknown",
                            _exact_h3_attempt_id(
                                result.attempt_id, state.get("h3_attempt_id")
                            ),
                        )
                    )
                status = _result_state(result, state.get("h3_attempt_id"))
                if (
                    fast_mode
                    and status[0] == "succeeded"
                    and not h3.output_is_reusable(
                        request,
                        expected_duration_s=_segment_duration_s(plan, segment),
                        allow_provider_duration_ceiling=True,
                    )
                ):
                    return (
                        "failed",
                        "long_video_segment_output_invalid",
                        status[2],
                    )
                return status
            except Exception as exc:
                record_error("startup-recover", exc, segment_index=segment.index)
                return (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(state.get("h3_attempt_id")),
                )

        if recoverable and fast_mode:
            parallel_update(recoverable, recover)
        elif recoverable:
            for segment in recoverable:
                status, error, attempt_id = recover(segment)
                changes = {"status": status, "error": error}
                if segment.index in native_audio_indices:
                    changes["h3_attempt_id"] = attempt_id
                states[segment.index].update(changes)
        if any(item.get("status") == "submission_unknown" for item in states.values()):
            persist("submission_unknown", "submission_unknown")
            return
        elif all(item.get("status") == "succeeded" for item in states.values()):
            try:
                _stitch(
                    settings,
                    cid,
                    plan,
                    dialogue_mode,
                    generation=exact_generation(),
                )
            except LongGenerationError as exc:
                record_error("startup-stitch", exc)
                persist("failed", exc.code, "stitch")
            except Exception as exc:
                record_error("startup-stitch", exc)
                persist("failed", "long_video_stitch_failed", "stitch")
            else:
                persist("succeeded", None, "stitch")
            return
        elif any(item.get("status") == "failed" for item in states.values()):
            persist("failed", "long_video_segment_failed")
            return
        # General startup remains GET-only. A prepared child still awaits
        # explicit confirmation, while provider failures remain terminal.
        persist("resume_required", "long_video_resume_required")
        return

    if fast_mode:
        # Phase 1: construct every immutable request before creating any paid
        # attempt. A local validation failure therefore guarantees zero POSTs.
        requests: dict[int, h3.H3Request] = {}
        try:
            for segment in plan.segments:
                state = states[segment.index]
                if state.get("status") == "succeeded":
                    continue
                if state.get("status") not in {
                    "not_started", "queued", "running", "resume_required",
                }:
                    continue
                child_id = state.get("child_request_id")
                if not isinstance(child_id, str) or not child_id:
                    child_id = child_request_id(parent_id, plan.receipt, segment.index)
                requests[segment.index] = _request(
                    settings, cid, plan, segment, parent_id, fit_mode,
                    frozen_child_id=child_id,
                    prepare_inputs=False,
                    fast_mode=True,
                )

            if context_ir_required:
                context_targets = [
                    segment for segment in plan.segments
                    if segment.index in requests
                ]

                def prepare_context(segment: FrozenSegment):
                    state = states[segment.index]
                    binding = state.get("context_ir")
                    source_request = requests[segment.index]
                    if (
                        isinstance(binding, Mapping)
                        and binding.get("status") == "succeeded"
                    ):
                        context = _freeze_segment_context_ir(
                            settings, plan, segment, source_request
                        )
                        return (
                            _bind_h3_operational_roots(
                                settings,
                                plan,
                                h3_project.apply_bound_context_ir(context, binding),
                            ),
                            binding,
                            "succeeded",
                            None,
                        )
                    frozen_context = (
                        _freeze_segment_context_ir(
                            settings, plan, segment, source_request
                        )
                        if isinstance(binding, Mapping)
                        and binding.get("status") == "failed"
                        else None
                    )
                    if not _context_ir_may_progress(
                        state,
                        source_request,
                        allow_create=state.get("status") == "not_started",
                        frozen_context=frozen_context,
                    ):
                        return (
                            None,
                            binding,
                            "submission_unknown",
                            "submission_unknown",
                        )
                    return _optimize_segment_context_ir(
                        settings, plan, segment, source_request
                    )

                with ThreadPoolExecutor(
                    max_workers=min(_FAST_MODE_WORKERS, len(context_targets))
                ) as pool:
                    futures = {
                        pool.submit(prepare_context, segment): segment
                        for segment in context_targets
                    }
                    for future in futures:
                        segment = futures[future]
                        final_request, binding, status, error = future.result()
                        states[segment.index]["context_ir"] = binding
                        if status == "succeeded" and final_request is not None:
                            requests[segment.index] = final_request
                        else:
                            states[segment.index].update(
                                status=status, error=error
                            )
                persist("running", None, "context_ir_native")
                context_statuses = {
                    states[segment.index].get("status")
                    for segment in context_targets
                    if states[segment.index].get("context_ir", {}).get("status")
                    != "succeeded"
                }
                if "submission_unknown" in context_statuses:
                    persist("submission_unknown", "submission_unknown", "context_ir_native")
                    return
                if "resume_required" in context_statuses:
                    persist("resume_required", "context_ir_resume_required", "context_ir_native")
                    return
                if context_statuses:
                    persist("failed", "long_video_context_ir_failed", "context_ir_native")
                    return

            # Phase 2: persist every unpaid child receipt before the first POST.
            for segment in plan.segments:
                state = states[segment.index]
                if state.get("status") != "not_started":
                    continue
                _revalidate_speaker_authority(
                    plan, segment, requests[segment.index]
                )
                result = h3.prepare(requests[segment.index])
                if result.status == "not_started":
                    prepared_status, prepared_error = "queued", None
                elif result.status == "h3_running":
                    prepared_status, prepared_error = "running", None
                elif result.status == "succeeded":
                    if not h3.output_is_reusable(
                        requests[segment.index],
                        expected_duration_s=_segment_duration_s(plan, segment),
                        allow_provider_duration_ceiling=True,
                    ):
                        prepared_status, prepared_error = (
                            "failed", "long_video_segment_output_invalid"
                        )
                    else:
                        prepared_status, prepared_error = "succeeded", None
                else:
                    prepared_status, prepared_error = _result_status(result)
                changes = dict(
                    status=prepared_status,
                    attempt=int(state.get("attempt", 0) or 0) + 1,
                    error=prepared_error,
                    child_request_id=requests[segment.index].client_request_id,
                )
                if segment.index in native_audio_indices:
                    changes["h3_attempt_id"] = _exact_h3_attempt_id(
                        result.attempt_id, state.get("h3_attempt_id")
                    )
                state.update(changes)
            persist("running", None)
        except (h3.H3Error, LongGenerationError, OSError, ValueError) as exc:
            code = exc.code if isinstance(exc, (h3.H3Error, LongGenerationError)) else "long_video_request_invalid"
            failed_index = next(
                (
                    segment.index for segment in plan.segments
                    if states[segment.index].get("status") == "not_started"
                ),
                None,
            )
            record_error("fast-prepare", exc, segment_index=failed_index)
            if failed_index is not None:
                states[failed_index].update(status="failed", error=code)
            persist("failed", "long_video_segment_failed")
            return

        prepared_statuses = {item.get("status") for item in states.values()}
        if "submission_unknown" in prepared_statuses:
            persist("submission_unknown", "submission_unknown")
            return
        if "failed" in prepared_statuses:
            persist("failed", "long_video_segment_failed")
            return

        def submit_one(segment: FrozenSegment):
            previous_attempt = states[segment.index].get("h3_attempt_id")
            try:
                _revalidate_speaker_authority(
                    plan, segment, requests[segment.index]
                )
                result = h3.submit(requests[segment.index])
                if result.status == "h3_running":
                    return (
                        "running",
                        None,
                        _exact_h3_attempt_id(result.attempt_id, previous_attempt),
                    )
                status = _result_state(result, previous_attempt)
                if status[0] == "succeeded" and not h3.output_is_reusable(
                    requests[segment.index],
                    expected_duration_s=_segment_duration_s(plan, segment),
                    allow_provider_duration_ceiling=True,
                ):
                    return (
                        "failed",
                        "long_video_segment_output_invalid",
                        status[2],
                    )
                return status
            except h3.H3Error as exc:
                record_error("fast-submit", exc, segment_index=segment.index)
                if exc.code == "attempt_not_prepared":
                    return (
                        "submission_unknown",
                        "submission_unknown",
                        _exact_h3_attempt_id(previous_attempt),
                    )
                try:
                    inspected = h3.inspect(requests[segment.index])
                    status = _result_state(inspected, previous_attempt)
                except Exception as inspect_exc:
                    record_error(
                        "fast-submit-inspect",
                        inspect_exc,
                        segment_index=segment.index,
                    )
                    status = (
                        "submission_unknown",
                        "submission_unknown",
                        _exact_h3_attempt_id(previous_attempt),
                    )
                if status[0] == "failed" and exc.code in {
                    "submission_unknown", "state_persist_failed", "h3_internal_error",
                }:
                    return "submission_unknown", "submission_unknown", status[2]
                return status
            except Exception as exc:
                record_error("fast-submit", exc, segment_index=segment.index)
                return (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )

        # Phase 3: fan out only the short POST boundary. No worker waits for a
        # provider result, so every queued child is submitted before polling.
        to_submit = [
            segment for segment in plan.segments
            if states[segment.index].get("status") == "queued"
        ]
        if to_submit:
            parallel_update(to_submit, submit_one)

        def poll_one(segment: FrozenSegment):
            request = requests[segment.index]
            previous_attempt = states[segment.index].get("h3_attempt_id")
            try:
                _revalidate_speaker_authority(plan, segment, request)
                result = h3.resume(request)
                if result.status == "not_started":
                    return (
                        "submission_unknown",
                        "submission_unknown",
                        _exact_h3_attempt_id(result.attempt_id, previous_attempt),
                    )
                status = _result_state(result, previous_attempt)
                if status[0] == "succeeded" and not h3.output_is_reusable(
                    request,
                    expected_duration_s=_segment_duration_s(plan, segment),
                    allow_provider_duration_ceiling=True,
                ):
                    return (
                        "failed",
                        "long_video_segment_output_invalid",
                        status[2],
                    )
                return status
            except h3.H3Error as exc:
                record_error("fast-poll", exc, segment_index=segment.index)
                return (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                ) if exc.code in {
                    "submission_unknown", "state_persist_failed", "h3_internal_error",
                } else (
                    "resume_required",
                    exc.code,
                    _exact_h3_attempt_id(previous_attempt),
                )
            except Exception as exc:
                record_error("fast-poll", exc, segment_index=segment.index)
                return (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )

        # Phase 4: bounded long-lived GET polling. Unknown children never get a
        # second POST, while known siblings are still allowed to finish.
        to_poll = [
            segment for segment in plan.segments
            if states[segment.index].get("status") in {"running", "resume_required"}
        ]
        if to_poll:
            parallel_update(to_poll, poll_one)

        statuses = {item.get("status") for item in states.values()}
        if statuses == {"succeeded"}:
            try:
                _stitch(
                    settings,
                    cid,
                    plan,
                    dialogue_mode,
                    generation=exact_generation(),
                )
            except LongGenerationError as exc:
                record_error("fast-stitch", exc)
                persist("failed", exc.code, "stitch")
            except Exception as exc:
                record_error("fast-stitch", exc)
                persist("failed", "long_video_stitch_failed", "stitch")
            else:
                persist("succeeded", None, "stitch")
        elif "submission_unknown" in statuses:
            persist("submission_unknown", "submission_unknown")
        elif "resume_required" in statuses or "running" in statuses:
            persist("resume_required", "long_video_resume_required")
        else:
            persist("failed", "long_video_segment_failed")
        return

    chains: dict[str, list[FrozenSegment]] = {}
    for segment in plan.segments:
        chains.setdefault(segment.chain_id, []).append(segment)

    attempted_indices: set[int] = set()

    def ready() -> list[FrozenSegment]:
        candidates = []
        for chain in chains.values():
            for segment in chain:
                state = states[segment.index]
                if state.get("status") == "succeeded":
                    continue
                if state.get("status") in {"not_started", "queued", "resume_required"}:
                    if segment.index in attempted_indices:
                        break
                    prior = [states[item.index].get("status") for item in chain if item.index < segment.index]
                    if all(value == "succeeded" for value in prior):
                        candidates.append(segment)
                    break
                break
        return candidates

    def worker(segment: FrozenSegment, action: str):
        if (
            action == "start"
            and _segment_duration_s(plan, segment)
            < long_video.SEGMENT_SOURCE_MIN_S
        ):
            return None, (
                "failed", "long_video_segment_below_provider_minimum", None
            )
        if (
            action == "start"
            and long_video.provider_duration_s(
                segment.start_s,
                segment.end_s,
                receipt_version=plan.receipt_version,
            )
            > long_video.SEGMENT_PROVIDER_MAX_DURATION_S
        ):
            return None, ("failed", "long_video_legacy_plan_read_only", None)
        existing_child_id = states[segment.index].get("child_request_id")
        previous_attempt = states[segment.index].get("h3_attempt_id")
        try:
            request = _request(
                settings, cid, plan, segment, parent_id, fit_mode,
                prepare_inputs=action == "start",
            )
            h3_action = action
            if context_ir_required:
                context_binding = states[segment.index].get("context_ir")
                recovering_failed_context = (
                    isinstance(context_binding, Mapping)
                    and context_binding.get("status") == "failed"
                )
                recovering_context_only = (
                    action == "resume"
                    and isinstance(context_binding, Mapping)
                    and context_binding.get("status") != "succeeded"
                    and existing_child_id is None
                    and previous_attempt is None
                )
                if (
                    isinstance(context_binding, Mapping)
                    and context_binding.get("status") == "succeeded"
                ):
                    context = _freeze_segment_context_ir(
                        settings, plan, segment, request
                    )
                    request = _bind_h3_operational_roots(
                        settings,
                        plan,
                        h3_project.apply_bound_context_ir(
                            context, context_binding
                        ),
                    )
                    if (
                        action == "resume"
                        and plan.prompt_fusion is not None
                        and existing_child_id is None
                        and previous_attempt is None
                    ):
                        h3_action = "start"
                else:
                    frozen_context = (
                        _freeze_segment_context_ir(
                            settings, plan, segment, request
                        )
                        if recovering_failed_context
                        else None
                    )
                    if not _context_ir_may_progress(
                        states[segment.index],
                        request,
                        allow_create=action == "start",
                        frozen_context=frozen_context,
                    ):
                        return (
                            states[segment.index].get("child_request_id"),
                            (
                                "submission_unknown",
                                "submission_unknown",
                                _exact_h3_attempt_id(previous_attempt),
                            ),
                        )
                    request, context_binding, status, error = (
                        _optimize_segment_context_ir(
                            settings, plan, segment, request
                        )
                    )
                    states[segment.index]["context_ir"] = context_binding
                    if status != "succeeded" or request is None:
                        return (
                            states[segment.index].get("child_request_id"),
                            (
                                status,
                                error,
                                _exact_h3_attempt_id(previous_attempt),
                            ),
                        )
                    if recovering_context_only:
                        return (
                            existing_child_id,
                            (
                                "resume_required",
                                "context_ir_ready",
                                _exact_h3_attempt_id(previous_attempt),
                            ),
                        )
                    if action == "resume":
                        h3_action = "start"
        except LongGenerationError as exc:
            record_error(
                "serial-request",
                exc,
                segment_index=segment.index,
                action=action,
            )
            if action == "resume":
                return existing_child_id, (
                    "submission_unknown", "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )
            return None, (
                "failed", exc.code, _exact_h3_attempt_id(previous_attempt)
            )
        except Exception as exc:
            record_error(
                "serial-request",
                exc,
                segment_index=segment.index,
                action=action,
            )
            if action == "resume":
                return existing_child_id, (
                    "submission_unknown", "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )
            return None, (
                "failed", "long_video_request_invalid",
                _exact_h3_attempt_id(previous_attempt),
            )
        try:
            _revalidate_speaker_authority(plan, segment, request)
            result = h3.start(request) if h3_action == "start" else h3.resume(request)
            if h3_action == "resume" and result.status == "not_started":
                return request.client_request_id, (
                    "submission_unknown", "submission_unknown",
                    _exact_h3_attempt_id(result.attempt_id, previous_attempt),
                )
            status = _result_state(result, previous_attempt)
            if status[0] == "succeeded" and not h3.output_is_reusable(
                request,
                expected_duration_s=_segment_duration_s(plan, segment),
                allow_provider_duration_ceiling=True,
            ):
                status = (
                    "failed",
                    "long_video_segment_output_invalid",
                    status[2],
                )
            return request.client_request_id, status
        except h3.H3Error as exc:
            record_error(
                "serial-provider",
                exc,
                segment_index=segment.index,
                action=h3_action,
            )
            try:
                inspected = h3.inspect(request)
                status = _result_state(inspected, previous_attempt)
            except Exception as inspect_exc:
                record_error(
                    "serial-provider-inspect",
                    inspect_exc,
                    segment_index=segment.index,
                    action=h3_action,
                )
                status = (
                    "submission_unknown",
                    "submission_unknown",
                    _exact_h3_attempt_id(previous_attempt),
                )
            if status[0] == "failed" and exc.code in {"submission_unknown", "state_persist_failed", "h3_internal_error"}:
                status = (
                    "submission_unknown", "submission_unknown", status[2]
                )
            return request.client_request_id, status
        except Exception as exc:
            record_error(
                "serial-provider",
                exc,
                segment_index=segment.index,
                action=h3_action,
            )
            return request.client_request_id, (
                "submission_unknown",
                "submission_unknown",
                _exact_h3_attempt_id(previous_attempt),
            )

    active = {}
    locked = False
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            if not locked:
                active_chains = {segment.chain_id for segment in active.values()}
                for segment in ready():
                    if len(active) >= 2:
                        break
                    if segment.chain_id in active_chains:
                        continue
                    state = states[segment.index]
                    is_new_child = state.get("status") == "not_started"
                    action = "start" if is_new_child else "resume"
                    state["status"], state["error"] = "running", None
                    if is_new_child:
                        state["attempt"] = int(state.get("attempt", 0) or 0) + 1
                    attempted_indices.add(segment.index)
                    persist("running", None)
                    future = pool.submit(worker, segment, action)
                    active[future] = segment
                    active_chains.add(segment.chain_id)
            if not active:
                break
            done, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                segment = active.pop(future)
                child_id, (status, error, attempt_id) = future.result()
                changes = {
                    "status": status,
                    "error": error,
                    "child_request_id": child_id,
                }
                if segment.index in native_audio_indices:
                    changes["h3_attempt_id"] = attempt_id
                states[segment.index].update(changes)
                if status == "submission_unknown":
                    locked = True
                persist("submission_unknown" if locked else "running",
                        "submission_unknown" if locked else None)

    if locked:
        persist("submission_unknown", "submission_unknown")
        return
    statuses = {item.get("status") for item in states.values()}
    if statuses == {"succeeded"}:
        try:
            _stitch(
                settings,
                cid,
                plan,
                dialogue_mode,
                generation=exact_generation(),
            )
        except LongGenerationError as exc:
            record_error("serial-stitch", exc)
            persist("failed", exc.code, "stitch")
        except Exception as exc:
            record_error("serial-stitch", exc)
            persist("failed", "long_video_stitch_failed", "stitch")
        else:
            persist("succeeded", None, "stitch")
    elif "resume_required" in statuses:
        persist("resume_required", "long_video_resume_required")
    else:
        persist("failed", "long_video_segment_failed")


def run(settings, cid: str, plan: FrozenPlan, *, startup: bool = False) -> None:
    """Daemon-safe public boundary: unexpected worker failures close durably."""
    try:
        _run_unchecked(settings, cid, plan, startup=startup)
    except BaseException as exc:
        error_trace.record(
            settings.data_dir / cid / "work" / "errors"
            / "long-generation-daemon.json",
            call_path=["pipeline", "long_generation", "daemon"],
            error=exc,
            logger=_LOGGER,
        )
        try:
            meta = storage.load_meta(settings.data_dir, cid)
            generation = meta.get("generation") if isinstance(meta, Mapping) else None
            if not isinstance(generation, dict):
                return
            segments = []
            for item in generation.get("segments", []):
                if not isinstance(item, dict):
                    continue
                if item.get("status") in {"succeeded", "failed"}:
                    segments.append(item)
                else:
                    segments.append({
                        **item,
                        "status": "submission_unknown",
                        "error": "submission_unknown",
                    })
            storage.update_meta(
                settings.data_dir,
                cid,
                generation={
                    **generation,
                    "segments": segments,
                    "status": "submission_unknown",
                    "error": "submission_unknown",
                },
            )
        except BaseException as persist_exc:
            error_trace.record(
                settings.data_dir / cid / "work" / "errors"
                / "long-generation-daemon-persist.json",
                call_path=["pipeline", "long_generation", "daemon", "persist"],
                error=persist_exc,
                logger=_LOGGER,
            )
