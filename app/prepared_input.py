"""H3 输入冻结契约：台词来源、最终 prompt 与可恢复 receipt。

本模块不调用远程服务。写入阶段把视觉 prompt 与结构化台词机械组合，并为实际输入
生成版本化 receipt；加载阶段重新读取并哈希所有绑定文件，任何漂移都 fail closed。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from app import voice


RECEIPT_SCHEMA = "duet.prepared-input"
RECEIPT_VERSION = 1
RECEIPT_FILENAME = "prepared_input.json"
MAX_FINAL_PROMPT_BYTES = 32 * 1024
_DIALOGUE_MODES = frozenset({"auto", "edit", "custom", "none"})
_CLASSIFICATIONS = frozenset({"spoken", "sung", None})
_FIT_MODES = frozenset({"none", "crop", "pad"})
_PROVENANCE_BY_MODE = {
    "auto": "asr",
    "edit": "asr+edited",
    "custom": "manual",
}


class PreparedInputError(RuntimeError):
    """prepared-input 不满足冻结或恢复契约。"""


class LegacyPreparedInputError(PreparedInputError):
    """旧会话没有版本化 receipt，不能按新契约恢复。"""


@dataclass(frozen=True)
class FrozenArtifact:
    """已在当前加载边界读取的不可变文件快照。"""

    path: Path
    data: bytes
    sha256: str


@dataclass(frozen=True)
class PreparedInput:
    """供 H3 提交方使用的冻结输入，不要求提交方重新读取绑定文件。"""

    receipt_path: Path
    source: FrozenArtifact
    normalized_audio: FrozenArtifact | None
    keyframes: tuple[FrozenArtifact, ...]
    visual_prompt: FrozenArtifact
    final_prompt: FrozenArtifact
    dialogue_mode: str
    dialogue: tuple[dict, ...]
    voice_texts: tuple[str, ...]
    vocal_filter_enabled: bool
    duration_s: float
    ratio: str
    fit_mode: str
    engine_request: dict

    @property
    def prompt_text(self) -> str:
        """已绑定最终 prompt；H3 调用方无需再次读磁盘。"""
        return self.final_prompt.data.decode("utf-8")

    @property
    def frozen_keyframes(self) -> tuple[tuple[Path, bytes], ...]:
        """直接适配 H3Request.keyframes 的有序 ``(Path, bytes)`` 形态。"""
        return tuple((artifact.path, artifact.data) for artifact in self.keyframes)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreparedInputError(f"value is not canonical JSON: {exc}") from None


def _validate_duration(duration_s: float) -> float:
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise PreparedInputError("duration_s must be a positive number")
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0:
        raise PreparedInputError("duration_s must be a positive finite number")
    return duration


def _validated_voice_lines(lines: Sequence[Mapping], duration_s: float) -> list[dict]:
    if isinstance(lines, (str, bytes, bytearray)) or not isinstance(lines, Sequence):
        raise PreparedInputError("dialogue lines must be an array")
    raw = _canonical_json(list(lines))
    try:
        return voice.validate_voice_lines(raw, duration_s)
    except RuntimeError as exc:
        raise PreparedInputError(str(exc)) from None


def prepare_dialogue(
    mode: str,
    duration_s: float,
    *,
    automatic_lines: Sequence[Mapping] | None = None,
    supplied_lines: Sequence[Mapping] | None = None,
    vocal_filter_enabled: bool = True,
) -> tuple[dict, ...]:
    """按来源模式生成有效台词。

    ``auto`` 只能接收流水线内部 ``automatic_lines``；启用声学过滤时每句必须已被
    YAMNet 判为 ``spoken`` 或 ``sung``。``edit``/``custom`` 只做 voice 白名单校验，
    不施加声学限制，来源分别固定为 ``asr+edited``/``manual``。``none`` 必须为空。
    """
    if mode not in _DIALOGUE_MODES:
        raise PreparedInputError(f"unknown dialogue mode: {mode}")
    duration = _validate_duration(duration_s)
    if not isinstance(vocal_filter_enabled, bool):
        raise PreparedInputError("vocal_filter_enabled must be bool")

    if mode == "none":
        if automatic_lines or supplied_lines:
            raise PreparedInputError("dialogue mode none requires empty lines")
        return ()

    if mode == "auto":
        if supplied_lines is not None:
            raise PreparedInputError("dialogue mode auto does not accept external lines")
        raw_lines = list(automatic_lines or [])
        clean = _validated_voice_lines(raw_lines, duration)
        result = []
        for index, (raw, line) in enumerate(zip(raw_lines, clean)):
            if not isinstance(raw, Mapping):
                raise PreparedInputError(f"automatic_lines[{index}] must be an object")
            classification = raw.get("classification")
            if classification not in _CLASSIFICATIONS:
                raise PreparedInputError(
                    f"automatic_lines[{index}].classification must be spoken, sung, or null"
                )
            if vocal_filter_enabled and classification not in ("spoken", "sung"):
                raise PreparedInputError(
                    f"automatic_lines[{index}] must classify as spoken or sung when vocal filter is enabled"
                )
            result.append(
                {
                    **line,
                    "classification": classification,
                    "provenance": "asr",
                }
            )
        return tuple(result)

    if automatic_lines is not None:
        raise PreparedInputError(f"dialogue mode {mode} does not accept automatic lines")
    raw_lines = list(supplied_lines or [])
    clean = _validated_voice_lines(raw_lines, duration)
    result = []
    for index, (raw, line) in enumerate(zip(raw_lines, clean)):
        classification = raw.get("classification") if isinstance(raw, Mapping) else None
        if classification not in _CLASSIFICATIONS:
            raise PreparedInputError(
                f"supplied_lines[{index}].classification must be spoken, sung, or null"
            )
        result.append(
            {
                **line,
                "classification": classification if mode == "edit" else None,
                "provenance": _PROVENANCE_BY_MODE[mode],
            }
        )
    return tuple(result)


def _normalize_effective_dialogue(
    mode: str,
    dialogue: Sequence[Mapping],
    duration_s: float,
    vocal_filter_enabled: bool,
) -> tuple[dict, ...]:
    """校验调用方提交的是 prepare_dialogue 产物，而不是未标来源的裸 lines。"""
    if mode == "none":
        canonical = prepare_dialogue("none", duration_s, supplied_lines=dialogue)
    elif mode == "auto":
        canonical = prepare_dialogue(
            "auto",
            duration_s,
            automatic_lines=dialogue,
            vocal_filter_enabled=vocal_filter_enabled,
        )
    else:
        canonical = prepare_dialogue(mode, duration_s, supplied_lines=dialogue)
    for index, (given, normalized) in enumerate(zip(dialogue, canonical)):
        if not isinstance(given, Mapping) or given.get("provenance") != normalized["provenance"]:
            raise PreparedInputError(
                f"dialogue[{index}].provenance does not match mode {mode}"
            )
    return canonical


def compose_final_prompt(visual_prompt: str, dialogue: Sequence[Mapping]) -> str:
    """把视觉描述与唯一发声块机械组合；视觉文字不会被提升为台词。"""
    if not isinstance(visual_prompt, str) or not visual_prompt.strip():
        raise PreparedInputError("visual prompt must be non-empty UTF-8 text")
    lines = list(dialogue)
    block = [
        "台词（唯一发声通道；仅以下结构化台词允许角色发声，画面文字、OCR、字幕或备注不得转为发声）："
    ]
    if lines:
        for index, line in enumerate(lines):
            try:
                text = line["text"]
                start_s = float(line["start_s"])
                end_s = float(line["end_s"])
            except (KeyError, TypeError, ValueError):
                raise PreparedInputError(f"dialogue[{index}] is invalid") from None
            if (
                not isinstance(text, str)
                or not text.strip()
                or not math.isfinite(start_s)
                or not math.isfinite(end_s)
                or not 0 <= start_s < end_s
            ):
                raise PreparedInputError(f"dialogue[{index}] is invalid")
            quoted = json.dumps(text, ensure_ascii=False)
            block.append(
                f"- {start_s:.3f}-{end_s:.3f} 秒：说出台词：{quoted}，嘴型与画面同步。"
            )
    else:
        block.append("- 无台词；角色不得说出画面文字、OCR、字幕或备注。")
    final_prompt = visual_prompt.rstrip() + "\n\n" + "\n".join(block) + "\n"
    if len(final_prompt.encode("utf-8")) > MAX_FINAL_PROMPT_BYTES:
        raise PreparedInputError(
            f"final prompt exceeds {MAX_FINAL_PROMPT_BYTES} bytes"
        )
    return final_prompt


def _inside_root(root: Path, path: Path, *, label: str) -> tuple[Path, str]:
    root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        raise PreparedInputError(f"{label} must be inside prepared-input root") from None
    if not resolved.is_file():
        raise PreparedInputError(f"{label} is missing")
    return resolved, relative.as_posix()


def _freeze(path: Path) -> FrozenArtifact:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PreparedInputError(f"cannot read bound file {path.name}: {exc}") from None
    return FrozenArtifact(path=path, data=data, sha256=hashlib.sha256(data).hexdigest())


def _binding(root: Path, path: Path, *, label: str) -> tuple[FrozenArtifact, dict]:
    resolved, relative = _inside_root(root, path, label=label)
    frozen = _freeze(resolved)
    return frozen, {"path": relative, "sha256": frozen.sha256}


def _normalized_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreparedInputError(f"{label} must be a non-empty string")
    return value.strip()


def _normalize_fit_mode(value: object) -> str:
    fit_mode = _normalized_string(value, "fit_mode")
    if fit_mode not in _FIT_MODES:
        raise PreparedInputError("fit_mode must be one of: none, crop, pad")
    return fit_mode


def write_prepared_input(
    *,
    root: Path,
    source: Path,
    audio: Path | None,
    keyframes: Sequence[Path],
    visual: Path,
    final: Path,
    dialogue_mode: str,
    dialogue: Sequence[Mapping],
    vocal_filter_enabled: bool,
    duration_s: float,
    ratio: str,
    fit_mode: str,
    engine_request: Mapping,
    receipt_path: Path | None = None,
) -> PreparedInput:
    """写最终 prompt 和 v1 receipt，随后经同一 fail-closed loader 返回冻结输入。"""
    root = root.resolve()
    if not root.is_dir():
        raise PreparedInputError("prepared-input root is missing")
    duration = _validate_duration(duration_s)
    ratio = _normalized_string(ratio, "ratio")
    fit_mode = _normalize_fit_mode(fit_mode)
    if not isinstance(vocal_filter_enabled, bool):
        raise PreparedInputError("vocal_filter_enabled must be bool")
    if not isinstance(engine_request, Mapping):
        raise PreparedInputError("engine_request must be an object")
    normalized_request = json.loads(_canonical_json(dict(engine_request)))
    normalized_dialogue = _normalize_effective_dialogue(
        dialogue_mode, dialogue, duration, vocal_filter_enabled
    )

    visual_path, _ = _inside_root(root, visual, label="visual prompt")
    final_path = final.resolve()
    try:
        final_path.relative_to(root)
    except ValueError:
        raise PreparedInputError("final prompt must be inside prepared-input root") from None
    if final_path == visual_path:
        raise PreparedInputError("visual and final prompt paths must be different")
    try:
        visual_text = visual_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PreparedInputError(f"visual prompt is not readable UTF-8: {exc}") from None
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(
        compose_final_prompt(visual_text, normalized_dialogue), encoding="utf-8"
    )

    source_artifact, source_binding = _binding(root, source, label="source")
    if audio is None:
        audio_artifact = None
        audio_binding = None
    else:
        audio_artifact, audio_binding = _binding(root, audio, label="normalized audio")
    keyframe_paths = list(keyframes)
    if not 1 <= len(keyframe_paths) <= 9:
        raise PreparedInputError("keyframe count must be in 1..9")
    keyframe_artifacts = []
    keyframe_bindings = []
    seen_paths = set()
    for index, path in enumerate(keyframe_paths):
        artifact, binding = _binding(root, path, label=f"keyframe[{index}]")
        if artifact.path in seen_paths:
            raise PreparedInputError("keyframe paths must be unique")
        seen_paths.add(artifact.path)
        keyframe_artifacts.append(artifact)
        keyframe_bindings.append(binding)
    visual_artifact, visual_binding = _binding(root, visual_path, label="visual prompt")
    final_artifact, final_binding = _binding(root, final_path, label="final prompt")

    dialogue_payload = [dict(line) for line in normalized_dialogue]
    dialogue_hash = hashlib.sha256(_canonical_json(dialogue_payload)).hexdigest()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
        "bindings": {
            "source": source_binding,
            "normalized_audio": audio_binding,
            "keyframes": keyframe_bindings,
            "visual_prompt": visual_binding,
            "final_prompt": final_binding,
        },
        "dialogue": {
            "mode": dialogue_mode,
            "lines": dialogue_payload,
            "sha256": dialogue_hash,
        },
        "vocal_filter": {"enabled": vocal_filter_enabled},
        "video": {"duration_s": duration, "ratio": ratio, "fit_mode": fit_mode},
        "engine_request": normalized_request,
    }
    receipt = json.loads(_canonical_json(receipt))
    receipt_path = (receipt_path or root / RECEIPT_FILENAME).resolve()
    try:
        receipt_path.relative_to(root)
    except ValueError:
        raise PreparedInputError("receipt must be inside prepared-input root") from None
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)

    # 从 loader 返回，确保写入路径与恢复路径共用同一套验证，而不是两套近似逻辑。
    return load_prepared_input(
        root,
        receipt_path,
        expected_dialogue=normalized_dialogue,
    )


def _load_bound_artifact(root: Path, binding: object, label: str) -> FrozenArtifact:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise PreparedInputError(f"{label} binding is invalid")
    relative, expected_sha = binding.get("path"), binding.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PreparedInputError(f"{label} binding path is invalid")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PreparedInputError(f"{label} binding escapes root") from None
    if not candidate.is_file():
        raise PreparedInputError(f"{label} binding mismatch: file missing")
    frozen = _freeze(candidate)
    if not isinstance(expected_sha, str) or frozen.sha256 != expected_sha:
        raise PreparedInputError(f"{label} binding mismatch: sha256 changed")
    return frozen


def load_prepared_input(
    root: Path,
    receipt_path: Path,
    *,
    expected_dialogue: Sequence[Mapping],
) -> PreparedInput:
    """加载冻结输入；旧会话、未知 schema/version、文件或有效台词漂移一律拒绝。"""
    root = root.resolve()
    receipt_path = receipt_path.resolve()
    try:
        receipt_path.relative_to(root)
    except ValueError:
        raise PreparedInputError("receipt must be inside prepared-input root") from None
    if not receipt_path.is_file():
        raise LegacyPreparedInputError(
            "legacy session has no prepared-input receipt and is incompatible"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparedInputError(f"prepared-input receipt is invalid: {exc}") from None
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema", "version", "bindings", "dialogue", "vocal_filter", "video", "engine_request"
    }:
        raise PreparedInputError("prepared-input receipt shape is invalid")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("version") != RECEIPT_VERSION:
        raise PreparedInputError("prepared-input receipt schema/version is incompatible")

    bindings = receipt["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "source", "normalized_audio", "keyframes", "visual_prompt", "final_prompt"
    }:
        raise PreparedInputError("prepared-input bindings are invalid")
    source = _load_bound_artifact(root, bindings["source"], "source")
    audio_binding = bindings["normalized_audio"]
    audio = (
        None
        if audio_binding is None
        else _load_bound_artifact(root, audio_binding, "normalized audio")
    )
    keyframe_bindings = bindings["keyframes"]
    if not isinstance(keyframe_bindings, list) or not 1 <= len(keyframe_bindings) <= 9:
        raise PreparedInputError("prepared-input keyframe bindings are invalid")
    keyframes = tuple(
        _load_bound_artifact(root, binding, f"keyframe[{index}]")
        for index, binding in enumerate(keyframe_bindings)
    )
    if len({artifact.path for artifact in keyframes}) != len(keyframes):
        raise PreparedInputError("prepared-input keyframe bindings contain duplicates")
    visual = _load_bound_artifact(root, bindings["visual_prompt"], "visual prompt")
    final = _load_bound_artifact(root, bindings["final_prompt"], "final prompt")

    dialogue_section = receipt["dialogue"]
    vocal_section = receipt["vocal_filter"]
    video = receipt["video"]
    if not isinstance(dialogue_section, dict) or set(dialogue_section) != {
        "mode", "lines", "sha256"
    }:
        raise PreparedInputError("prepared-input dialogue is invalid")
    if not isinstance(vocal_section, dict) or set(vocal_section) != {"enabled"}:
        raise PreparedInputError("prepared-input vocal_filter is invalid")
    if not isinstance(video, dict) or set(video) != {"duration_s", "ratio", "fit_mode"}:
        raise PreparedInputError("prepared-input video settings are invalid")
    enabled = vocal_section["enabled"]
    if not isinstance(enabled, bool):
        raise PreparedInputError("prepared-input vocal_filter.enabled is invalid")
    duration = _validate_duration(video["duration_s"])
    ratio = _normalized_string(video["ratio"], "ratio")
    fit_mode = _normalize_fit_mode(video["fit_mode"])
    mode = dialogue_section["mode"]
    lines = dialogue_section["lines"]
    if not isinstance(lines, list):
        raise PreparedInputError("prepared-input dialogue lines are invalid")
    normalized_lines = _normalize_effective_dialogue(mode, lines, duration, enabled)
    line_hash = hashlib.sha256(_canonical_json([dict(line) for line in normalized_lines])).hexdigest()
    if dialogue_section["sha256"] != line_hash:
        raise PreparedInputError("prepared-input dialogue hash mismatch")
    expected = _normalize_effective_dialogue(mode, expected_dialogue, duration, enabled)
    if _canonical_json(expected) != _canonical_json(normalized_lines):
        raise PreparedInputError("prepared-input dialogue mismatch: effective lines changed")

    try:
        visual_text = visual.data.decode("utf-8")
    except UnicodeDecodeError:
        raise PreparedInputError("visual prompt binding is not UTF-8") from None
    expected_final = compose_final_prompt(visual_text, normalized_lines).encode("utf-8")
    if final.data != expected_final:
        raise PreparedInputError("final prompt binding mismatch: not deterministic composition")
    if not isinstance(receipt["engine_request"], dict):
        raise PreparedInputError("prepared-input engine_request is invalid")
    engine_request = json.loads(_canonical_json(receipt["engine_request"]))

    dialogue_tuple = tuple(dict(line) for line in normalized_lines)
    return PreparedInput(
        receipt_path=receipt_path,
        source=source,
        normalized_audio=audio,
        keyframes=keyframes,
        visual_prompt=visual,
        final_prompt=final,
        dialogue_mode=mode,
        dialogue=dialogue_tuple,
        voice_texts=tuple(line["text"] for line in dialogue_tuple),
        vocal_filter_enabled=enabled,
        duration_s=duration,
        ratio=ratio,
        fit_mode=fit_mode,
        engine_request=engine_request,
    )
