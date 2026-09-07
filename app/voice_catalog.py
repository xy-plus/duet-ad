"""Immutable, source-labelled voice references for project-wide H3 consistency."""

from __future__ import annotations

import hashlib
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VOICE_LIBRARY_ROOT = ROOT / "assets" / "voice-library"
VOICE_CATALOG_PATH = VOICE_LIBRARY_ROOT / "catalog.json"
VOICE_CATALOG_NAME = "duet.voice-library"
VOICE_CATALOG_VERSION = 1

_CHINESE_LANGUAGE_MARKERS = ("chinese", "mandarin", "zh", "中文", "汉语", "普通话")


@dataclass(frozen=True)
class VoiceLibrary:
    root: Path
    voices: tuple[dict[str, Any], ...]
    files: dict[str, Path]
    digest: str

    def voice(self, voice_id: str) -> dict[str, Any]:
        for voice in self.voices:
            if voice["voice_id"] == voice_id:
                return voice
        raise KeyError(voice_id)

    def model_line(self, voice_id: str) -> str:
        voice = self.voice(voice_id)
        features = voice["features"]

        def feature(key: str) -> str:
            value = features.get(key)
            return str(value).strip() if value is not None else ""

        parts = [
            voice["voice_id"],
            voice["file"],
            feature("language") or "unspecified",
            feature("gender") or "unspecified",
        ]
        for key, label in (
            ("race", "race"),
            ("accent", "accent"),
            ("region", "region"),
            ("age", "age"),
            ("performance_style", "style"),
            ("intensity", "intensity"),
        ):
            value = feature(key)
            if value and value not in {"not_reported", "unspecified"}:
                parts.append(f"{label}:{value}")
        return " - ".join(str(part) for part in parts)

    def model_lines(self) -> str:
        """One terse ``-`` separated sentence per voice; never JSON."""
        return "\n".join(self.model_line(voice["voice_id"]) for voice in self.voices)

    def default_voice(self, target_language: str | None = None) -> dict[str, Any]:
        """Fallback voice: the first catalog voice in the target language
        (Chinese targets prefer Chinese voices), else the first catalog
        voice."""
        target = (target_language or "").strip().lower()
        if any(marker in target for marker in _CHINESE_LANGUAGE_MARKERS):
            target = "chinese"
        for voice in self.voices:
            language = str(
                voice["features"].get("language") or ""
            ).strip().lower()
            if any(marker in language for marker in _CHINESE_LANGUAGE_MARKERS):
                language = "chinese"
            if target and language == target:
                return voice
        return self.voices[0]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"voice catalog {field} must be a non-empty trimmed string")
    return value


def load_voice_library(path: Path = VOICE_CATALOG_PATH) -> VoiceLibrary:
    catalog_path = path.resolve(strict=True)
    root = catalog_path.parent
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema", "version", "label_policy", "voices"}
    ):
        raise ValueError("voice catalog root is invalid")
    if (
        raw["schema"] != VOICE_CATALOG_NAME
        or type(raw["version"]) is not int
        or raw["version"] != VOICE_CATALOG_VERSION
    ):
        raise ValueError("voice catalog identity is invalid")
    label_policy = raw["label_policy"]
    if (
        not isinstance(label_policy, dict)
        or set(label_policy) != {"source_truth", "personality", "identity"}
        or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in label_policy.values()
        )
    ):
        raise ValueError("voice catalog label policy is invalid")
    voices = raw["voices"]
    if not isinstance(voices, list) or not voices:
        raise ValueError("voice catalog must contain at least one asset")

    ids: set[str] = set()
    files: dict[str, Path] = {}
    validated: list[dict[str, Any]] = []
    for index, value in enumerate(voices, 1):
        required_keys = {
            "voice_id", "file", "sha256", "duration_s", "media_type",
            "source", "features",
        }
        if not isinstance(value, dict) or set(value) != required_keys:
            raise ValueError(f"voice catalog entry {index} is invalid")
        voice_id = _non_empty_string(value["voice_id"], "voice_id")
        if voice_id in ids:
            raise ValueError("voice catalog voice_id is duplicated")
        ids.add(voice_id)
        relative = Path(_non_empty_string(value["file"], "file"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("voice catalog file must be a safe relative path")
        resolved = (root / relative).resolve(strict=True)
        if root not in resolved.parents or not resolved.is_file():
            raise ValueError("voice catalog file is outside the library")
        digest = _non_empty_string(value["sha256"], "sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("voice catalog sha256 is invalid")
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != digest:
            raise ValueError(f"voice asset checksum mismatch: {voice_id}")
        duration = value["duration_s"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError("voice catalog duration is invalid")
        duration = float(duration)
        if not math.isfinite(duration):
            raise ValueError("voice catalog duration is invalid")
        if value["media_type"] != "audio/wav" or resolved.suffix.lower() != ".wav":
            raise ValueError("voice asset must be declared as audio/wav")
        with wave.open(str(resolved), "rb") as handle:
            if (
                handle.getnchannels() != 1
                or handle.getsampwidth() != 2
                or handle.getcomptype() != "NONE"
            ):
                raise ValueError("voice asset must be mono 16-bit PCM WAV")
            actual_duration = handle.getnframes() / handle.getframerate()
        if not 2.0 <= actual_duration <= 5.0:
            raise ValueError("voice asset duration must be between 2 and 5 seconds")
        if not isinstance(value["source"], dict):
            raise ValueError("voice catalog source metadata is invalid")
        features = value["features"]
        if not isinstance(features, dict):
            raise ValueError("voice catalog features are invalid")
        files[voice_id] = resolved
        validated.append(value)
    return VoiceLibrary(
        root=root,
        voices=tuple(validated),
        files=files,
        digest=hashlib.sha256(_canonical(raw)).hexdigest(),
    )
