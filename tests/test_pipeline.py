from __future__ import annotations

import base64
import hashlib
import json
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from app import face_mask
from app import media
from app.config import SEEDREAM_API_BASE, SEEDREAM_PRO_MODEL, Settings
from app.fusion_schema import AUTHORITY_BEGIN, AUTHORITY_END, FUSION_OUTPUT_NAME
from app.object_replace_schema import (
    OBJECT_REPLACE_OUTPUT_NAME,
    OBJECT_REPLACE_OUTPUT_VERSION,
)
from app.pipeline import (
    DIALOGUE_GENERATION_SCHEMA,
    H3_PREPARED_ARTIFACTS_VERSION,
    PICSART_RECEIPT_VERSION,
    FACE_MASK_MAX_ATTEMPTS,
    Pipeline,
    PicsartTaskHungError,
    PreparedPicsartRequest,
    REPLACEMENT_SHEET_VERSION,
    SAME_AS_SOURCE_LANGUAGE,
    TRANSLATION_SCHEMA,
    _validate_dialogue_generation_shape,
    _validate_translation_shape,
)
from app.providers import CodexProcessError, ProviderFailure, ProviderResult
from app.replacement_sheet import (
    build_replacement_sheet_spec,
    finalize_replacement_sheet,
    materialize_replacement_sheet_inputs,
    replacement_sheet_spec_sha256,
)
from app.research_log import ResearchLog
from app.seedream import (
    SEEDREAM_OUTPUT_HEIGHT,
    SEEDREAM_OUTPUT_SIZE,
    SEEDREAM_OUTPUT_WIDTH,
    SeedreamError,
)
from app.store import ProjectStore
from app.voice_catalog import load_voice_library
from app.video_maker_schemas import (
    VISUAL_PROMPT_SCHEMA,
    normalize_project_index,
    project_index_schema,
)


_TEST_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVQIHWOsr69n"
    "YGBgrK+vZ2BgAAAZcAL9Q/C0aAAAAABJRU5ErkJggg=="
)


def _settings(path: Path, **changes: Any) -> Settings:
    values = {
        "access_token": "test-token",
        "data_dir": path,
        "public_base_url": "https://pipeline.test",
        "enable_mediakit": False,
        "provider_poll_interval_s": 0,
    }
    values.update(changes)
    return Settings(**values)


def _effective_request(
    *,
    resolution: str = "768p",
    aspect_ratio: str = "16:9",
    target_language: str = SAME_AS_SOURCE_LANGUAGE,
    dialogue_mode: str = "auto_rewrite",
    brief: str | None = None,
    fusion_agent: str = "deepseek",
    replacement_instruction: str | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "output": {
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "fit_mode": "auto",
        },
        "processing": {
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": (
            {
                "mode": "generate",
                "target_language": target_language,
                "brief": brief,
            }
            if dialogue_mode == "generate"
            else {
                "mode": "auto_rewrite",
                "target_language": target_language,
            }
        ),
        "replacement_guidance": (
            {
                "instruction": replacement_instruction,
                "image_field": "replacement_image",
            }
            if replacement_instruction is not None
            else None
        ),
        "fusion": {"agent": fusion_agent},
    }


def test_effective_request_passes_through_regardless_of_shape() -> None:
    # 中间层不做形态校验：自己写下的状态原样放行，消费点自行容错。
    request = _effective_request()
    assert Pipeline._validated_effective_request(request) is request
    request.pop("output")
    request["processing"]["optimize_image"] = False
    request["dialogue"] = {"mode": "manual"}
    request["unsupported"] = True
    assert Pipeline._validated_effective_request(request) is request
    with pytest.raises(ValueError, match="effective request"):
        Pipeline._validated_effective_request(["not", "a", "dict"])


def test_stored_voice_references_bind_by_hash_not_path_name(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "voice-path-version",
    )
    library = load_voice_library()
    voice = library.voices[0]
    directory = project / "work" / "voice-references"
    directory.mkdir(parents=True)
    renamed = directory / "renamed.wav"
    renamed.write_bytes(library.files[voice["voice_id"]].read_bytes())
    state = {
        "voice_references": [{
            "audio_reference": "Audio 1",
            "voice_id": voice["voice_id"],
            "asset_path": "work/voice-references/renamed.wav",
            "asset_sha256": voice["sha256"],
            "features": library.model_line(voice["voice_id"]),
        }],
    }

    assert Pipeline._stored_voice_references(project, state) == [
        renamed.resolve()
    ]

    state["voice_references"][0]["asset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum does not match"):
        Pipeline._stored_voice_references(project, state)

    state["voice_references"][0]["asset_sha256"] = voice["sha256"]
    state["voice_references"][0]["asset_path"] = (
        "work/voice-references/missing.wav"
    )
    with pytest.raises(ValueError, match="voice reference"):
        Pipeline._stored_voice_references(project, state)


class _UnusedMediaKit:
    def erase(self, *_args: Any, **_kwargs: Any) -> ProviderResult[Path]:
        raise AssertionError("MediaKit is disabled in this test")


class _FlowMediaKit:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[list[str]] = []

    def erase(
        self, source: Path, destination: Path, scenes: list[str],
    ) -> ProviderResult[Path]:
        destination.write_bytes(source.read_bytes() + b"-processed")
        with self.lock:
            self.calls.append(list(scenes))
        return ProviderResult(
            destination,
            {"stages": list(scenes), "raw": "mediakit-response"},
            [destination],
        )

class _FlowSeedream:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []
        self.replay_calls: list[dict[str, Path]] = []

    def generate(
        self,
        *,
        prompt: str,
        images: list[Path],
        destination: Path,
        receipt_path: Path,
    ) -> ProviderResult[Path]:
        self.calls.append({
            "prompt": prompt,
            "images": list(images),
            "destination": destination,
            "receipt_path": receipt_path,
        })
        if self.events is not None:
            self.events.append("seedream")
        result_payload = images[0].read_bytes()
        destination.write_bytes(result_payload)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        input_sha256 = [Pipeline._sha256(path) for path in images]
        endpoint = f"{SEEDREAM_API_BASE}/images/generations"
        request_binding = {
            "version": 1,
            "endpoint": endpoint,
            "model": SEEDREAM_PRO_MODEL,
            "prompt_sha256": prompt_sha256,
            "input_sha256": input_sha256,
            "size": SEEDREAM_OUTPUT_SIZE,
            "response_format": "b64_json",
            "watermark": False,
        }
        request_sha256 = Pipeline._json_sha256(request_binding)
        result_sha256 = hashlib.sha256(result_payload).hexdigest()
        receipt_path.with_name(receipt_path.name + ".post-claim").write_text(
            json.dumps({
                "version": 1,
                "kind": "seedream_paid_post",
                "request_sha256": request_sha256,
            }),
            encoding="utf-8",
        )
        receipt_path.with_name(receipt_path.name + ".result").write_bytes(
            result_payload,
        )
        receipt_path.with_name(receipt_path.name + ".response.json").write_text(
            json.dumps({
                "version": 1,
                "kind": "seedream_response",
                "request_sha256": request_sha256,
                "result_file": receipt_path.name + ".result",
                "result_bytes": len(result_payload),
                "result_sha256": result_sha256,
                "media_type": "image/png",
                "width": SEEDREAM_OUTPUT_WIDTH,
                "height": SEEDREAM_OUTPUT_HEIGHT,
            }),
            encoding="utf-8",
        )
        receipt_path.write_text(
            json.dumps({
                **request_binding,
                "provider": "ark_seedream",
                "status": "succeeded",
                "request_sha256": request_sha256,
                "result_file": receipt_path.name + ".result",
                "result_bytes": len(result_payload),
                "result_sha256": result_sha256,
                "response_marker_file": receipt_path.name + ".response.json",
                "output_sha256": result_sha256,
            }),
            encoding="utf-8",
        )
        return ProviderResult(
            destination,
            {"provider": "ark_seedream", "outcome": "succeeded"},
            [destination],
        )

    def replay_received(
        self,
        *,
        destination: Path,
        receipt_path: Path,
    ) -> ProviderResult[Path]:
        self.replay_calls.append({
            "destination": destination,
            "receipt_path": receipt_path,
        })
        result_path = receipt_path.with_name(receipt_path.name + ".result")
        destination.write_bytes(result_path.read_bytes())
        return ProviderResult(
            destination,
            {"provider": "ark_seedream", "outcome": "succeeded", "replayed": True},
            [destination],
        )


class _RejectSeedream:
    def generate(self, **_kwargs: Any) -> None:
        raise AssertionError("Seedream must not run while reusing frozen artifacts")

    def replay_received(self, **_kwargs: Any) -> None:
        raise AssertionError("Seedream must not replay while reusing frozen artifacts")


def _materialize_test_replacement_sheet(
    pipeline: Pipeline,
    project_id: str,
    project: Path,
    prepared: list[dict[str, Any]],
    enhanced_element_index: dict[str, Any],
    *,
    replacement: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    state = pipeline.store.read(project_id)
    existing_segments = {
        int(segment["index"]): segment
        for segment in state.get("segments", [])
        if isinstance(segment, dict)
        and isinstance(segment.get("index"), int)
        and not isinstance(segment.get("index"), bool)
    }
    frozen_prepared: list[dict[str, Any]] = []
    for raw in prepared:
        index = int(raw["index"])
        item = {**existing_segments.get(index, {}), **raw}
        item.setdefault("start_s", float(index - 1) * 5.0)
        item.setdefault("duration_s", 5.0)
        item.setdefault("end_s", float(item["start_s"]) + float(item["duration_s"]))
        item.setdefault("join_mode", "hard_cut" if index == 1 else "continue")
        item.setdefault("visual_prompt", f"segment {index} visual")
        item.setdefault("dialogue", [])
        source = item.get("source")
        if not isinstance(source, Path):
            source = project / "work" / "segments" / f"{index:04d}" / "source.mp4"
            source.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists():
                source.write_bytes(f"source-{index}".encode("ascii"))
            item["source"] = source
        frozen_prepared.append(item)
    object_replace = dict(state.get("object_replace") or {})
    object_replace.setdefault("version", OBJECT_REPLACE_OUTPUT_VERSION)
    object_replace.setdefault("input_sha256", "a" * 64)
    pipeline.store.update(
        project_id,
        segment_count=len(frozen_prepared),
        segments=[Pipeline._public_segment(item) for item in frozen_prepared],
        enhanced_element_index=enhanced_element_index,
        object_replace=object_replace,
        voice_references=state.get("voice_references") or [],
    )
    return pipeline._run_replacement_sheet(
        project_id,
        project,
        frozen_prepared,
        enhanced_element_index,
        object_replace,
        replacement,
        ResearchLog(project),
    )


def _failed_replacement_sheet_checkpoint(
    tmp_path: Path,
    *,
    prepared: bool,
) -> tuple[ProjectStore, str, Path, dict[str, Any], _FlowSeedream]:
    request = _effective_request()
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(request, "sheet-checkpoint")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    enhanced_element_index = {
        "people": {
            "person-01": {
                "source_visual_description": "the source person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a different person",
                    "appearance_description": "an isolated different person",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }
    seedream = _FlowSeedream()
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
        seedream=seedream,
    )
    replacement_sheet, _sheet_path = _materialize_test_replacement_sheet(
        pipeline,
        project_id,
        project,
        [{
            "index": 1,
            "start_s": 0.0,
            "end_s": 5.0,
            "duration_s": 5.0,
            "join_mode": "hard_cut",
            "visual_prompt": "frozen visual prompt",
            "dialogue": [],
            "analysis_paths": [],
        }],
        enhanced_element_index,
    )
    if prepared:
        replacement_sheet = json.loads(json.dumps(replacement_sheet))
        replacement_sheet["status"] = "prepared"
        replacement_sheet["output"] = None
        replacement_sheet["seedream"]["receipt"] = None
        replacement_sheet["h3_reference_images"] = [
            reference
            for reference in replacement_sheet["h3_reference_images"]
            if reference.get("role") == "user_replacement"
        ]
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        replacement_sheet=replacement_sheet,
        project_progress={
            "status": "failed", "percent": 55, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    return store, project_id, project, replacement_sheet, seedream


class _FlowDeepSeek:
    def __init__(self) -> None:
        self.visual_barrier = threading.Barrier(2)
        self.branch_barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.structured_calls: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.common_ready = False

    def complete_structured(
        self,
        instruction: str,
        payload: Any,
        schema: dict[str, Any],
        images: list[Path] | None = None,
        validator: Any = None,
    ) -> ProviderResult[dict[str, Any]]:
        with self.lock:
            self.structured_calls.append({
                "instruction": instruction,
                "payload": payload,
                "schema": schema,
                "images": list(images or []),
            })
        if schema == VISUAL_PROMPT_SCHEMA:
            index = int(payload["segment"]["index"])
            with self.lock:
                self.events.append(f"visual-start-{index}")
            self.visual_barrier.wait(timeout=3)
            return ProviderResult(
                {"prompt": f"visual-{index}"},
                {"response": {"api": "responses", "segment": index}},
            )
        output_name = (
            ((schema.get("properties") or {}).get("schema") or {}).get("enum")
            or [None]
        )[0]
        if output_name == OBJECT_REPLACE_OUTPUT_NAME:
            with self.lock:
                assert self.common_ready
                assert "translation" in self.events
                self.events.append("object-replace")
            category_schemas = {
                category: schema["properties"][category]["properties"]
                for category in ("people", "entities", "scenes")
            }
            people_schema = category_schemas["people"]
            voice_ids = next(iter(people_schema.values()))["properties"][
                "voice_id"
            ]["anyOf"][0]["enum"]
            people = {}
            for offset, stable_id in enumerate(people_schema):
                matches_user_request = offset == 0
                people[stable_id] = {
                    "matches_user_request": matches_user_request,
                    "replacement": {
                        "source": (
                            "image_1"
                            if matches_user_request
                            else "generated_variant"
                        ),
                        "target_visual_description": (
                            f"replacement appearance for {stable_id}"
                        ),
                        "appearance_description": (
                            f"isolated replacement person {stable_id}"
                        ),
                    },
                    "voice_id": voice_ids[offset],
                }
            people_keys = list(people_schema)
            return ProviderResult(
                {
                    "schema": OBJECT_REPLACE_OUTPUT_NAME,
                    "version": OBJECT_REPLACE_OUTPUT_VERSION,
                    "input_sha256": schema["properties"]["input_sha256"]["enum"][0],
                    "dialogue_voices": [
                        people_keys[offset % len(people_keys)] if people_keys else None
                        for offset, _line in enumerate(
                            payload["translated_dialogue"]
                        )
                    ],
                    "people": people,
                    "entities": {
                        stable_id: {
                            "matches_user_request": False,
                            "replacement": {
                                "source": "generated_variant",
                                "target_visual_description": (
                                    f"replacement object for {stable_id}"
                                ),
                                "appearance_description": (
                                    f"isolated replacement entity {stable_id}"
                                ),
                            },
                        }
                        for stable_id in category_schemas["entities"]
                    },
                    "scenes": {
                        stable_id: {
                            "matches_user_request": False,
                            "replacement": {
                                "source": "generated_variant",
                                "target_visual_description": (
                                    f"replacement layout for {stable_id}"
                                ),
                                "appearance_description": (
                                    f"empty replacement scene {stable_id}"
                                ),
                            },
                        }
                        for stable_id in category_schemas["scenes"]
                    },
                },
                {"response": {"api": "responses", "phase": "object-replace"}},
            )
        if output_name == FUSION_OUTPUT_NAME:
            index = int(payload["segment_index"])
            with self.lock:
                assert self.common_ready
                assert "translation" in self.events
                self.events.append(f"branch-start-{index}")
            self.branch_barrier.wait(timeout=3)
            return ProviderResult(
                {
                    "schema": FUSION_OUTPUT_NAME,
                    "version": 1,
                    "input_sha256": schema["properties"]["input_sha256"]["enum"][0],
                    "segment_index": index,
                    "prompt": (
                        f"r2v cinematic motion prompt {index}; preserve timing, "
                        "camera, lighting, contact, and continuity"
                    ),
                },
                {"response": {"api": "responses", "segment": index}},
            )
        if schema == TRANSLATION_SCHEMA:
            with self.lock:
                self.events.append("translation")
            return ProviderResult(
                {"lines": [{"text": "translated line"}]},
                {"response": {"api": "responses", "phase": "translation"}},
            )
        assert payload["phase"] == "project_index"
        segment_indices = [
            segment["segment_index"] for segment in payload["segments"]
        ]
        assert schema == project_index_schema(segment_indices)
        with self.lock:
            self.events.append("project-index")
            self.common_ready = True
        raw = {
            "people": [
                {
                    "key": "local-person-a",
                    "source_visual_description": "the main visible person",
                    "occurrences": [
                        {"segment_index": 1, "frame_orders": [1]},
                        {"segment_index": 2, "frame_orders": [1]},
                    ],
                    "replaceable": ["face", "clothing"],
                    "preserve": ["motion"],
                },
                {
                    "key": "local-person-b",
                    "source_visual_description": "a second visible person",
                    "occurrences": [
                        {"segment_index": 1, "frame_orders": [1]},
                    ],
                    "replaceable": ["face", "clothing"],
                    "preserve": ["motion"],
                },
                {
                    "key": "local-person-c",
                    "source_visual_description": "a third visible person",
                    "occurrences": [
                        {"segment_index": 2, "frame_orders": [1]},
                    ],
                    "replaceable": ["face", "clothing"],
                    "preserve": ["motion"],
                },
            ],
            "entities": [{
                "key": "local-entity",
                "source_visual_description": "a visible product",
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1]},
                    {"segment_index": 2, "frame_orders": [1]},
                ],
                "replaceable": ["appearance"],
                "preserve": ["function", "contact"],
            }],
            "scenes": [{
                "key": "local-scene",
                "source_visual_description": "a visible kitchen",
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1]},
                    {"segment_index": 2, "frame_orders": [1]},
                ],
                "replaceable": ["layout"],
                "preserve": ["lighting"],
            }],
            "relations": [{
                "key": "local-relation",
                "subject_key": "local-person-a",
                "predicate": "holds",
                "object_key": "local-entity",
                "occurrences": [{
                    "segment_index": 1,
                    "frames": [{
                        "frame_order": 1,
                        "state": "held",
                        "geometry": "right hand",
                    }],
                }],
                "preserve": ["hand contact"],
                "replace_together": True,
            }],
        }
        return ProviderResult(
            raw,
            {"response": {"api": "responses", "phase": "project_index"}},
        )


class _FlowAutodlTTS:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tts_calls: list[dict[str, Any]] = []

    def submit_tts(self, *, text: str, voice_path: Path) -> ProviderResult[str]:
        with self.lock:
            task_id = f"tts-task-{len(self.tts_calls) + 1}"
            self.tts_calls.append({"text": text, "voice_path": voice_path})
        return ProviderResult(
            task_id, {"outcome": "accepted", "task_id": task_id},
        )

    def poll(self, task_id: str) -> ProviderResult[str]:
        return ProviderResult(
            f"https://result.invalid/{task_id}.wav", {"task_id": task_id},
        )

    def download(self, url: str, destination: Path) -> ProviderResult[Path]:
        destination.write_bytes(b"tts-audio")
        return ProviderResult(
            destination, {"request": {"url": url}}, [destination],
        )


class _FlowPicsart:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.submits: list[dict[str, Any]] = []
        self.poll_counts: dict[str, int] = {}

    def submit(self, **kwargs: Any) -> ProviderResult[str]:
        with self.lock:
            task_id = f"task-{len(self.submits) + 1}"
            self.submits.append(dict(kwargs))
        return ProviderResult(
            task_id,
            {"outcome": "accepted", "task_id": task_id, "request": kwargs},
        )

    def poll(self, task_id: str) -> ProviderResult[str]:
        with self.lock:
            count = self.poll_counts.get(task_id, 0) + 1
            self.poll_counts[task_id] = count
        if task_id == "task-1" and count == 1:
            raise ProviderFailure(
                {"task_id": task_id, "polls": [{"status": "network-error"}]},
                [],
                OSError("temporary poll failure"),
            )
        return ProviderResult(
            f"https://result.invalid/{task_id}.mp4",
            {"task_id": task_id, "polls": [{"status": "COMPLETED"}]},
        )

    def download(self, url: str, destination: Path) -> ProviderResult[Path]:
        destination.write_bytes(b"provider-video")
        return ProviderResult(
            destination,
            {"request": {"url": url}, "response": {"status_code": 200}},
            [destination],
        )


def _install_fake_media(monkeypatch: Any, events: list[str]) -> None:
    def probe(_source: Path) -> dict[str, Any]:
        return {"duration_s": 9.2, "width": 1280, "height": 720, "has_audio": True}

    def frames(_source: Path, destination: Path) -> dict[str, Any]:
        destination.mkdir(parents=True, exist_ok=True)
        frame = destination / "frame-0001.png"
        frame.write_bytes(b"frame")
        return {
            "duration_seconds": 9.2,
            "frames": [{"index": 1, "time_seconds": 0.0, "file": frame.name}],
        }

    def extract_audio(_source: Path, destination: Path) -> dict[str, Any]:
        destination.write_bytes(b"audio")
        return {"stdout": "", "stderr": ""}

    def transcribe(
        _audio: Path, destination: Path, _cli: Path, _model: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        lines = [{"text": "source line", "start_s": 0.5, "end_s": 1.5}]
        destination.write_text(json.dumps(lines), encoding="utf-8")
        return lines, {"provider_raw": {"transcription": ["raw line"]}}

    def filter_spoken(
        lines: list[dict[str, Any]], _analysis: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return lines, [{"classification": "spoken", "kept": True}]

    def planned(
        _scenes: list[dict[str, Any]],
        _duration_s: float,
        dialogue: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        assert [line["text"] for line in dialogue] == ["source line"]
        return [
            {
                "index": 1,
                "start_s": 0.0,
                "end_s": 4.2,
                "duration_s": 4.2,
                "chain_id": "chain-1",
                "join_mode": "hard_cut",
                "scene_indices": [1],
                "source_cut_timeline": [],
            },
            {
                "index": 2,
                "start_s": 4.2,
                "end_s": 9.2,
                "duration_s": 5.0,
                "chain_id": "chain-2",
                "join_mode": "hard_cut",
                "scene_indices": [2],
                "source_cut_timeline": [],
            },
        ]

    def selection(_scenes: Any, segment: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"order": order, "decode_frame_index": order + int(segment["index"]) * 10}
            for order in range(1, 10)
        ]

    def cut(
        _source: Path,
        _start_s: float,
        _duration_s: float,
        destination: Path,
    ) -> dict[str, Any]:
        destination.write_bytes(b"segment")
        return {"stdout": "", "stderr": ""}

    def materialize(
        _source: Path,
        _selection: list[dict[str, Any]],
        destination: Path,
    ) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for order in range(1, 10):
            path = destination / f"{order:02d}.png"
            path.write_bytes(_TEST_PNG)
            paths.append(path)
        return paths

    def half(source: Path, destination: Path) -> Path:
        destination.write_bytes(source.read_bytes())
        return destination

    def provider_reference(source: Path, destination: Path) -> Path:
        destination.write_bytes(
            b"\x89PNG\r\n\x1a\n" + source.read_bytes()
        )
        return destination

    def localize(
        lines: list[dict[str, Any]],
        segment: dict[str, Any],
        *,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del segments
        localized = []
        for line in lines:
            start = max(float(line["start_s"]), float(segment["start_s"]))
            end = min(float(line["end_s"]), float(segment["end_s"]))
            if start < end:
                localized.append({
                    "text": line["text"],
                    "start_s": start - float(segment["start_s"]),
                    "end_s": end - float(segment["start_s"]),
                })
        return localized

    def mute(source: Path, destination: Path) -> dict[str, Any]:
        destination.write_bytes(source.read_bytes())
        return {"stdout": "", "stderr": ""}

    def mask_faces(
        source: Path, destination: Path, **_kwargs: Any,
    ) -> dict[str, Any]:
        destination.write_bytes(source.read_bytes() + b"-masked")
        return {"frames": 1, "frames_with_face": 1, "faces_total": 1}

    def trim(source: Path, duration_s: float, destination: Path) -> dict[str, Any]:
        events.append(f"trim-{duration_s}")
        destination.write_bytes(source.read_bytes())
        return {"fps": 24, "duration_s": duration_s}

    def concat(
        paths: list[Path],
        destination: Path,
        *,
        target_durations: list[float] | None = None,
        join_modes: list[str] | None = None,
    ) -> dict[str, Any]:
        assert target_durations == [4.2, 5.0]
        assert join_modes == ["hard_cut", "hard_cut"]
        destination.write_bytes(b"".join(path.read_bytes() for path in paths))
        return {"fps": 24}

    def assemble(
        lines: list[dict[str, Any]],
        total_duration_s: float,
        destination: Path,
    ) -> dict[str, Any]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            b"".join(line["audio"].read_bytes() for line in lines)
        )
        return {"duration_s": total_duration_s, "lines": list(lines)}

    def slice_track(
        source: Path,
        start_s: float,
        duration_s: float,
        destination: Path,
    ) -> dict[str, Any]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return {"stdout": "", "stderr": ""}

    def replace_audio(
        video: Path, audio: Path, destination: Path,
    ) -> dict[str, Any]:
        destination.write_bytes(video.read_bytes() + b"|tts-track")
        return {"duration_s": 9.2}

    monkeypatch.setattr(media, "probe", probe)
    monkeypatch.setattr(media, "extract_frames_4fps", frames, raising=False)
    monkeypatch.setattr(media, "extract_audio", extract_audio)
    monkeypatch.setattr(
        media,
        "yamnet_analyze",
        lambda _audio, _model: {
            "windows": [{
                "start_ms": 0,
                "end_ms": 2000,
                "spoken": 0.9,
                "sung": 0.0,
                "music": 0.0,
            }],
            "has_bgm": False,
        },
    )
    monkeypatch.setattr(media, "transcribe", transcribe)
    monkeypatch.setattr(media, "filter_spoken_lines", filter_spoken, raising=False)
    monkeypatch.setattr(
        media,
        "detect_scenes",
        lambda _source, _duration: [{"index": 1, "frames": [{}]}],
    )
    monkeypatch.setattr(media, "plan_segments", planned)
    monkeypatch.setattr(media, "select_segment_keyframes", selection, raising=False)
    monkeypatch.setattr(media, "cut_segment", cut)
    monkeypatch.setattr(media, "materialize_keyframes", materialize, raising=False)
    monkeypatch.setattr(media, "half_resolution_png", half, raising=False)
    monkeypatch.setattr(
        media, "provider_reference_png", provider_reference, raising=False,
    )
    monkeypatch.setattr(media, "localize_dialogue", localize, raising=False)
    monkeypatch.setattr(media, "mute", mute)
    monkeypatch.setattr(face_mask, "mask_faces", mask_faces)
    monkeypatch.setattr(media, "trim_24fps", trim, raising=False)
    monkeypatch.setattr(media, "concat", concat)
    monkeypatch.setattr(
        media, "assemble_dialogue_track", assemble, raising=False,
    )
    monkeypatch.setattr(
        media, "slice_dialogue_track", slice_track, raising=False,
    )
    monkeypatch.setattr(
        media, "replace_audio_track", replace_audio, raising=False,
    )


def test_full_pipeline_uses_clean_structured_index_barrier_and_split_poll_retry(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request(
        target_language="日语",
        replacement_instruction="把主角替换成用户参考形象",
    )
    project_id, project, _state = store.create(request, "request-1")
    source = project / "inputs" / "source.mp4"
    source.write_bytes(b"source")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"replacement")
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
    )
    media_events: list[str] = []
    _install_fake_media(monkeypatch, media_events)
    provider_reference_calls: list[tuple[Path, Path]] = []
    provider_reference_png = media.provider_reference_png

    def tracked_provider_reference(source: Path, destination: Path) -> Path:
        provider_reference_calls.append((source, destination))
        return provider_reference_png(source, destination)

    monkeypatch.setattr(
        media, "provider_reference_png", tracked_provider_reference,
    )
    deepseek = _FlowDeepSeek()
    picsart = _FlowPicsart()
    mediakit = _FlowMediaKit()
    seedream = _FlowSeedream(deepseek.events)
    autodl = _FlowAutodlTTS()
    registered: list[Path] = []

    def register(_project_id: str, path: Path) -> str:
        registered.append(path)
        return f"https://assets.invalid/{path.parent.name}/{path.name}"

    pipeline = Pipeline(
        _settings(tmp_path, enable_mediakit=True),
        store,
        register,
        deepseek=deepseek,
        mediakit=mediakit,
        picsart=picsart,
        seedream=seedream,
        autodl=autodl,
    )

    pipeline.run(project_id)

    state = store.read(project_id)
    canonical_reference = (
        project / "work" / "global-reference-sheet" / "user-reference.png"
    )
    assert state["status"] == "succeeded"
    assert state["project_progress"] == {
        "status": "succeeded",
        "percent": 100,
        "stage": "completed",
    }
    assert len(picsart.submits) == 2
    assert len(seedream.calls) == 1
    assert len(mediakit.calls) == 18
    assert all(
        call == ["full_screen_text_erase", "full_screen_icon_erase"]
        for call in mediakit.calls
    )
    assert sorted(call["duration"] for call in picsart.submits) == [5, 5]
    assert picsart.poll_counts["task-1"] == 2
    assert picsart.poll_counts["task-2"] == 1
    assert media_events == ["trim-4.2", "trim-5.0"] or sorted(media_events) == [
        "trim-4.2",
        "trim-5.0",
    ]
    # Both segments have replaced people: every reference video is
    # face-masked before submission.
    assert sorted(
        scope.name
        for scope in (project / "logs" / "mask-reference-faces").iterdir()
    ) == ["segment-0001", "segment-0002"]
    assert [
        path.name for path in registered
    ].count("source-faces-masked.mp4") == 2

    visual_calls = [
        call for call in deepseek.structured_calls
        if call["schema"] == VISUAL_PROMPT_SCHEMA
    ]
    index_calls = [
        call for call in deepseek.structured_calls
        if call["payload"].get("phase") == "project_index"
    ]
    fusion_calls = [
        call for call in deepseek.structured_calls
        if (
            ((call["schema"].get("properties") or {}).get("schema") or {}).get("enum")
            == [FUSION_OUTPUT_NAME]
        )
    ]
    object_replace_calls = [
        call for call in deepseek.structured_calls
        if (
            ((call["schema"].get("properties") or {}).get("schema") or {}).get("enum")
            == [OBJECT_REPLACE_OUTPUT_NAME]
        )
    ]
    translation_calls = [
        call for call in deepseek.structured_calls
        if call["schema"] == TRANSLATION_SCHEMA
    ]
    assert len(visual_calls) == 2
    assert len(index_calls) == 1
    assert len(fusion_calls) == 2
    assert len(object_replace_calls) == 1
    assert len(translation_calls) == 1
    assert all(
        "submit_result" not in call["instruction"]
        for call in deepseek.structured_calls
    )
    assert all("JSON Schema" in call["instruction"] for call in visual_calls)
    assert "JSON Schema" in translation_calls[0]["instruction"]
    assert translation_calls[0]["schema"]["additionalProperties"] is False
    assert translation_calls[0]["schema"]["properties"]["lines"]["items"]["additionalProperties"] is False
    assert "像素帧是视觉事实的唯一依据" in index_calls[0]["instruction"]
    assert "visual_prompt 只用于汇总逐段已发现对象" in index_calls[0]["instruction"]
    assert [
        segment["segment_index"]
        for segment in index_calls[0]["payload"]["segments"]
    ] == [1, 2]
    index_schema = index_calls[0]["schema"]
    for category in ("people", "entities", "scenes"):
        assert (
            index_schema["properties"][category]["items"]["properties"]
            ["occurrences"]["items"]["properties"]["segment_index"]["enum"]
        ) == [1, 2]
    assert (
        index_schema["properties"]["relations"]["items"]["properties"]
        ["occurrences"]["items"]["properties"]["segment_index"]["enum"]
    ) == [1, 2]
    assert sorted(call["payload"]["segment_index"] for call in fusion_calls) == [1, 2]
    object_replace_call = object_replace_calls[0]
    compact_catalog = object_replace_call["payload"]["voice_catalog_lines"]
    assert len(compact_catalog.splitlines()) == 30
    assert len(compact_catalog) < 3500
    assert "{" not in compact_catalog
    assert all(" - " in line for line in compact_catalog.splitlines())
    assert object_replace_call["payload"]["user_replacement_instruction"] == (
        "把主角替换成用户参考形象"
    )
    assert object_replace_call["payload"]["has_user_image"] is True
    assert object_replace_call["payload"]["user_replacement_image"] == {
        "name": "Image 1",
        "sha256": Pipeline._sha256(canonical_reference),
    }
    raw_object_index = state["element_index"]
    assert object_replace_call["payload"]["replacement_set"] == {
        category: sorted(raw_object_index[category])
        for category in ("people", "entities", "scenes")
    }
    planning_index = object_replace_call["payload"]["global_element_index"]
    assert object_replace_call["payload"]["segment_visual_prompts"] == [
        {"segment_index": 1, "visual_prompt": "visual-1"},
        {"segment_index": 2, "visual_prompt": "visual-2"},
    ]
    assert planning_index["relations"] == raw_object_index["relations"]
    for category in ("people", "entities", "scenes"):
        for stable_id, record in planning_index[category].items():
            assert "preserve" not in record
            assert "replaceable" not in record
            expected = dict(raw_object_index[category][stable_id])
            expected.pop("preserve", None)
            expected.pop("replaceable", None)
            assert record == expected
    assert object_replace_call["images"] == [canonical_reference]
    assert deepseek.events.index("project-index") < deepseek.events.index(
        "object-replace"
    )
    assert deepseek.events.index("object-replace") < min(
        deepseek.events.index("branch-start-1"),
        deepseek.events.index("branch-start-2"),
    )
    assert deepseek.events.index("object-replace") < deepseek.events.index(
        "seedream"
    )
    assert deepseek.events.index("seedream") < min(
        deepseek.events.index("branch-start-1"),
        deepseek.events.index("branch-start-2"),
    )
    assert all(call["schema"]["additionalProperties"] is False for call in fusion_calls)
    assert all(call["images"] == [] for call in fusion_calls)
    assert all(
        set(call["payload"]) == {
            "segment_index",
            "visual_prompt",
            "global_element_index",
            "translated_dialogue",
            "h3_reference_images",
            "replacement_sheet",
            "authoritative_constraints",
        }
        for call in fusion_calls
    )
    assert all(
        call["payload"]["authoritative_constraints"].startswith(
            "BEGIN AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN\n"
        )
        for call in fusion_calls
    )
    assert all(
        "user_replacement_instruction" not in json.dumps(
            call["payload"], ensure_ascii=False,
        )
        and "user_replacement_image" not in json.dumps(
            call["payload"], ensure_ascii=False,
        )
        and "selected_voice_reference" not in json.dumps(
            call["payload"], ensure_ascii=False,
        )
        for call in fusion_calls
    )
    assert all(
        call["payload"]["generation_config"] == {
            "remove_subtitle": True,
            "remove_watermark": True,
        }
        for call in visual_calls
    )
    index_payload = index_calls[0]["payload"]
    assert index_payload["generation_config"] == {
        "remove_subtitle": True,
        "remove_watermark": True,
    }
    assert [
        (segment["segment_index"], segment["visual_prompt"])
        for segment in index_payload["segments"]
    ] == [(1, "visual-1"), (2, "visual-2")]
    expected_index_images = []
    for segment in index_payload["segments"]:
        assert set(segment) == {"segment_index", "visual_prompt", "frames"}
        assert len(segment["frames"]) == 9
        assert [frame["frame_order"] for frame in segment["frames"]] == list(
            range(1, 10)
        )
        assert all(
            set(frame) == {"frame_order", "path", "sha256"}
            for frame in segment["frames"]
        )
        for frame in segment["frames"]:
            frame_path = project / frame["path"]
            expected_index_images.append(frame_path)
            assert frame["sha256"] == Pipeline._sha256(frame_path)
    assert index_calls[0]["images"] == expected_index_images
    raw_index = state["element_index"]
    enhanced_index = state["enhanced_element_index"]
    assert all(
        json.loads(json.dumps(call["payload"]["global_element_index"]))
        == enhanced_index
        for call in fusion_calls
    )
    assert set(raw_index) == {
        "people", "entities", "scenes", "relations",
    }
    assert list(raw_index["people"]) == [
        "person-01", "person-02", "person-03",
    ]
    assert json.loads(
        (project / "work" / "element_index.json").read_text(encoding="utf-8")
    ) == raw_index
    assert json.loads(
        (project / "work" / "enhanced_element_index.json").read_text(
            encoding="utf-8",
        )
    ) == enhanced_index
    assert enhanced_index != raw_index
    assert all(
        "replacement" not in record and "voice" not in record
        for category in ("people", "entities", "scenes")
        for record in raw_index[category].values()
    )
    assert enhanced_index["people"]["person-01"]["replacement"]["source"] == (
        "image_1"
    )
    assert state["object_replace"]["version"] == OBJECT_REPLACE_OUTPUT_VERSION
    replacement_sheet = state["replacement_sheet"]
    assert replacement_sheet["status"] == "succeeded"
    assert replacement_sheet["h3_image_reference"] == "Image 1"
    canonical_receipt = Pipeline._project_artifact_receipt(
        project,
        canonical_reference,
        "work/global-reference-sheet/user-reference.png",
    )
    assert replacement_sheet["h3_reference_images"] == [
        {
            "image_reference": "Image 1",
            "role": "global_replacement_sheet",
            **replacement_sheet["output"],
            "spec_sha256": replacement_sheet["spec_sha256"],
        },
        {
            "image_reference": "Image 2",
            "role": "user_replacement",
            **canonical_receipt,
        },
    ]
    assert [
        (tile["tile_id"], tile["stable_key"])
        for tile in replacement_sheet["spec"]["tiles"]
    ] == [
        ("TILE_01", "person-01"),
        ("TILE_02", "person-02"),
        ("TILE_03", "person-03"),
        ("TILE_04", "entity-01"),
        ("TILE_05", "scene-01"),
    ]
    assert all(
        tile["stable_key"] != "relation-01"
        for tile in replacement_sheet["spec"]["tiles"]
    )
    assert [path.name for path in seedream.calls[0]["images"]] == [
        "canvas.png", "user-reference.png",
    ]
    assert [
        receipt["path"] for receipt in replacement_sheet["seedream"]["inputs"]
    ] == [
        "work/global-reference-sheet/canvas.png",
        "work/global-reference-sheet/user-reference.png",
    ]
    expected_image_urls = [
        "https://assets.invalid/global-reference-sheet/reference-sheet.png",
        "https://assets.invalid/global-reference-sheet/user-reference.png",
    ]
    assert all(
        call["image_urls"] == expected_image_urls for call in picsart.submits
    )
    assert provider_reference_calls == [(replacement, canonical_reference)]
    registered_images = [
        path for path in registered
        if path.name in {"user-reference.png", "reference-sheet.png"}
    ]
    assert registered_images.count(canonical_reference.resolve()) == 2
    assert registered_images.count(
        (project / "work" / "global-reference-sheet" / "reference-sheet.png")
        .resolve()
    ) == 2
    assert [reference["audio_reference"] for reference in state["voice_references"]] == [
        "Audio 1", "Audio 2", "Audio 3",
    ]
    assert all(
        (project / reference["asset_path"]).is_file()
        for reference in state["voice_references"]
    )
    # The only translated line is attributed to person-01 and synthesized
    # with that person's Audio 1 voice.
    assert [call["text"] for call in autodl.tts_calls] == ["translated line"]
    assert [call["voice_path"].name for call in autodl.tts_calls] == [
        "audio-01.wav",
    ]
    full_track = project / "work" / "dialogue-audio" / "full-track.wav"
    assert full_track.is_file()
    assert (project / "work" / "dialogue-audio" / "line-0001.wav").is_file()
    segment_slice = (
        project / "work" / "segments" / "0001" / "dialogue-track.wav"
    )
    assert segment_slice.is_file()
    assert not (
        project / "work" / "segments" / "0002" / "dialogue-track.wav"
    ).exists()
    # H3 receives the per-segment dialogue speech slice as Audio 1, not the
    # voice-library wavs; the segment without dialogue gets no audio input.
    assert sorted(call["audio_urls"] for call in picsart.submits) == [
        [],
        ["https://assets.invalid/0001/dialogue-track.wav"],
    ]
    assert state["enhanced_element_index"]["dialogue_voices"] == ["person-01"]
    assert (project / "output" / "generated.mp4").read_bytes().endswith(
        b"|tts-track"
    )
    ledger = state["picsart_submission_receipts"]
    preparation_sha256 = Pipeline._json_sha256(
        state["h3_prepared_artifacts"]
    )
    assert ledger["version"] == 2
    assert all(
        receipt["preparation_sha256"] == preparation_sha256
        for records in ledger["segments"].values()
        for receipt in records
    )
    serialized_index = json.dumps(raw_index)
    for forbidden in ("provider_record", '"id"', "subject_id", "object_id"):
        assert forbidden not in serialized_index

    for call in fusion_calls:
        segment_index = call["payload"]["segment_index"]
        prompt = (
            project / "work" / "segments" / f"{segment_index:04d}"
            / "h3_prompt.txt"
        ).read_text(encoding="utf-8")
        segment_audio: list[str] = []
        requires_image_1 = False
        for category in ("people", "entities", "scenes"):
            for stable_id, record in enhanced_index[category].items():
                if not any(
                    occurrence["segment_index"] == segment_index
                    for occurrence in record["occurrences"]
                ):
                    continue
                replacement_plan = record["replacement"]
                if replacement_plan is not None:
                    assert stable_id in prompt
                    requires_image_1 = (
                        requires_image_1
                        or replacement_plan["source"] == "image_1"
                    )
                if category == "people":
                    audio_reference = record["voice"]["audio_reference"]
                    if audio_reference not in segment_audio:
                        segment_audio.append(audio_reference)
        assert requires_image_1 is True
        assert "Image 1" in prompt
        assert "Image 2" in prompt
        assert segment_audio
        assert all(reference in prompt for reference in segment_audio)
        assert prompt.count(AUTHORITY_BEGIN) == 1
        assert prompt.count(AUTHORITY_END) == 1
        for stable_id, person in enhanced_index["people"].items():
            if not any(
                occurrence["segment_index"] == segment_index
                for occurrence in person["occurrences"]
            ):
                continue
            authority_line = next(
                line for line in prompt.splitlines()
                if line.startswith(f"- {stable_id} | REPLACE |")
            )
            assert (
                f"| voice: {person['voice']['audio_reference']} |"
                in authority_line
            )
            tile = next(
                tile for tile in replacement_sheet["spec"]["tiles"]
                if tile["stable_key"] == stable_id
            )
            assert f"reference: Image 1, {tile['tile_id']} " in authority_line

    translation_output = json.loads(
        (project / "logs" / "deepseek-translation" / "project" / "attempt-1" / "output.json").read_text(encoding="utf-8")
    )
    assert translation_output["value"][0]["text"] == "translated line"
    assert translation_output["provider_record"]["response"] == {
        "api": "responses", "phase": "translation",
    }

    poll_root = project / "logs" / "picsart-poll"
    retry_scopes = [
        scope for scope in poll_root.iterdir()
        if (scope / "attempt-2").is_dir()
    ]
    assert len(retry_scopes) == 1
    assert (retry_scopes[0] / "attempt-1" / "output.json").is_file()
    assert (retry_scopes[0] / "attempt-1" / "error.json").is_file()
    assert stat.S_IMODE(
        (retry_scopes[0] / "attempt-1" / "output.json").stat().st_mode
    ) == 0o600
    assert not any(
        (scope / "attempt-2").exists()
        for scope in (project / "logs" / "picsart-submit").iterdir()
    )


def test_h3_prompt_fusion_selects_codex_and_keeps_schema_binding(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(fusion_agent="codex"), "codex-fusion",
    )
    segment_dir = project / "work" / "segments" / "segment-001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    source.write_bytes(b"source")
    calls: list[dict[str, Any]] = []
    library = load_voice_library()
    voice = library.voices[0]
    voice_directory = project / "work" / "voice-references"
    voice_directory.mkdir(parents=True)
    voice_reference = voice_directory / "audio-01.wav"
    voice_reference.write_bytes(library.files[voice["voice_id"]].read_bytes())
    enhanced_index = {
        "people": {
            "person-01": {
                "source_visual_description": "a visible person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replaceable": ["face", "clothing"],
                "preserve": ["motion"],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a different visible person",
                    "appearance_description": "an isolated different visible person",
                },
                "voice": {
                    "audio_reference": "Audio 1",
                    "voice_id": voice["voice_id"],
                    "asset_path": "work/voice-references/audio-01.wav",
                    "asset_sha256": voice["sha256"],
                    "features": library.model_line(voice["voice_id"]),
                },
            },
        },
        "entities": {},
        "scenes": {
            "scene-01": {
                "source_visual_description": "a visible kitchen",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replaceable": ["layout"],
                "preserve": ["lighting"],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a replacement kitchen",
                    "appearance_description": "an empty replacement kitchen",
                },
            },
        },
        "relations": {},
    }

    class FakeCodex:
        def complete_structured(
            self,
            instruction: str,
            payload: dict[str, Any],
            schema: dict[str, Any],
            images: list[Path],
            validator: Any = None,
        ) -> ProviderResult[dict[str, Any]]:
            calls.append({
                "instruction": instruction,
                "payload": payload,
                "schema": schema,
                "images": images,
            })
            return ProviderResult({
                "schema": FUSION_OUTPUT_NAME,
                "version": 1,
                "input_sha256": schema["properties"]["input_sha256"]["enum"][0],
                "segment_index": 1,
                "prompt": "Preserve the source motion, camera, light, and timing",
            }, {"provider": "codex", "account": "acct2"})

    class RejectDeepSeek:
        def complete_structured(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("DeepSeek must not run for a Codex-frozen project")

    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: f"https://assets.invalid/{path.name}",
        codex=FakeCodex(),
        deepseek=RejectDeepSeek(),
        mediakit=object(),
        picsart=object(),
    )
    monkeypatch.setattr(
        media, "mute", lambda input_path, target: target.write_bytes(input_path.read_bytes()),
    )
    monkeypatch.setattr(
        face_mask,
        "mask_faces",
        lambda input_path, target, **_kwargs: (
            target.write_bytes(input_path.read_bytes() + b"-masked")
            or {"frames": 1, "frames_with_face": 1, "faces_total": 1}
        ),
    )
    replacement_sheet = {
        "spec_sha256": "s" * 64,
        "h3_image_reference": "Image 1",
        "h3_reference_images": [{
            "image_reference": "Image 1",
            "role": "global_replacement_sheet",
            "sha256": "f" * 64,
            "spec_sha256": "s" * 64,
        }],
        "user_image_present": False,
        "spec": {"tiles": [
            {
                "stable_key": "person-01",
                "tile_id": "TILE_01",
                "row": 1,
                "column": 1,
            },
            {
                "stable_key": "scene-01",
                "tile_id": "TILE_02",
                "row": 1,
                "column": 2,
            },
        ]},
    }
    result_item, result_prompt, result_muted, fusion_input_sha256 = (
        pipeline._prepare_h3_segment(
        project,
        {
            "index": 1,
            "source": source,
            "visual_prompt": "a person in a kitchen",
            "dialogue": [{"start_s": 0.0, "end_s": 1.0, "text": "你好"}],
            },
            enhanced_index,
            replacement_sheet,
            _effective_request(fusion_agent="codex"),
        ResearchLog(project),
        )
    )

    assert result_item["index"] == 1
    assert result_prompt.startswith("FACE REPLACEMENT IS MANDATORY")
    assert "Preserve the source motion" in result_prompt
    assert result_muted == segment_dir / "source-faces-masked.mp4"
    assert len(fusion_input_sha256) == 64
    assert len(calls) == 1
    assert calls[0]["payload"]["segment_index"] == 1
    assert calls[0]["payload"]["global_element_index"] == enhanced_index
    assert calls[0]["payload"]["h3_reference_images"] == (
        replacement_sheet["h3_reference_images"]
    )
    assert calls[0]["payload"]["replacement_sheet"] == {
        "spec_sha256": "s" * 64,
        "h3_image_reference": "Image 1",
        "tiles": replacement_sheet["spec"]["tiles"],
    }
    assert calls[0]["schema"]["additionalProperties"] is False
    assert calls[0]["images"] == []
    prompt = (segment_dir / "h3_prompt.txt").read_text(encoding="utf-8")
    assert prompt.startswith("FACE REPLACEMENT IS MANDATORY")
    assert "Preserve the source motion" in prompt
    assert prompt.index("FACE REPLACEMENT IS MANDATORY") < prompt.index(
        "Preserve the source motion"
    )
    assert "Audio 1" in prompt
    assert "reference: Image 1, TILE_01" in prompt
    assert "[0.000-1.000s] 你好" in prompt
    assert "Overlay removal:" in prompt
    assert "burned-in subtitles, captions, and on-screen text" in prompt
    assert "watermarks and logos" in prompt
    assert prompt.index("Overlay removal:") < prompt.index("[0.000-1.000s] 你好")


def test_h3_barrier_never_submits_if_any_fusion_preparation_fails(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "h3-barrier",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    submits: list[int] = []

    def prepare(
        _project: Path,
        item: dict[str, Any],
        *_args: Any,
    ) -> tuple[dict[str, Any], str, Path]:
        if item["index"] == 2:
            raise ValueError("fusion contract rejected segment 2")
        return item, "prompt", project / "work" / "muted.mp4"

    def submit(_project_id: str, item: dict[str, Any], *_args: Any, **_kwargs: Any) -> Path:
        submits.append(item["index"])
        return project / "output" / "unexpected.mp4"

    monkeypatch.setattr(pipeline, "_prepare_h3_segment", prepare)
    monkeypatch.setattr(pipeline, "_generate_picsart_segment", submit)

    with pytest.raises(ValueError, match="segment 2"):
        pipeline._run_h3_segments(
            project_id,
            project,
            [{"index": 1}, {"index": 2}],
            {},
            None,
            {},
            None,
            {},
            ResearchLog(project),
        )
    assert submits == []


def test_h3_without_user_image_uses_sheet_as_image_1_and_keeps_tile_authority(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, "sheet-only-image")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    muted = segment_dir / "source-muted.mp4"
    muted.write_bytes(b"muted")
    enhanced_element_index = {
        "people": {
            "person-01": {
                "source_visual_description": "the source person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a different person",
                    "appearance_description": "an isolated different person",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }
    registered: list[Path] = []
    seedream = _FlowSeedream()
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: (
            registered.append(path) or f"https://assets.invalid/{path.name}"
        ),
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
        seedream=seedream,
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )

    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [{"index": 1, "analysis_paths": []}],
            enhanced_element_index,
        )
    )
    assert replacement_sheet_path is not None
    assert len(seedream.calls) == 1
    assert replacement_sheet["h3_image_reference"] == "Image 1"
    assert replacement_sheet["h3_reference_images"] == [{
        "image_reference": "Image 1",
        "role": "global_replacement_sheet",
        **replacement_sheet["output"],
        "spec_sha256": replacement_sheet["spec_sha256"],
    }]
    assert [path.name for path in seedream.calls[0]["images"]] == [
        "canvas.png",
    ]
    assert [
        receipt["path"] for receipt in replacement_sheet["seedream"]["inputs"]
    ] == [
        "work/global-reference-sheet/canvas.png",
    ]
    authority = Pipeline._fusion_authority_block(
        enhanced_element_index,
        1,
        replacement_sheet,
    )
    assert "- person-01 | REPLACE |" in authority
    assert "reference: Image 1, TILE_01 (row 1, column 1)" in authority

    source = segment_dir / "source.mp4"
    (segment_dir / "h3_prompt.txt").write_text(
        "frozen prompt", encoding="utf-8",
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(
            {
                "index": 1,
                "duration_s": 5.0,
                "join_mode": "hard_cut",
                "source": source,
            },
            "frozen prompt",
            muted,
            "a" * 64,
        )],
        replacement_sheet=replacement_sheet,
        request=request,
    )
    prepared = pipeline._prepare_picsart_segment(
        project_id,
        project,
        {"index": 1, "duration_s": 5.0},
        None,
        replacement_sheet_path,
        request,
        ResearchLog(project),
        final_prompt="frozen prompt",
        muted=muted,
    )

    assert prepared.provider_images == (replacement_sheet_path.resolve(),)
    assert prepared.image_urls == (
        "https://assets.invalid/reference-sheet.png",
    )
    assert [path.name for path in registered] == [
        "source-muted.mp4", "reference-sheet.png",
    ]


def test_h3_does_not_send_an_unanchored_user_image_when_sheet_is_not_needed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, "unanchored-image")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"replacement")
    store.update(
        project_id,
        replacement_image_path="inputs/replacement-image",
    )
    registered: list[Path] = []
    seedream = _FlowSeedream()
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: (
            registered.append(path) or f"https://assets.invalid/{path.name}"
        ),
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
        seedream=seedream,
    )
    monkeypatch.setattr(
        media,
        "provider_reference_png",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an unanchored user image must not be materialized")
        ),
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    empty_index = {
        "people": {}, "entities": {}, "scenes": {}, "relations": {},
    }
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [{"index": 1, "analysis_paths": []}],
            empty_index,
            replacement=replacement,
        )
    )
    assert replacement_sheet_path is None
    assert replacement_sheet["status"] == "not_needed"
    assert replacement_sheet["user_image_present"] is False
    assert replacement_sheet["h3_reference_images"] == []
    raw_upload_receipt = replacement_sheet["upstream_bindings"][
        "replacement_image"
    ]
    assert raw_upload_receipt["path"] == "inputs/replacement-image"
    assert raw_upload_receipt["sha256"] == Pipeline._sha256(replacement)
    assert seedream.calls == []
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    muted = segment_dir / "source-muted.mp4"
    muted.write_bytes(b"muted")
    source = segment_dir / "source.mp4"
    (segment_dir / "h3_prompt.txt").write_text(
        "frozen prompt", encoding="utf-8",
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(
            {
                "index": 1,
                "duration_s": 5.0,
                "join_mode": "hard_cut",
                "source": source,
            },
            "frozen prompt",
            muted,
            "a" * 64,
        )],
        replacement=replacement,
        replacement_sheet=replacement_sheet,
        request=request,
    )

    prepared = pipeline._prepare_picsart_segment(
        project_id,
        project,
        {"index": 1, "duration_s": 5.0},
        replacement,
        None,
        request,
        ResearchLog(project),
        final_prompt="frozen prompt",
        muted=muted,
    )

    assert prepared.provider_images == ()
    assert prepared.image_urls == ()
    assert registered == [muted]


def test_resume_r2v_reuses_prompt_and_muted_segments_without_upstream_calls(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request(resolution="480p", aspect_ratio="9:16")
    project_id, project, _state = store.create(request, "resume-only")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"replacement")
    segments = [
        {
            "index": 1,
            "start_s": 0.0,
            "end_s": 4.2,
            "duration_s": 4.2,
            "chain_id": "chain-1",
            "join_mode": "hard_cut",
            "dialogue": [],
            "visual_prompt": "visual-1",
            "status": "running",
        },
        {
            "index": 2,
            "start_s": 4.2,
            "end_s": 9.2,
            "duration_s": 5.0,
            "chain_id": "chain-2",
            "join_mode": "hard_cut",
            "dialogue": [],
            "visual_prompt": "visual-2",
            "status": "running",
        },
    ]
    for segment in segments:
        segment_dir = project / "work" / "segments" / f"{segment['index']:04d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        (segment_dir / "source.mp4").write_bytes(b"source")
        (segment_dir / "source-muted.mp4").write_bytes(b"muted")
        (segment_dir / "h3_prompt.txt").write_text(
            f"frozen prompt {segment['index']}", encoding="utf-8",
        )
        mute_log = (
            project / "logs" / "mute-reference-video"
            / f"segment-{segment['index']:04d}" / "attempt-1"
        )
        mute_log.mkdir(parents=True)
        (mute_log / "output.json").write_text("{}", encoding="utf-8")
        fusion_log = (
            project / "logs" / "h3-max-r2v-prompt"
            / f"segment-{segment['index']:04d}" / "attempt-1"
        )
        fusion_log.mkdir(parents=True)
        (fusion_log / "output.json").write_text(
            json.dumps(f"frozen prompt {segment['index']}"),
            encoding="utf-8",
        )
    library = load_voice_library()
    voice_directory = project / "work" / "voice-references"
    voice_directory.mkdir(parents=True)
    references: list[dict[str, Any]] = []
    for slot, voice in enumerate(library.voices[:3], 1):
        asset_path = f"work/voice-references/audio-{slot:02d}.wav"
        destination = project / asset_path
        destination.write_bytes(
            library.files[voice["voice_id"]].read_bytes(),
        )
        references.append({
            "audio_reference": f"Audio {slot}",
            "voice_id": voice["voice_id"],
            "asset_path": asset_path,
            "asset_sha256": voice["sha256"],
            "features": library.model_line(voice["voice_id"]),
        })
    voice_state = {
        "voice_references": references,
        "object_replace": {
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "0" * 64,
        },
        "enhanced_element_index": {
            "people": {
                f"person-{slot:02d}": {
                    "source_visual_description": f"source person {slot}",
                    "occurrences": [
                        {"segment_index": 1, "frame_orders": [1]},
                        {"segment_index": 2, "frame_orders": [1]},
                    ],
                    "replacement": {
                        "reason": "automatic",
                        "source": "generated_variant",
                        "target_visual_description": (
                            f"replacement person {slot}"
                        ),
                        "appearance_description": (
                            f"isolated replacement person {slot}"
                        ),
                    },
                    "voice": dict(reference),
                }
                for slot, reference in enumerate(references, 1)
            },
            "entities": {},
            "scenes": {},
            "relations": {},
        },
    }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
        segment_count=2,
        segments=segments,
        project_progress={"status": "failed", "percent": 57, "stage": "pipeline_failed"},
        generation={"status": "failed", "stage": "pipeline_failed", "error": "failed"},
        **voice_state,
    )
    accepted = (
        project / "logs" / "picsart-submit"
        / "segment-0001" / "attempt-1"
    )
    accepted.mkdir(parents=True)
    (accepted / "output.json").write_text(json.dumps({
        "value": "existing-task",
        "provider_record": {
            "outcome": "accepted", "task_id": "existing-task",
        },
    }), encoding="utf-8")
    failed = (
        project / "logs" / "picsart-submit"
        / "segment-0002" / "attempt-1"
    )
    failed.mkdir(parents=True)
    (failed / "output.json").write_text(json.dumps({
        "value": "failed-task",
        "provider_record": {
            "outcome": "accepted", "task_id": "failed-task",
        },
    }), encoding="utf-8")
    terminal = (
        project / "logs" / "picsart-poll"
        / "segment-0002" / "attempt-1"
    )
    terminal.mkdir(parents=True)
    (terminal / "output.json").write_text(json.dumps({
        "provider_record": {
            "task_id": "failed-task",
            "polls": [{"response": {
                "status_code": 422,
                "body": {
                    "status": "error",
                    "reason": "minimax/h3-max/reference-to-video_failed",
                    "message": (
                        "minimax/h3-max/reference-to-video failed with message: "
                        "ValidationError: reference video download failed"
                    ),
                },
            }}],
        },
    }), encoding="utf-8")
    events: list[str] = []
    _install_fake_media(monkeypatch, events)
    picsart = _FlowPicsart()
    seedream = _FlowSeedream()
    registered: list[Path] = []
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: (
            registered.append(path) or f"https://assets.invalid/{path.name}"
        ),
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
        seedream=seedream,
    )
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [dict(segment, analysis_paths=[]) for segment in segments],
            voice_state["enhanced_element_index"],
            replacement=replacement,
        )
    )
    assert replacement_sheet_path == (
        project / "work" / "global-reference-sheet"
        / "reference-sheet.png"
    )
    assert len(seedream.calls) == 1
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [
            (
                dict(
                    segment,
                    source=(
                        project / "work" / "segments"
                        / f"{segment['index']:04d}" / "source.mp4"
                    ),
                ),
                f"frozen prompt {segment['index']}",
                (
                    project / "work" / "segments"
                    / f"{segment['index']:04d}" / "source-muted.mp4"
                ),
                f"{segment['index']}" * 64,
            )
            for segment in segments
        ],
        replacement=replacement,
        replacement_sheet=replacement_sheet,
        request=request,
    )
    preparation_sha256 = Pipeline._json_sha256(
        store.read(project_id)["h3_prepared_artifacts"]
    )
    store.update(
        project_id,
        picsart_submission_receipts={
            "version": PICSART_RECEIPT_VERSION,
            "segments": {
                "1": [{
                    "segment_index": 1,
                    "run_identity": "picsart",
                    "request_sha256": "a" * 64,
                    "preparation_sha256": preparation_sha256,
                    "status": "accepted",
                    "task_id": "existing-task",
                }],
                "2": [{
                    "segment_index": 2,
                    "run_identity": "picsart",
                    "request_sha256": "b" * 64,
                    "preparation_sha256": preparation_sha256,
                    "status": "accepted",
                    "task_id": "failed-task",
                }],
            },
        },
    )
    pipeline.seedream = _RejectSeedream()
    monkeypatch.setattr(
        media,
        "provider_reference_png",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("canonical user PNG must not be regenerated")
        ),
    )

    pipeline.validate_r2v_resume(project_id)
    reconciled = store.read(project_id)["picsart_submission_receipts"]
    assert reconciled["segments"]["1"][0]["status"] == "accepted"
    assert reconciled["segments"]["2"][0]["status"] == "provider_failed"
    pipeline.resume_r2v(project_id)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    assert state["has_video"] is True
    assert len(picsart.submits) == 1
    assert [call["prompt"] for call in picsart.submits] == ["frozen prompt 2"]
    assert picsart.poll_counts["existing-task"] == 1
    assert "failed-task" not in picsart.poll_counts
    assert [
        receipt["status"]
        for receipt in state["picsart_submission_receipts"]["segments"]["2"]
    ] == ["provider_failed", "accepted"]
    expected_registered = {
        "source-muted.mp4", "reference-sheet.png",
    }
    assert picsart.submits[0]["image_urls"] == [
        "https://assets.invalid/reference-sheet.png",
    ]
    assert len(seedream.calls) == 1
    assert {path.name for path in registered} == expected_registered
    # Both frozen segments have no dialogue: no TTS slice is registered and
    # H3 receives no reference audio.
    assert picsart.submits[0]["audio_urls"] == []
    assert (project / "output" / "generated.mp4").is_file()
    assert (project / "logs" / "picsart-resume-submit").is_dir()
    assert not (
        project / "logs" / "picsart-resume-submit" / "segment-0001"
    ).exists()
    assert not any(
        (scope / "attempt-2").exists()
        for scope in (project / "logs" / "h3-max-r2v-prompt").iterdir()
    )


def test_resume_r2v_blocks_an_unknown_submit_receipt(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "unknown-submit",
    )
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "chain_id": "chain-1",
        "join_mode": "hard_cut",
        "dialogue": [],
        "visual_prompt": "frozen",
        "status": "failed",
    }
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "source.mp4").write_bytes(b"source")
    (segment_dir / "source-muted.mp4").write_bytes(b"muted")
    (segment_dir / "h3_prompt.txt").write_text("frozen", encoding="utf-8")
    mute_log = (
        project / "logs" / "mute-reference-video"
        / "segment-0001" / "attempt-1"
    )
    mute_log.mkdir(parents=True)
    (mute_log / "output.json").write_text("{}", encoding="utf-8")
    fusion_log = (
        project / "logs" / "h3-max-r2v-prompt"
        / "segment-0001" / "attempt-1"
    )
    fusion_log.mkdir(parents=True)
    (fusion_log / "output.json").write_text(
        json.dumps("frozen"), encoding="utf-8",
    )
    receipt = (
        project / "logs" / "picsart-submit" / "segment-0001" / "attempt-1"
    )
    receipt.mkdir(parents=True)
    (receipt / "output.json").write_text(
        json.dumps({
            "provider_record": {
                "outcome": "unknown",
                "request": {"method": "POST"},
            },
        }),
        encoding="utf-8",
    )
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[segment],
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    picsart = _FlowPicsart()
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
    )
    empty_index = {
        "people": {}, "entities": {}, "scenes": {}, "relations": {},
    }
    replacement_sheet, _sheet_path = _materialize_test_replacement_sheet(
        pipeline,
        project_id,
        project,
        [{**segment, "source": segment_dir / "source.mp4"}],
        empty_index,
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(
            {**segment, "source": segment_dir / "source.mp4"},
            "frozen",
            segment_dir / "source-muted.mp4",
            "a" * 64,
        )],
        replacement_sheet=replacement_sheet,
        request=_effective_request(),
    )
    preparation_sha256 = Pipeline._json_sha256(
        store.read(project_id)["h3_prepared_artifacts"],
    )
    store.update(
        project_id,
        picsart_submission_receipts={
            "version": PICSART_RECEIPT_VERSION,
            "segments": {
                "1": [{
                    "segment_index": 1,
                    "run_identity": "picsart",
                    "request_sha256": "b" * 64,
                    "preparation_sha256": preparation_sha256,
                    "status": "unknown",
                    "task_id": None,
                }],
            },
        },
    )

    with pytest.raises(ValueError, match="outcome is unknown"):
        pipeline.validate_r2v_resume(project_id)
    assert picsart.submits == []


def test_same_language_translation_is_an_explicit_skipped_attempt(
    tmp_path: Path,
) -> None:
    class NoCallDeepSeek:
        def complete_structured(self, *_args: Any, **_kwargs: Any) -> ProviderResult[Any]:
            raise AssertionError("same-language translation must not call DeepSeek")

    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "request-2",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=NoCallDeepSeek(),
        mediakit=_UnusedMediaKit(),
        picsart=object(),
    )
    log = ResearchLog(project)
    source = [{"text": "原文", "start_s": 0.0, "end_s": 1.0}]

    translated = pipeline._node(
        log,
        "deepseek-translation",
        "project",
        {"target_language": SAME_AS_SOURCE_LANGUAGE, "source_lines": source},
        [],
        lambda: pipeline._translation_request(
            source, SAME_AS_SOURCE_LANGUAGE, "visual context",
        ),
        transform=lambda value: pipeline._normalize_translation(value, source),
    )

    assert translated == source
    output = json.loads(
        (project / "logs" / "deepseek-translation" / "project" / "attempt-1" / "output.json").read_text(encoding="utf-8")
    )
    assert output["provider_record"] == {
        "outcome": "skipped",
        "reason": "same_language",
    }
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert '"event": "node_skipped"' in trace


@pytest.mark.parametrize(
    "value",
    [
        {"lines": []},
        {"lines": [{"text": "   "}]},
        {"lines": "not-a-list"},
        {},
    ],
)
def test_translation_quality_is_not_a_local_blocker(value: Any) -> None:
    source = [{"text": "原文", "start_s": 0.0, "end_s": 1.0}]

    assert Pipeline._normalize_translation(value, source) == source


class _RecordingDeepSeek:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value
        self.calls: list[dict[str, Any]] = []

    def complete_structured(
        self,
        instruction: str,
        payload: Any,
        schema: dict[str, Any],
        images: list[Path] | None = None,
        validator: Any = None,
    ) -> ProviderResult[dict[str, Any]]:
        self.calls.append({
            "instruction": instruction,
            "payload": payload,
            "schema": schema,
            "images": list(images or []),
            "validator": validator,
        })
        return ProviderResult(
            self.value,
            {"response": {"api": "responses", "phase": "dialogue-generation"}},
        )


def _generation_prepared() -> list[dict[str, Any]]:
    return [
        {"index": 1, "start_s": 0.0, "end_s": 5.0, "visual_prompt": "开场画面"},
        {"index": 2, "start_s": 5.0, "end_s": 8.0, "visual_prompt": "结尾画面"},
    ]


def test_dialogue_generation_node_carries_brief_and_segment_context(
    tmp_path: Path,
) -> None:
    deepseek = _RecordingDeepSeek({
        "lines": [
            {"segment_index": 1, "text": "第一句台词呀"},
            {"segment_index": 1, "text": "第二句"},
            {"segment_index": 2, "text": "结尾"},
            {"segment_index": 99, "text": "非法段落"},
            {"segment_index": 2, "text": "   "},
        ],
    })
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(dialogue_mode="generate", brief="介绍产品"), "request-generate",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=deepseek,
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    prepared = _generation_prepared()

    generated = pipeline._node(
        log,
        "deepseek-dialogue-generation",
        "project",
        pipeline._dialogue_generation_payload("介绍产品", "中文", prepared),
        [],
        lambda: pipeline._dialogue_generation_request("介绍产品", "中文", prepared),
        transform=lambda value: pipeline._normalize_generated_dialogue(
            value, prepared,
        ),
    )

    assert len(deepseek.calls) == 1
    call = deepseek.calls[0]
    assert call["schema"] == DIALOGUE_GENERATION_SCHEMA
    assert call["payload"] == {
        "brief": "介绍产品",
        "target_language": "中文",
        "duration_s": 8.0,
        "segments": [
            {"index": 1, "start_s": 0.0, "end_s": 5.0, "visual_prompt": "开场画面"},
            {"index": 2, "start_s": 5.0, "end_s": 8.0, "visual_prompt": "结尾画面"},
        ],
    }
    assert "# dialogue-writer" in call["instruction"]
    assert "台词语言为中文" in call["instruction"]
    assert call["validator"] is _validate_dialogue_generation_shape
    # 段内多条按字数比例切分：第一句 6 字、第二句 3 字，共 9 字切 5 秒。
    assert generated == [
        {"start_s": 0.0, "end_s": pytest.approx(5.0 * 6 / 9), "text": "第一句台词呀"},
        {"start_s": pytest.approx(5.0 * 6 / 9), "end_s": 5.0, "text": "第二句"},
        {"start_s": 5.0, "end_s": 8.0, "text": "结尾"},
    ]
    output = json.loads(
        (project / "logs" / "deepseek-dialogue-generation" / "project" / "attempt-1" / "output.json").read_text(encoding="utf-8")
    )
    assert output["provider_record"] == {
        "response": {"api": "responses", "phase": "dialogue-generation"},
    }


def test_dialogue_generation_same_language_still_calls_deepseek(
    tmp_path: Path,
) -> None:
    deepseek = _RecordingDeepSeek({"lines": [{"segment_index": 1, "text": "口播"}]})
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(dialogue_mode="generate", brief="介绍产品"),
        "request-generate-same-language",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=deepseek,
        mediakit=object(),
        picsart=object(),
    )

    result = pipeline._dialogue_generation_request(
        "介绍产品", SAME_AS_SOURCE_LANGUAGE, _generation_prepared(),
    )

    assert result.value == {"lines": [{"segment_index": 1, "text": "口播"}]}
    assert len(deepseek.calls) == 1
    assert "没有源台词可抄" in deepseek.calls[0]["instruction"]


@pytest.mark.parametrize(
    "value",
    [
        {"lines": []},
        {"lines": [{"segment_index": 9, "text": "无此段"}]},
        {"lines": [{"segment_index": 1, "text": "   "}]},
        {"lines": [{"segment_index": "1", "text": "类型错误"}]},
        {"lines": "not-a-list"},
        {},
    ],
)
def test_generated_dialogue_drops_unusable_entries(value: Any) -> None:
    prepared = _generation_prepared()

    assert Pipeline._normalize_generated_dialogue(value, prepared) == []


@pytest.mark.parametrize(
    ("mode", "expected_node"),
    [
        ("auto_rewrite", "deepseek-translation"),
        ("generate", "deepseek-dialogue-generation"),
    ],
)
def test_continue_after_prepared_selects_dialogue_node_by_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_node: str,
) -> None:
    request = (
        _effective_request(dialogue_mode="generate", brief="介绍产品")
        if mode == "generate"
        else _effective_request()
    )
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(request, f"dialogue-mode-{mode}")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    calls: list[tuple[str, Any]] = []

    def spy(
        log: ResearchLog,
        node: str,
        scope: str,
        inputs: Any,
        input_media: list[Path],
        operation: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        calls.append((node, inputs))
        raise RuntimeError("stop after the dialogue node")

    monkeypatch.setattr(pipeline, "_node", spy)

    with pytest.raises(RuntimeError, match="stop after the dialogue node"):
        pipeline._continue_after_prepared(
            project_id, project, _generation_prepared(), [], None, request,
            ResearchLog(project),
        )

    assert [node for node, _inputs in calls] == [expected_node]
    if mode == "generate":
        assert calls[0][1]["brief"] == "介绍产品"
        assert calls[0][1]["duration_s"] == 8.0
        assert [
            segment["visual_prompt"] for segment in calls[0][1]["segments"]
        ] == ["开场画面", "结尾画面"]
        assert store.read(project_id)["project_progress"]["stage"] == (
            "dialogue-generation"
        )
    else:
        assert store.read(project_id)["project_progress"]["stage"] == "translation"


def test_no_audio_skips_audio_analysis_without_creating_fake_media(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "request-no-audio",
    )
    source = project / "inputs" / "source.mp4"
    source.write_bytes(b"silent-video")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    for name in (
        "extract_audio",
        "yamnet_analyze",
        "transcribe",
        "filter_spoken_lines",
    ):
        monkeypatch.setattr(
            media,
            name,
            lambda *_args, name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"{name} must not run without an audio track")
            ),
        )
    log = ResearchLog(project)

    yamnet, asr_lines, spoken_lines, decisions = (
        pipeline._analyze_audio_dialogue(
            project_id,
            project,
            source,
            has_audio=False,
            log=log,
        )
    )

    assert yamnet == {"windows": [], "has_bgm": False}
    assert asr_lines == []
    assert spoken_lines == []
    assert decisions == []
    assert not (project / "work" / "source.wav").exists()
    assert not (project / "work" / "voice_lines.json").exists()
    rows = [
        json.loads(line)
        for line in (project / "logs" / "trace.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    skipped = [row for row in rows if row["event"] == "node_skipped"]
    assert [row["node"] for row in skipped] == [
        "extract-audio",
        "yamnet",
        "whisper-asr",
        "spoken-dialogue-filter",
    ]
    for node in (row["node"] for row in skipped):
        output = json.loads(
            (
                project / "logs" / node / "project" / "attempt-1" / "output.json"
            ).read_text(encoding="utf-8")
        )
        assert output["provider_record"] == {
            "outcome": "skipped",
            "reason": "source_has_no_audio",
        }
        assert not (
            project / "logs" / node / "project" / "attempt-2"
        ).exists()


def test_project_index_normalizer_mechanically_maps_ids_without_metadata() -> None:
    normalized = normalize_project_index({
        "people": [{
            "key": "p",
            "source_visual_description": "person",
            "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
            "replaceable": [],
            "preserve": [],
        }],
        "entities": [{
            "key": "e",
            "source_visual_description": "cup",
            "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
            "replaceable": [],
            "preserve": [],
        }],
        "scenes": [{
            "key": "s",
            "source_visual_description": "kitchen",
            "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
            "replaceable": [],
            "preserve": [],
        }],
        "relations": [{
            "key": "r",
            "subject_key": "p",
            "predicate": "holds",
            "object_key": "e",
            "occurrences": [{
                "segment_index": 1,
                "frames": [{"frame_order": 1, "state": "held", "geometry": "hand"}],
            }],
            "preserve": [],
            "replace_together": False,
        }],
    })

    assert list(normalized["people"]) == ["person-01"]
    assert list(normalized["entities"]) == ["entity-01"]
    assert list(normalized["scenes"]) == ["scene-01"]
    assert list(normalized["relations"]) == ["relation-01"]
    relation = normalized["relations"]["relation-01"]
    assert relation["subject_key"] == "person-01"
    assert relation["object_key"] == "entity-01"
    serialized = json.dumps(normalized)
    assert '"id"' not in serialized
    assert "provider_record" not in serialized
    assert "subject_id" not in serialized


def test_nullable_object_replace_plan_is_compiled_without_a_quality_gate(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, _project, _state = store.create(
        _effective_request(), "nullable-plan",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    patch = {
        "people": {
            "person-01": {
                "matches_user_request": False,
                "replacement": None,
                "voice_id": None,
            },
        },
        "entities": {},
        "scenes": {
            "scene-01": {
                "matches_user_request": False,
                "replacement": None,
            },
        },
    }

    metadata, references, paths, fallback_used = (
        pipeline._materialize_voice_references(
            store.path(project_id), patch,
        )
    )
    authority = pipeline._fusion_authority_block({
        "people": {
            "person-01": {
                "occurrences": [{"segment_index": 1}],
                "replacement": None,
                "voice": None,
            },
        },
        "entities": {},
        "scenes": {
            "scene-01": {
                "occurrences": [{"segment_index": 1}],
                "replacement": None,
            },
        },
        "relations": {},
    }, 1)

    assert metadata == {}
    assert references == []
    assert paths == []
    assert fallback_used is False
    assert "person-01 | KEEP" in authority
    assert "scene-01 | KEEP" in authority
    assert "Audio " not in authority


def test_fusion_authority_replace_uses_only_safe_category_invariants() -> None:
    authority = Pipeline._fusion_authority_block({
        "people": {
            "person-01": {
                "source_visual_description": "the person at frame center",
                "occurrences": [{"segment_index": 1}],
                "preserve": ["facial identity", "original haircut"],
                "replacement": {
                    "source": "generated_variant",
                    "target_visual_description": (
                        "a new woman with an angular face and a braided bob"
                    ),
                    "appearance_description": (
                        "one isolated woman with an angular face and braided bob"
                    ),
                },
            },
            "person-02": {
                "source_visual_description": "only a forearm is visible",
                "occurrences": [{"segment_index": 1}],
                "preserve": ["source hand identity", "source skin texture"],
                "replacement": {
                    "source": "generated_variant",
                    "target_visual_description": (
                        "a different visible forearm with broader proportions, "
                        "warmer skin tone, and a small geometric birthmark"
                    ),
                    "appearance_description": (
                        "one isolated broader forearm with warm skin and a birthmark"
                    ),
                },
            },
        },
        "entities": {
            "entity-01": {
                "source_visual_description": "the held personal-care item",
                "occurrences": [{"segment_index": 1}],
                "preserve": ["toothbrush form", "white color"],
                "replacement": {
                    "source": "image_1",
                    "target_visual_description": (
                        "the black bear-shaped jewelry case shown in Image 1"
                    ),
                    "appearance_description": (
                        "the isolated black bear-shaped jewelry case from Image 1"
                    ),
                },
            },
            "entity-02": {
                "source_visual_description": "the background cup",
                "occurrences": [{"segment_index": 1}],
                "preserve": ["hand-painted logo", "glazed finish"],
                "replacement": None,
            },
            "entity-03": {
                "source_visual_description": "the background tray",
                "occurrences": [{"segment_index": 1}],
                "preserve": ["brushed metal finish"],
                "replacement": None,
            },
        },
        "scenes": {
            "scene-01": {
                "source_visual_description": "the bathroom",
                "occurrences": [{"segment_index": 1}],
                "preserve": ["original cabinet fronts"],
                "replacement": {
                    "source": "generated_variant",
                    "target_visual_description": (
                        "a bathroom with ribbed oak cabinets and terrazzo walls"
                    ),
                    "appearance_description": (
                        "an empty bathroom with ribbed oak cabinets and terrazzo walls"
                    ),
                },
            },
        },
        "relations": {
            "relation-01": {
                "subject_key": "person-01",
                "predicate": "holds the source toothbrush",
                "object_key": "entity-01",
                "occurrences": [{"segment_index": 1}],
                "preserve": [
                    "hand grip around the toothbrush handle",
                    "source brush-head angle",
                ],
                "replace_together": True,
            },
            "relation-02": {
                "subject_key": "entity-02",
                "predicate": "rests beside",
                "object_key": "entity-03",
                "occurrences": [{"segment_index": 1}],
                "preserve": ["cup logo faces the tray"],
                "replace_together": False,
            },
        },
    }, 1)

    lines = authority.splitlines()
    person_line = next(
        line for line in lines if line.startswith("- person-01 | REPLACE |")
    )
    partial_person_line = next(
        line for line in lines if line.startswith("- person-02 | REPLACE |")
    )
    entity_line = next(
        line for line in lines if line.startswith("- entity-01 | REPLACE |")
    )
    keep_line = next(
        line for line in lines if line.startswith("- entity-02 | KEEP |")
    )
    scene_line = next(
        line for line in lines if line.startswith("- scene-01 | REPLACE |")
    )
    relation_line = next(
        line for line in lines if line.startswith("- relation-01 | RELATION |")
    )
    keep_relation_line = next(
        line for line in lines if line.startswith("- relation-02 | RELATION |")
    )

    assert "a new woman with an angular face and a braided bob" in person_line
    assert "facial identity" not in person_line
    assert "original haircut" not in person_line
    assert "source role, action, pose, position, and continuity" in person_line
    assert "keep only explicitly stated gender and age group" in person_line
    assert "replace with a clearly different identity" in person_line
    assert "must differ visibly from the source in at least two of" in person_line
    assert "when clothing is visible, change its design" in person_line
    assert "slightly different" not in person_line

    assert "broader proportions" in partial_person_line
    assert "warmer skin tone" in partial_person_line
    assert "geometric birthmark" in partial_person_line
    assert "source hand identity" not in partial_person_line
    assert "source skin texture" not in partial_person_line
    assert "different face" not in partial_person_line
    assert "clothing design" not in partial_person_line
    assert "do not infer unseen traits" in partial_person_line

    assert "the black bear-shaped jewelry case shown in Image 1" in entity_line
    assert "toothbrush form" not in entity_line
    assert "white color" not in entity_line
    assert "source function, action interface, scale, contact" in entity_line

    assert "original cabinet fronts" not in scene_line
    assert "source scene type, color tone, composition" in scene_line
    assert "camera position and angle, shot movement" in scene_line
    assert "same scene type, use, and layout" in scene_line
    assert "clearly different instance and visible design details" in scene_line
    assert "slightly different" not in scene_line

    assert "hand-painted logo; glazed finish" in keep_line
    assert "endpoints: person-01 -> entity-01" in relation_line
    assert (
        "interaction type, contact, relative position, and motion continuity"
        in relation_line
    )
    assert "holds the source toothbrush" not in relation_line
    assert "toothbrush handle" not in relation_line
    assert "brush-head angle" not in relation_line

    assert "entity-02 rests beside entity-03" in keep_relation_line
    assert "cup logo faces the tray" in keep_relation_line


def test_fusion_authority_keeps_image_one_bound_to_its_matching_element() -> None:
    index = {
        "people": {
            "person-01": {
                "source_visual_description": "the presenter",
                "occurrences": [
                    {"segment_index": 1},
                    {"segment_index": 2},
                ],
                "replacement": {
                    "source": "generated_variant",
                    "target_visual_description": "one fixed new presenter design",
                    "appearance_description": "one isolated new presenter design",
                },
            },
        },
        "entities": {
            "entity-01": {
                "source_visual_description": "the user-selected cup",
                "occurrences": [{"segment_index": 1}],
                "replacement": {
                    "source": "image_1",
                    "target_visual_description": "the bear-shaped cup in Image 1",
                    "appearance_description": "the isolated bear-shaped cup in Image 1",
                },
            },
            "entity-02": {
                "source_visual_description": "the table lamp",
                "occurrences": [{"segment_index": 2}],
                "replacement": {
                    "source": "generated_variant",
                    "target_visual_description": "one fixed sculptural brass lamp",
                    "appearance_description": "one isolated sculptural brass lamp",
                },
            },
        },
        "scenes": {
            "scene-01": {
                "source_visual_description": "the warm kitchen",
                "occurrences": [
                    {"segment_index": 1},
                    {"segment_index": 2},
                ],
                "replacement": {
                    "source": "generated_variant",
                    "target_visual_description": "one fixed alternate warm kitchen",
                    "appearance_description": "one empty alternate warm kitchen",
                },
            },
        },
        "relations": {},
    }

    first = Pipeline._fusion_authority_block(index, 1)
    second = Pipeline._fusion_authority_block(index, 2)

    assert "entity-01 | REPLACE" in first
    assert "reference: Image 1" in first
    assert "entity-02" not in first
    assert "Image 1" not in second
    assert "entity-02 | REPLACE" in second
    assert "reference: generated variant" in second
    first_lines = first.splitlines()
    second_lines = second.splitlines()
    for stable_id in ("person-01", "scene-01"):
        first_line = next(
            line for line in first_lines if line.startswith(f"- {stable_id} |")
        )
        second_line = next(
            line for line in second_lines if line.startswith(f"- {stable_id} |")
        )
        assert first_line == second_line
    for stable_target in (
        "one fixed new presenter design",
        "one fixed alternate warm kitchen",
    ):
        assert stable_target in first
        assert stable_target in second


def test_fusion_authority_keeps_global_index_element_with_empty_occurrences() -> None:
    authority = Pipeline._fusion_authority_block({
        "people": {},
        "entities": {
            "entity-01": {
                "source_visual_description": "the indexed product",
                "occurrences": [],
                "replacement": {
                    "source": "generated_variant",
                    "target_visual_description": "one fixed new product design",
                    "appearance_description": "one isolated new product design",
                },
            },
        },
        "scenes": {},
        "relations": {},
    }, 7)

    assert "entity-01 | REPLACE" in authority
    assert "one fixed new product design" in authority


def test_node_records_the_failed_attempt_and_process_streams(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "request-3",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    calls = 0

    def fail() -> str:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise media.CommandError(["fake"], 1, f"stdout-{calls}", f"stderr-{calls}")
        return "ok"

    # CommandError (a non-zero ffmpeg exit) is transient: the node retries
    # until the command succeeds instead of escaping on the first failure.
    assert pipeline._node(log, "command", "project", {}, [], fail) == "ok"

    assert calls == 3
    for attempt in (1, 2):
        root = project / "logs" / "command" / "project" / f"attempt-{attempt}"
        assert (root / "error.json").is_file()
        assert (root / "stdout.log").read_text(encoding="utf-8") == f"stdout-{attempt}"
        assert (root / "stderr.log").read_text(encoding="utf-8") == f"stderr-{attempt}"
        assert stat.S_IMODE((root / "stdout.log").stat().st_mode) == 0o600
        assert stat.S_IMODE((root / "stderr.log").stat().st_mode) == 0o600


def test_pipeline_keeps_raw_failure_in_logs_but_not_public_state(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "request-public-error",
    )
    source = project / "inputs" / "source.mp4"
    source.write_bytes(b"source")
    store.update(project_id, source_path="inputs/source.mp4", has_source=True)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _source: (_ for _ in ()).throw(
            RuntimeError("private provider response and /private/path")
        ),
    )

    pipeline.run(project_id)

    state = store.read(project_id)
    assert state["status"] == "failed"
    assert state["generation"]["error"] == "pipeline_failed"
    assert state["error"] == {
        "code": "pipeline_failed",
        "message": "流水线执行失败",
    }
    assert "private provider response" not in json.dumps(state)
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "private provider response" in trace


def test_submit_node_retries_only_with_an_explicit_safe_policy(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "request-4",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    unknown_calls = 0

    def unknown() -> ProviderResult[str]:
        nonlocal unknown_calls
        unknown_calls += 1
        raise ProviderFailure(
            {"outcome": "unknown", "request": {"method": "POST"}},
            [],
            OSError("connection ended before response"),
        )

    with pytest.raises(ProviderFailure):
        pipeline._node(
            log,
            "picsart-submit",
            "unknown",
            {},
            [],
            unknown,
            retry_if=lambda _exc: False,
        )
    assert unknown_calls == 1
    assert not (
        project / "logs" / "picsart-submit" / "unknown" / "attempt-2"
    ).exists()

    rejected_calls = 0

    def rejected() -> None:
        nonlocal rejected_calls
        rejected_calls += 1
        raise ProviderFailure(
            {"outcome": "rejected", "response": {"status_code": 400}},
            [],
            RuntimeError("bad request"),
        )

    try:
        pipeline._node(
            log,
            "picsart-submit",
            "rejected",
            {},
            [],
            rejected,
            retry_if=lambda _exc: False,
        )
    except ProviderFailure:
        pass
    else:
        raise AssertionError("a rejected submit must escape")
    assert rejected_calls == 1
    assert not (
        project / "logs" / "picsart-submit" / "rejected" / "attempt-2"
    ).exists()


def test_pre_picsart_resume_rejects_state_receipt_without_log_evidence(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, _project, _state = store.create(
        _effective_request(), "state-only-receipt",
    )
    store.update(
        project_id,
        status="failed",
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        picsart_submission_receipts={
            "version": PICSART_RECEIPT_VERSION,
            "segments": {},
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )

    with pytest.raises(ValueError, match="submission evidence"):
        pipeline.validate_pre_picsart_resume(project_id)


@pytest.mark.parametrize(
    "failure", ["reference_png", "sheet_png", "asset_register"],
)
def test_global_picsart_preflight_failure_keeps_every_segment_zero_submit(
    tmp_path: Path,
    monkeypatch: Any,
    failure: str,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, f"preflight-{failure}")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"replacement")
    enhanced_element_index = {
        "people": {
            "person-01": {
                "source_visual_description": "a visible person",
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1]},
                    {"segment_index": 2, "frame_orders": [1]},
                ],
                "replacement": {
                    "reason": "automatic",
                    "source": (
                        "image_1"
                        if failure == "reference_png"
                        else "generated_variant"
                    ),
                    "target_visual_description": "a different visible person",
                    "appearance_description": "an isolated different visible person",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }
    store.update(
        project_id,
        **(
            {"replacement_image_path": "inputs/replacement-image"}
            if failure == "reference_png"
            else {}
        ),
        enhanced_element_index=enhanced_element_index,
        voice_references=[],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
    )
    items: list[dict[str, Any]] = []
    for index in (1, 2):
        segment_dir = project / "work" / "segments" / f"{index:04d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        source = segment_dir / "source.mp4"
        source.write_bytes(f"source-{index}".encode())
        items.append({
            "index": index,
            "start_s": float(index - 1) * 5,
            "end_s": float(index) * 5,
            "duration_s": 5.0,
            "join_mode": "hard_cut" if index == 1 else "continue",
            "source": source,
            "status": "running",
        })

    def prepare(
        _project: Path,
        item: dict[str, Any],
        *_args: Any,
    ) -> tuple[dict[str, Any], str, Path, str]:
        segment_dir = item["source"].parent
        prompt = f"frozen prompt {item['index']}"
        (segment_dir / "h3_prompt.txt").write_text(prompt, encoding="utf-8")
        muted = segment_dir / "source-muted.mp4"
        muted.write_bytes(f"muted-{item['index']}".encode())
        return item, prompt, muted, f"{item['index']}" * 64

    def provider_reference(_source: Path, destination: Path) -> Path:
        if failure == "reference_png":
            raise TypeError("canonical image conversion failed")
        destination.write_bytes(b"\x89PNG\r\n\x1a\nreference")
        return destination

    registered: list[Path] = []

    def register(_project_id: str, path: Path) -> str:
        registered.append(path)
        if (
            failure == "asset_register"
            and path.parent.name == "0002"
            and path.name == "source-muted.mp4"
        ):
            raise TypeError("segment 2 asset registration failed")
        return f"https://assets.invalid/{path.parent.name}/{path.name}"

    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    monkeypatch.setattr(media, "provider_reference_png", provider_reference)
    picsart = _FlowPicsart()
    seedream = _FlowSeedream()
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        register,
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
        seedream=seedream,
    )
    effective_replacement = replacement if failure == "reference_png" else None
    if failure == "reference_png":
        with pytest.raises(TypeError, match="canonical image conversion failed"):
            _materialize_test_replacement_sheet(
                pipeline,
                project_id,
                project,
                [dict(item, analysis_paths=[]) for item in items],
                enhanced_element_index,
                replacement=effective_replacement,
            )
        assert seedream.calls == []
        assert picsart.submits == []
        assert "h3_prepared_artifacts" not in store.read(project_id)
        return
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [dict(item, analysis_paths=[]) for item in items],
            enhanced_element_index,
            replacement=effective_replacement,
        )
    )
    assert replacement_sheet_path is not None
    assert len(seedream.calls) == 1
    if failure == "sheet_png":
        replacement_sheet_path.write_bytes(b"not-a-png")
    monkeypatch.setattr(pipeline, "_prepare_h3_segment", prepare)

    expected_error = (
        "artifact receipt does not match"
        if failure == "sheet_png"
        else "segment 2"
    )
    expected_type = ValueError if failure == "sheet_png" else TypeError
    with pytest.raises(expected_type, match=expected_error):
        pipeline._run_h3_segments(
            project_id,
            project,
            items,
            enhanced_element_index,
            effective_replacement,
            replacement_sheet,
            replacement_sheet_path,
            request,
            ResearchLog(project),
        )

    assert picsart.submits == []
    assert not (project / "logs" / "picsart-submit").exists()
    assert (
        "h3_prepared_artifacts" in store.read(project_id)
    ) is (failure == "asset_register")
    assert registered or failure in {"reference_png", "sheet_png"}


def test_seedream_failure_happens_before_any_picsart_submit(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "seedream-preflight",
    )
    enhanced_element_index = {
        "people": {
            "person-01": {
                "source_visual_description": "a visible person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a different visible person",
                    "appearance_description": "an isolated different visible person",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }

    class FailingSeedream:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_kwargs: Any) -> None:
            self.calls += 1
            raise TypeError("Seedream rejected the frozen sheet request")

    picsart = _FlowPicsart()
    seedream = FailingSeedream()
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
        seedream=seedream,
    )

    with pytest.raises(TypeError, match="Seedream rejected"):
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [{"index": 1, "analysis_paths": []}],
            enhanced_element_index,
        )

    assert seedream.calls == 1
    assert picsart.submits == []
    assert store.read(project_id)["replacement_sheet"]["status"] == "prepared"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "different-model"),
        ("prompt_sha256", "0" * 64),
        ("input_sha256", ["0" * 64]),
        ("request_sha256", "0" * 64),
    ],
)
def test_stored_sheet_rejects_seedream_receipt_binding_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), f"seedream-drift-{field}",
    )
    enhanced_element_index = {
        "people": {
            "person-01": {
                "source_visual_description": "the source person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a different person",
                    "appearance_description": "an isolated different person",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=_FlowSeedream(),
    )
    _materialize_test_replacement_sheet(
        pipeline,
        project_id,
        project,
        [{"index": 1, "analysis_paths": []}],
        enhanced_element_index,
    )
    receipt_path = (
        project / "work" / "global-reference-sheet" / "seedream-receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    state = store.read(project_id)
    replacement_sheet = json.loads(json.dumps(state["replacement_sheet"]))
    replacement_sheet["seedream"]["receipt"] = (
        pipeline._project_artifact_receipt(
            project,
            receipt_path,
            "work/global-reference-sheet/seedream-receipt.json",
        )
    )
    store.update(project_id, replacement_sheet=replacement_sheet)

    with pytest.raises(ValueError, match="Seedream receipt"):
        pipeline._validated_stored_replacement_sheet(
            project, store.read(project_id),
        )


def test_stored_sheet_rebuilds_tile_mapping_from_global_index(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "sheet-tile-rebind",
    )
    enhanced_element_index = {
        "people": {
            "person-01": {
                "source_visual_description": "the source person",
                "occurrences": [],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "one fixed new person",
                    "appearance_description": "one isolated fixed new person",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=_FlowSeedream(),
    )
    _materialize_test_replacement_sheet(
        pipeline,
        project_id,
        project,
        [{"index": 1, "analysis_paths": []}],
        enhanced_element_index,
    )
    state = store.read(project_id)
    changed_sheet = json.loads(json.dumps(state["replacement_sheet"]))
    changed_sheet["spec"]["tiles"][0]["stable_key"] = "person-99"
    changed_sheet["spec"]["spec_sha256"] = replacement_sheet_spec_sha256(
        changed_sheet["spec"]
    )
    changed_sheet["spec_sha256"] = changed_sheet["spec"]["spec_sha256"]
    store.update(project_id, replacement_sheet=changed_sheet)

    with pytest.raises(ValueError, match="tiles do not match the global index"):
        pipeline._validated_stored_replacement_sheet(
            project, store.read(project_id),
        )


def test_pre_picsart_resume_replays_prepared_sheet_without_upstream_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, project_id, _project, _sheet, initial_seedream = (
        _failed_replacement_sheet_checkpoint(tmp_path, prepared=True)
    )
    replay_seedream = _FlowSeedream()
    pipeline = Pipeline(
        _settings(
            tmp_path,
            seedream_api_key="",
            seedream_api_base="https://drifted-seedream.invalid/api/v9",
            seedream_model="drifted-model",
        ),
        store,
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("asset registration must be downstream of this test")
        ),
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=replay_seedream,
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_segment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("segment preparation must not rerun")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_continue_after_prepared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("project index and Object Replace must not rerun")
        ),
    )
    downstream: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        pipeline,
        "_continue_after_replacement_sheet",
        lambda *args, **_kwargs: downstream.append(args),
    )

    pipeline.validate_pre_picsart_resume(project_id)
    pipeline.resume_pre_picsart(project_id)

    assert len(initial_seedream.calls) == 1
    assert replay_seedream.calls == []
    assert len(replay_seedream.replay_calls) == 1
    assert len(downstream) == 1
    completed = store.read(project_id)["replacement_sheet"]
    assert completed["status"] == "succeeded"
    assert completed["seedream"]["endpoint"] == (
        f"{SEEDREAM_API_BASE}/images/generations"
    )


def test_pre_picsart_resume_reuses_succeeded_sheet_without_seedream_or_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, project_id, _project, frozen_sheet, _initial_seedream = (
        _failed_replacement_sheet_checkpoint(tmp_path, prepared=False)
    )
    pipeline = Pipeline(
        _settings(
            tmp_path,
            seedream_api_base="https://drifted-seedream.invalid/api/v9",
            seedream_model="drifted-model",
        ),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=_RejectSeedream(),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_segment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("segment preparation must not rerun")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_continue_after_prepared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("project index and Object Replace must not rerun")
        ),
    )
    downstream: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        pipeline,
        "_continue_after_replacement_sheet",
        lambda *args, **_kwargs: downstream.append(args),
    )

    pipeline.validate_pre_picsart_resume(project_id)
    pipeline.resume_pre_picsart(project_id)

    assert len(downstream) == 1
    assert store.read(project_id)["replacement_sheet"] == frozen_sheet


def test_pre_picsart_resume_backfills_default_voice_for_voiceless_frozen_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, project_id, project, frozen_sheet, _initial_seedream = (
        _failed_replacement_sheet_checkpoint(tmp_path, prepared=False)
    )
    # The frozen plan predates the object-replace voice fallback: dialogue
    # waits for TTS but no person was assigned a voice.
    assert store.read(project_id)["voice_references"] == []
    store.update(
        project_id,
        translated_voice_lines=[
            {"text": "第一句", "start_s": 0.0, "end_s": 2.0},
        ],
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=_RejectSeedream(),
    )
    downstream: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        pipeline,
        "_continue_after_replacement_sheet",
        lambda *args, **_kwargs: downstream.append(args),
    )

    pipeline.resume_pre_picsart(project_id)

    assert len(downstream) == 1
    voice_paths = downstream[0][4]
    fallback = pipeline.voice_library.voices[0]
    assert [path.name for path in voice_paths] == ["audio-01.wav"]
    assert voice_paths[0].is_file()
    state = store.read(project_id)
    assert [
        reference["voice_id"] for reference in state["voice_references"]
    ] == [fallback["voice_id"]]
    assert state["object_replace"]["voice_reference_fallback"] == {
        "voice_id": fallback["voice_id"],
        "reason": (
            "resume: dialogue requires TTS but the frozen plan assigned "
            "no voice"
        ),
    }
    # The frozen sheet keeps everything but its upstream bindings, which now
    # honestly reflect the added Audio 1 reference.
    updated_sheet = state["replacement_sheet"]
    assert {
        key: value
        for key, value in updated_sheet.items()
        if key != "upstream_bindings"
    } == {
        key: value
        for key, value in frozen_sheet.items()
        if key != "upstream_bindings"
    }
    updated_bindings = updated_sheet["upstream_bindings"]
    old_bindings = frozen_sheet["upstream_bindings"]
    assert updated_bindings["voice_references_sha256"] == (
        Pipeline._json_sha256(state["voice_references"])
    )
    assert (
        updated_bindings["voice_references_sha256"]
        != old_bindings["voice_references_sha256"]
    )
    assert (
        updated_bindings["segments_sha256"] == old_bindings["segments_sha256"]
    )
    fallback_events = [
        event
        for event in _trace_events(project)
        if event["event"] == "voice_reference_default_fallback"
    ]
    assert len(fallback_events) == 1
    assert fallback_events[0]["voice_id"] == fallback["voice_id"]
    assert fallback_events[0]["reason"].startswith("resume:")

    # The updated state is a consistent frozen set: the load gate passes and
    # a later resume (after another failure) does not backfill again.
    store.update(
        project_id,
        status="failed",
        project_progress={
            "status": "failed", "percent": 56, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    pipeline.validate_pre_picsart_resume(project_id)
    pipeline.resume_pre_picsart(project_id)
    assert len(downstream) == 2
    assert [path.resolve() for path in downstream[1][4]] == [
        path.resolve() for path in voice_paths
    ]
    assert [
        event
        for event in _trace_events(project)
        if event["event"] == "voice_reference_default_fallback"
    ] == fallback_events


def test_pre_picsart_resume_without_dialogue_keeps_the_empty_voice_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, project_id, project, frozen_sheet, _initial_seedream = (
        _failed_replacement_sheet_checkpoint(tmp_path, prepared=False)
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=_RejectSeedream(),
    )
    downstream: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        pipeline,
        "_continue_after_replacement_sheet",
        lambda *args, **_kwargs: downstream.append(args),
    )

    pipeline.resume_pre_picsart(project_id)

    assert len(downstream) == 1
    assert downstream[0][4] == []
    state = store.read(project_id)
    assert state["voice_references"] == []
    assert "voice_reference_fallback" not in state["object_replace"]
    assert state["replacement_sheet"] == frozen_sheet
    assert "voice_reference_default_fallback" not in (
        project / "logs" / "trace.jsonl"
    ).read_text(encoding="utf-8")


def test_pre_picsart_resume_reuses_not_needed_sheet_and_keeps_upload_unbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _effective_request()
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(request, "no-sheet-checkpoint")
    source = project / "inputs" / "source.mp4"
    source.write_bytes(b"project-source")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"raw-user-upload")
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
    )
    initial = Pipeline(
        _settings(tmp_path),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=_RejectSeedream(),
    )
    frozen_sheet, sheet_path = _materialize_test_replacement_sheet(
        initial,
        project_id,
        project,
        [{"index": 1, "analysis_paths": []}],
        {"people": {}, "entities": {}, "scenes": {}, "relations": {}},
        replacement=replacement,
    )
    assert sheet_path is None
    assert frozen_sheet["status"] == "not_needed"
    assert frozen_sheet["user_image_present"] is False
    raw_upload_receipt = frozen_sheet["upstream_bindings"]["replacement_image"]
    assert raw_upload_receipt["path"] == "inputs/replacement-image"
    assert raw_upload_receipt["sha256"] == Pipeline._sha256(replacement)
    store.update(
        project_id,
        status="failed",
        project_progress={
            "status": "failed", "percent": 55, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=_RejectSeedream(),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_segment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("segment preparation must not rerun")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_continue_after_prepared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("project index and Object Replace must not rerun")
        ),
    )
    downstream: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        pipeline,
        "_continue_after_replacement_sheet",
        lambda *args, **_kwargs: downstream.append(args),
    )

    pipeline.validate_pre_picsart_resume(project_id)
    pipeline.resume_pre_picsart(project_id)

    assert len(downstream) == 1
    assert store.read(project_id)["replacement_sheet"] == frozen_sheet


def test_pre_picsart_resume_submits_only_when_prepared_has_no_attempt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, project_id, project, _sheet, _initial_seedream = (
        _failed_replacement_sheet_checkpoint(tmp_path, prepared=True)
    )
    directory = project / "work" / "global-reference-sheet"
    for name in (
        "seedream-receipt.json",
        "seedream-receipt.json.post-claim",
        "seedream-receipt.json.response.json",
        "seedream-receipt.json.result",
        "provider-output.bin",
        "reference-sheet.png",
    ):
        (directory / name).unlink(missing_ok=True)
    seedream = _FlowSeedream()
    pipeline = Pipeline(
        _settings(tmp_path, seedream_api_key="configured-test-key"),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=seedream,
    )
    monkeypatch.setattr(
        pipeline,
        "_continue_after_prepared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("upstream nodes must not rerun")
        ),
    )
    downstream: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        pipeline,
        "_continue_after_replacement_sheet",
        lambda *args, **_kwargs: downstream.append(args),
    )

    pipeline.validate_pre_picsart_resume(project_id)
    pipeline.resume_pre_picsart(project_id)

    assert len(seedream.calls) == 1
    assert seedream.replay_calls == []
    assert [path.name for path in seedream.calls[0]["images"]] == [
        "canvas.png",
    ]
    assert len(downstream) == 1
    assert store.read(project_id)["replacement_sheet"]["status"] == "succeeded"


@pytest.mark.parametrize(
    "damage",
    ["orphan_claim", "corrupt_result", "missing_claim", "request"],
)
def test_pre_picsart_checkpoint_damage_fails_before_resume_writes_or_provider_calls(
    tmp_path: Path,
    damage: str,
) -> None:
    store, project_id, project, _sheet, _initial_seedream = (
        _failed_replacement_sheet_checkpoint(tmp_path, prepared=True)
    )
    directory = project / "work" / "global-reference-sheet"
    receipt_path = directory / "seedream-receipt.json"
    claim_path = directory / "seedream-receipt.json.post-claim"
    result_path = directory / "seedream-receipt.json.result"
    if damage == "orphan_claim":
        receipt_path.unlink()
    elif damage == "corrupt_result":
        result_path.write_bytes(_TEST_PNG[:-8])
    elif damage == "missing_claim":
        claim_path.unlink()
    else:
        state = store.read(project_id)
        changed = json.loads(json.dumps(state["replacement_sheet"]))
        changed["seedream"]["request_sha256"] = "0" * 64
        store.update(project_id, replacement_sheet=changed)

    seedream = _FlowSeedream()
    pipeline = Pipeline(
        _settings(tmp_path, seedream_api_key="configured-test-key"),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=seedream,
    )
    state_path = project / "state.json"
    trace_path = project / "logs" / "trace.jsonl"
    state_before = state_path.read_bytes()
    trace_before = trace_path.read_bytes()

    with pytest.raises(ValueError):
        pipeline.resume_pre_picsart(project_id)

    assert state_path.read_bytes() == state_before
    assert trace_path.read_bytes() == trace_before
    assert seedream.calls == []
    assert seedream.replay_calls == []


@pytest.mark.parametrize("terminal_status", ["rejected", "submission_unknown"])
def test_pre_picsart_resume_regenerates_a_terminal_seedream_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    store, project_id, project, _sheet, _initial_seedream = (
        _failed_replacement_sheet_checkpoint(tmp_path, prepared=True)
    )
    directory = project / "work" / "global-reference-sheet"
    receipt_path = directory / "seedream-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = terminal_status
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    seedream = _FlowSeedream()
    pipeline = Pipeline(
        _settings(tmp_path, seedream_api_key="configured-test-key"),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=seedream,
    )
    monkeypatch.setattr(
        pipeline,
        "_continue_after_prepared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("upstream nodes must not rerun")
        ),
    )
    downstream: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        pipeline,
        "_continue_after_replacement_sheet",
        lambda *args, **_kwargs: downstream.append(args),
    )

    # A terminally rejected/unknown Seedream attempt no longer blocks the
    # resume paths: its artifacts are reset and the frozen request re-POSTs.
    pipeline.validate_pre_picsart_resume(project_id)
    pipeline.resume_pre_picsart(project_id)

    assert len(seedream.calls) == 1
    assert seedream.replay_calls == []
    assert len(downstream) == 1
    sheet = store.read(project_id)["replacement_sheet"]
    assert sheet["status"] == "succeeded"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "succeeded"
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "seedream_receipt_reset" in trace
    assert f'"prior_status": "{terminal_status}"' in trace


def test_pre_picsart_resume_rejects_existing_h3_manifest_before_writes(
    tmp_path: Path,
) -> None:
    store, project_id, project, _sheet, _initial_seedream = (
        _failed_replacement_sheet_checkpoint(tmp_path, prepared=False)
    )
    store.update(
        project_id,
        h3_prepared_artifacts={"version": H3_PREPARED_ARTIFACTS_VERSION},
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda *_args: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=_RejectSeedream(),
    )
    state_path = project / "state.json"
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match="frozen H3 artifacts"):
        pipeline.resume_pre_picsart(project_id)

    assert state_path.read_bytes() == state_before


def _prepared_submit(
    store: ProjectStore,
    project_id: str,
    project: Path,
) -> PreparedPicsartRequest:
    manifest = {
        "version": H3_PREPARED_ARTIFACTS_VERSION,
        "bindings": {},
        "segments": {},
    }
    store.update(project_id, h3_prepared_artifacts=manifest)
    muted = project / "work" / "segments" / "0001" / "source-muted.mp4"
    muted.parent.mkdir(parents=True, exist_ok=True)
    muted.write_bytes(b"muted")
    payload = {
        "workflow": "minimax/h3-max/reference-to-video",
        "prompt": "frozen prompt",
        "reference_video_urls": ["https://assets.invalid/video.mp4"],
        "reference_image_urls": [],
        "reference_audio_urls": [],
        "resolution": "768P",
        "duration": 5,
        "aspect_ratio": "16:9",
        "prompt_expansion_mode": "quality",
    }
    return PreparedPicsartRequest(
        project_id=project_id,
        segment_index=1,
        scope="segment-0001",
        run_identity="picsart",
        artifact_suffix="",
        final_prompt="frozen prompt",
        muted=muted,
        provider_images=(),
        dialogue_track=None,
        video_url="https://assets.invalid/video.mp4",
        image_urls=(),
        audio_urls=(),
        provider_duration=5,
        resolution="768p",
        aspect_ratio="16:9",
        target_duration_s=5.0,
        request_sha256=Pipeline._json_sha256(payload),
        preparation_sha256=Pipeline._json_sha256(manifest),
    )


def test_submit_receipt_is_accepted_without_research_log_evidence(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "accepted-state-receipt",
    )

    class AcceptedPicsart:
        def submit(self, **_kwargs: Any) -> ProviderResult[str]:
            return ProviderResult(
                "paid-task",
                {"outcome": "accepted", "task_id": "paid-task"},
            )

    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=AcceptedPicsart(),
    )
    prepared = _prepared_submit(store, project_id, project)
    pipeline._begin_submission_batch([prepared])

    result = pipeline._submit_prepared_picsart(prepared)

    assert result.value == "paid-task"
    assert not (project / "logs" / "picsart-submit").exists()
    has_state_receipt, task_id = pipeline._state_submission_task(
        store.read(project_id), 1,
    )
    assert has_state_receipt is True
    assert task_id == "paid-task"


def test_picsart_receipt_v2_rejects_manifest_drift_before_task_reuse() -> None:
    manifest = {
        "version": H3_PREPARED_ARTIFACTS_VERSION,
        "bindings": {},
        "segments": {},
    }
    state = {
        "h3_prepared_artifacts": manifest,
        "picsart_submission_receipts": {
            "version": PICSART_RECEIPT_VERSION,
            "segments": {
                "1": [{
                    "segment_index": 1,
                    "run_identity": "picsart",
                    "request_sha256": "a" * 64,
                    "preparation_sha256": Pipeline._json_sha256(manifest),
                    "status": "accepted",
                    "task_id": "paid-task",
                }],
            },
        },
    }
    assert Pipeline._state_submission_task(state, 1) == (True, "paid-task")

    state["h3_prepared_artifacts"] = {
        "version": H3_PREPARED_ARTIFACTS_VERSION,
        "bindings": {"changed": True},
        "segments": {},
    }
    with pytest.raises(ValueError, match="receipt fields"):
        Pipeline._state_submission_task(state, 1)


def test_picsart_receipt_unsupported_version_is_rejected() -> None:
    state = {
        "h3_prepared_artifacts": {
            "version": H3_PREPARED_ARTIFACTS_VERSION,
            "bindings": {},
            "segments": {},
        },
        "picsart_submission_receipts": {
            "version": 1,
            "segments": {
                "1": [{
                    "segment_index": 1,
                    "run_identity": "picsart",
                    "request_sha256": "a" * 64,
                    "status": "accepted",
                    "task_id": "paid-task",
                }],
            },
        },
    }

    with pytest.raises(ValueError, match="receipt ledger is invalid"):
        Pipeline._state_submission_task(state, 1)


def test_submit_response_then_accepted_state_write_failure_becomes_unknown(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "accepted-write-crash",
    )
    calls = 0

    class AcceptedPicsart:
        def submit(self, **_kwargs: Any) -> ProviderResult[str]:
            nonlocal calls
            calls += 1
            return ProviderResult(
                "paid-task",
                {"outcome": "accepted", "task_id": "paid-task"},
            )

    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=AcceptedPicsart(),
    )
    prepared = _prepared_submit(store, project_id, project)
    pipeline._begin_submission_batch([prepared])
    original_update = store.update
    failed_accepted_write = False

    def update(project: str, **changes: Any) -> dict[str, Any]:
        nonlocal failed_accepted_write
        ledger = changes.get("picsart_submission_receipts")
        records = (
            ledger.get("segments", {}).get("1", [])
            if isinstance(ledger, dict) else []
        )
        if (
            not failed_accepted_write
            and records
            and records[-1].get("status") == "accepted"
        ):
            failed_accepted_write = True
            raise OSError("simulated crash before accepted receipt commit")
        return original_update(project, **changes)

    monkeypatch.setattr(store, "update", update)

    with pytest.raises(OSError, match="simulated crash"):
        pipeline._submit_prepared_picsart(prepared)

    assert calls == 1
    receipt = store.read(project_id)["picsart_submission_receipts"]["segments"]["1"][-1]
    assert receipt["status"] == "unknown"
    assert receipt["task_id"] is None
    with pytest.raises(ValueError, match="outcome is unknown"):
        pipeline._state_submission_task(store.read(project_id), 1)


def test_poll_http_4xx_is_not_provider_task_terminal_failure() -> None:
    assert Pipeline._poll_record_is_terminal_failure({
        "polls": [{"response": {"status_code": 404, "body": {}}}],
    }) is False
    assert Pipeline._poll_record_is_terminal_failure({
        "polls": [{"response": {
            "status_code": 500,
            "body": {"response": {"status": "FAILED"}},
        }}],
    }) is False
    assert Pipeline._poll_record_is_terminal_failure({
        "polls": [{"response": {
            "status_code": 200,
            "body": {"response": {"status": "FAILED"}},
        }}],
    }) is True
    workflow_failed = {
        "status": "error",
        "reason": "minimax/h3-max/reference-to-video_failed",
        "message": (
            "minimax/h3-max/reference-to-video failed with message: "
            "ValidationError: reference video download failed"
        ),
    }
    assert Pipeline._poll_record_is_terminal_failure({
        "polls": [{"response": {
            "status_code": 422,
            "body": workflow_failed,
        }}],
    }) is True
    for status_code in (404, 429, 500):
        assert Pipeline._poll_record_is_terminal_failure({
            "polls": [{"response": {
                "status_code": status_code,
                "body": workflow_failed,
            }}],
        }) is False
    assert Pipeline._poll_record_is_terminal_failure({
        "polls": [{"response": {
            "status_code": 422,
            "body": {
                "status": "error",
                "reason": "validation_error",
                "message": workflow_failed["message"],
            },
        }}],
    }) is False


@pytest.mark.parametrize(
    "segments",
    [
        [
            {"index": 1, "duration_s": 5.0},
            {"index": 1, "duration_s": 5.0},
        ],
        [{"index": 1, "duration_s": float("nan")}],
        [{"index": 1, "duration_s": 0.001}],
        [{"index": 1, "duration_s": 5.0, "join_mode": "bogus"}],
    ],
)
def test_resume_invalid_segment_structure_fails_before_any_submit(
    tmp_path: Path,
    segments: list[dict[str, Any]],
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "invalid-resume-segments",
    )
    (project / "inputs" / "source.mp4").write_bytes(b"source")
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=len(segments),
        segments=segments,
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    picsart = _FlowPicsart()
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
    )

    with pytest.raises(ValueError, match="stored segment"):
        pipeline.validate_r2v_resume(project_id)
    assert picsart.submits == []


@pytest.mark.parametrize(
    "manifest_version",
    [1, 2, 3, H3_PREPARED_ARTIFACTS_VERSION + 1],
)
def test_h3_manifest_unsupported_version_is_rejected(
    tmp_path: Path,
    manifest_version: int,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "version-mismatch",
    )
    enhanced_index = {
        "people": {}, "entities": {}, "scenes": {}, "relations": {},
    }
    object_replace = {
        "version": OBJECT_REPLACE_OUTPUT_VERSION,
        "input_sha256": "a" * 64,
    }
    replacement_sheet = {"version": REPLACEMENT_SHEET_VERSION}
    effective_request = _effective_request()
    bindings = {
        "enhanced_element_index_sha256": Pipeline._json_sha256(
            enhanced_index,
        ),
        "voice_references_sha256": Pipeline._json_sha256([]),
        "object_replace_input_sha256": object_replace["input_sha256"],
        "replacement_image": None,
        "output_request_sha256": Pipeline._json_sha256(
            effective_request["output"]
        ),
    }
    bindings["replacement_sheet_sha256"] = Pipeline._json_sha256(
        replacement_sheet,
    )
    state = {
        "enhanced_element_index": enhanced_index,
        "voice_references": [],
        "object_replace": object_replace,
        "effective_request": effective_request,
        "replacement_sheet": replacement_sheet,
        "h3_prepared_artifacts": {
            "version": manifest_version,
            "bindings": bindings,
            "segments": {},
        },
    }

    with pytest.raises(ValueError, match="manifest is invalid"):
        Pipeline._validated_h3_prepared_artifacts(project, state, [])


def test_freeze_h3_rejects_unsupported_object_replace_version(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, "freeze-version")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    (segment_dir / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    enhanced_index = {
        "people": {}, "entities": {}, "scenes": {}, "relations": {},
    }
    store.update(
        project_id,
        enhanced_element_index=enhanced_index,
        voice_references=[],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [{"index": 1, "analysis_paths": []}],
            enhanced_index,
        )
    )
    assert replacement_sheet_path is None
    store.update(
        project_id,
        object_replace={"version": 2, "input_sha256": "a" * 64},
    )

    with pytest.raises(ValueError, match="requires Object Replace"):
        pipeline._freeze_h3_prepared_artifacts(
            project_id,
            project,
            [({
                "index": 1,
                "join_mode": "hard_cut",
                "source": source,
            }, "prompt", muted, "b" * 64)],
            replacement_sheet=replacement_sheet,
            request=request,
        )
    assert "h3_prepared_artifacts" not in store.read(project_id)


def test_h3_manifest_rejects_new_voice_mapping_with_old_prompt_binding(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, "voice-binding")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    prompt_path = segment_dir / "h3_prompt.txt"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    prompt_path.write_text("old prompt", encoding="utf-8")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "dialogue": [],
        "visual_prompt": "visual",
        "status": "failed",
    }
    library = load_voice_library()
    voice_directory = project / "work" / "voice-references"
    voice_directory.mkdir(parents=True)

    def reference(voice: dict[str, Any]) -> dict[str, Any]:
        return {
            "audio_reference": "Audio 1",
            "voice_id": voice["voice_id"],
            "asset_path": "work/voice-references/audio-01.wav",
            "asset_sha256": voice["sha256"],
            "features": library.model_line(voice["voice_id"]),
        }

    first = reference(library.voices[0])
    audio = voice_directory / "audio-01.wav"
    audio.write_bytes(library.files[first["voice_id"]].read_bytes())
    first_index = {
        "people": {"person-01": {"voice": dict(first)}},
        "entities": {}, "scenes": {}, "relations": {},
    }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[segment],
        enhanced_element_index=first_index,
        voice_references=[first],
        object_replace={"input_sha256": "a" * 64},
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [dict(segment, source=source, analysis_paths=[])],
            first_index,
        )
    )
    assert replacement_sheet_path is None
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(dict(segment, source=source), "old prompt", muted, "b" * 64)],
        replacement_sheet=replacement_sheet,
        request=request,
    )
    second = reference(library.voices[1])
    audio.write_bytes(library.files[second["voice_id"]].read_bytes())
    second_index = {
        "people": {"person-01": {"voice": dict(second)}},
        "entities": {}, "scenes": {}, "relations": {},
    }
    store.update(
        project_id,
        enhanced_element_index=second_index,
        voice_references=[second],
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )

    with pytest.raises(ValueError, match="preparation bindings do not match"):
        pipeline.validate_r2v_resume(project_id)


def test_h3_manifest_rejects_a_changed_replacement_image(
    tmp_path: Path,
) -> None:
    request = _effective_request()
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(request, "replacement-binding")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    replacement = project / "inputs" / "replacement-image"
    replacement.write_bytes(b"original-image")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    (segment_dir / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "dialogue": [],
        "visual_prompt": "visual",
        "status": "failed",
    }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
        segment_count=1,
        segments=[segment],
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
        },
        voice_references=[],
        object_replace={"input_sha256": "a" * 64},
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [dict(segment, source=source, analysis_paths=[])],
            store.read(project_id)["enhanced_element_index"],
            replacement=replacement,
        )
    )
    assert replacement_sheet_path is None
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(dict(segment, source=source), "prompt", muted, "b" * 64)],
        replacement=replacement,
        replacement_sheet=replacement_sheet,
        request=request,
    )
    replacement.write_bytes(b"changed-image")

    with pytest.raises(ValueError, match="artifact receipt does not match"):
        pipeline.validate_r2v_resume(project_id)


@pytest.mark.parametrize(
    "drift_target", ["sheet_bytes", "user_bytes", "user_receipt"],
)
def test_resume_rejects_frozen_replacement_sheet_drift_without_provider_calls(
    tmp_path: Path,
    monkeypatch: Any,
    drift_target: str,
) -> None:
    request = _effective_request()
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(request, "sheet-drift")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    replacement = None
    if drift_target.startswith("user_"):
        replacement = project / "inputs" / "replacement-image"
        replacement.write_bytes(b"replacement")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    (segment_dir / "h3_prompt.txt").write_text(
        "frozen prompt", encoding="utf-8",
    )
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "dialogue": [],
        "visual_prompt": "visual",
        "status": "failed",
    }
    enhanced_element_index = {
        "people": {
            "person-01": {
                "source_visual_description": "the source person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replacement": {
                    "reason": "automatic",
                    "source": (
                        "image_1"
                        if replacement is not None
                        else "generated_variant"
                    ),
                    "target_visual_description": "a different person",
                    "appearance_description": "an isolated different person",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        **(
            {"replacement_image_path": "inputs/replacement-image"}
            if replacement is not None
            else {}
        ),
        segment_count=1,
        segments=[segment],
        enhanced_element_index=enhanced_element_index,
        voice_references=[],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    picsart = _FlowPicsart()
    seedream = _FlowSeedream()
    registered: list[Path] = []
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: (
            registered.append(path) or "https://assets.invalid/value"
        ),
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
        seedream=seedream,
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    monkeypatch.setattr(
        media,
        "provider_reference_png",
        lambda _source, destination: (
            destination.write_bytes(b"\x89PNG\r\n\x1a\ncanonical")
            and destination
        ),
    )
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [dict(segment, source=source, analysis_paths=[])],
            enhanced_element_index,
            replacement=replacement,
        )
    )
    assert replacement_sheet_path is not None
    assert len(seedream.calls) == 1
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(dict(segment, source=source), "frozen prompt", muted, "c" * 64)],
        replacement=replacement,
        replacement_sheet=replacement_sheet,
        request=request,
    )
    preparation_sha256 = Pipeline._json_sha256(
        store.read(project_id)["h3_prepared_artifacts"]
    )
    store.update(
        project_id,
        picsart_submission_receipts={
            "version": PICSART_RECEIPT_VERSION,
            "segments": {
                "1": [{
                    "segment_index": 1,
                    "run_identity": "picsart",
                    "request_sha256": "b" * 64,
                    "preparation_sha256": preparation_sha256,
                    "status": "provider_failed",
                    "task_id": "failed-task",
                }],
            },
        },
    )
    if drift_target == "sheet_bytes":
        replacement_sheet_path.write_bytes(
            replacement_sheet_path.read_bytes() + b"drift"
        )
    elif drift_target == "user_bytes":
        canonical_user = (
            project / "work" / "global-reference-sheet"
            / "user-reference.png"
        )
        canonical_user.write_bytes(canonical_user.read_bytes() + b"drift")
    else:
        changed_sheet = dict(store.read(project_id)["replacement_sheet"])
        changed_references = [
            dict(reference)
            for reference in changed_sheet["h3_reference_images"]
        ]
        changed_references[0]["sha256"] = "0" * 64
        changed_sheet["h3_reference_images"] = changed_references
        store.update(project_id, replacement_sheet=changed_sheet)
    pipeline.seedream = _RejectSeedream()
    monkeypatch.setattr(
        media,
        "provider_reference_png",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider reference conversion must not run")
        ),
    )

    with pytest.raises(
        ValueError,
        match="artifact receipt|replacement sheet|H3 image reference",
    ):
        pipeline.validate_r2v_resume(project_id)

    assert len(seedream.calls) == 1
    assert picsart.submits == []
    assert registered == []


def test_current_state_requires_h3_manifest(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "new-state-no-manifest",
    )
    (project / "inputs" / "source.mp4").write_bytes(b"source")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "source.mp4").write_bytes(b"source")
    (segment_dir / "source-muted.mp4").write_bytes(b"muted")
    (segment_dir / "h3_prompt.txt").write_text("stale prompt", encoding="utf-8")
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[{
            "index": 1,
            "duration_s": 5.0,
            "join_mode": "hard_cut",
        }],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
        },
        voice_references=[],
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )

    with pytest.raises(ValueError, match="missing the current H3 preparation"):
        pipeline.validate_r2v_resume(project_id)


def _frozen_current_after_global_preflight_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> tuple[ProjectStore, str, Path, list[dict[str, Any]], _FlowSeedream]:
    request = _effective_request()
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(request, "current-preflight-failure")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    enhanced_element_index = {
        "people": {
            "person-01": {
                "source_visual_description": "the source person",
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1]},
                    {"segment_index": 2, "frame_orders": [1]},
                ],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a different person",
                    "appearance_description": "an isolated different person",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }
    items: list[dict[str, Any]] = []
    for index, duration in ((1, 4.2), (2, 5.0)):
        segment_dir = project / "work" / "segments" / f"{index:04d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        source = segment_dir / "source.mp4"
        source.write_bytes(f"source-{index}".encode("ascii"))
        items.append({
            "index": index,
            "start_s": 0.0 if index == 1 else 4.2,
            "end_s": 4.2 if index == 1 else 9.2,
            "duration_s": duration,
            "join_mode": "hard_cut",
            "visual_prompt": f"segment {index} visual",
            "dialogue": [],
            "source": source,
            "status": "running",
        })
    seedream = _FlowSeedream()
    register_calls: list[Path] = []

    def fail_register(_project_id: str, path: Path) -> str:
        register_calls.append(path)
        raise TypeError("simulated asset registration failure")

    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        fail_register,
        codex=object(),
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
        seedream=seedream,
    )
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [dict(item, analysis_paths=[]) for item in items],
            enhanced_element_index,
        )
    )
    assert replacement_sheet_path is not None

    def prepare_h3(
        _project: Path,
        item: dict[str, Any],
        *_args: Any,
    ) -> tuple[dict[str, Any], str, Path, str]:
        segment_dir = item["source"].parent
        prompt = f"frozen prompt {item['index']}"
        (segment_dir / "h3_prompt.txt").write_text(prompt, encoding="utf-8")
        muted = segment_dir / "source-muted.mp4"
        muted.write_bytes(f"muted-{item['index']}".encode("ascii"))
        return item, prompt, muted, str(item["index"]) * 64

    monkeypatch.setattr(pipeline, "_prepare_h3_segment", prepare_h3)
    with pytest.raises(TypeError, match="asset registration failure"):
        pipeline._run_h3_segments(
            project_id,
            project,
            items,
            enhanced_element_index,
            None,
            replacement_sheet,
            replacement_sheet_path,
            request,
            ResearchLog(project),
        )

    frozen_state = store.read(project_id)
    assert frozen_state["h3_prepared_artifacts"]["version"] == (
        H3_PREPARED_ARTIFACTS_VERSION
    )
    assert "picsart_submission_receipts" not in frozen_state
    assert register_calls
    assert not any(
        path.is_dir() for path in (project / "logs").glob("picsart*-submit")
    )
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    return store, project_id, project, items, seedream


def _install_resume_output_media(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )

    def trim(source: Path, duration_s: float, destination: Path) -> dict[str, Any]:
        destination.write_bytes(source.read_bytes())
        return {"fps": 24, "duration_s": duration_s}

    def concat(
        paths: list[Path],
        destination: Path,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        destination.write_bytes(b"".join(path.read_bytes() for path in paths))
        return {"fps": 24}

    monkeypatch.setattr(media, "trim_24fps", trim)
    monkeypatch.setattr(media, "concat", concat)


def test_current_manifest_without_submission_evidence_can_resume_first_batch(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store, project_id, project, _items, initial_seedream = (
        _frozen_current_after_global_preflight_failure(tmp_path, monkeypatch)
    )
    _install_resume_output_media(monkeypatch)
    registered: list[Path] = []

    class LedgerCheckingPicsart:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.submits: list[dict[str, Any]] = []
            self.polls: list[str] = []

        def submit(self, **kwargs: Any) -> ProviderResult[str]:
            ledger = store.read(project_id).get("picsart_submission_receipts")
            assert isinstance(ledger, dict)
            assert ledger["version"] == 2
            assert set(ledger["segments"]) == {"1", "2"}
            assert all(
                len(records) == 1
                and records[0]["status"] in {"submission_started", "accepted"}
                for records in ledger["segments"].values()
            )
            with self.lock:
                self.submits.append(dict(kwargs))
                task_id = f"resumed-task-{len(self.submits)}"
            return ProviderResult(
                task_id,
                {"outcome": "accepted", "task_id": task_id},
            )

        def poll(self, task_id: str) -> ProviderResult[str]:
            with self.lock:
                self.polls.append(task_id)
            return ProviderResult(
                f"https://result.invalid/{task_id}.mp4",
                {"task_id": task_id, "polls": [{"status": "COMPLETED"}]},
            )

        def download(self, _url: str, destination: Path) -> ProviderResult[Path]:
            destination.write_bytes(b"provider-video")
            return ProviderResult(destination, {"response": {"status_code": 200}})

    picsart = LedgerCheckingPicsart()
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: (
            registered.append(path)
            or f"https://assets.invalid/{path.parent.name}/{path.name}"
        ),
        codex=object(),
        deepseek=object(),
        mediakit=object(),
        picsart=picsart,
        seedream=_RejectSeedream(),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_h3_segment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Fusion and H3 prompt preparation must not rerun")
        ),
    )

    pipeline.validate_r2v_resume(project_id)
    pipeline.resume_r2v(project_id)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    assert len(initial_seedream.calls) == 1
    assert len(picsart.submits) == 2
    assert sorted(call["prompt"] for call in picsart.submits) == [
        "frozen prompt 1", "frozen prompt 2",
    ]
    assert sorted(picsart.polls) == ["resumed-task-1", "resumed-task-2"]
    assert set(state["picsart_submission_receipts"]["segments"]) == {"1", "2"}
    assert all(
        [receipt["status"] for receipt in records] == ["accepted"]
        for records in state["picsart_submission_receipts"]["segments"].values()
    )
    assert len(list((project / "logs" / "picsart-resume-submit").glob(
        "segment-*/attempt-1",
    ))) == 2
    assert len(registered) == 4


@pytest.mark.parametrize("damage", ["submit_directory", "partial_ledger"])
def test_current_first_submit_resume_rejects_ambiguous_submission_evidence(
    tmp_path: Path,
    monkeypatch: Any,
    damage: str,
) -> None:
    store, project_id, project, _items, _seedream = (
        _frozen_current_after_global_preflight_failure(tmp_path, monkeypatch)
    )
    if damage == "submit_directory":
        (project / "logs" / "picsart-resume-submit").mkdir(parents=True)
    else:
        manifest = store.read(project_id)["h3_prepared_artifacts"]
        store.update(
            project_id,
            picsart_submission_receipts={
                "version": PICSART_RECEIPT_VERSION,
                "segments": {
                    "1": [{
                        "segment_index": 1,
                        "run_identity": "picsart",
                        "request_sha256": "a" * 64,
                        "preparation_sha256": Pipeline._json_sha256(manifest),
                        "status": "rejected",
                        "task_id": None,
                    }],
                },
            },
        )

    provider_calls: list[str] = []

    class RejectProvider:
        def submit(self, **_kwargs: Any) -> None:
            provider_calls.append("submit")
            raise AssertionError("provider submit must not run")

        def poll(self, _task_id: str) -> None:
            provider_calls.append("poll")
            raise AssertionError("provider poll must not run")

        def download(self, _url: str, _destination: Path) -> None:
            provider_calls.append("download")
            raise AssertionError("provider download must not run")

    registered: list[Path] = []
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: (
            registered.append(path) or "https://assets.invalid/value"
        ),
        codex=object(),
        deepseek=object(),
        mediakit=object(),
        picsart=RejectProvider(),
        seedream=_RejectSeedream(),
    )

    with pytest.raises(ValueError, match="submission evidence|complete Picsart"):
        pipeline.resume_r2v(project_id)

    assert registered == []
    assert provider_calls == []


def _prepare_h3_mask_fixture(
    tmp_path: Path,
    monkeypatch: Any,
) -> tuple[
    Pipeline, Path, dict[str, Any], dict[str, Any], dict[str, Any],
    list[tuple[Path, Path]],
]:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "face-mask",
    )
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    source.write_bytes(b"source")
    enhanced_index = {
        "people": {
            "person-01": {
                "source_visual_description": "a visible subject",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a replacement subject",
                    "appearance_description": "an isolated replacement subject",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }
    spec = build_replacement_sheet_spec(enhanced_index)
    sheet_directory = project / "work" / "global-reference-sheet"
    inputs = materialize_replacement_sheet_inputs(sheet_directory, spec)
    finalize_replacement_sheet(
        inputs["canvas_path"], spec, sheet_directory / "reference-sheet.png",
    )
    replacement_sheet = {
        "spec_sha256": spec["spec_sha256"],
        "h3_image_reference": "Image 1",
        "h3_reference_images": [],
        "user_image_present": False,
        "spec": spec,
    }
    item = {
        "index": 1,
        "source": source,
        "visual_prompt": "a person close-up",
        "dialogue": [],
    }

    class FakeFusion:
        def complete_structured(
            self,
            _instruction: str,
            _payload: dict[str, Any],
            schema: dict[str, Any],
            _images: list[Path],
            validator: Any = None,
        ) -> ProviderResult[dict[str, Any]]:
            return ProviderResult({
                "schema": FUSION_OUTPUT_NAME,
                "version": 1,
                "input_sha256": schema["properties"]["input_sha256"]["enum"][0],
                "segment_index": 1,
                "prompt": "Preserve the source motion, camera, light, and timing",
            }, {"provider": "deepseek"})

    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: f"https://assets.invalid/{path.name}",
        deepseek=FakeFusion(),
        mediakit=object(),
        picsart=object(),
    )
    monkeypatch.setattr(
        media,
        "mute",
        lambda input_path, target: target.write_bytes(input_path.read_bytes()),
    )
    mask_calls: list[tuple[Path, Path]] = []

    def fake_mask_faces(
        source_path: Path, destination: Path, **_kwargs: Any,
    ) -> dict[str, Any]:
        mask_calls.append((source_path, destination))
        destination.write_bytes(source_path.read_bytes() + b"-masked")
        return {"frames": 1, "frames_with_face": 1, "faces_total": 1}

    monkeypatch.setattr(face_mask, "mask_faces", fake_mask_faces)
    return (
        pipeline, project, item, enhanced_index, replacement_sheet, mask_calls,
    )


def test_h3_segment_with_a_replaced_person_masks_faces(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    pipeline, project, item, enhanced_index, replacement_sheet, mask_calls = (
        _prepare_h3_mask_fixture(tmp_path, monkeypatch)
    )
    segment_dir = project / "work" / "segments" / "0001"

    _item, _prompt, muted, _fusion_sha256 = pipeline._prepare_h3_segment(
        project,
        item,
        enhanced_index,
        replacement_sheet,
        _effective_request(),
        ResearchLog(project),
    )

    assert muted == segment_dir / "source-faces-masked.mp4"
    assert muted.read_bytes() == b"source-masked"
    assert mask_calls == [
        (segment_dir / "source-muted.mp4", muted),
    ]
    output = json.loads(
        (
            project / "logs" / "mask-reference-faces" / "segment-0001"
            / "attempt-1" / "output.json"
        ).read_text(encoding="utf-8")
    )
    assert output == {"frames": 1, "frames_with_face": 1, "faces_total": 1}


def _dialogue_audio_fixture(
    tmp_path: Path,
    monkeypatch: Any,
    autodl: Any,
) -> tuple[ProjectStore, str, Path, Pipeline, list[dict[str, Any]], dict[str, Any], list[Path]]:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "dialogue-audio",
    )
    store.update(
        project_id,
        translated_voice_lines=[
            {"text": "第一句", "start_s": 0.0, "end_s": 2.0},
            {"text": "第二句", "start_s": 5.0, "end_s": 7.0},
        ],
    )
    library = load_voice_library()
    voice_directory = project / "work" / "voice-references"
    voice_directory.mkdir(parents=True)
    voice_paths: list[Path] = []
    people: dict[str, Any] = {}
    for slot, voice in enumerate(library.voices[:2], 1):
        destination = voice_directory / f"audio-{slot:02d}.wav"
        destination.write_bytes(library.files[voice["voice_id"]].read_bytes())
        voice_paths.append(destination)
        people[f"person-{slot:02d}"] = {
            "voice": {
                "audio_reference": f"Audio {slot}",
                "voice_id": voice["voice_id"],
                "asset_path": f"work/voice-references/audio-{slot:02d}.wav",
                "asset_sha256": voice["sha256"],
                "features": library.model_line(voice["voice_id"]),
            },
        }
    enhanced_index = {
        "people": people,
        "entities": {},
        "scenes": {},
        "relations": {},
        "dialogue_voices": ["person-02", None],
    }
    prepared = [
        {
            "index": 1,
            "start_s": 0.0,
            "end_s": 5.0,
            "duration_s": 5.0,
            "join_mode": "hard_cut",
            "dialogue": [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}],
        },
        {
            "index": 2,
            "start_s": 5.0,
            "end_s": 10.0,
            "duration_s": 5.0,
            "join_mode": "continue",
            "dialogue": [{"text": "第二句", "start_s": 0.0, "end_s": 2.0}],
        },
    ]

    def assemble(
        lines: list[dict[str, Any]],
        total_duration_s: float,
        destination: Path,
    ) -> dict[str, Any]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            b"".join(line["audio"].read_bytes() for line in lines)
        )
        return {"duration_s": total_duration_s}

    def slice_track(
        source: Path,
        start_s: float,
        duration_s: float,
        destination: Path,
    ) -> dict[str, Any]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return {"stdout": "", "stderr": ""}

    monkeypatch.setattr(media, "assemble_dialogue_track", assemble)
    monkeypatch.setattr(media, "slice_dialogue_track", slice_track)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        autodl=autodl,
    )
    return store, project_id, project, pipeline, prepared, enhanced_index, voice_paths


def test_dialogue_tts_synthesizes_each_line_with_the_attributed_voice(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    autodl = _FlowAutodlTTS()
    (
        _store, project_id, project, pipeline,
        prepared, enhanced_index, voice_paths,
    ) = _dialogue_audio_fixture(tmp_path, monkeypatch, autodl)

    full_track = pipeline._run_dialogue_audio(
        project_id,
        project,
        prepared,
        enhanced_index,
        voice_paths,
        ResearchLog(project),
    )

    assert full_track == (
        project / "work" / "dialogue-audio" / "full-track.wav"
    )
    assert full_track.is_file()
    # person-02's line uses Audio 2; the unattributed line falls back to
    # Audio 1.
    assert [call["text"] for call in autodl.tts_calls] == ["第一句", "第二句"]
    assert [call["voice_path"] for call in autodl.tts_calls] == [
        voice_paths[1].resolve(),
        voice_paths[0].resolve(),
    ]
    for offset in (1, 2):
        assert (
            project / "work" / "dialogue-audio" / f"line-{offset:04d}.wav"
        ).is_file()
    output = json.loads(
        (
            project / "logs" / "dialogue-tts-line" / "project"
            / "attempt-1" / "output.json"
        ).read_text(encoding="utf-8")
    )
    assert output["provider_record"]["submit"]["outcome"] == "accepted"
    for index in (1, 2):
        assert (
            project / "work" / "segments" / f"{index:04d}" / "dialogue-track.wav"
        ).is_file()


def test_dialogue_tts_moderation_rejection_retries_until_success(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    class FlakyAutodl(_FlowAutodlTTS):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def submit_tts(self, **kwargs: Any) -> ProviderResult[str]:
            self.calls += 1
            if self.calls <= 4:
                raise ProviderFailure(
                    {"error": "content policy violation"},
                    [],
                    RuntimeError("autodl tts content policy rejection"),
                )
            return super().submit_tts(**kwargs)

    autodl = FlakyAutodl()
    (
        _store, project_id, project, pipeline,
        prepared, enhanced_index, voice_paths,
    ) = _dialogue_audio_fixture(tmp_path, monkeypatch, autodl)
    delays: list[float] = []
    monkeypatch.setattr(pipeline, "_retry_sleep", delays.append)

    pipeline._run_dialogue_audio(
        project_id,
        project,
        prepared,
        enhanced_index,
        voice_paths,
        ResearchLog(project),
    )

    # Moderation rejections retry with backoff instead of failing closed.
    # Two dialogue lines: the first burns four failures plus its success,
    # the second succeeds on its first attempt.
    assert autodl.calls == 6
    # Retries 2-3 rerun immediately; attempts 3 and 4 back off 60s and 120s.
    assert delays == [60.0, 120.0]
    assert (project / "work" / "dialogue-audio" / "full-track.wav").is_file()


def test_dialogue_tts_without_any_voice_reference_fails_closed(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    (
        _store, project_id, project, pipeline,
        prepared, enhanced_index, _voice_paths,
    ) = _dialogue_audio_fixture(tmp_path, monkeypatch, _FlowAutodlTTS())

    with pytest.raises(ValueError, match="voice reference"):
        pipeline._run_dialogue_audio(
            project_id,
            project,
            prepared,
            enhanced_index,
            [],
            ResearchLog(project),
        )


class _ObjectReplaceOnlyDeepSeek:
    """Object-replace node stub with a configurable voice assignment."""

    def __init__(self, voice_id: str | None = None) -> None:
        self.voice_id = voice_id

    def complete_structured(
        self,
        instruction: str,
        payload: Any,
        schema: dict[str, Any],
        images: list[Path] | None = None,
        validator: Any = None,
    ) -> ProviderResult[dict[str, Any]]:
        people = {
            stable_id: {
                "matches_user_request": False,
                "replacement": {
                    "source": "generated_variant",
                    "target_visual_description": (
                        f"replacement appearance for {stable_id}"
                    ),
                    "appearance_description": (
                        f"isolated replacement person {stable_id}"
                    ),
                },
                "voice_id": self.voice_id,
            }
            for stable_id in schema["properties"]["people"]["properties"]
        }
        return ProviderResult(
            {
                "schema": OBJECT_REPLACE_OUTPUT_NAME,
                "version": OBJECT_REPLACE_OUTPUT_VERSION,
                "input_sha256": schema["properties"]["input_sha256"]["enum"][0],
                "dialogue_voices": [
                    "person-01" if self.voice_id is not None else None
                    for _line in payload["translated_dialogue"]
                ],
                "people": people,
                "entities": {},
                "scenes": {},
            },
            {"response": {"api": "responses", "phase": "object-replace"}},
        )


def _object_replace_fixture(
    tmp_path: Path,
    deepseek: Any,
    *,
    target_language: str = "中文",
) -> tuple[
    ProjectStore, str, Path, Pipeline, dict[str, Any], ResearchLog,
]:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request(target_language=target_language)
    project_id, project, _state = store.create(
        request, "object-replace-fallback",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=deepseek,
        mediakit=object(),
        picsart=object(),
        autodl=_FlowAutodlTTS(),
    )
    return store, project_id, project, pipeline, request, ResearchLog(project)


_OBJECT_REPLACE_ELEMENT_INDEX = {
    "people": {
        "person-01": {
            "source_visual_description": "only a hand holding a comb",
            "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
        },
    },
    "entities": {},
    "scenes": {},
    "relations": {},
}

_OBJECT_REPLACE_PREPARED = [
    {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "visual_prompt": "visual",
    },
]

_OBJECT_REPLACE_TRANSLATED_LINES = [
    {"text": "第一句", "start_s": 0.0, "end_s": 2.0},
]


def _trace_events(project: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (project / "logs" / "trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_object_replace_voiceless_people_fall_back_to_default_voice(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    library = load_voice_library()
    (
        store, project_id, project, pipeline, request, log,
    ) = _object_replace_fixture(
        tmp_path, _ObjectReplaceOnlyDeepSeek(), target_language="中文",
    )

    enhanced_index, metadata, references, paths = pipeline._run_object_replace(
        project,
        [dict(item) for item in _OBJECT_REPLACE_PREPARED],
        [dict(line) for line in _OBJECT_REPLACE_TRANSLATED_LINES],
        {
            category: dict(records)
            for category, records in _OBJECT_REPLACE_ELEMENT_INDEX.items()
        },
        None,
        request,
        log,
    )

    # The catalog has no Chinese voice yet, so the first voice is picked.
    fallback = library.default_voice("中文")
    assert fallback == library.voices[0]
    assert [reference["voice_id"] for reference in references] == [
        fallback["voice_id"]
    ]
    assert [path.name for path in paths] == ["audio-01.wav"]
    assert paths[0].is_file()
    # The model's deliberate "no voice" decision is kept in the index; the
    # fallback only materializes the Audio 1 reference.
    assert enhanced_index["people"]["person-01"]["voice"] is None
    assert metadata["voice_reference_fallback"] == {
        "voice_id": fallback["voice_id"],
        "reason": "dialogue requires TTS but no person was assigned a voice",
    }
    fallback_events = [
        event
        for event in _trace_events(project)
        if event["event"] == "voice_reference_default_fallback"
    ]
    assert len(fallback_events) == 1
    assert fallback_events[0]["voice_id"] == fallback["voice_id"]

    # Resume path: the fallback entry is a regular frozen Audio 1 reference.
    store.update(
        project_id,
        voice_references=references,
        translated_voice_lines=[
            dict(line) for line in _OBJECT_REPLACE_TRANSLATED_LINES
        ],
    )
    assert Pipeline._stored_voice_references(
        project, store.read(project_id),
    ) == [paths[0].resolve()]

    # The dialogue TTS chain no longer fails closed on voiceless people.
    def assemble(
        lines: list[dict[str, Any]],
        total_duration_s: float,
        destination: Path,
    ) -> dict[str, Any]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            b"".join(line["audio"].read_bytes() for line in lines)
        )
        return {"duration_s": total_duration_s}

    monkeypatch.setattr(media, "assemble_dialogue_track", assemble)
    full_track = pipeline._run_dialogue_audio(
        project_id,
        project,
        [dict(item) for item in _OBJECT_REPLACE_PREPARED],
        enhanced_index,
        paths,
        log,
    )
    assert full_track == (
        project / "work" / "dialogue-audio" / "full-track.wav"
    )
    assert full_track.is_file()
    assert [
        call["voice_path"] for call in pipeline.autodl.tts_calls
    ] == [paths[0].resolve()]


def test_object_replace_with_assigned_voice_never_falls_back(
    tmp_path: Path,
) -> None:
    library = load_voice_library()
    assigned = library.voices[1]["voice_id"]
    (
        _store, _project_id, project, pipeline, request, log,
    ) = _object_replace_fixture(
        tmp_path, _ObjectReplaceOnlyDeepSeek(assigned),
    )

    enhanced_index, metadata, references, paths = pipeline._run_object_replace(
        project,
        [dict(item) for item in _OBJECT_REPLACE_PREPARED],
        [dict(line) for line in _OBJECT_REPLACE_TRANSLATED_LINES],
        {
            category: dict(records)
            for category, records in _OBJECT_REPLACE_ELEMENT_INDEX.items()
        },
        None,
        request,
        log,
    )

    assert [reference["voice_id"] for reference in references] == [assigned]
    assert enhanced_index["people"]["person-01"]["voice"]["voice_id"] == assigned
    assert metadata["voice_reference_fallback"] is None
    assert "voice_reference_default_fallback" not in (
        project / "logs" / "trace.jsonl"
    ).read_text(encoding="utf-8")


def test_object_replace_without_dialogue_materializes_no_voice(
    tmp_path: Path,
) -> None:
    (
        _store, _project_id, project, pipeline, request, log,
    ) = _object_replace_fixture(tmp_path, _ObjectReplaceOnlyDeepSeek())

    _enhanced_index, metadata, references, paths = pipeline._run_object_replace(
        project,
        [dict(item) for item in _OBJECT_REPLACE_PREPARED],
        [],
        {
            category: dict(records)
            for category, records in _OBJECT_REPLACE_ELEMENT_INDEX.items()
        },
        None,
        request,
        log,
    )

    assert references == []
    assert paths == []
    assert metadata["voice_reference_fallback"] is None
    assert "voice_reference_default_fallback" not in (
        project / "logs" / "trace.jsonl"
    ).read_text(encoding="utf-8")


def test_no_dialogue_skips_the_whole_tts_chain(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    (
        store, project_id, project, pipeline,
        prepared, enhanced_index, voice_paths,
    ) = _dialogue_audio_fixture(tmp_path, monkeypatch, _FlowAutodlTTS())
    store.update(project_id, translated_voice_lines=[])
    monkeypatch.setattr(
        media,
        "assemble_dialogue_track",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("assemble must not run without dialogue")
        ),
    )

    assert pipeline._run_dialogue_audio(
        project_id,
        project,
        [{**item, "dialogue": []} for item in prepared],
        enhanced_index,
        voice_paths,
        ResearchLog(project),
    ) is None


def test_picsart_segment_uses_the_dialogue_slice_as_audio_1(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, "dialogue-audio-url")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    dialogue_track = segment_dir / "dialogue-track.wav"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    dialogue_track.write_bytes(b"dialogue-slice")
    (segment_dir / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "dialogue": [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}],
        "visual_prompt": "visual",
        "status": "running",
    }
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[Pipeline._public_segment(segment)],
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
        },
        voice_references=[],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
    )
    registered: list[Path] = []
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: (
            registered.append(path) or f"https://assets.invalid/{path.name}"
        ),
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [dict(segment, source=source, analysis_paths=[])],
            store.read(project_id)["enhanced_element_index"],
        )
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(dict(segment, source=source), "prompt", muted, "b" * 64)],
        replacement_sheet=replacement_sheet,
        request=request,
    )

    prepared = pipeline._prepare_picsart_segment(
        project_id,
        project,
        dict(segment, source=source),
        None,
        replacement_sheet_path,
        request,
        ResearchLog(project),
        final_prompt="prompt",
        muted=muted,
    )

    assert prepared.dialogue_track == dialogue_track.resolve()
    assert prepared.audio_urls == (
        "https://assets.invalid/dialogue-track.wav",
    )
    assert dialogue_track.resolve() in [path.resolve() for path in registered]


def test_h3_segment_without_a_replaced_person_keeps_the_muted_reference(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    pipeline, project, item, enhanced_index, replacement_sheet, mask_calls = (
        _prepare_h3_mask_fixture(tmp_path, monkeypatch)
    )
    # The replaced person never appears in segment 1.
    enhanced_index["people"]["person-01"]["occurrences"] = [
        {"segment_index": 2, "frame_orders": [1]},
    ]
    segment_dir = project / "work" / "segments" / "0001"

    _item, _prompt, muted, _fusion_sha256 = pipeline._prepare_h3_segment(
        project,
        item,
        enhanced_index,
        replacement_sheet,
        _effective_request(),
        ResearchLog(project),
    )

    assert muted == segment_dir / "source-muted.mp4"
    assert mask_calls == []
    assert not (segment_dir / "source-faces-masked.mp4").exists()


def test_masked_reference_is_frozen_and_resumed(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, "masked-resume")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    masked = segment_dir / "source-faces-masked.mp4"
    source.write_bytes(b"source")
    masked.write_bytes(b"masked")
    (segment_dir / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    mute_log = (
        project / "logs" / "mask-reference-faces" / "segment-0001" / "attempt-1"
    )
    mute_log.mkdir(parents=True)
    (mute_log / "output.json").write_text("{}", encoding="utf-8")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "dialogue": [],
        "visual_prompt": "visual",
        "status": "failed",
    }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[segment],
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
        },
        voice_references=[],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    registered: list[Path] = []
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: (
            registered.append(path) or f"https://assets.invalid/{path.name}"
        ),
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    replacement_sheet, replacement_sheet_path = (
        _materialize_test_replacement_sheet(
            pipeline,
            project_id,
            project,
            [dict(segment, source=source, analysis_paths=[])],
            store.read(project_id)["enhanced_element_index"],
        )
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(dict(segment, source=source), "prompt", masked, "b" * 64)],
        replacement_sheet=replacement_sheet,
        request=request,
    )

    manifest = store.read(project_id)["h3_prepared_artifacts"]
    assert manifest["segments"]["1"]["muted_video"]["path"] == (
        "work/segments/0001/source-faces-masked.mp4"
    )

    pipeline.validate_r2v_resume(project_id)

    prepared = pipeline._prepare_picsart_segment(
        project_id,
        project,
        dict(segment, source=source),
        None,
        replacement_sheet_path,
        request,
        ResearchLog(project),
        final_prompt="prompt",
        muted=masked,
    )

    assert prepared.muted == masked.resolve()
    assert [path.name for path in registered] == ["source-faces-masked.mp4"]


def test_freeze_binds_dialogue_track_and_resume_requires_it(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, "dialogue-freeze")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    dialogue_track = segment_dir / "dialogue-track.wav"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    dialogue_track.write_bytes(b"dialogue-slice")
    (segment_dir / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "dialogue": [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}],
        "visual_prompt": "visual",
        "status": "failed",
    }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[segment],
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
            "dialogue_voices": [None],
        },
        voice_references=[],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: f"https://assets.invalid/{path.name}",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    replacement_sheet, _sheet_path = _materialize_test_replacement_sheet(
        pipeline,
        project_id,
        project,
        [dict(segment, source=source, analysis_paths=[])],
        store.read(project_id)["enhanced_element_index"],
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [(dict(segment, source=source), "prompt", muted, "b" * 64)],
        replacement_sheet=replacement_sheet,
        request=request,
    )

    manifest = store.read(project_id)["h3_prepared_artifacts"]
    receipt = manifest["segments"]["1"]["dialogue_track"]
    assert receipt["path"] == "work/segments/0001/dialogue-track.wav"
    assert receipt["sha256"] == Pipeline._sha256(dialogue_track)

    pipeline.validate_r2v_resume(project_id)

    dialogue_track.unlink()
    with pytest.raises(FileNotFoundError, match="dialogue-track.wav"):
        pipeline.validate_r2v_resume(project_id)


def test_h3_manifest_rejects_dialogue_track_mismatch(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request()
    project_id, project, _state = store.create(request, "dialogue-mismatch")
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True, exist_ok=True)
    source = segment_dir / "source.mp4"
    muted = segment_dir / "source-muted.mp4"
    source.write_bytes(b"source")
    muted.write_bytes(b"muted")
    (segment_dir / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    segment = {
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "dialogue": [{"text": "第一句", "start_s": 0.0, "end_s": 2.0}],
        "visual_prompt": "visual",
        "status": "failed",
    }
    store.update(
        project_id,
        status="failed",
        source_path="inputs/source.mp4",
        segment_count=1,
        segments=[segment],
        enhanced_element_index={
            "people": {}, "entities": {}, "scenes": {}, "relations": {},
        },
        voice_references=[],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
        project_progress={
            "status": "failed", "percent": 57, "stage": "pipeline_failed",
        },
        generation={
            "status": "failed", "stage": "pipeline_failed", "error": "failed",
        },
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, path: f"https://assets.invalid/{path.name}",
        deepseek=object(),
        mediakit=object(),
        picsart=_FlowPicsart(),
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {
            "duration_s": 5.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
        },
    )
    replacement_sheet, _sheet_path = _materialize_test_replacement_sheet(
        pipeline,
        project_id,
        project,
        [dict(segment, source=source, analysis_paths=[])],
        store.read(project_id)["enhanced_element_index"],
    )
    pipeline._freeze_h3_prepared_artifacts(
        project_id,
        project,
        [
            (
                {**segment, "dialogue": [], "source": source},
                "prompt",
                segment_dir / "source-muted.mp4",
                "b" * 64,
            )
        ],
        replacement_sheet=replacement_sheet,
        request=request,
    )


    with pytest.raises(ValueError, match="dialogue track does not match"):
        pipeline.validate_r2v_resume(project_id)


class _FetchFailPicsart(_FlowPicsart):
    """Poll fails terminally with a provider-side asset fetch/validation error."""

    def __init__(
        self,
        *,
        fail_once_for: int = 0,
        fail_always: bool = False,
        message: str,
    ) -> None:
        super().__init__()
        self._fail_once_for = fail_once_for
        self._fail_always = fail_always
        self._message = message

    def poll(self, task_id: str) -> ProviderResult[str]:
        with self.lock:
            count = self.poll_counts.get(task_id, 0) + 1
            self.poll_counts[task_id] = count
        submit_number = int(task_id.rsplit("-", 1)[1])
        if self._fail_always or (
            count == 1 and submit_number <= self._fail_once_for
        ):
            raise ProviderFailure(
                {
                    "task_id": task_id,
                    "polls": [
                        {
                            "request": {
                                "method": "GET",
                                "url": f"https://picsart.invalid/{task_id}",
                            },
                            "response": {
                                "status_code": 200,
                                "body": {
                                    "response": {
                                        "status": "FAILED",
                                        "error": {"message": self._message},
                                    }
                                },
                            },
                        }
                    ],
                },
                [],
                RuntimeError(f"Picsart task failed: {self._message}"),
            )
        return ProviderResult(
            f"https://result.invalid/{task_id}.mp4",
            {"task_id": task_id, "polls": [{"status": "COMPLETED"}]},
        )


def _run_h3_with_picsart(
    tmp_path: Path,
    monkeypatch: Any,
    picsart: _FlowPicsart,
    customize: Any = None,
) -> tuple[ProjectStore, str, Path]:
    store = ProjectStore(tmp_path / "projects")
    request = _effective_request(
        target_language="日语",
        replacement_instruction="把主角替换成用户参考形象",
    )
    project_id, project, _state = store.create(request, "request-fetch-retry")
    (project / "inputs" / "source.mp4").write_bytes(b"source")
    (project / "inputs" / "replacement-image").write_bytes(b"replacement")
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        replacement_image_path="inputs/replacement-image",
    )
    _install_fake_media(monkeypatch, [])
    if customize is not None:
        customize()
    deepseek = _FlowDeepSeek()
    pipeline = Pipeline(
        _settings(tmp_path, enable_mediakit=True),
        store,
        lambda _project_id, path: (
            f"https://assets.invalid/{path.parent.name}/{path.name}"
        ),
        deepseek=deepseek,
        mediakit=_FlowMediaKit(),
        picsart=picsart,
        seedream=_FlowSeedream(deepseek.events),
        autodl=_FlowAutodlTTS(),
    )
    pipeline.run(project_id)
    return store, project_id, project


def test_h3_asset_fetch_failure_is_resubmitted_with_frozen_inputs(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    picsart = _FetchFailPicsart(
        fail_once_for=2,
        message=(
            "ValidationError: Failed to download audio file: "
            "All validated IPs unreachable"
        ),
    )
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    # Each segment: original submit (provider-side fetch failure) + one
    # automatic resubmit with the same frozen prompt.
    assert len(picsart.submits) == 4
    prompts = [call["prompt"] for call in picsart.submits]
    assert len(set(prompts)) == 2
    assert sorted(prompts.count(prompt) for prompt in set(prompts)) == [2, 2]
    ledger = state["picsart_submission_receipts"]["segments"]
    for key in ("1", "2"):
        assert [r["status"] for r in ledger[key]] == ["provider_failed", "accepted"]
        assert ledger[key][0]["request_sha256"] == ledger[key][1]["request_sha256"]
        assert ledger[key][1]["run_identity"].endswith("-resubmit-01")
    assert (project / "logs" / "picsart-resubmit-01-submit").is_dir()
    assert (project / "logs" / "picsart-resubmit-01-poll").is_dir()
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count("picsart_terminal_failure_resubmit") == 2
    assert trace.count('"reason": "asset_fetch"') == 2


def test_h3_asset_fetch_failure_resubmits_without_a_cap(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    picsart = _FetchFailPicsart(
        fail_once_for=6,
        message="Failed to download video file: All validated IPs unreachable",
    )
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    # No resubmit cap: each segment burns the initial submit plus three
    # resubmits (beyond the old two-resubmit limit) before succeeding.
    assert len(picsart.submits) == 8
    ledger = state["picsart_submission_receipts"]["segments"]
    for key in ("1", "2"):
        records = ledger[key]
        assert [r["status"] for r in records] == ["provider_failed"] * 3 + [
            "accepted"
        ]
        assert [r["run_identity"] for r in records] == [
            "picsart",
            "picsart-resubmit-01",
            "picsart-resubmit-02",
            "picsart-resubmit-03",
        ]
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count("picsart_terminal_failure_resubmit") == 6


def test_h3_content_moderation_failure_is_also_resubmitted(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    picsart = _FetchFailPicsart(
        fail_once_for=1,
        message="content policy violation: image rejected by moderation",
    )
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    # Every terminal task failure — moderation included — resubmits the
    # frozen segment under a fresh run identity.  Exactly one segment fails
    # terminally once; the other succeeds on its initial submit.
    assert len(picsart.submits) == 3
    ledger = state["picsart_submission_receipts"]["segments"]
    statuses = [r["status"] for key in ("1", "2") for r in ledger[key]]
    assert sorted(statuses) == ["accepted", "accepted", "provider_failed"]
    assert (project / "logs" / "picsart-resubmit-01-submit").is_dir()
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count("picsart_terminal_failure_resubmit") == 1
    assert trace.count('"reason": "terminal_failure"') == 1


def test_node_retries_provider_failures_until_success_with_backoff(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "request-retry-schedule",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    delays: list[float] = []
    monkeypatch.setattr(pipeline, "_retry_sleep", delays.append)
    calls = 0

    def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls <= 4:
            raise ProviderFailure(
                {"outcome": "unknown", "attempt": calls},
                [],
                OSError("connection ended"),
            )
        return "ok"

    result = pipeline._node(log, "deepseek-call", "project", {}, [], flaky)

    assert result == "ok"
    assert calls == 5
    # Attempts 2 and 3 rerun immediately; attempt 4 waits 60s, attempt 5 120s.
    assert delays == [60.0, 120.0]
    for attempt in (1, 2, 3, 4):
        assert (
            project / "logs" / "deepseek-call" / "project" / f"attempt-{attempt}"
        ).is_dir()


def test_node_retries_moderation_rejections_with_backoff_and_logs(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "request-rejection",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    delays: list[float] = []
    monkeypatch.setattr(pipeline, "_retry_sleep", delays.append)
    calls = 0

    def rejected() -> str:
        nonlocal calls
        calls += 1
        if calls <= 6:
            raise ProviderFailure(
                {"outcome": "rejected", "error": "image_rejected by moderation"},
                [],
                RuntimeError("rejected"),
            )
        return "ok"

    # Moderation rejections never fail the project: the node backs off and
    # retries, logging a prominent event once the same rejection persists.
    assert pipeline._node(log, "seedream-submit", "project", {}, [], rejected) == "ok"
    assert calls == 7
    assert delays[:2] == [60.0, 120.0]
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count("node_moderation_persistent") == 2  # attempts 5 and 6
    assert '"consecutive": 6' in trace


def test_node_retries_transient_model_output_errors(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "request-output-parse",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    cases = (
        json.JSONDecodeError("Expecting ',' delimiter", "{", 0),
        ValueError("visual prompt output is invalid"),
        media.MediaError("ASR output is invalid"),
    )
    for index, exc in enumerate(cases):
        calls = 0

        def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise exc
            return "ok"

        node = f"deepseek-call-{index}"
        assert pipeline._node(log, node, "project", {}, [], flaky) == "ok"
        assert calls == 2


def test_node_retries_value_media_and_assertion_errors(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "request-other-value-error",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    cases = (
        ValueError("stored source path must be inputs/source.mp4"),
        ValueError("output is invalid but keeps talking"),
        media.MediaError("audio stream is missing"),
        AssertionError("defensive leftover"),
    )
    for index, exc in enumerate(cases):
        calls = 0

        def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise exc
            return "ok"

        node = f"probe-{index}"
        assert pipeline._node(log, node, "project", {}, [], flaky) == "ok"
        assert calls == 2


def test_node_never_retries_programmer_bug_errors(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create(
        _effective_request(), "request-programmer-bug",
    )
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    log = ResearchLog(project)
    cases = (
        TypeError("bug"),
        KeyError("bug"),
        AttributeError("bug"),
    )
    for index, exc in enumerate(cases):
        calls = 0

        def broken() -> None:
            nonlocal calls
            calls += 1
            raise exc

        with pytest.raises(type(exc)):
            pipeline._node(log, f"bug-{index}", "project", {}, [], broken)
        assert calls == 1


def test_node_retry_delay_schedule() -> None:
    assert Pipeline._node_retry_delay(1) == 0.0
    assert Pipeline._node_retry_delay(2) == 0.0
    assert Pipeline._node_retry_delay(3) == 60.0
    assert Pipeline._node_retry_delay(4) == 120.0
    assert Pipeline._node_retry_delay(5) == 240.0
    assert Pipeline._node_retry_delay(20) == 1800.0


def _node_pipeline(tmp_path: Path, client_request_id: str) -> tuple[Pipeline, ResearchLog]:
    store = ProjectStore(tmp_path / "projects")
    _project_id, project, _state = store.create({}, client_request_id)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    return pipeline, ResearchLog(project)


def test_retryable_node_error_covers_subprocess_and_process_failures() -> None:
    assert Pipeline._retryable_node_error(subprocess.TimeoutExpired(["ffmpeg"], 10))
    assert Pipeline._retryable_node_error(
        media.CommandError(["ffmpeg"], 1, "", "boom")
    )
    assert Pipeline._retryable_node_error(CodexProcessError("codex exited"))
    assert Pipeline._retryable_node_error(ValueError("bad input"))
    assert Pipeline._retryable_node_error(AssertionError("leftover"))
    assert not Pipeline._retryable_node_error(
        face_mask.FaceMaskError("could not open video: x")
    )


def test_node_retries_subprocess_timeouts_until_success(tmp_path: Path) -> None:
    pipeline, log = _node_pipeline(tmp_path, "request-timeout-retry")
    calls = 0

    def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(["ffmpeg", "-i", "in"], 900)
        return "ok"

    assert pipeline._node(log, "ffmpeg-concat", "project", {}, [], flaky) == "ok"
    assert calls == 2
    assert (
        Path(log.root) / "ffmpeg-concat" / "project" / "attempt-1" / "error.json"
    ).is_file()


def test_face_mask_failure_degrades_to_unmasked_reference(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    pipeline, project, item, enhanced_index, replacement_sheet, mask_calls = (
        _prepare_h3_mask_fixture(tmp_path, monkeypatch)
    )

    def broken_mask(_source: Path, _destination: Path, **_kwargs: Any) -> None:
        raise face_mask.FaceMaskError("could not open video: source-muted.mp4")

    monkeypatch.setattr(face_mask, "mask_faces", broken_mask)

    _item, _prompt, muted, _fusion_sha256 = pipeline._prepare_h3_segment(
        project,
        item,
        enhanced_index,
        replacement_sheet,
        _effective_request(),
        ResearchLog(project),
    )

    # Masking is load-bearing but must never block the render: after
    # FACE_MASK_MAX_ATTEMPTS failures the unmasked reference is used and a
    # prominent degradation event is logged.
    segment_dir = project / "work" / "segments" / "0001"
    assert muted == segment_dir / "source-muted.mp4"
    assert mask_calls == []  # the patched mask_faces replaced the fake
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "face_mask_degraded" in trace
    assert '"attempts": 3' in trace
    attempts = (
        project / "logs" / "mask-reference-faces" / "segment-0001"
    ).iterdir()
    assert sorted(path.name for path in attempts) == [
        f"attempt-{index}" for index in range(1, FACE_MASK_MAX_ATTEMPTS + 1)
    ]


def test_face_mask_missing_model_stays_fatal(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    pipeline, project, item, enhanced_index, replacement_sheet, _mask_calls = (
        _prepare_h3_mask_fixture(tmp_path, monkeypatch)
    )
    calls = 0

    def missing_model(_source: Path, _destination: Path, **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise face_mask.FaceMaskError("face detection model is missing")

    monkeypatch.setattr(face_mask, "mask_faces", missing_model)

    # A missing detection model is a configuration error: no retry, no
    # degrade — the segment fails loudly.
    with pytest.raises(face_mask.FaceMaskError, match="model is missing"):
        pipeline._prepare_h3_segment(
            project,
            item,
            enhanced_index,
            replacement_sheet,
            _effective_request(),
            ResearchLog(project),
        )
    assert calls == 1
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "face_mask_degraded" not in trace


def test_h3_task_hang_limit_resubmits_with_fresh_identity(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    class _HungPicsart(_FlowPicsart):
        def poll(self, task_id: str) -> ProviderResult[str]:
            if task_id == "task-1":
                clock["now"] += 22000  # past the default 21600s hang limit
                raise ProviderFailure(
                    {"task_id": task_id, "polls": [{"status": "RUNNING"}]},
                    [],
                    RuntimeError("Picsart task still running"),
                )
            return super().poll(task_id)

    picsart = _HungPicsart()
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    ledger = state["picsart_submission_receipts"]["segments"]
    # Both segments were submitted before either polled, so the frozen fake
    # clock is already past the hang limit when each task is first polled:
    # both hang once and resubmit under a fresh identity.
    for key in ("1", "2"):
        assert [r["status"] for r in ledger[key]] == ["provider_failed", "accepted"]
        assert [r["run_identity"] for r in ledger[key]] == [
            "picsart",
            "picsart-resubmit-01",
        ]
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count('"reason": "task_hung"') == 2
    assert trace.count("picsart_terminal_failure_resubmit") == 2


def test_missing_full_track_degrades_to_stitched_video(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    def customize() -> None:
        base_concat = media.concat

        def concat(
            paths: list[Path],
            destination: Path,
            **kwargs: Any,
        ) -> dict[str, Any]:
            result = base_concat(paths, destination, **kwargs)
            # Simulate the assembled dialogue track going missing between
            # dialogue-audio assembly and the final audio replacement.
            for track in (tmp_path / "projects").glob(
                "*/work/dialogue-audio/full-track.wav"
            ):
                track.unlink()
            return result

        monkeypatch.setattr(media, "concat", concat)

    store, project_id, project = _run_h3_with_picsart(
        tmp_path, monkeypatch, _FlowPicsart(), customize,
    )

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    assert state["has_video"] is True
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "dialogue_track_missing_degraded" in trace
    assert not (project / "logs" / "replace-audio-track").exists()


def test_pipeline_failure_persists_actionable_error_code(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "request-actionable-error",
    )
    source = project / "inputs" / "source.mp4"
    source.write_bytes(b"source")
    store.update(project_id, source_path="inputs/source.mp4", has_source=True)
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    monkeypatch.setattr(
        media,
        "probe",
        lambda _source: (_ for _ in ()).throw(
            media.MediaError("source video has no audio track")
        ),
    )
    # MediaError is retryable under the node gate (uncertainty self-heals);
    # disable the gate here to exercise the terminal error-code mapping.
    monkeypatch.setattr(
        Pipeline,
        "_retryable_node_error",
        classmethod(lambda _cls, _exc: False),
    )

    pipeline.run(project_id)

    state = store.read(project_id)
    assert state["status"] == "failed"
    # The public error map keys off this code: it must be the actionable
    # MediaError code, not the generic pipeline_failed.
    assert state["error"] == {
        "code": "source video has no audio track",
        "message": "输入视频不满足处理要求",
    }
    assert state["generation"]["error"] == "source video has no audio track"


class _RejectOnceSeedream(_FlowSeedream):
    """First POST ends in a terminal outcome; the re-POST succeeds."""

    def __init__(self, code: str) -> None:
        super().__init__()
        self._code = code
        self._failed_once = False

    def generate(
        self,
        *,
        prompt: str,
        images: list[Path],
        destination: Path,
        receipt_path: Path,
    ) -> ProviderResult[Path]:
        if not self._failed_once:
            self._failed_once = True
            self.calls.append({
                "prompt": prompt,
                "images": list(images),
                "destination": destination,
                "receipt_path": receipt_path,
                "failed": True,
            })
            raise ProviderFailure(
                {"provider": "ark_seedream", "outcome": "rejected"},
                [],
                SeedreamError(self._code, f"Seedream attempt ended: {self._code}"),
            )
        result = _FlowSeedream.generate(
            self,
            prompt=prompt,
            images=images,
            destination=destination,
            receipt_path=receipt_path,
        )
        self.calls[-1]["failed"] = False
        return result


def _run_sheet_with_seedream(
    tmp_path: Path,
    monkeypatch: Any,
    seedream: _FlowSeedream,
) -> tuple[ProjectStore, str, Path, list[float]]:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create(
        _effective_request(), "sheet-regeneration",
    )
    (project / "inputs" / "source.mp4").write_bytes(b"project-source")
    enhanced_element_index = {
        "people": {
            "person-01": {
                "source_visual_description": "the source person",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replacement": {
                    "reason": "automatic",
                    "source": "generated_variant",
                    "target_visual_description": "a different person",
                    "appearance_description": "an isolated different person",
                },
            },
        },
        "entities": {},
        "scenes": {},
        "relations": {},
    }
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=seedream,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(pipeline, "_retry_sleep", sleeps.append)
    _materialize_test_replacement_sheet(
        pipeline,
        project_id,
        project,
        [{
            "index": 1,
            "start_s": 0.0,
            "end_s": 5.0,
            "duration_s": 5.0,
            "join_mode": "hard_cut",
            "visual_prompt": "frozen visual prompt",
            "dialogue": [],
            "analysis_paths": [],
        }],
        enhanced_element_index,
    )
    return store, project_id, project, sleeps


def test_seedream_rejection_regenerates_with_perturbed_prompt(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    seedream = _RejectOnceSeedream("provider_rejected")
    store, project_id, project, sleeps = _run_sheet_with_seedream(
        tmp_path, monkeypatch, seedream,
    )

    assert len(seedream.calls) == 2
    assert [call["failed"] for call in seedream.calls] == [True, False]
    # Content rejections perturb the prompt before re-POSTing.
    assert "Regeneration 1" in seedream.calls[1]["prompt"]
    assert seedream.calls[0]["prompt"] != seedream.calls[1]["prompt"]
    assert sleeps == [60.0]
    sheet = store.read(project_id)["replacement_sheet"]
    assert sheet["status"] == "succeeded"
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "seedream_submission_regenerated" in trace
    assert '"reason": "provider_rejected"' in trace


def test_seedream_submission_unknown_reposts_the_frozen_prompt(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    seedream = _RejectOnceSeedream("submission_unknown")
    store, project_id, project, sleeps = _run_sheet_with_seedream(
        tmp_path, monkeypatch, seedream,
    )

    assert len(seedream.calls) == 2
    assert [call["failed"] for call in seedream.calls] == [True, False]
    # Unknown submission outcomes re-POST the frozen prompt unchanged.
    assert seedream.calls[0]["prompt"] == seedream.calls[1]["prompt"]
    assert sleeps == [60.0]
    assert store.read(project_id)["replacement_sheet"]["status"] == "succeeded"
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert "seedream_submission_regenerated" in trace
    assert '"reason": "submission_unknown"' in trace


def test_node_attempt_numbering_continues_past_failed_evidence(
    tmp_path: Path,
) -> None:
    """A resumed node must not overwrite a failed attempt's raw evidence."""
    store = ProjectStore(tmp_path / "projects")
    pipeline = Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
    )
    project = tmp_path / "project"
    log = ResearchLog(project)
    failed = log.attempt_dir("some-node", "project", 1)
    log.write_json(failed / "input.json", {"original": "failed-input"})
    log.error(failed / "error.json", RuntimeError("original failure"))

    result = pipeline._node(
        log, "some-node", "project", {"fresh": 1}, [], lambda: {"ok": True},
    )

    assert result == {"ok": True}
    # The failed attempt keeps its original input and error untouched.
    assert json.loads((failed / "input.json").read_text(encoding="utf-8")) == {
        "original": "failed-input"
    }
    assert (failed / "error.json").is_file()
    continued = project / "logs" / "some-node" / "project" / "attempt-2"
    assert json.loads(
        (continued / "input.json").read_text(encoding="utf-8")
    ) == {"fresh": 1}
    assert json.loads(
        (continued / "output.json").read_text(encoding="utf-8")
    ) == {"ok": True}
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert '"attempt": 2' in trace


def test_translation_and_dialogue_shape_validators() -> None:
    _validate_translation_shape({"lines": []})
    _validate_translation_shape({"lines": [{"text": "你好"}]})
    with pytest.raises(ValueError, match="translation output is invalid"):
        _validate_translation_shape({})
    with pytest.raises(ValueError, match="translation output is invalid"):
        _validate_translation_shape({"lines": [{"text": 3}]})

    _validate_dialogue_generation_shape(
        {"lines": [{"segment_index": 1, "text": "口播"}]}
    )
    with pytest.raises(
        ValueError, match="dialogue generation output is invalid",
    ):
        _validate_dialogue_generation_shape(
            {"lines": [{"segment_index": "1", "text": "口播"}]}
        )
    with pytest.raises(
        ValueError, match="dialogue generation output is invalid",
    ):
        _validate_dialogue_generation_shape({"lines": [{"segment_index": 1}]})
