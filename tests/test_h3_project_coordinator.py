import hashlib
import json
import subprocess
import threading
import wave
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from fastapi.testclient import TestClient

from app import (
    context_ir_bridge, h3, h3_project, long_generation, long_video,
    prepared_input, stitch, storage,
)
from app.main import (
    _SubmitError,
    _finish_generation,
    _freeze_submission,
    _has_valid_generated_video,
    _replace_source_prompt,
    _run_generation,
    _validate_generated_video_uncached,
    create_app,
)
from conftest import AUTH, make_settings


def _png(path: Path, value: int) -> None:
    image = np.full((160, 90, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _wav(path: Path, *, frequency: int = 330, seconds: int = 2) -> None:
    sample_rate = 8000
    samples = (
        np.sin(2 * np.pi * frequency * np.arange(sample_rate * seconds) / sample_rate)
        * 12000
    ).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def _video(path: Path, *, frequency: int, duration: float, color: str = "black") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=90x160:r=24:d={duration}",
            "-f", "lavfi", "-i",
            f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
        capture_output=True,
    )


def _plan(visual_prompt: str, dialogue_source_sha256: str) -> dict:
    return {
        "version": 2,
        "phase": "multimodal_audio",
        "eligible": True,
        "reason": None,
        "visual_prompt": visual_prompt,
        "dialogue_source_sha256": dialogue_source_sha256,
        "subjects": [
            {"subject_id": "S1", "picture_refs": [1], "voice_ref": 1},
        ],
        "audio_refs": [
            {"audio_index": 1, "purpose": "voice", "subject_id": "S1"},
            {"audio_index": 2, "purpose": "ambience", "subject_id": None},
        ],
        "speech_bindings": [
            {
                "line_index": 1,
                "delivery": "on_screen",
                "subject_id": "S1",
                "language": "Chinese",
                "voice_ref": None,
            },
            {
                "line_index": 2,
                "delivery": "off_screen_voiceover",
                "subject_id": None,
                "language": "Chinese",
                "voice_ref": 1,
            },
        ],
        "sound_design": {
            "ambience_refs": [
                {"audio_index": 2, "description": "远处连续雨声"},
            ],
            "effects": [],
        },
    }


def _multimodal_source(
    work: Path, visual_prompt: str, *, plan: dict,
    keyframes: list[Path] | None = None,
) -> Path:
    plan_path = work / "h3_prompt_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    audio_bindings = []
    for item in plan["audio_refs"]:
        order = item["audio_index"]
        purpose = item["purpose"]
        audio = work / f"reference-{order}-{purpose}.wav"
        _wav(audio, frequency=90 + order * 120)
        audio_bindings.append({
            "order": order,
            "path": audio.name,
            "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            "purpose": purpose,
        })
    selected_keyframes = keyframes or sorted((work / "keyframes").glob("*.png"))
    skill_input = {
        "schema": h3_project.SKILL_INPUT_SCHEMA,
        "version": h3_project.SKILL_INPUT_VERSION,
        "visual_prompt": {
            "path": "visual_prompt.txt",
            "sha256": hashlib.sha256(visual_prompt.encode("utf-8")).hexdigest(),
        },
        "keyframes": [
            {
                "order": order,
                "path": path.resolve().relative_to(work.resolve()).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for order, path in enumerate(selected_keyframes, 1)
        ],
        "dialogue_source_sha256": plan["dialogue_source_sha256"],
        "reference_audios": audio_bindings,
    }
    skill_input_path = work / h3_project.SKILL_INPUT_FILENAME
    skill_input_path.write_text(
        json.dumps(skill_input, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "schema": "duet.h3-multimodal-source",
        "version": h3_project.SOURCE_VERSION,
        "mode": "multimodal",
        "approved_skill_plan_sha256": h3.canonical_json_sha256(plan),
        "multimodal_input": {
            "path": skill_input_path.name,
            "sha256": hashlib.sha256(skill_input_path.read_bytes()).hexdigest(),
        },
        "skill_plan": {
            "path": plan_path.name,
            "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        },
        "reference_audios": audio_bindings,
    }
    path = work / h3_project.SOURCE_FILENAME
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def _projection_plan(
    visual_prompt: str, *, speech: bool, dialogue_source_sha256: str,
) -> dict:
    return {
        "version": 2,
        "phase": "multimodal_audio",
        "eligible": True,
        "reason": None,
        "visual_prompt": visual_prompt,
        "dialogue_source_sha256": dialogue_source_sha256,
        "subjects": ([
            {"subject_id": "S1", "picture_refs": [1], "voice_ref": 1},
        ] if speech else []),
        "audio_refs": ([
            {"audio_index": 1, "purpose": "voice", "subject_id": "S1"},
        ] if speech else [
            {"audio_index": 1, "purpose": "ambience", "subject_id": None},
        ]),
        "speech_bindings": ([
            {
                "line_index": 1,
                "delivery": "on_screen",
                "subject_id": "S1",
                "language": "Chinese",
                "voice_ref": None,
            },
        ] if speech else []),
        "sound_design": {
            "ambience_refs": ([] if speech else [
                {"audio_index": 1, "description": "远处连续雨声"},
            ]),
            "effects": [],
        },
    }


class _Gateway:
    def __init__(self, outputs: list[bytes], *, ambiguous: bool = False) -> None:
        self.outputs = outputs
        self.ambiguous = ambiguous
        self.posts: list[dict] = []
        self._lock = threading.Lock()
        self._task_outputs: dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            if request.method == "POST" and request.url.path == "/v1/videos":
                body = json.loads(request.content)
                self.posts.append(body)
                if self.ambiguous:
                    raise httpx.ReadTimeout("ambiguous", request=request)
                task_id = f"task-{len(self.posts)}"
                prompt = body.get("prompt", "")
                output_index = (
                    0 if len(self.outputs) == 2 and "第1段" in prompt
                    else 1 if len(self.outputs) == 2 and "第2段" in prompt
                    else len(self.posts) - 1
                )
                self._task_outputs[task_id] = output_index
                return httpx.Response(
                    201, json={"task_id": task_id}
                )
            if request.method == "GET" and request.url.path.endswith("/content"):
                task_id = request.url.path.split("/")[-2]
                index = self._task_outputs[task_id]
                return httpx.Response(200, content=self.outputs[index])
            if request.method == "GET" and request.url.path.startswith("/v1/videos/task-"):
                return httpx.Response(200, json={"status": "succeeded"})
        raise AssertionError(f"unexpected fake Gateway call: {request.method} {request.url}")


class _ContextGateway:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._file_index = 0
        self._task_index = 0
        self.tasks: dict[str, str] = {}
        self.posts: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            if request.method == "POST" and request.url.path == "/v1/files/upload":
                self._file_index += 1
                return httpx.Response(200, json={
                    "file": {"file_id": str(427752006353317 + self._file_index)},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                })
            if request.method == "POST" and request.url.path == "/v2/h3_context_ir":
                body = json.loads(request.content)
                self.posts.append(body)
                self._task_index += 1
                task_id = f"context-task-{self._task_index}"
                self.tasks[task_id] = body["content"][0]["text"]
                return httpx.Response(200, json={"task_id": task_id})
            prefix = "/v2/query/video_generation/"
            if request.method == "GET" and request.url.path.startswith(prefix):
                task_id = request.url.path.removeprefix(prefix)
                source_prompt = self.tasks[task_id]
                return httpx.Response(200, json={"task": {
                    "id": task_id,
                    "task_type": "h3_context_ir",
                    "status": "succeeded",
                    "modality": "text",
                    "content": {
                        "prompt": source_prompt
                        + "\nContext IR retained the exact speech contract.",
                    },
                }})
        raise AssertionError(
            f"unexpected fake Context IR call: {request.method} {request.url}"
        )


def _client(gateway: _Gateway) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(gateway))


def _install_fake_context(monkeypatch, gateway: _ContextGateway) -> None:
    real_optimize = context_ir_bridge.optimize_h3_prompt

    def optimize(frozen):
        with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
            return real_optimize(frozen, client=client)

    monkeypatch.setattr(context_ir_bridge, "optimize_h3_prompt", optimize)


def _install_fake_h3(monkeypatch, gateway: _Gateway) -> None:
    real_start = h3.start

    def start(request):
        with _client(gateway) as client:
            return real_start(request, client=client)

    monkeypatch.setattr(h3, "start", start)


def _decode_audio(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
            "-f", "f32le", "-ac", "1", "-ar", "48000", "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def _tone(samples: np.ndarray, frequency: int) -> float:
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(samples), 1 / 48000)
    return float(spectrum[np.argmin(np.abs(frequencies - frequency))])


def test_short_project_freezes_skill_audio_contract_and_stitches_only_h3_audio(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        autodl_art_token="not-sent-to-gateway",
        minimax_api_key="not-sent-to-context",
    )
    created = storage.new_conversation(
        settings.data_dir, "short project", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    work = root / "work"
    key = work / "keyframes" / "01.png"
    visual = work / "visual_prompt.txt"
    final = work / "prompt.txt"
    source = root / "source.mp4"
    _png(key, 40)
    visual.parent.mkdir(parents=True, exist_ok=True)
    visual.write_text("雨夜车站，人物面对镜头。", encoding="utf-8")
    _video(source, frequency=220, duration=4)
    dialogue = prepared_input.prepare_dialogue(
        "custom",
        4,
        supplied_lines=[
            {"text": "我会准时回来。", "start_s": 0.2, "end_s": 1.8},
            {"text": "雨仍然没有停。", "start_s": 2.0, "end_s": 3.8},
        ],
    )
    # The pipeline-time Skill output is stale by design: custom dialogue only
    # becomes authoritative at submit. The first submit must freeze it and ask
    # the external Skill stage to refresh its binding-only plan.
    _multimodal_source(
        work,
        visual.read_text(encoding="utf-8"),
        plan=_projection_plan(
            visual.read_text(encoding="utf-8"),
            speech=False,
            dialogue_source_sha256=h3.canonical_json_sha256([]),
        ),
    )

    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        duration_s=4,
        keyframes=["01.png"],
        vocal_filter_enabled=True,
        voice_lines=[],
        voice_line_provenance=[],
        source_width=90,
        source_height=160,
        fit_required=False,
        fit_profiles={
            "9:16": {"fit_required": False, "default_fit_mode": "none"},
            "16:9": {"fit_required": True, "default_fit_mode": "crop"},
        },
    )
    provider = tmp_path / "provider-short.mp4"
    _video(provider, frequency=440, duration=4, color="red")
    gateway = _Gateway([provider.read_bytes()])
    context_gateway = _ContextGateway()
    _install_fake_context(monkeypatch, context_gateway)
    _install_fake_h3(monkeypatch, gateway)
    payload = {
        "confirm": True,
        "client_request_id": "short-project-request",
        "dialogue_mode": "custom",
        "fit_mode": "none",
        "aspect_ratio": "9:16",
        "resolution": "768p",
        "lines": [
            {key: line[key] for key in ("text", "start_s", "end_s")}
            for line in dialogue
        ],
    }
    with TestClient(create_app(settings)) as client:
        refresh = client.post(
            f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
        )
        assert refresh.status_code == 409
        assert refresh.json() == {"detail": "multimodal_plan_refresh_required"}
        refreshed = prepared_input.load_prepared_input(
            root,
            root / prepared_input.RECEIPT_FILENAME,
            expected_dialogue=dialogue,
        )
        assert refreshed.dialogue == dialogue
        _multimodal_source(
            work,
            visual.read_text(encoding="utf-8"),
            plan=_plan(
                visual.read_text(encoding="utf-8"),
                refreshed.dialogue_sha256,
            ),
        )
        response = client.post(
            f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
        )
        assert response.status_code == 202

    assert len(gateway.posts) == 1
    assert len(context_gateway.posts) == 1
    body = gateway.posts[0]
    assert set(body) == {
        "mode", "prompt", "duration_sec", "aspect_ratio", "resolution",
        "images", "audios",
    }
    assert "<d>[Chinese]我会准时回来。</d>" in body["prompt"]
    assert "visible lips articulate" in body["prompt"]
    assert "off-screen voiceover" in body["prompt"]
    assert "lips remain completely closed" in body["prompt"]
    assert "overall_soundscape" in body["prompt"]
    source_prompt = context_gateway.posts[0]["content"][0]["text"]
    assert body["prompt"] != source_prompt
    assert body["prompt"].endswith("Context IR retained the exact speech contract.")
    assert [item["kind"] for item in body["audios"]] == ["voice", "sound"]
    samples = _decode_audio(root / "generated.mp4")
    assert _tone(samples, 440) > 20 * _tone(samples, 220)
    receipt = json.loads((root / stitch.RECEIPT_FILENAME).read_text())
    assert receipt["audio"]["mode"] == "provider_generated"
    provider_attempt_id = receipt["audio"]["provider_segments"][0]["attempt_id"]
    stored = storage.load_meta(settings.data_dir, cid)["generation"]
    assert stored["status"] == "succeeded"
    assert stored["stage"] == "stitch"
    assert stored["h3_attempt_id"] == provider_attempt_id
    assert stored["context_ir"]["status"] == "succeeded"
    complete_meta = storage.load_meta(settings.data_dir, cid)
    assert _has_valid_generated_video(settings, complete_meta) is True
    context_receipt = Path(str(stored["context_ir"]["receipt_path"]))
    context_receipt_bytes = context_receipt.read_bytes()
    context_receipt.write_text("{}", encoding="utf-8")
    assert _has_valid_generated_video(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is False
    context_receipt.write_bytes(context_receipt_bytes)
    assert _has_valid_generated_video(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is True
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={**stored, "h3_attempt_id": "999999"},
    )
    assert _validate_generated_video_uncached(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is False


def test_partial_multimodal_contract_fails_before_attempt_or_post(tmp_path):
    root = tmp_path / "partial"
    work = root / "work"
    root.mkdir(parents=True)
    work.mkdir()
    plan = _plan("视觉", "0" * 64)
    (work / "h3_prompt_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(h3_project.ProjectMultimodalError, match="multimodal_source_missing"):
        h3_project.freeze_optional(root, work)
    assert not (root / ".h3").exists()


def test_short_query_unknown_with_missing_context_attempt_never_posts(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        autodl_art_token="not-sent",
        minimax_api_key="not-sent-to-context",
    )
    created = storage.new_conversation(settings.data_dir, "short resume", "source.mp4")
    cid = created["id"]
    root = settings.data_dir / cid
    work = root / "work"
    source = root / "source.mp4"
    key = work / "keyframes" / "01.png"
    visual = work / "visual_prompt.txt"
    final = work / "prompt.txt"
    _video(source, frequency=220, duration=4)
    _png(key, 40)
    visual.parent.mkdir(parents=True, exist_ok=True)
    visual.write_text("无人物的雨夜街道。", encoding="utf-8")
    _multimodal_source(
        work,
        visual.read_text(encoding="utf-8"),
        plan=_projection_plan(
            visual.read_text(encoding="utf-8"),
            speech=False,
            dialogue_source_sha256=h3.canonical_json_sha256([]),
        ),
    )
    frozen = prepared_input.write_prepared_input(
        root=root,
        source=source,
        audio=None,
        keyframes=[key],
        visual=visual,
        final=final,
        dialogue_mode="none",
        dialogue=(),
        vocal_filter_enabled=True,
        duration_s=4,
        ratio="9:16",
        fit_mode="none",
        engine_request={"h3": {
            "workflow": h3.H3_MULTIMODAL_WORKFLOW,
            "duration": 4,
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "provider_resolution": "768p竖",
        }},
        multimodal=h3_project.freeze_optional(root, work),
    )
    request = h3_project.build_request(
        frozen=frozen,
        cid=cid,
        workdir=work / "h3-native",
        client_request_id="short-query-unknown",
        duration=4,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token="not-sent",
    )
    context = h3_project.freeze_context_ir(
        source_request=request,
        upstream_dialogue_sha256=frozen.dialogue_sha256,
        upstream_artifact_path=frozen.receipt_path,
        upstream_artifact_sha256=frozen.receipt_sha256,
        upstream_dialogue_sha256_path=("dialogue", "sha256"),
        minimax_api_key=settings.minimax_api_key,
        request_timeout_s=1,
        poll_timeout_s=1,
        poll_interval_s=0,
    )
    context_gateway = _ContextGateway()
    with httpx.Client(
        transport=httpx.MockTransport(context_gateway)
    ) as context_client:
        result = context_ir_bridge.optimize_h3_prompt(
            context, client=context_client
        )
    binding = h3_project.context_ir_binding(result)
    binding["status"] = "query_unknown"
    receipt_path = Path(str(binding["receipt_path"]))
    receipt_path.unlink()
    receipt_path.with_name("attempt.json").unlink()
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        dialogue_mode="none",
        prepared_dialogue=[],
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
        fit_mode="none",
        aspect_ratio="9:16",
        resolution="768p",
        generation={
            "status": "resume_required",
            "error": "context_ir_query_unknown",
            "attempt": 1,
            "client_request_id": request.client_request_id,
            "stage": "context_ir_native",
            "audio_route": dict(h3_project.AUDIO_ROUTE),
            "h3_attempt_id": None,
            "context_ir": binding,
        },
    )
    monkeypatch.setattr(
        context_ir_bridge,
        "optimize_h3_prompt",
        lambda *_args, **_kwargs: pytest.fail("lost attempt must not POST Context IR"),
    )
    monkeypatch.setattr(
        h3,
        "resume",
        lambda *_args, **_kwargs: pytest.fail("lost Context receipt must not reach H3"),
    )

    _run_generation(settings, cid, request, "resume")

    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation["status"] == "submission_unknown"
    assert len(context_gateway.posts) == 1


def test_short_submit_resumes_reconcilable_context_failure_with_same_client_and_task(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        autodl_art_token="not-sent",
        minimax_api_key="not-sent-to-context",
        h3_poll_interval_s=0,
    )
    created = storage.new_conversation(
        settings.data_dir, "legacy Context recovery", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    work = root / "work"
    source = root / "source.mp4"
    key = work / "keyframes" / "01.png"
    visual = work / "visual_prompt.txt"
    final = work / "prompt.txt"
    _video(source, frequency=220, duration=4)
    _png(key, 40)
    visual.parent.mkdir(parents=True, exist_ok=True)
    visual.write_text("雨夜车站，人物面对镜头。", encoding="utf-8")
    dialogue = prepared_input.prepare_dialogue(
        "custom",
        4,
        supplied_lines=[
            {"text": "准备好了。", "start_s": 0.4, "end_s": 2.2},
        ],
    )
    _multimodal_source(
        work,
        visual.read_text(encoding="utf-8"),
        plan=_projection_plan(
            visual.read_text(encoding="utf-8"),
            speech=True,
            dialogue_source_sha256=h3.canonical_json_sha256(list(dialogue)),
        ),
    )
    frozen = prepared_input.write_prepared_input(
        root=root,
        source=source,
        audio=None,
        keyframes=[key],
        visual=visual,
        final=final,
        dialogue_mode="custom",
        dialogue=dialogue,
        vocal_filter_enabled=True,
        duration_s=4,
        ratio="9:16",
        fit_mode="none",
        engine_request={"h3": {
            "workflow": h3.H3_MULTIMODAL_WORKFLOW,
            "duration": 4,
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "provider_resolution": "768p竖",
        }},
        multimodal=h3_project.freeze_optional(root, work),
    )
    request = h3_project.build_request(
        frozen=frozen,
        cid=cid,
        workdir=work / "h3-native",
        client_request_id="legacy-context-recovery",
        duration=4,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token=settings.autodl_art_token,
    )
    context = h3_project.freeze_context_ir(
        source_request=request,
        upstream_dialogue_sha256=frozen.dialogue_sha256,
        upstream_artifact_path=frozen.receipt_path,
        upstream_artifact_sha256=frozen.receipt_sha256,
        upstream_dialogue_sha256_path=("dialogue", "sha256"),
        minimax_api_key=settings.minimax_api_key,
        request_timeout_s=1,
        poll_timeout_s=1,
        poll_interval_s=0,
    )
    context_gateway = _ContextGateway()

    def first_attempt(request_: httpx.Request) -> httpx.Response:
        if request_.method == "GET":
            raise httpx.ReadTimeout("ambiguous query", request=request_)
        return context_gateway(request_)

    with httpx.Client(transport=httpx.MockTransport(first_attempt)) as client:
        first = context_ir_bridge.optimize_h3_prompt(context, client=client)
    assert first.status == "query_unknown"
    binding = h3_project.context_ir_binding(first)
    binding["status"] = "failed"
    attempt_path = (
        request.workdir / ".context-ir" / "attempts" / "000001" / "attempt.json"
    )
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["status"] = "failed"
    attempt["error"] = "context_ir_semantic_mismatch"
    attempt_path.write_text(
        json.dumps(attempt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        duration_s=4,
        vocal_filter_enabled=True,
        dialogue_mode="custom",
        prepared_dialogue=[dict(line) for line in dialogue],
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
        fit_mode="none",
        aspect_ratio="9:16",
        resolution="768p",
        fit_required=False,
        fit_profiles={
            "9:16": {"fit_required": False, "default_fit_mode": "none"},
            "16:9": {"fit_required": True, "default_fit_mode": "crop"},
        },
        generation={
            "status": "failed",
            "error": "context_ir_semantic_mismatch",
            "attempt": 1,
            "client_request_id": request.client_request_id,
            "stage": "context_ir_native",
            "audio_route": dict(h3_project.AUDIO_ROUTE),
            "h3_attempt_id": None,
            "context_ir": binding,
        },
    )
    events: list[str] = []
    query_count = 0
    source_prompt = context_gateway.tasks[str(first.provider_task_id)]

    def recovery_gateway(request_: httpx.Request) -> httpx.Response:
        nonlocal query_count
        assert request_.method == "GET"
        assert request_.url.path.endswith(f"/{first.provider_task_id}")
        query_count += 1
        status = "running" if query_count == 1 else "succeeded"
        events.append(f"context-{status}")
        return httpx.Response(200, json={"task": {
            "id": first.provider_task_id,
            "task_type": "h3_context_ir",
            "status": status,
            "content": {
                "prompt": "" if status == "running" else (
                    source_prompt + "\nContext IR preserved the complete prompt."
                ),
            },
        }})

    real_optimize = context_ir_bridge.optimize_h3_prompt

    def optimize(frozen_context):
        with httpx.Client(
            transport=httpx.MockTransport(recovery_gateway)
        ) as client:
            return real_optimize(frozen_context, client=client)

    monkeypatch.setattr(context_ir_bridge, "optimize_h3_prompt", optimize)

    def start(effective_request):
        events.append("h3")
        assert "Context IR preserved the complete prompt." in effective_request.prompt
        return h3.H3Result("h3_running", "000001")

    monkeypatch.setattr(h3, "start", start)
    payload = {
        "confirm": True,
        "client_request_id": request.client_request_id,
        "dialogue_mode": "custom",
        "fit_mode": "none",
        "aspect_ratio": "9:16",
        "resolution": "768p",
        "lines": [
            {key_: line[key_] for key_ in ("text", "start_s", "end_s")}
            for line in dialogue
        ],
    }
    with TestClient(create_app(settings)) as api:
        drifted = {
            **payload,
            "lines": [{"text": "输入已漂移。", "start_s": 0.4, "end_s": 2.2}],
        }
        rejected = api.post(
            f"/api/conversations/{cid}/submit", headers=AUTH, json=drifted
        )
        assert rejected.status_code == 409
        assert events == []

        stored = storage.load_meta(settings.data_dir, cid)["generation"]
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={**stored, "error": "context_ir_provider_failed"},
        )
        unrelated = api.post(
            f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
        )
        assert unrelated.status_code == 409
        assert unrelated.json() == {"detail": "new client_request_id required"}
        assert events == []
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={**stored, "error": "context_ir_semantic_mismatch"},
        )

        resumed = api.post(
            f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
        )
        assert resumed.status_code == 202, resumed.text

    assert events == ["context-running", "context-succeeded", "h3"]
    assert query_count == 2
    assert len(context_gateway.posts) == 1
    recovered = storage.load_meta(settings.data_dir, cid)["generation"]
    assert recovered["context_ir"]["provider_task_id"] == first.provider_task_id
    assert recovered["context_ir"]["attempt_id"] == "000001"


@pytest.mark.parametrize("mutation", ["bytes", "order"])
def test_skill_input_keyframe_binding_rejects_drift_before_h3(tmp_path, mutation):
    root = tmp_path / f"keyframe-{mutation}"
    work = root / "work"
    visual = work / "visual_prompt.txt"
    first = work / "keyframes" / "01.png"
    second = work / "keyframes" / "02.png"
    _png(first, 20)
    _png(second, 80)
    visual.parent.mkdir(parents=True, exist_ok=True)
    visual.write_text("人物在真实车站内面对镜头。", encoding="utf-8")
    dialogue = prepared_input.prepare_dialogue(
        "custom",
        4,
        supplied_lines=[{"text": "现在出发。", "start_s": 0.2, "end_s": 2.0}],
    )
    dialogue_sha256 = h3.canonical_json_sha256(list(dialogue))
    plan = _projection_plan(
        visual.read_text(encoding="utf-8"),
        speech=True,
        dialogue_source_sha256=dialogue_sha256,
    )
    _multimodal_source(
        work,
        visual.read_text(encoding="utf-8"),
        plan=plan,
        keyframes=[first, second],
    )
    frozen = h3_project.freeze_optional(root, work)
    assert frozen is not None
    keyframes = h3.freeze_keyframes((first, second))
    if mutation == "bytes":
        keyframes = ((keyframes[0][0], b"changed-after-skill"), keyframes[1])
    else:
        keyframes = tuple(reversed(keyframes))

    with pytest.raises(
        h3_project.ProjectMultimodalError,
        match="multimodal_input_runtime_mismatch",
    ):
        h3_project.build_request_from_parts(
            multimodal=frozen,
            visual_prompt=visual.read_text(encoding="utf-8"),
            keyframes=keyframes,
            upstream_dialogue=dialogue,
            upstream_dialogue_receipt_sha256=dialogue_sha256,
            cid=f"keyframe-{mutation}",
            workdir=work / "h3-native",
            client_request_id=f"keyframe-{mutation}",
            duration=4,
            resolution="768p",
            aspect_ratio="9:16",
            autodl_token="not-sent",
        )
    assert not (work / "h3-native" / ".h3").exists()


@pytest.mark.parametrize("dialogue_mode", ["auto", "none"])
def test_two_segment_long_project_builds_each_exact_multimodal_request(
    tmp_path, monkeypatch, dialogue_mode,
):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        autodl_art_token="not-sent",
        minimax_api_key="not-sent-to-context",
    )
    created = storage.new_conversation(
        settings.data_dir, "long project", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    source = root / "source.mp4"
    _video(source, frequency=220, duration=16)
    public_segments = []
    receipt_segments = []
    for index in (1, 2):
        segdir = root / "work" / "segments" / str(index)
        work = segdir / "work"
        segment_source = segdir / "source.mp4"
        key = work / "keyframes" / "01.png"
        first = work / "anchors" / "first.png"
        last = work / "anchors" / "last.png"
        visual = work / "visual_prompt.txt"
        final = work / "prompt.txt"
        _video(segment_source, frequency=220, duration=8)
        _png(key, 20 + index)
        _png(first, 40 + index)
        _png(last, 60 + index)
        visual.parent.mkdir(parents=True, exist_ok=True)
        visual.write_text(f"第{index}段人物面对镜头。", encoding="utf-8")
        segment_dialogue = [
            {
                "text": "我会准时回来。",
                "start_s": 0.2,
                "end_s": 3.0,
                "classification": "spoken",
                "provenance": "asr",
            },
            {
                "text": "雨仍然没有停。",
                "start_s": 3.2,
                "end_s": 7.8,
                "classification": "spoken",
                "provenance": "asr",
            },
        ]
        final.write_text(
            "不要生成背景音乐\n"
            + prepared_input.compose_final_prompt(
                long_video.compose_segment_visual_prompt(
                    visual.read_text(encoding="utf-8")
                ),
                segment_dialogue,
            ),
            encoding="utf-8",
        )
        effective_dialogue = segment_dialogue if dialogue_mode == "auto" else []
        dialogue_source_sha256 = hashlib.sha256(
            (
                json.dumps(
                    effective_dialogue,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        _multimodal_source(
            work,
            visual.read_text(encoding="utf-8"),
            plan=(
                _plan(
                    visual.read_text(encoding="utf-8"),
                    dialogue_source_sha256,
                )
                if dialogue_mode == "auto"
                else _projection_plan(
                    visual.read_text(encoding="utf-8"),
                    speech=False,
                    dialogue_source_sha256=dialogue_source_sha256,
                )
            ),
        )
        start_s = float((index - 1) * 8)
        end_s = float(index * 8)
        public_segments.append(
            {
                "index": index,
                "start_s": start_s,
                "end_s": end_s,
                "chain_id": f"chain-{index}",
                "join_mode": "hard_cut",
                "source": f"segments/{index}/source.mp4",
                "keyframes": ["01.png"],
                "keyframe_paths": [f"segments/{index}/work/keyframes/01.png"],
                "first_frame_path": f"segments/{index}/work/anchors/first.png",
                "last_frame_path": f"segments/{index}/work/anchors/last.png",
                "visual_prompt": visual.read_text(encoding="utf-8"),
                "prompt": final.read_text(encoding="utf-8"),
                "dialogue": segment_dialogue,
                "lines": [line["text"] for line in segment_dialogue],
            }
        )
        receipt_segments.append(
            {
                **public_segments[-1],
                "source_path": segment_source,
                "keyframe_paths": [key],
                "first_frame_path": first,
                "last_frame_path": last,
                "visual_prompt_path": visual,
                "final_prompt_path": final,
            }
        )
    receipt_path = long_video.write_plan_receipt(
        root,
        source=source,
        duration_s=16,
        segments=receipt_segments,
        workflow=h3.H3_WORKFLOW,
    )
    receipt = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    meta = {
        "duration_s": 16,
        "segments": public_segments,
        "long_video_plan_receipt": receipt_path.name,
    }
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        duration_s=16,
        voice_mode="keep",
        fit_mode="none",
        dialogue_mode="auto",
        aspect_ratio="9:16",
        resolution="768p",
        segments=public_segments,
        long_video_plan_receipt=receipt_path.name,
    )
    first_provider = tmp_path / "provider-long-1.mp4"
    second_provider = tmp_path / "provider-long-2.mp4"
    _video(first_provider, frequency=440, duration=8, color="red")
    _video(second_provider, frequency=880, duration=8, color="blue")
    gateway = _Gateway([
        first_provider.read_bytes(),
        second_provider.read_bytes(),
    ])
    context_gateway = _ContextGateway()
    _install_fake_context(monkeypatch, context_gateway)
    _install_fake_h3(monkeypatch, gateway)
    payload = {
        "confirm": True,
        "client_request_id": "long-parent-request",
        "dialogue_mode": dialogue_mode,
        "fit_mode": "none",
        "aspect_ratio": "9:16",
        "resolution": "768p",
        "expected_plan_receipt": receipt,
    }
    with TestClient(create_app(settings)) as client:
        refresh = client.post(
            f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
        )
        assert refresh.status_code == 409
        assert refresh.json()["detail"]["code"] == "long_video_plan_changed"
        promoted = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        assert promoted != receipt
        payload["expected_plan_receipt"] = promoted
        submitted = client.post(
            f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
        )
        assert submitted.status_code == 202, submitted.text

    assert len(gateway.posts) == 2
    assert len(context_gateway.posts) == 2
    assert all(set(body) == {
        "mode", "prompt", "duration_sec", "aspect_ratio", "resolution",
        "images", "audios",
    } for body in gateway.posts)
    assert [body["duration_sec"] for body in gateway.posts] == [8, 8]
    if dialogue_mode == "auto":
        assert all("<d>[Chinese]我会准时回来。</d>" in body["prompt"]
                   for body in gateway.posts)
    else:
        assert all("<d>[" not in body["prompt"] for body in gateway.posts)
        assert all("No audible speech is specified" in body["prompt"]
                   for body in gateway.posts)
    source_prompts = [item["content"][0]["text"] for item in context_gateway.posts]
    assert all(
        body["prompt"] != source_prompt
        and body["prompt"].endswith(
            "Context IR retained the exact speech contract."
        )
        for body, source_prompt in zip(gateway.posts, source_prompts, strict=True)
    )
    samples = _decode_audio(root / "generated.mp4")
    first = samples[int(0.5 * 48000):int(7.5 * 48000)]
    second = samples[int(8.5 * 48000):int(15.5 * 48000)]
    assert _tone(first, 440) > 20 * _tone(first, 220)
    assert _tone(second, 880) > 20 * _tone(second, 220)
    stitched = json.loads((root / stitch.RECEIPT_FILENAME).read_text())
    assert stitched["audio"]["mode"] == "provider_generated"
    assert [
        item["attempt_id"]
        for item in stitched["audio"]["provider_segments"]
    ] == [
        item["h3_attempt_id"]
        for item in storage.load_meta(settings.data_dir, cid)["generation"]["segments"]
    ]
    stored = storage.load_meta(settings.data_dir, cid)
    assert stored["dialogue_mode"] == dialogue_mode
    assert stored["generation"]["status"] == "succeeded"
    assert all(
        item["context_ir"]["status"] == "succeeded"
        for item in stored["generation"]["segments"]
    )
    assert _has_valid_generated_video(settings, stored) is True
    workflow_changed = {
        **stored,
        "generation": {
            **stored["generation"],
            "workflow": h3.H3_WORKFLOW,
        },
    }
    assert _has_valid_generated_video(settings, workflow_changed) is False
    first_context_receipt = Path(str(
        stored["generation"]["segments"][0]["context_ir"]["receipt_path"]
    ))
    first_context_bytes = first_context_receipt.read_bytes()
    first_context_receipt.write_text("{}", encoding="utf-8")
    assert _has_valid_generated_video(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is False
    first_context_receipt.write_bytes(first_context_bytes)
    assert _has_valid_generated_video(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is True
    promoted_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    multimodal_input_binding = (
        promoted_payload["segments"][0]["multimodal"]["multimodal_input"]
    )
    first_multimodal_input = root / multimodal_input_binding["path"]
    first_multimodal_input_bytes = first_multimodal_input.read_bytes()
    first_multimodal_input.write_text("{}", encoding="utf-8")
    assert _has_valid_generated_video(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is False
    first_multimodal_input.write_bytes(first_multimodal_input_bytes)
    assert _has_valid_generated_video(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is True
    damaged_generation = dict(stored["generation"])
    damaged_segments = [dict(item) for item in damaged_generation["segments"]]
    damaged_generation.pop("audio_route")
    damaged_generation.pop("workflow")
    for item in damaged_segments:
        item.pop("h3_attempt_id")
        item.pop("context_ir")
    damaged_generation["segments"] = damaged_segments
    storage.update_meta(
        settings.data_dir, cid, generation=damaged_generation
    )
    monkeypatch.setattr(
        h3,
        "legacy_succeeded_output_is_valid",
        lambda *_args, **_kwargs: pytest.fail(
            "native-audio validation must not fall back to legacy source audio"
        ),
    )
    assert _validate_generated_video_uncached(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is False
    promoted_receipt_bytes = receipt_path.read_bytes()
    receipt_path.write_text("{}", encoding="utf-8")
    assert _validate_generated_video_uncached(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is False
    receipt_path.unlink()
    assert _validate_generated_video_uncached(
        settings, storage.load_meta(settings.data_dir, cid)
    ) is False
    receipt_path.write_bytes(promoted_receipt_bytes)
    if dialogue_mode == "auto":
        plan = long_generation.freeze_plan(
            root,
            stored,
            promoted,
            "none",
            dialogue_mode,
            aspect_ratio="9:16",
            resolution="768p",
            prepare_fit=False,
        )
        query_generation = {
            **stored["generation"],
            "status": "resume_required",
            "error": "context_ir_query_unknown",
        }
        query_segments = []
        for item in stored["generation"]["segments"]:
            binding = dict(item["context_ir"])
            binding["status"] = "query_unknown"
            receipt_path = Path(str(binding["receipt_path"]))
            receipt_path.unlink()
            receipt_path.with_name("attempt.json").unlink()
            query_segments.append({
                **item,
                "status": "resume_required",
                "error": "context_ir_query_unknown",
                "child_request_id": None,
                "h3_attempt_id": None,
                "context_ir": binding,
            })
        query_generation["segments"] = query_segments
        storage.update_meta(
            settings.data_dir, cid, generation=query_generation
        )
        monkeypatch.setattr(
            context_ir_bridge,
            "optimize_h3_prompt",
            lambda *_args, **_kwargs: pytest.fail(
                "lost long Context attempt must not POST"
            ),
        )
        monkeypatch.setattr(
            h3,
            "resume",
            lambda *_args, **_kwargs: pytest.fail(
                "lost long Context receipt must not reach H3"
            ),
        )

        long_generation.run(settings, cid, plan, startup=True)

        resumed = storage.load_meta(settings.data_dir, cid)["generation"]
        assert resumed["status"] == "submission_unknown"
        assert len(context_gateway.posts) == 2
        assert len(gateway.posts) == 2


def test_submission_unknown_has_zero_stitch_and_zero_followup_post(tmp_path, monkeypatch):
    root = tmp_path / "unknown-project"
    work = root / "work"
    key = work / "keyframes" / "01.png"
    visual = work / "visual_prompt.txt"
    final = work / "prompt.txt"
    source = root / "source.mp4"
    _png(key, 40)
    visual.parent.mkdir(parents=True, exist_ok=True)
    visual.write_text("雨夜车站，人物面对镜头。", encoding="utf-8")
    _video(source, frequency=220, duration=4)
    _multimodal_source(
        work,
        visual.read_text(encoding="utf-8"),
        plan=_projection_plan(
            visual.read_text(encoding="utf-8"),
            speech=False,
            dialogue_source_sha256=h3.canonical_json_sha256([]),
        ),
    )
    multimodal = h3_project.freeze_optional(root, work)
    frozen = prepared_input.write_prepared_input(
        root=root,
        source=source,
        audio=None,
        keyframes=[key],
        visual=visual,
        final=final,
        dialogue_mode="none",
        dialogue=(),
        vocal_filter_enabled=True,
        duration_s=4,
        ratio="9:16",
        fit_mode="none",
        engine_request={"h3": {
            "workflow": h3.H3_MULTIMODAL_WORKFLOW,
            "duration": 4,
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "provider_resolution": "768p竖",
        }},
        multimodal=multimodal,
    )
    request = h3_project.build_request(
        frozen=frozen,
        cid="unknown-project",
        workdir=work / "h3-native",
        client_request_id="unknown-request",
        duration=4,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token="not-sent",
        timeouts=h3.Timeouts(
            request_s=1, h3_poll_s=1, download_s=1, poll_interval_s=0,
            retry_count=0, retry_interval_s=0,
        ),
    )
    gateway = _Gateway([], ambiguous=True)
    monkeypatch.setattr(
        stitch,
        "stitch_video",
        lambda **_kwargs: pytest.fail("submission_unknown must never stitch"),
    )
    with pytest.raises(h3.ReceiptError, match="context_ir_receipt_required"):
        h3.output_is_reusable(request)
    with _client(gateway) as client:
        with pytest.raises(h3.ReceiptError, match="context_ir_receipt_required"):
            h3.start(request, client=client)
    assert gateway.posts == []
    context = h3_project.freeze_context_ir(
        source_request=request,
        upstream_dialogue_sha256=frozen.dialogue_sha256,
        upstream_artifact_path=frozen.receipt_path,
        upstream_artifact_sha256=frozen.receipt_sha256,
        upstream_dialogue_sha256_path=("dialogue", "sha256"),
        minimax_api_key="not-sent-to-context",
        request_timeout_s=1,
        poll_timeout_s=1,
        poll_interval_s=0,
    )
    context_gateway = _ContextGateway()
    with httpx.Client(
        transport=httpx.MockTransport(context_gateway)
    ) as context_client:
        optimized = context_ir_bridge.optimize_h3_prompt(
            context, client=context_client
        )
    request = context_ir_bridge.apply_effective_prompt(
        context, optimized.receipt_path
    )
    context_receipt = Path(str(request.context_ir_receipt_path))
    context_receipt_bytes = context_receipt.read_bytes()
    context_receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(h3.ReceiptError, match="context_ir_receipt_mismatch"):
        h3.output_is_reusable(request)
    with _client(gateway) as client:
        with pytest.raises(h3.ReceiptError, match="context_ir_receipt_mismatch"):
            h3.start(request, client=client)
    assert gateway.posts == []
    context_receipt.write_bytes(context_receipt_bytes)
    with _client(gateway) as client:
        with pytest.raises(h3.H3Error, match="submission_unknown"):
            h3.start(request, client=client)
        assert h3.resume(request, client=client).status == "submission_unknown"
    assert len(gateway.posts) == 1


def test_multimodal_visual_prompt_edit_requires_skill_plan_refresh(tmp_path):
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "editable project", "source.mp4"
    )
    cid = created["id"]
    root = settings.data_dir / cid
    work = root / "work"
    key = work / "keyframes" / "01.png"
    visual = work / "visual_prompt.txt"
    final = work / "prompt.txt"
    source = root / "source.mp4"
    _png(key, 40)
    visual.parent.mkdir(parents=True, exist_ok=True)
    visual.write_text("雨夜车站，人物面对镜头。", encoding="utf-8")
    _video(source, frequency=220, duration=4)
    _multimodal_source(
        work,
        visual.read_text(encoding="utf-8"),
        plan=_projection_plan(
            visual.read_text(encoding="utf-8"),
            speech=False,
            dialogue_source_sha256=h3.canonical_json_sha256([]),
        ),
    )
    frozen_multimodal = h3_project.freeze_optional(root, work)
    frozen = prepared_input.write_prepared_input(
        root=root,
        source=source,
        audio=None,
        keyframes=[key],
        visual=visual,
        final=final,
        dialogue_mode="none",
        dialogue=(),
        vocal_filter_enabled=True,
        duration_s=4,
        ratio="9:16",
        fit_mode="none",
        engine_request={"h3": {
            "workflow": h3.H3_MULTIMODAL_WORKFLOW,
            "duration": 4,
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "provider_resolution": "768p竖",
        }},
        multimodal=frozen_multimodal,
    )

    with pytest.raises(_SubmitError) as caught:
        _replace_source_prompt(
            settings,
            cid,
            {"prepared_input_receipt": prepared_input.RECEIPT_FILENAME},
            frozen.visual_prompt.sha256,
            "晴天公园，人物背对镜头。",
        )

    assert caught.value.status == 409
    assert caught.value.detail == "multimodal_plan_refresh_required"
    reloaded = prepared_input.load_prepared_input(
        root, frozen.receipt_path, expected_dialogue=()
    )
    assert reloaded.multimodal is not None
    assert reloaded.visual_prompt.data.decode("utf-8") == "雨夜车站，人物面对镜头。"


def test_provider_generated_final_output_rejects_av_timeline_skew(
    tmp_path, monkeypatch
):
    output = tmp_path / "provider.mp4"
    _video(output, frequency=440, duration=4)

    def reject_timeline(*_args, **_kwargs):
        raise h3.H3Error("download_invalid_video", retryable=False)

    monkeypatch.setattr(h3, "_probe_media_timeline", reject_timeline)
    with pytest.raises(stitch.StitchError, match="final native-audio timeline invalid"):
        stitch._validate_output(output, 4, "provider_generated", False)


def test_existing_edited_dialogue_is_the_exact_h3_speech_source(tmp_path):
    root = tmp_path / "edited-dialogue"
    work = root / "work"
    key = work / "keyframes" / "01.png"
    visual = work / "visual_prompt.txt"
    final = work / "prompt.txt"
    source = root / "source.mp4"
    _png(key, 40)
    visual.parent.mkdir(parents=True, exist_ok=True)
    visual.write_text("人物面对镜头。", encoding="utf-8")
    _video(source, frequency=220, duration=4)
    dialogue = prepared_input.prepare_dialogue(
        "custom",
        4,
        supplied_lines=[{
            "text": "这是编辑后的唯一台词。",
            "start_s": 0.5,
            "end_s": 2.5,
        }],
    )
    plan = _projection_plan(
        visual.read_text(encoding="utf-8"),
        speech=True,
        dialogue_source_sha256=h3.canonical_json_sha256(list(dialogue)),
    )
    _multimodal_source(
        work, visual.read_text(encoding="utf-8"), plan=plan
    )
    frozen_multimodal = h3_project.freeze_optional(root, work)
    frozen = prepared_input.write_prepared_input(
        root=root,
        source=source,
        audio=None,
        keyframes=[key],
        visual=visual,
        final=final,
        dialogue_mode="custom",
        dialogue=dialogue,
        vocal_filter_enabled=True,
        duration_s=4,
        ratio="9:16",
        fit_mode="none",
        engine_request={"h3": {
            "workflow": h3.H3_MULTIMODAL_WORKFLOW,
            "duration": 4,
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "provider_resolution": "768p竖",
        }},
        multimodal=frozen_multimodal,
    )

    request = h3_project.build_request(
        frozen=frozen,
        cid="edited-dialogue",
        workdir=work / "h3-native",
        client_request_id="edited-dialogue-request",
        duration=4,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token="not-sent",
    )

    assert "<d>[Chinese]这是编辑后的唯一台词。</d>" in request.prompt
    assert "0.500-2.500" in request.prompt
    assert request.voice_texts == ("这是编辑后的唯一台词。",)
    assert request.upstream_dialogue_receipt_sha256 == frozen.dialogue_sha256
    assert "text" not in plan["speech_bindings"][0]


def test_existing_none_dialogue_cannot_gain_skill_sidecar_speech(tmp_path):
    root = tmp_path / "no-dialogue"
    work = root / "work"
    key = work / "keyframes" / "01.png"
    visual = work / "visual_prompt.txt"
    final = work / "prompt.txt"
    source = root / "source.mp4"
    _png(key, 40)
    visual.parent.mkdir(parents=True, exist_ok=True)
    visual.write_text("无人物的雨夜街道。", encoding="utf-8")
    _video(source, frequency=220, duration=4)
    plan = _projection_plan(
        visual.read_text(encoding="utf-8"),
        speech=False,
        dialogue_source_sha256=h3.canonical_json_sha256([]),
    )
    _multimodal_source(
        work, visual.read_text(encoding="utf-8"), plan=plan
    )
    frozen_multimodal = h3_project.freeze_optional(root, work)
    frozen = prepared_input.write_prepared_input(
        root=root,
        source=source,
        audio=None,
        keyframes=[key],
        visual=visual,
        final=final,
        dialogue_mode="none",
        dialogue=(),
        vocal_filter_enabled=True,
        duration_s=4,
        ratio="9:16",
        fit_mode="none",
        engine_request={"h3": {
            "workflow": h3.H3_MULTIMODAL_WORKFLOW,
            "duration": 4,
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "provider_resolution": "768p竖",
        }},
        multimodal=frozen_multimodal,
    )

    request = h3_project.build_request(
        frozen=frozen,
        cid="no-dialogue",
        workdir=work / "h3-native",
        client_request_id="no-dialogue-request",
        duration=4,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token="not-sent",
    )

    assert "<d>[" not in request.prompt
    assert "No audible speech is specified" in request.prompt
    assert request.voice_texts == ()
    assert request.upstream_dialogue_receipt_sha256 == frozen.dialogue_sha256
    assert not (root / ".h3").exists()
