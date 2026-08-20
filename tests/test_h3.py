import base64
import hashlib
import json
import socket
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app import h3


VOICE_TEXTS = ("第一句台词", "第二句台词")
OPTIMIZED_PROMPT = (
    "镜头保持稳定。<d>[Chinese]第一句台词</d>随后说<d>第二句台词</d>。"
)


def _request(tmp_path: Path, *, request_id: str = "request-1") -> h3.H3Request:
    first = tmp_path / "01.png"
    second = tmp_path / "02.png"
    first.write_bytes(b"first-frame")
    second.write_bytes(b"second-frame")
    frozen = h3.freeze_keyframes((first, second))
    voice_receipt = h3.voice_texts_receipt(VOICE_TEXTS)
    return h3.H3Request(
        cid="cid-1",
        workdir=tmp_path / "session",
        client_request_id=request_id,
        prompt='原始 prompt，包含"第一句台词"和"第二句台词"。',
        keyframes=frozen,
        voice_texts=VOICE_TEXTS,
        voice_receipt=voice_receipt,
        duration=10,
        ratio="9:16",
        minimax_api_key="mm-secret",
        autodl_token="art-secret",
        timeouts=h3.Timeouts(
            request_s=0.1,
            upload_s=0.1,
            ir_poll_s=0.05,
            h3_poll_s=0.05,
            download_s=0.1,
            poll_interval_s=0,
        ),
    )


class HappyProvider:
    video_bytes = b""

    def __init__(
        self,
        *,
        ir_prompt: str = OPTIMIZED_PROMPT,
        result_url: str = "https://download.invalid/video.mp4",
    ) -> None:
        self.ir_prompt = ir_prompt
        self.result_url = result_url
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/files/upload":
            return httpx.Response(
                200,
                json={"file": {"file_id": f"file-{len(self.uploads)}"}},
            )
        if path == "/v2/h3_context_ir":
            return httpx.Response(200, json={"task_id": "ir-task-local"})
        if path == "/v2/query/video_generation":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "ir-task-local",
                            "status": "succeeded",
                            "content": {"prompt": self.ir_prompt},
                        }
                    ]
                },
            )
        if path.endswith("/minimax_h3_lightx2v_v5"):
            return httpx.Response(
                200, json={"data": {"task_id": "h3-task-local"}}
            )
        if path.endswith("/result/h3-task-local"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "status": "SUCCESS",
                        "results": [{"url": self.result_url}],
                    }
                },
            )
        if request.url.host == "download.invalid":
            return _download_response(200, content=self.video_bytes)
        raise AssertionError(f"unexpected request path: {path}")

    @property
    def uploads(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == "/v1/files/upload"]

    @property
    def ir_posts(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == "/v2/h3_context_ir"]

    @property
    def h3_posts(self) -> list[httpx.Request]:
        return [
            r
            for r in self.requests
            if r.url.path.endswith("/minimax_h3_lightx2v_v5")
        ]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class FakeNetworkStream:
    def __init__(self, server_addr=("93.184.216.34", 443)) -> None:
        self.server_addr = server_addr

    def get_extra_info(self, name):
        return self.server_addr if name == "server_addr" else None


def _download_response(status_code=200, *, peer=("93.184.216.34", 443), **kwargs):
    extensions = kwargs.pop("extensions", {})
    extensions["network_stream"] = FakeNetworkStream(peer)
    return httpx.Response(status_code, extensions=extensions, **kwargs)


@pytest.fixture(scope="session")
def valid_video_bytes(tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("h3-video") / "valid.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=5:d=0.4",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path.read_bytes()


@pytest.fixture(autouse=True)
def public_download_host(monkeypatch, valid_video_bytes):
    HappyProvider.video_bytes = valid_video_bytes
    if hasattr(h3, "socket"):
        monkeypatch.setattr(
            h3.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ],
        )


def _attempt_file(request: h3.H3Request, number: int = 1) -> Path:
    return request.workdir / ".h3" / "attempts" / f"{number:06d}" / "attempt.json"


def test_start_keeps_payload_and_frozen_bytes_and_writes_state(tmp_path):
    request = _request(tmp_path)
    # The remote calls must use the already frozen bytes, not a later disk read.
    (tmp_path / "01.png").write_bytes(b"mutated-on-disk")
    provider = HappyProvider()

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert result.output == request.workdir / "generated.mp4"
    assert result.output.read_bytes() == provider.video_bytes
    assert len(provider.uploads) == 2
    assert b"first-frame" in provider.uploads[0].content
    assert b"second-frame" in provider.uploads[1].content
    assert len(provider.ir_posts) == 1
    ir_body = json.loads(provider.ir_posts[0].content)
    assert ir_body == {
        "model": "MiniMax-H3",
        "content": [
            {"type": "text", "text": request.prompt},
            {
                "type": "image_url",
                "image_url": {"url": "mm_file://file-1"},
                "role": "reference_image",
            },
            {
                "type": "image_url",
                "image_url": {"url": "mm_file://file-2"},
                "role": "reference_image",
            },
        ],
        "duration": 10,
        "ratio": "9:16",
    }
    assert len(provider.h3_posts) == 1
    h3_body = json.loads(provider.h3_posts[0].content)
    assert h3_body == {
        "prompt": OPTIMIZED_PROMPT,
        "duration": 10,
        "resolution": "768p竖",
        "ref_image_0": "data:image/png;base64,"
        + base64.b64encode(b"first-frame").decode("ascii"),
        "ref_image_1": "data:image/png;base64,"
        + base64.b64encode(b"second-frame").decode("ascii"),
    }

    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["schema_version"] == 1
    assert state["cid"] == request.cid
    assert state["client_request_id"] == request.client_request_id
    assert state["status"] == "succeeded"
    assert state["ir"]["task_id"] == "ir-task-local"
    assert state["h3"]["task_id"] == "h3-task-local"
    assert state["h3"]["output"] == {
        "name": "generated.mp4",
        "sha256": hashlib.sha256(provider.video_bytes).hexdigest(),
        "size": len(provider.video_bytes),
    }
    assert "download.invalid" not in json.dumps(state)
    assert "result_url" not in state["h3"]
    assert "mm-secret" not in json.dumps(state)
    assert "art-secret" not in json.dumps(state)


def test_existing_generated_video_is_idempotent_without_network(tmp_path):
    request = _request(tmp_path)
    request.workdir.mkdir(parents=True)
    output = request.workdir / "generated.mp4"
    output.write_bytes(b"already-done")

    def no_network(_request):
        raise AssertionError("idempotent success must not use network")

    with _client(no_network) as client:
        first = h3.start(request, client=client)
        second = h3.resume(request, client=client)

    assert first.status == second.status == "succeeded"
    assert output.read_bytes() == b"already-done"


def test_empty_frozen_voice_texts_are_valid_and_bound_to_empty_json_array(tmp_path):
    request = _request(tmp_path)
    request = replace(
        request,
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
    )
    provider = HappyProvider(ir_prompt="静态产品镜头，环境保持安静")

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["input"]["voice_texts_sha256"] == h3.canonical_json_sha256([])
    assert state["ir"]["receipt"]["voice_texts_sha256"] == h3.canonical_json_sha256([])


def test_inspect_is_read_only_when_session_has_not_started(tmp_path):
    request = _request(tmp_path)

    result = h3.inspect(request)

    assert result.status == "not_started"
    assert result.attempt_id is None
    assert result.error_code is None
    assert not (request.workdir / ".h3").exists()


def test_context_ir_snapshot_is_absent_before_start_and_available_after_video(tmp_path):
    request = _request(tmp_path)

    before = h3.inspect_context_ir(request.workdir, request.cid)

    assert before == h3.ContextIRSnapshot("not_started", None, None)
    assert not (request.workdir / ".h3").exists()

    provider = HappyProvider()
    with _client(provider) as client:
        result = h3.start(request, client=client)
    assert result.status == "succeeded"
    assert (request.workdir / "generated.mp4").is_file()

    after = h3.inspect_context_ir(request.workdir, request.cid)

    assert after == h3.ContextIRSnapshot(
        "succeeded",
        OPTIMIZED_PROMPT,
        hashlib.sha256(OPTIMIZED_PROMPT.encode("utf-8")).hexdigest(),
    )


def test_prepare_context_ir_allows_rewritten_dialogue_before_h3(tmp_path):
    request = _request(tmp_path)
    raw = "镜头稳定。<d>第一句被改写</d><d>第二句台词</d>"
    provider = HappyProvider(ir_prompt=raw)

    with _client(provider) as client:
        result = h3.prepare_context_ir(request, client=client)

    assert result.status == "ready_for_h3"
    assert len(provider.ir_posts) == 1
    assert provider.h3_posts == []
    snapshot = h3.inspect_context_ir(request.workdir, request.cid)
    assert snapshot.prompt == raw
    assert snapshot.sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_context_ir_can_be_edited_before_h3_and_uses_expected_sha(tmp_path):
    request = _request(tmp_path)
    raw = "镜头稳定。<d>第一句被改写</d><d>第二句台词</d>"
    provider = HappyProvider(ir_prompt=raw)
    with _client(provider) as client:
        h3.prepare_context_ir(request, client=client)

        with pytest.raises(h3.H3Error, match="context_ir_version_conflict"):
            h3.edit_context_ir(request, "0" * 64, OPTIMIZED_PROMPT)
        revised = "镜头推进。<d>完全改写的台词</d><d>新增台词</d>"
        edited = h3.edit_context_ir(
            request,
            hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            revised,
        )
        completed = h3.start(request, client=client)

    assert edited.prompt == revised
    assert edited.sha256 == hashlib.sha256(revised.encode("utf-8")).hexdigest()
    assert completed.status == "succeeded"
    assert len(provider.ir_posts) == 1
    assert len(provider.h3_posts) == 1


@pytest.mark.parametrize("tamper", ["hash", "status"])
def test_context_ir_snapshot_fails_closed_on_hash_or_status_tamper(tmp_path, tamper):
    request = _request(tmp_path)
    provider = HappyProvider()
    with _client(provider) as client:
        h3.start(request, client=client)
    state_path = _attempt_file(request)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if tamper == "hash":
        state["ir"]["optimized_prompt_sha256"] = "0" * 64
    else:
        state["ir"]["status"] = "running"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(h3.ReceiptError, match="context_ir_mismatch"):
        h3.inspect_context_ir(request.workdir, request.cid)


def test_start_is_idempotent_for_the_same_client_request_id(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider()
    with _client(provider) as client:
        first = h3.start(request, client=client)
        second = h3.start(request, client=client)

    assert first.status == second.status == "succeeded"
    assert len(provider.ir_posts) == 1
    assert len(provider.h3_posts) == 1


def test_startup_resume_only_queries_existing_ir_task(tmp_path):
    request = _request(tmp_path)
    calls: list[httpx.Request] = []

    def first_run(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if req.url.path == "/v1/files/upload":
            return httpx.Response(200, json={"file": {"file_id": "file-local"}})
        if req.url.path == "/v2/h3_context_ir":
            return httpx.Response(200, json={"task_id": "ir-task-local"})
        if req.url.path == "/v2/query/video_generation":
            return httpx.Response(
                200,
                json={"items": [{"id": "ir-task-local", "status": "running"}]},
            )
        raise AssertionError(req.url.path)

    with _client(first_run) as client:
        first = h3.start(request, client=client)
    assert first.status == "retryable_failure"

    resume_calls: list[httpx.Request] = []

    def recovery(req: httpx.Request) -> httpx.Response:
        resume_calls.append(req)
        assert req.method == "GET"
        assert req.url.path == "/v2/query/video_generation"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "ir-task-local",
                        "status": "succeeded",
                        "content": {"prompt": OPTIMIZED_PROMPT},
                    }
                ]
            },
        )

    with _client(recovery) as client:
        recovered = h3.resume(request, client=client)

    assert recovered.status == "ready_for_h3"
    assert resume_calls and all(call.method == "GET" for call in resume_calls)

    state_path = _attempt_file(request)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["ir"]["optimized_prompt"] = OPTIMIZED_PROMPT.replace(
        "第二句台词", "第二句被改写"
    )
    state["ir"]["optimized_prompt_sha256"] = hashlib.sha256(
        state["ir"]["optimized_prompt"].encode("utf-8")
    ).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    provider = HappyProvider()
    with _client(provider) as client:
        completed = h3.start(request, client=client)

    assert completed.status == "succeeded"
    assert len(provider.h3_posts) == 1


def test_startup_resume_only_queries_and_downloads_existing_h3_task(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider()

    def h3_pending(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/result/h3-task-local"):
            provider.requests.append(req)
            return httpx.Response(200, json={"data": {"status": "RUNNING"}})
        return provider(req)

    with _client(h3_pending) as client:
        first = h3.start(request, client=client)
    assert first.status == "retryable_failure"
    assert len(provider.ir_posts) == len(provider.h3_posts) == 1

    recovery_calls: list[httpx.Request] = []

    def recovery(req: httpx.Request) -> httpx.Response:
        recovery_calls.append(req)
        assert req.method == "GET"
        if req.url.path.endswith("/result/h3-task-local"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "status": "COMPLETED",
                        "results": [{"url": "https://download.invalid/video.mp4"}],
                    }
                },
            )
        if req.url.host == "download.invalid":
            return _download_response(200, content=HappyProvider.video_bytes)
        raise AssertionError(req.url.path)

    with _client(recovery) as client:
        recovered = h3.resume(request, client=client)

    assert recovered.status == "succeeded"
    assert recovered.output.read_bytes() == HappyProvider.video_bytes
    assert recovery_calls and all(call.method == "GET" for call in recovery_calls)


@pytest.mark.parametrize(
    "ir_prompt",
    [
        "<d>第一句台词</d>",
        "<d>第一句台词</d><d>第二句被改写</d>",
        "<d>第一句台词</d><d>第二句台词</d><d>额外台词</d>",
        "<d>第二句台词</d><d>第一句台词</d>",
    ],
)
def test_ir_dialogue_content_does_not_have_to_match_frozen_voice_texts(
    tmp_path, ir_prompt
):
    request = _request(tmp_path)
    provider = HappyProvider(ir_prompt=ir_prompt)
    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(provider.h3_posts) == 1


def test_rewritten_ir_is_edited_without_reposting(tmp_path):
    request = _request(tmp_path)
    rejected = HappyProvider(ir_prompt="<d>第一句台词</d>")
    with _client(rejected) as client:
        prepared = h3.prepare_context_ir(request, client=client)
        snapshot = h3.inspect_context_ir(request.workdir, request.cid)
        h3.edit_context_ir(request, snapshot.sha256, OPTIMIZED_PROMPT)
        result = h3.start(request, client=client)

    assert prepared.status == "ready_for_h3"
    assert result.status == "succeeded"
    assert len(rejected.ir_posts) == 1
    assert len(rejected.h3_posts) == 1


@pytest.mark.parametrize(
    "ir_prompt",
    [
        "角色朗读画面字幕：限时优惠",
        "the character speaks the OCR text: limited offer",
        "人物开口：请扫描屏幕上的二维码。",
        "The actor reads the on-screen subtitle aloud: Limited offer.",
    ],
)
def test_context_ir_content_is_not_rejected_without_frozen_voice(tmp_path, ir_prompt):
    request = _request(tmp_path)
    request = replace(
        request,
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
    )
    provider = HappyProvider(ir_prompt=ir_prompt)

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(provider.h3_posts) == 1


def test_empty_voice_texts_allow_ir_to_add_tagged_dialogue(tmp_path):
    request = replace(
        _request(tmp_path),
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
    )
    provider = HappyProvider(ir_prompt="无原始台词，但<d>新增一句</d>")

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(provider.h3_posts) == 1


def test_context_ir_extra_speech_is_not_rejected(tmp_path):
    request = _request(tmp_path)
    prompt = OPTIMIZED_PROMPT + "随后人物又开口：额外一句。"
    provider = HappyProvider(ir_prompt=prompt)

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(provider.h3_posts) == 1


@pytest.mark.parametrize(
    ("voice_texts", "ir_prompt"),
    [
        (
            VOICE_TEXTS,
            "<d>[Chinese]第一句台词</d><d>第二句台词</d>，随后人物回答：额外一句。",
        ),
        ((), "人物念道屏幕字幕：限时优惠。"),
        ((), "The actor recites the on-screen caption: Limited offer."),
    ],
)
def test_ir_speech_prose_does_not_block_h3_post(
    tmp_path, voice_texts, ir_prompt
):
    request = _request(tmp_path)
    request = replace(
        request,
        voice_texts=voice_texts,
        voice_receipt=h3.voice_texts_receipt(voice_texts),
    )
    provider = HappyProvider(ir_prompt=ir_prompt)

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(provider.h3_posts) == 1


def test_real_temp10_speaks_clearly_then_dialogue_is_bound(tmp_path):
    source = Path("temp/10-restore-h3-vocal-face/ir_prompt.txt").read_text(
        encoding="utf-8"
    )
    start = source.index("[Shot 1] The shot begins")
    end = source.index("\n\n[Shot 2]", start)
    shot = source[start:end]
    request = _request(tmp_path)
    request = replace(
        request,
        voice_texts=("Kalung Ayatul Kursi.",),
        voice_receipt=h3.voice_texts_receipt(("Kalung Ayatul Kursi.",)),
    )
    provider = HappyProvider(ir_prompt=shot)

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(provider.h3_posts) == 1


def test_ir_allows_contentless_speech_context_around_exact_frozen_dialogue(
    tmp_path,
):
    voice_texts = ("Kalung Ayatul Kursi.",)
    summary = "Subject 1 speaks a brief introductory line at the beginning."
    intervening_sections = "\n\n".join(
        f"[Visual section {index}] " + "Detailed silent visual direction. " * 14
        for index in range(5)
    )
    timeline = (
        "Subject 1 speaks, <d>[Indonesian] Kalung Ayatul Kursi.</d> "
        "As he speaks, the pendant remains centered, responding to gentle body "
        "breathing and natural torso movement."
    )
    prompt = f"summary:\n{summary}\n\n{intervening_sections}\n\n[Timeline]\n{timeline}"
    assert prompt.index("<d>") - prompt.index("speaks") > 2151

    request = replace(
        _request(tmp_path),
        voice_texts=voice_texts,
        voice_receipt=h3.voice_texts_receipt(voice_texts),
    )
    provider = HappyProvider(ir_prompt=prompt)

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(provider.h3_posts) == 1


@pytest.mark.parametrize(
    ("voice_texts", "ir_prompt"),
    [
        (
            ("Kalung Ayatul Kursi.",),
            "Subject says extra words then <d>Kalung Ayatul Kursi.</d>",
        ),
        (
            ("Kalung Ayatul Kursi.",),
            "[Summary]\nSubject speaks an opening line reading BUY NOW at the "
            "start.\n\n[Timeline]\nSubject speaks, "
            "<d>Kalung Ayatul Kursi.</d>",
        ),
        (
            ("Kalung Ayatul Kursi.",),
            "<d>Kalung Ayatul Kursi.</d> The actor responds to camera motion with "
            "an extra slogan.",
        ),
        (
            ("Kalung Ayatul Kursi.",),
            "<d>Kalung Ayatul Kursi.</d> As the words BUY NOW play he speaks "
            "softly.",
        ),
        ((), "The actor does not hesitate and speaks extra words."),
        ((), "The actor silently whispers buy now."),
        ((), "The actor talks to camera and delivers a sales pitch."),
        (
            ("Kalung Ayatul Kursi.",),
            "<d>Kalung Ayatul Kursi.</d> He states BUY NOW.",
        ),
        (
            ("Kalung Ayatul Kursi.",),
            "<d>Kalung Ayatul Kursi.</d> Then he speaks softly.",
        ),
    ],
)
def test_context_ir_prose_is_reviewed_by_user_not_blocked_by_server(
    tmp_path,
    voice_texts,
    ir_prompt,
):
    request = replace(
        _request(tmp_path),
        voice_texts=voice_texts,
        voice_receipt=h3.voice_texts_receipt(voice_texts),
    )
    provider = HappyProvider(ir_prompt=ir_prompt)

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(provider.h3_posts) == 1


def test_receipt_tampering_blocks_recovery_before_network(tmp_path):
    request = _request(tmp_path)

    def pending(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/files/upload":
            return httpx.Response(200, json={"file": {"file_id": "file-local"}})
        if req.url.path == "/v2/h3_context_ir":
            return httpx.Response(200, json={"task_id": "ir-task-local"})
        if req.url.path == "/v2/query/video_generation":
            return httpx.Response(
                200,
                json={"items": [{"id": "ir-task-local", "status": "running"}]},
            )
        raise AssertionError(req.url.path)

    with _client(pending) as client:
        assert h3.start(request, client=client).status == "retryable_failure"

    path = _attempt_file(request)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["ir"]["receipt"]["prompt_sha256"] = "0" * 64
    path.write_text(json.dumps(state), encoding="utf-8")

    def no_network(_request):
        raise AssertionError("tampered receipt must fail before network")

    with _client(no_network) as client:
        with pytest.raises(h3.ReceiptError, match="receipt_mismatch"):
            h3.resume(request, client=client)


@pytest.mark.parametrize("crash_stage", ["ir", "h3"])
def test_post_crash_without_persisted_task_id_becomes_submission_unknown(
    tmp_path, monkeypatch, crash_stage
):
    request = _request(tmp_path)
    provider = HappyProvider()
    original = h3._atomic_write_json
    crashed = False

    def crash_after_post(path, payload):
        nonlocal crashed
        stage = payload.get(crash_stage, {})
        if not crashed and stage.get("task_id"):
            crashed = True
            raise OSError("simulated power loss")
        return original(path, payload)

    monkeypatch.setattr(h3, "_atomic_write_json", crash_after_post)
    with _client(provider) as client:
        with pytest.raises(h3.H3Error, match="state_persist_failed"):
            h3.start(request, client=client)

    monkeypatch.setattr(h3, "_atomic_write_json", original)
    inspected = h3.inspect(request)
    assert inspected.status == f"{crash_stage}_submitting"
    assert inspected.retryable is False
    assert inspected.error_code is None
    recovery_calls: list[httpx.Request] = []

    def no_network(req: httpx.Request):
        recovery_calls.append(req)
        raise AssertionError("unknown submission must never be resent or queried")

    with _client(no_network) as client:
        result = h3.resume(request, client=client)

    assert result.status == "submission_unknown"
    assert recovery_calls == []
    assert len(provider.ir_posts) == 1
    assert len(provider.h3_posts) == (1 if crash_stage == "h3" else 0)


def test_manual_retry_uses_a_new_attempt_and_never_reposts_old_attempt(tmp_path):
    request = _request(tmp_path)
    ir_submissions = 0

    def failed(req: httpx.Request) -> httpx.Response:
        nonlocal ir_submissions
        if req.url.path == "/v1/files/upload":
            return httpx.Response(200, json={"file": {"file_id": "file-local"}})
        if req.url.path == "/v2/h3_context_ir":
            ir_submissions += 1
            return httpx.Response(200, json={"task_id": f"ir-{ir_submissions}"})
        if req.url.path == "/v2/query/video_generation":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": f"ir-{ir_submissions}", "status": "failed"}
                    ]
                },
            )
        raise AssertionError(req.url.path)

    with _client(failed) as client:
        assert h3.start(request, client=client).status == "failed"
        same = h3.retry(request, "request-1", client=client)
        assert same.attempt_id == "000001"
        newer = h3.retry(request, "request-2", client=client)

    assert newer.attempt_id == "000002"
    assert newer.status == "failed"
    assert ir_submissions == 2
    assert _attempt_file(request, 1).is_file()
    assert _attempt_file(request, 2).is_file()

    # Detail/startup callers inspect the latest attempt without knowing its
    # retry id and without parsing the private JSON schema themselves.
    inspected = h3.inspect(request)
    assert inspected.status == "failed"
    assert inspected.attempt_id == "000002"
    assert inspected.error_code == "ir_provider_failed"


def test_nonblocking_session_lock_rejects_same_cid_concurrency(tmp_path):
    request = _request(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    provider = HappyProvider()

    def blocked(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/files/upload" and not entered.is_set():
            entered.set()
            assert release.wait(2)
        return provider(req)

    error: list[BaseException] = []

    def run_first() -> None:
        try:
            with _client(blocked) as client:
                h3.start(request, client=client)
        except BaseException as exc:  # test captures the worker failure for assertion
            error.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert entered.wait(2)
    try:
        with _client(provider) as client:
            with pytest.raises(h3.H3BusyError, match="session_busy"):
                h3.start(request, client=client)
    finally:
        release.set()
        thread.join(2)

    assert not thread.is_alive()
    assert error == []


def test_generated_video_is_published_with_atomic_replace(tmp_path, monkeypatch):
    request = _request(tmp_path)
    provider = HappyProvider()
    original_replace = h3.os.replace
    replacements: list[tuple[Path, Path]] = []

    def observe_replace(src, dst):
        src_path, dst_path = Path(src), Path(dst)
        if dst_path.name == "generated.mp4":
            assert src_path.is_file()
            assert not dst_path.exists()
            replacements.append((src_path, dst_path))
        return original_replace(src, dst)

    monkeypatch.setattr(h3.os, "replace", observe_replace)
    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(replacements) == 1
    assert replacements[0][1] == request.workdir / "generated.mp4"
    assert list(request.workdir.glob(".generated.mp4.*.tmp")) == []


@pytest.mark.parametrize(
    "result_url",
    [
        "http://download.invalid/video.mp4",
        "https://user:password@download.invalid/video.mp4",
        "https://127.0.0.1/video.mp4",
        "https://localhost/video.mp4",
    ],
)
def test_download_rejects_non_https_userinfo_and_loopback(tmp_path, result_url):
    request = _request(tmp_path)
    provider = HappyProvider(result_url=result_url)

    with _client(provider) as client:
        with pytest.raises(h3.H3Error, match="download_url_rejected"):
            h3.start(request, client=client)

    assert not (request.workdir / "generated.mp4").exists()
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["error"] == {"code": "download_url_rejected"}
    assert result_url not in json.dumps(state)


def test_download_rejects_hostname_resolving_to_private_address(tmp_path, monkeypatch):
    request = _request(tmp_path)
    provider = HappyProvider()
    monkeypatch.setattr(
        h3.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 443))
        ],
    )

    with _client(provider) as client:
        with pytest.raises(h3.H3Error, match="download_url_rejected"):
            h3.start(request, client=client)

    assert not (request.workdir / "generated.mp4").exists()


def test_download_dns_failure_is_retryable_with_known_h3_task(tmp_path, monkeypatch):
    request = _request(tmp_path)
    provider = HappyProvider()

    def dns_down(*_args, **_kwargs):
        raise OSError("temporary resolver outage")

    monkeypatch.setattr(h3.socket, "getaddrinfo", dns_down)
    with _client(provider) as client:
        with pytest.raises(h3.H3Error, match="download_dns_failed") as caught:
            h3.start(request, client=client)

    assert caught.value.retryable is True
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["status"] == "retryable_failure"
    assert state["error"] == {"code": "download_dns_failed"}
    assert state["h3"]["task_id"] == "h3-task-local"


@pytest.mark.parametrize("peer_ip", ["127.0.0.1", "10.23.4.5"])
def test_download_rebinding_private_peer_is_rejected_before_body(
    tmp_path, peer_ip
):
    request = _request(tmp_path)
    provider = HappyProvider()

    class GuardedStream(httpx.SyncByteStream):
        iterated = False

        def __iter__(self):
            self.iterated = True
            yield HappyProvider.video_bytes

    body = GuardedStream()

    def rebound(req: httpx.Request) -> httpx.Response:
        if req.url.host == "download.invalid":
            return _download_response(
                200,
                peer=(peer_ip, 443),
                stream=body,
            )
        return provider(req)

    with _client(rebound) as client:
        with pytest.raises(h3.H3Error, match="download_url_rejected"):
            h3.start(request, client=client)

    assert body.iterated is False
    assert not (request.workdir / "generated.mp4").exists()


def test_download_unverifiable_peer_is_retryable_before_body(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider()

    class GuardedStream(httpx.SyncByteStream):
        iterated = False

        def __iter__(self):
            self.iterated = True
            yield HappyProvider.video_bytes

    body = GuardedStream()

    def no_peer(req: httpx.Request) -> httpx.Response:
        if req.url.host == "download.invalid":
            return httpx.Response(200, stream=body)
        return provider(req)

    with _client(no_peer) as client:
        with pytest.raises(h3.H3Error, match="download_peer_unverified") as caught:
            h3.start(request, client=client)

    assert caught.value.retryable is True
    assert body.iterated is False
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["status"] == "retryable_failure"
    assert state["error"] == {"code": "download_peer_unverified"}


def test_download_does_not_follow_redirects(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider()
    download_calls: list[httpx.Request] = []

    def redirect(req: httpx.Request) -> httpx.Response:
        if req.url.host == "download.invalid":
            download_calls.append(req)
            return _download_response(
                302, headers={"Location": "https://127.0.0.1/private.mp4"}
            )
        return provider(req)

    with _client(redirect) as client:
        with pytest.raises(h3.H3Error, match="download_redirect_rejected"):
            h3.start(request, client=client)

    assert len(download_calls) == 1
    assert not (request.workdir / "generated.mp4").exists()


@pytest.mark.parametrize("limit_source", ["content_length", "actual_stream"])
def test_download_enforces_declared_and_actual_size_limit(
    tmp_path, monkeypatch, limit_source
):
    request = _request(tmp_path)
    provider = HappyProvider()
    monkeypatch.setattr(h3, "MAX_VIDEO_BYTES", 16, raising=False)

    def oversized(req: httpx.Request) -> httpx.Response:
        if req.url.host != "download.invalid":
            return provider(req)
        if limit_source == "content_length":
            return _download_response(
                200, headers={"Content-Length": "17"}, content=b"x"
            )
        return _download_response(200, stream=httpx.ByteStream(b"x" * 17))

    with _client(oversized) as client:
        with pytest.raises(h3.H3Error, match="download_too_large"):
            h3.start(request, client=client)

    assert not (request.workdir / "generated.mp4").exists()
    assert list(request.workdir.glob(".generated.mp4.*.tmp")) == []


def test_download_rejects_non_video_before_atomic_publish(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider()

    def non_video(req: httpx.Request) -> httpx.Response:
        if req.url.host == "download.invalid":
            return _download_response(200, content=b"not-an-mp4")
        return provider(req)

    with _client(non_video) as client:
        with pytest.raises(h3.H3Error, match="download_invalid_video"):
            h3.start(request, client=client)

    assert not (request.workdir / "generated.mp4").exists()
    assert list(request.workdir.glob(".generated.mp4.*.tmp")) == []


@pytest.mark.parametrize(
    "probe_error",
    [
        FileNotFoundError("ffprobe missing"),
        subprocess.TimeoutExpired("ffprobe", 1),
        OSError("ffprobe unavailable"),
    ],
)
def test_ffprobe_infrastructure_failure_is_retryable(
    tmp_path, monkeypatch, probe_error
):
    request = _request(tmp_path)
    provider = HappyProvider()

    def probe_unavailable(*_args, **_kwargs):
        raise probe_error

    monkeypatch.setattr(h3.subprocess, "run", probe_unavailable)
    with _client(provider) as client:
        with pytest.raises(h3.H3Error, match="output_probe_failed") as caught:
            h3.start(request, client=client)

    assert caught.value.retryable is True
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["status"] == "retryable_failure"
    assert state["error"] == {"code": "output_probe_failed"}
    assert state["h3"]["task_id"] == "h3-task-local"
    assert not (request.workdir / "generated.mp4").exists()


def test_credentials_and_remote_identifiers_never_enter_exception_text(tmp_path):
    request = _request(tmp_path)

    def explode(req: httpx.Request):
        raise httpx.ConnectError(
            "art-secret mm-secret https://provider.invalid/task/private-task-id",
            request=req,
        )

    with _client(explode) as client:
        with pytest.raises(h3.H3Error) as caught:
            h3.start(request, client=client)

    text = str(caught.value)
    assert "art-secret" not in text
    assert "mm-secret" not in text
    assert "provider.invalid" not in text
    assert "private-task-id" not in text


def test_voice_receipt_is_canonical_and_required(tmp_path):
    assert h3.voice_texts_receipt(("甲", "乙")) == h3.canonical_json_sha256(
        ["甲", "乙"]
    )
    request = _request(tmp_path)
    with pytest.raises(h3.ReceiptError, match="voice_receipt_mismatch"):
        replace(request, voice_receipt="0" * 64)


def test_duration_above_provider_limit_is_rejected_by_request(tmp_path):
    request = _request(tmp_path)
    with pytest.raises(h3.H3Error, match="invalid_duration"):
        replace(request, duration=16)
