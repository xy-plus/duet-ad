from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from app import providers
from app.config import Settings
from app.fusion_schema import FUSION_OUTPUT_NAME, fusion_prompt_schema
from app.providers import (
    CodexClient,
    DeepSeekClient,
    MediaKitClient,
    PicsartClient,
    ProviderFailure,
    ProviderResult,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://providers.test",
        "deepseek_api_key": "deepseek-secret",
        "deepseek_api_base": "https://deepseek.test/beta",
        "deepseek_model": "deepseek-test-model",
        "mediakit_api_key": "mediakit-secret",
        "mediakit_api_base": "https://mediakit.test/api/v1",
        "picsart_api_key": "picsart-secret",
        "picsart_api_base": "https://picsart.test",
        "request_timeout_s": 1.0,
        "provider_poll_interval_s": 0.0,
        "provider_poll_timeout_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


def test_deepseek_complete_uses_canonical_url_json_mode_and_preserves_record(
    tmp_path: Path,
) -> None:
    image = tmp_path / "reference.png"
    image.write_bytes(b"reference-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://deepseek.test/chat/completions")
        assert request.headers["authorization"] == "Bearer deepseek-secret"
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert body["thinking"] == {"type": "enabled"}
        assert "max_tokens" not in body
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        return httpx.Response(
            200,
            headers={"X-Provider": "deepseek"},
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    result = DeepSeekClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).complete("instruction", {"input": "原样"}, [image], json_mode=True)

    assert isinstance(result, ProviderResult)
    assert result.value == '{"ok":true}'
    assert result.evidence_media == [image]
    assert result.record["request"]["method"] == "POST"
    assert result.record["request"]["url"] == "https://deepseek.test/chat/completions"
    assert result.record["response"]["status_code"] == 200
    assert result.record["response"]["body"] == {
        "choices": [{"message": {"content": '{"ok":true}'}}]
    }


def test_deepseek_image_mime_uses_bytes_when_processing_suffix_is_stale(tmp_path: Path) -> None:
    image = tmp_path / "mediakit-output.png"
    image.write_bytes(b"RIFF\x04\x00\x00\x00WEBP")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith(
            "data:image/webp;base64,"
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    result = DeepSeekClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).complete("instruction", {}, [image])
    assert result.value == "ok"


def test_deepseek_structured_uses_responses_json_schema_and_preserves_exchange(
    tmp_path: Path,
) -> None:
    schema = {
        "type": "object",
        "properties": {"visual_prompt": {"type": "string"}},
        "required": ["visual_prompt"],
        "additionalProperties": False,
    }
    image = tmp_path / "reference.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    response_payload = {
        "id": "resp_test",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "rs_test",
                "type": "reasoning",
                "summary": [],
            },
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"visual_prompt":"move left"}',
                        "annotations": [],
                    }
                ],
            },
        ],
    }
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://deepseek.test/responses")
        assert request.headers["authorization"] == "Bearer deepseek-secret"
        body = json.loads(request.content)
        observed["body"] = body
        assert set(body) == {"model", "instructions", "input", "reasoning", "text"}
        assert body["model"] == "deepseek-test-model"
        assert body["instructions"] == "skill"
        assert body["reasoning"] == {"effort": "high"}
        assert body["text"] == {
            "format": {
                "type": "json_schema",
                "name": "duet_structured_result",
                "schema": schema,
            }
        }
        assert body["input"][0]["role"] == "user"
        assert body["input"][0]["content"][0] == {
            "type": "input_text",
            "text": '{"segment": 1}',
        }
        assert body["input"][0]["content"][1]["type"] == "input_image"
        assert body["input"][0]["content"][1]["image_url"].startswith(
            "data:image/png;base64,"
        )
        encoded_body = json.dumps(body, ensure_ascii=False)
        assert "tools" not in body
        assert "tool_choice" not in body
        assert "max_output_tokens" not in body
        assert "submit_result" not in encoded_body
        return httpx.Response(
            200,
            headers={"X-Provider": "deepseek"},
            json=response_payload,
        )

    result = DeepSeekClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).complete_structured("skill", {"segment": 1}, schema, [image])

    assert isinstance(result, ProviderResult)
    assert result.value == {"visual_prompt": "move left"}
    assert result.evidence_media == [image]
    assert result.record["request"]["method"] == "POST"
    assert result.record["request"]["url"] == "https://deepseek.test/responses"
    assert result.record["request"]["body"] == observed["body"]
    assert result.record["response"]["status_code"] == 200
    assert result.record["response"]["headers"]["x-provider"] == "deepseek"
    assert result.record["response"]["body"] == response_payload


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(502, json={"error": {"message": "upstream failed"}}),
        httpx.Response(200, json={"choices": [{"message": {}}]}),
    ],
)
def test_deepseek_http_and_structure_failures_keep_exchange(
    tmp_path: Path, response: httpx.Response
) -> None:
    client = DeepSeekClient(
        _settings(tmp_path), httpx.MockTransport(lambda request: response)
    )
    with pytest.raises(ProviderFailure) as caught:
        client.complete("instruction", {"input": 1})
    assert isinstance(caught.value.cause, Exception)
    assert caught.value.record["request"]["url"] == "https://deepseek.test/chat/completions"
    assert caught.value.record["response"]["status_code"] == response.status_code
    assert caught.value.record["response"]["body"] == response.json()


def test_deepseek_structured_http_error_keeps_exchange_and_media(tmp_path: Path) -> None:
    image = tmp_path / "reference.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    payload = {"error": {"message": "upstream failed"}}
    client = DeepSeekClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda request: httpx.Response(502, json=payload)),
    )

    with pytest.raises(ProviderFailure) as caught:
        client.complete_structured("skill", {"segment": 1}, {"type": "object"}, [image])

    assert isinstance(caught.value.cause, httpx.HTTPStatusError)
    assert caught.value.evidence_media == [image]
    assert caught.value.record["request"]["url"] == "https://deepseek.test/responses"
    assert caught.value.record["request"]["body"]["input"][0]["content"][1]["type"] == (
        "input_image"
    )
    assert caught.value.record["response"]["status_code"] == 502
    assert caught.value.record["response"]["body"] == payload


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_deepseek_structured_non_completed_status_keeps_raw_exchange(
    tmp_path: Path,
    status: str,
) -> None:
    payload = {
        "status": status,
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": '{"ok":true}'}],
            },
        ],
    }
    client = DeepSeekClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )

    with pytest.raises(ProviderFailure) as caught:
        client.complete_structured("skill", {}, {"type": "object"})

    assert isinstance(caught.value.cause, ValueError)
    assert caught.value.record["request"]["url"] == "https://deepseek.test/responses"
    assert caught.value.record["response"]["status_code"] == 200
    assert caught.value.record["response"]["body"] == payload


@pytest.mark.parametrize(
    "case",
    [
        "output_not_list",
        "missing_message",
        "duplicate_message",
        "incomplete_message",
        "missing_output_text",
        "duplicate_output_text",
        "non_string_output_text",
    ],
)
def test_deepseek_structured_requires_unique_assistant_message_and_output_text(
    tmp_path: Path,
    case: str,
) -> None:
    reasoning = {"type": "reasoning", "summary": []}

    def message(*texts: str) -> dict[str, object]:
        return {
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text} for text in texts],
        }

    if case == "output_not_list":
        output: object = {}
    elif case == "missing_message":
        output = [reasoning]
    elif case == "duplicate_message":
        output = [reasoning, message('{"ok":true}'), message('{"ok":true}')]
    elif case == "incomplete_message":
        output = [
            reasoning,
            {
                "type": "message",
                "status": "incomplete",
                "role": "assistant",
                "content": [{"type": "output_text", "text": '{"ok":true}'}],
            },
        ]
    elif case == "missing_output_text":
        output = [
            reasoning,
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "cannot comply"}],
            },
        ]
    elif case == "duplicate_output_text":
        output = [reasoning, message('{"ok":true}', '{"extra":true}')]
    else:
        output = [
            reasoning,
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": {"ok": True}}],
            },
        ]

    payload = {"status": "completed", "output": output}
    client = DeepSeekClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )

    with pytest.raises(ProviderFailure) as caught:
        client.complete_structured("skill", {}, {"type": "object"})

    assert isinstance(caught.value.cause, (TypeError, ValueError))
    assert caught.value.record["request"]["url"] == "https://deepseek.test/responses"
    assert caught.value.record["response"]["body"] == payload


@pytest.mark.parametrize(
    ("text", "error_type"),
    [
        ("{not-json", json.JSONDecodeError),
        ("[]", TypeError),
    ],
)
def test_deepseek_structured_invalid_output_text_keeps_raw_exchange(
    tmp_path: Path,
    text: str,
    error_type: type[Exception],
) -> None:
    payload = {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
    }
    client = DeepSeekClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )

    with pytest.raises(ProviderFailure) as caught:
        client.complete_structured("skill", {}, {"type": "object"})

    assert isinstance(caught.value.cause, error_type)
    assert caught.value.record["request"]["url"] == "https://deepseek.test/responses"
    assert caught.value.record["response"]["body"] == payload


def test_codex_structured_uses_acct2_sol_low_and_output_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"segment_index": 3, "visual_prompt": "walk right"}
    schema, input_sha256 = fusion_prompt_schema(payload)
    image = tmp_path / "reference.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nreference")
    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 987654

        def __init__(self, command: list[str], **kwargs: object) -> None:
            observed["command"] = command
            observed["env"] = kwargs["env"]
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text(json.dumps({
                "schema": FUSION_OUTPUT_NAME,
                "version": 1,
                "input_sha256": input_sha256,
                "segment_index": 3,
                "prompt": "replace every visible person and scene",
            }), encoding="utf-8")

        def wait(self, timeout: float | None = None) -> int:
            observed["timeout"] = timeout
            return 0

        def poll(self) -> int:
            return 0

    monkeypatch.setenv("PICSART_API_KEY", "must-not-reach-codex")
    monkeypatch.setattr(providers.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(providers, "_terminate_process_group", lambda _process: None)

    result = CodexClient(_settings(tmp_path)).complete_structured(
        "skill", payload, schema, [image],
    )

    command = observed["command"]
    assert isinstance(command, list)
    account_index = command.index("/home/xy/.local/bin/codex-acct")
    inner = command[account_index:]
    assert inner[:3] == ["/home/xy/.local/bin/codex-acct", "acct2", "exec"]
    assert inner[inner.index("-m") + 1] == "gpt-5.6-sol"
    assert 'model_reasoning_effort="low"' in inner
    assert "--output-schema" in inner
    assert "--ignore-user-config" in inner
    assert "--ignore-rules" in inner
    assert "PICSART_API_KEY" not in observed["env"]
    assert observed["env"]["PATH"].split(":")[0] == "/home/xy/.local/bin"
    assert result.value["prompt"] == "replace every visible person and scene"
    assert result.evidence_media == [image]
    assert result.record["request"]["account"] == "acct2"
    assert result.record["request"]["model"] == "gpt-5.6-sol"
    assert result.record["request"]["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda value: value,
        lambda value: "```json\n" + value + "\n```",
        lambda value: value + "\ntokens used\n1,234\n",
    ],
)
def test_codex_structured_transport_compatibility(wrapper) -> None:
    body = json.dumps({"prompt": "ok"})
    assert providers._parse_codex_object(wrapper(body).encode("utf-8")) == {
        "prompt": "ok",
    }


def test_mediakit_multi_scene_supports_documented_and_nested_upload_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.jpg"
    source.write_bytes(b"jpeg-source")
    destination = tmp_path / "out" / "final.png"
    init_count = 0
    erase_count = 0
    client_kwargs: list[dict[str, object]] = []
    real_client = httpx.Client

    def client_factory(*args: object, **kwargs: object) -> httpx.Client:
        client_kwargs.append(dict(kwargs))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(providers.httpx, "Client", client_factory)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal init_count, erase_count
        if request.url.path.endswith("request-media-upload-url"):
            init_count += 1
            if init_count == 1:
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "result": {
                            "file_id": "mediakit://one",
                            "upload_headers": [{"Key": "X-Upload", "Value": "one"}],
                            "upload_url": "https://upload.test/one",
                        },
                    },
                )
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {
                        "upload": {
                            "file_id": "mediakit://two",
                            "method": "POST",
                            "headers": {"X-Upload": "two"},
                            "url": "https://upload.test/two",
                        }
                    },
                },
            )
        if request.url.host == "upload.test":
            if request.url.path == "/one":
                assert request.method == "PUT"
                assert request.headers["content-type"] == "image/jpeg"
                assert request.headers["x-upload"] == "one"
                assert request.content == b"jpeg-source"
            else:
                assert request.method == "POST"
                assert request.headers["content-type"] == "image/webp"
                assert request.headers["x-upload"] == "two"
                assert request.content == b"first-stage"
            return httpx.Response(204)
        if request.url.path.endswith("erase-image"):
            erase_count += 1
            body = json.loads(request.content)
            assert body["standard_scene"] == ["erase-text", "erase-logo"][erase_count - 1]
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {"image_url": f"https://result.test/{erase_count}"},
                },
            )
        if request.url.host == "result.test":
            if request.url.path == "/1":
                return httpx.Response(
                    200, content=b"first-stage", headers={"Content-Type": "image/webp"}
                )
            return httpx.Response(
                200, content=b"final-stage", headers={"Content-Type": "image/png"}
            )
        raise AssertionError(str(request.url))

    transport = httpx.MockTransport(handler)
    result = MediaKitClient(_settings(tmp_path), transport).erase(
        source, destination, ["erase-text", "erase-logo"]
    )

    assert result.value == destination
    assert destination.read_bytes() == b"final-stage"
    assert [path.read_bytes() for path in result.evidence_media] == [
        b"first-stage",
        b"final-stage",
    ]
    assert len(result.record["stages"]) == 2
    assert len(client_kwargs) == 1
    assert client_kwargs[0]["transport"] is transport
    assert client_kwargs[0]["trust_env"] is False
    first_download = result.record["stages"][0]["download"]
    assert first_download["response"]["body"] == {
        "content_type": "image/webp",
        "length": len(b"first-stage"),
        "sha256": hashlib.sha256(b"first-stage").hexdigest(),
    }
    assert "first-stage" not in json.dumps(first_download)
    assert result.record["stages"][1]["upload"]["request"]["method"] == "POST"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            httpx.Response(503, json={"success": False, "error": "unavailable"}),
            "503",
        ),
        (
            httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {
                        "file_id": "mediakit://invalid-method",
                        "method": 123,
                        "upload_url": "https://upload.test/file",
                        "upload_headers": {},
                    },
                },
            ),
            "method",
        ),
    ],
)
def test_mediakit_http_and_init_shape_failures_keep_stage_record(
    tmp_path: Path, response: httpx.Response, message: str
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"input")
    client = MediaKitClient(
        _settings(tmp_path), httpx.MockTransport(lambda request: response)
    )
    with pytest.raises(ProviderFailure, match=message) as caught:
        client.erase(source, tmp_path / "output.png", ["erase-text"])
    stage = caught.value.record["stages"][0]
    assert stage["initialize"]["response"]["status_code"] == response.status_code
    assert stage["initialize"]["response"]["body"] == response.json()


def test_picsart_submit_and_poll_are_split_and_poll_one_task(
    tmp_path: Path,
) -> None:
    poll_paths: list[str] = []
    poll_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.url.path.endswith("/submit"):
            body = json.loads(request.content)
            params = body["params"]
            assert request.url.path == (
                "/workflows/minimax/h3-max/reference-to-video/submit"
            )
            assert set(body) == {"params"}
            assert "model" not in params
            assert params["prompt_expansion_mode"] == "quality"
            assert params["reference_video_urls"] == [
                "https://assets.test/source.mp4"
            ]
            assert params["reference_image_urls"] == [
                "https://assets.test/reference.png"
            ]
            assert params["reference_audio_urls"] == [
                "https://assets.test/voice.wav"
            ]
            assert params["resolution"] == "768P"
            assert params["aspect_ratio"] == "16:9"
            return httpx.Response(200, json={"response": {"id": "task-fixed"}})
        poll_paths.append(request.url.path)
        poll_count += 1
        if poll_count == 1:
            return httpx.Response(200, json={"response": {"status": "IN_PROGRESS"}})
        return httpx.Response(
            200,
            json={
                "response": {
                    "status": "COMPLETED",
                    "result": {
                        "expanded_prompt": "expanded",
                        "video": {"url": "https://result.test/video.mp4"},
                    },
                }
            },
        )

    client = PicsartClient(_settings(tmp_path), httpx.MockTransport(handler))
    submitted = client.submit(
        prompt="replace subject",
        video_url="https://assets.test/source.mp4",
        image_url="https://assets.test/reference.png",
        audio_urls=["https://assets.test/voice.wav"],
        duration=5,
        resolution="768p",
        aspect_ratio="16:9",
    )
    polled = client.poll(submitted.value)

    assert submitted.value == "task-fixed"
    assert submitted.record["outcome"] == "accepted"
    assert polled.value == "https://result.test/video.mp4"
    assert polled.record["task_id"] == "task-fixed"
    assert poll_paths == [
        "/workflows/minimax/h3-max/reference-to-video/task-fixed/result",
        "/workflows/minimax/h3-max/reference-to-video/task-fixed/result",
    ]


def test_picsart_advanced_request_omits_empty_optional_inputs(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {
            "params": {
                "prompt": "prompt",
                "reference_video_urls": ["https://assets.test/source.mp4"],
                "resolution": "480P",
                "duration": 5,
                "aspect_ratio": "9:16",
                "prompt_expansion_mode": "quality",
            }
        }
        return httpx.Response(200, json={"response": {"id": "task-no-image"}})

    client = PicsartClient(_settings(tmp_path), httpx.MockTransport(handler))
    result = client.submit(
        prompt="prompt",
        video_url="https://assets.test/source.mp4",
        image_url=None,
        audio_urls=[],
        duration=5,
        resolution="480p",
        aspect_ratio="9:16",
    )

    assert result.value == "task-no-image"


def test_picsart_submit_preserves_three_audio_urls_in_order(tmp_path: Path) -> None:
    audio_urls = [
        "https://assets.test/first.wav",
        "https://assets.test/second.wav",
        "https://assets.test/third.wav",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            json.loads(request.content)["params"]["reference_audio_urls"]
            == audio_urls
        )
        return httpx.Response(200, json={"response": {"id": "task-three-audios"}})

    result = PicsartClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).submit(
        prompt="prompt",
        video_url="https://assets.test/source.mp4",
        image_url=None,
        audio_urls=audio_urls,
        duration=5,
        resolution="768p",
        aspect_ratio="16:9",
    )

    assert result.value == "task-three-audios"


@pytest.mark.parametrize(
    ("audio_urls", "message"),
    [
        (
            [
                "https://assets.test/one.wav",
                "https://assets.test/two.wav",
                "https://assets.test/three.wav",
                "https://assets.test/four.wav",
            ],
            "at most 3",
        ),
        (
            [
                "https://assets.test/repeated.wav",
                "https://assets.test/repeated.wav",
            ],
            "duplicates",
        ),
        ([""], "non-empty strings"),
        ([" https://assets.test/voice.wav"], "trimmed"),
    ],
)
def test_picsart_submit_rejects_invalid_audio_urls_before_network(
    tmp_path: Path, audio_urls: list[str], message: str
) -> None:
    network_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        return httpx.Response(200, json={"response": {"id": "unexpected"}})

    client = PicsartClient(_settings(tmp_path), httpx.MockTransport(handler))
    with pytest.raises(ValueError, match=message):
        client.submit(
            prompt="prompt",
            video_url="https://assets.test/source.mp4",
            image_url=None,
            audio_urls=audio_urls,
            duration=5,
            resolution="768p",
            aspect_ratio="16:9",
        )

    assert network_calls == 0


@pytest.mark.parametrize(
    ("handler", "outcome"),
    [
        (
            lambda request: (_ for _ in ()).throw(
                httpx.ReadTimeout("lost response", request=request)
            ),
            "unknown",
        ),
        (
            lambda request: httpx.Response(400, json={"error": "bad request"}),
            "rejected",
        ),
        (
            lambda request: httpx.Response(500, json={"error": "upstream"}),
            "unknown",
        ),
        (
            lambda request: httpx.Response(429, json={"error": "rate limited"}),
            "unknown",
        ),
        (
            lambda request: httpx.Response(200, json={"response": {}}),
            "unknown",
        ),
    ],
)
def test_picsart_submit_failure_outcome_is_explicit(
    tmp_path: Path, handler: object, outcome: str
) -> None:
    client = PicsartClient(_settings(tmp_path), httpx.MockTransport(handler))
    with pytest.raises(ProviderFailure) as caught:
        client.submit(
            prompt="prompt",
            video_url="https://assets.test/source.mp4",
            image_url=None,
            duration=5,
            resolution="768p",
            aspect_ratio="16:9",
        )
    assert caught.value.record["outcome"] == outcome


def test_picsart_poll_failure_keeps_task_and_download_records_binary(
    tmp_path: Path,
) -> None:
    def failed_poll(request: httpx.Request) -> httpx.Response:
        assert request.url.path == (
            "/workflows/minimax/h3-max/reference-to-video/same-task/result"
        )
        return httpx.Response(
            200,
            json={"response": {"status": "FAILED", "error": {"message": "nope"}}},
        )

    client = PicsartClient(_settings(tmp_path), httpx.MockTransport(failed_poll))
    with pytest.raises(ProviderFailure) as caught:
        client.poll("same-task")
    assert caught.value.record["task_id"] == "same-task"
    assert len(caught.value.record["polls"]) == 1

    payload = b"\x00\xffvideo-bytes"
    downloader = PicsartClient(
        _settings(tmp_path),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=payload, headers={"Content-Type": "video/mp4"}
            )
        ),
    )
    destination = tmp_path / "downloads" / "output.mp4"
    downloaded = downloader.download("https://result.test/video.mp4", destination)
    assert downloaded.value == destination
    assert downloaded.evidence_media == [destination]
    assert destination.read_bytes() == payload
    assert downloaded.record["request"]["method"] == "GET"
    assert downloaded.record["request"]["url"] == "https://result.test/video.mp4"
    assert downloaded.record["response"]["body"] == {
        "content_type": "video/mp4",
        "length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_picsart_poll_preserves_workflow_failed_422_exchange(
    tmp_path: Path,
) -> None:
    body = {
        "status": "error",
        "reason": "minimax/h3-max/reference-to-video_failed",
        "message": (
            "minimax/h3-max/reference-to-video failed with message: "
            "ValidationError: reference video download failed"
        ),
    }
    calls = 0

    def failed_poll(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == (
            "/workflows/minimax/h3-max/reference-to-video/failed-task/result"
        )
        return httpx.Response(422, json=body)

    client = PicsartClient(_settings(tmp_path), httpx.MockTransport(failed_poll))
    with pytest.raises(ProviderFailure) as caught:
        client.poll("failed-task")

    assert calls == 1
    assert caught.value.record["task_id"] == "failed-task"
    assert caught.value.record["polls"][-1]["response"]["status_code"] == 422
    assert caught.value.record["polls"][-1]["response"]["body"] == body
