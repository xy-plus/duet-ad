from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.providers import (
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


def test_deepseek_structured_uses_beta_strict_submit_result(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"visual_prompt": {"type": "string"}},
        "required": ["visual_prompt"],
        "additionalProperties": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://deepseek.test/beta/chat/completions")
        body = json.loads(request.content)
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "submit_result",
                    "description": "Submit the complete structured result.",
                    "strict": True,
                    "parameters": schema,
                },
            }
        ]
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": "submit_result"},
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "submit_result",
                                        "arguments": '{"visual_prompt":"move left"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        )

    result = DeepSeekClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).complete_structured("skill", {"segment": 1}, schema)
    assert result.value == {"visual_prompt": "move left"}
    assert result.record["request"]["body"]["tools"][0]["function"]["strict"] is True


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


def test_deepseek_structured_invalid_arguments_keep_raw_response(tmp_path: Path) -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_result",
                                "arguments": "{not-json",
                            }
                        }
                    ]
                }
            }
        ]
    }
    client = DeepSeekClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    with pytest.raises(ProviderFailure) as caught:
        client.complete_structured("skill", {}, {"type": "object"})
    assert isinstance(caught.value.cause, json.JSONDecodeError)
    assert caught.value.record["response"]["body"] == payload


def test_mediakit_multi_scene_supports_documented_and_nested_upload_shapes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.jpg"
    source.write_bytes(b"jpeg-source")
    destination = tmp_path / "out" / "final.png"
    init_count = 0
    erase_count = 0

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

    result = MediaKitClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).erase(source, destination, ["erase-text", "erase-logo"])

    assert result.value == destination
    assert destination.read_bytes() == b"final-stage"
    assert [path.read_bytes() for path in result.evidence_media] == [
        b"first-stage",
        b"final-stage",
    ]
    assert len(result.record["stages"]) == 2
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


def test_picsart_advanced_request_omits_absent_optional_image(tmp_path: Path) -> None:
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
        duration=5,
        resolution="480p",
        aspect_ratio="9:16",
    )

    assert result.value == "task-no-image"


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
