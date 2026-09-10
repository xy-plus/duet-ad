from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

from app.safe_download import ReferenceDownloadError, download_reference_url


PUBLIC_V4 = "93.184.216.34"
PUBLIC_V6 = "2606:4700:4700::1111"


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def run_download(url: str, destination: Path, transport: httpx.MockTransport, resolver, **overrides) -> int:
    return asyncio.run(download_reference_url(
        url,
        destination,
        max_bytes=overrides.pop("max_bytes", 1024),
        timeout_s=overrides.pop("timeout_s", 1),
        transport=transport,
        resolver=resolver,
        **overrides,
    ))


def test_download_pins_numeric_ip_and_preserves_host_and_sni_across_redirect(tmp_path: Path):
    seen: list[httpx.Request] = []

    async def resolver(hostname: str, _port: int):
        return {
            "media.example": [PUBLIC_V4, PUBLIC_V6],
            "cdn.example": ["8.8.8.8"],
        }[hostname]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"Location": "https://cdn.example/final.mp4"})
        return httpx.Response(200, content=b"video-bytes")

    destination = tmp_path / "source.mp4"
    size = run_download(
        "https://media.example/original.mp4",
        destination,
        httpx.MockTransport(handler),
        resolver,
    )

    assert size == len(b"video-bytes")
    assert destination.read_bytes() == b"video-bytes"
    assert os.stat(destination).st_mode & 0o777 == 0o600
    assert [request.url.host for request in seen] == [PUBLIC_V4, "8.8.8.8"]
    assert [request.headers["host"] for request in seen] == ["media.example", "cdn.example"]
    assert [request.extensions["sni_hostname"] for request in seen] == ["media.example", "cdn.example"]


def test_redirect_to_private_target_is_rejected_before_second_request(tmp_path: Path):
    calls = 0

    async def resolver(hostname: str, _port: int):
        return [PUBLIC_V4] if hostname == "public.example" else ["10.0.0.7"]

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(302, headers={"Location": "http://internal.example/secret"})

    destination = tmp_path / "source.mp4"
    with pytest.raises(ReferenceDownloadError) as caught:
        run_download("https://public.example/video", destination, httpx.MockTransport(handler), resolver)

    assert caught.value.code == "reference_url_forbidden"
    assert calls == 1
    assert not destination.exists()


@pytest.mark.parametrize("answers,selected", [
    ([PUBLIC_V4, "127.0.0.1"], PUBLIC_V4),
    (["2001::1", "10.0.0.1", PUBLIC_V4], PUBLIC_V4),
    (["::1", PUBLIC_V6, "224.0.0.1"], PUBLIC_V6),
    ([PUBLIC_V6], PUBLIC_V6),
    (["::ffff:127.0.0.1", "::ffff:93.184.216.34"], "::ffff:5db8:d822"),
])
def test_mixed_dns_pins_only_public_candidate(tmp_path: Path, answers, selected):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"video")

    async def resolver(_hostname: str, _port: int):
        return answers

    assert run_download("https://mixed.example/video", tmp_path / "source.mp4",
                        httpx.MockTransport(handler), resolver) == 5
    assert len(seen) == 1
    assert seen[0].url.host == selected
    assert seen[0].headers["host"] == "mixed.example"
    assert seen[0].extensions["sni_hostname"] == "mixed.example"


@pytest.mark.parametrize("answers,code", [
    (["127.0.0.1", "10.0.0.1", "::1", "2001::1", "224.0.0.1"], "reference_url_forbidden"),
    (["169.254.169.254", "::ffff:127.0.0.1"], "reference_url_forbidden"),
    ([], "invalid_reference_url"),
    (None, "invalid_reference_url"),
])
def test_no_safe_dns_candidate_rejects_before_transport(tmp_path: Path, answers, code):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not receive an unapproved target")

    async def resolver(_hostname: str, _port: int):
        if answers is None:
            raise OSError("DNS unavailable")
        return answers

    with pytest.raises(ReferenceDownloadError) as caught:
        run_download(
            "https://mixed.example/video",
            tmp_path / "source.mp4",
            httpx.MockTransport(handler),
            resolver,
        )
    assert caught.value.code == code


def test_each_redirect_filters_dns_and_preserves_pin(tmp_path: Path):
    seen = []
    targets = ["one.example", "two.example", "three.example"]

    async def resolver(hostname: str, _port: int):
        return ["127.0.0.1", PUBLIC_V6 if hostname == "two.example" else PUBLIC_V4]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) < 3:
            return httpx.Response(302, headers={"Location": "https://" + targets[len(seen)] + "/video"})
        return httpx.Response(200, content=b"video")

    assert run_download("https://one.example/video", tmp_path / "source.mp4",
                        httpx.MockTransport(handler), resolver) == 5
    assert [r.url.host for r in seen] == [PUBLIC_V4, PUBLIC_V6, PUBLIC_V4]
    assert [r.headers["host"] for r in seen] == targets
    assert [r.extensions["sni_hostname"] for r in seen] == targets


@pytest.mark.parametrize(
    "url,code",
    [
        ("https://user@public.example/video", "invalid_reference_url"),
        ("https://public.example:444/video", "invalid_reference_url"),
        ("https://public.example:/video", "invalid_reference_url"),
        ("file:///etc/passwd", "invalid_reference_url"),
        ("http://127.0.0.1/video", "reference_url_forbidden"),
        ("http://[::1]/video", "reference_url_forbidden"),
        ("http://224.0.0.1/video", "reference_url_forbidden"),
        ("http://[ff02::1]/video", "reference_url_forbidden"),
    ],
)
def test_unsafe_url_shapes_never_reach_transport(tmp_path: Path, url: str, code: str):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not receive an unapproved target")

    async def resolver(_hostname: str, _port: int):
        return [PUBLIC_V4]

    with pytest.raises(ReferenceDownloadError) as caught:
        run_download(url, tmp_path / "source.mp4", httpx.MockTransport(handler), resolver)
    assert caught.value.code == code


@pytest.mark.parametrize("url", [" https://public.example/video", "https://public.example/video\n", "https://public.example/vi\tdeo"])
def test_whitespace_and_control_characters_are_rejected_without_normalization(tmp_path: Path, url: str):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not receive a malformed target")

    async def resolver(_hostname: str, _port: int):
        return [PUBLIC_V4]

    with pytest.raises(ReferenceDownloadError) as caught:
        run_download(url, tmp_path / "source.mp4", httpx.MockTransport(handler), resolver)
    assert caught.value.code == "invalid_reference_url"


def test_wall_clock_timeout_also_applies_to_mock_transport(tmp_path: Path):
    async def resolver(_hostname: str, _port: int):
        return [PUBLIC_V4]

    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, content=b"late")

    with pytest.raises(ReferenceDownloadError) as caught:
        run_download(
            "https://public.example/video",
            tmp_path / "source.mp4",
            httpx.MockTransport(handler),
            resolver,
            timeout_s=0.01,
        )
    assert caught.value.code == "reference_download_timeout"


def test_content_length_and_streamed_byte_limits_publish_nothing_partial(tmp_path: Path):
    async def resolver(_hostname: str, _port: int):
        return [PUBLIC_V4]

    destination = tmp_path / "source.mp4"
    destination.write_bytes(b"existing")

    declared = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"Content-Length": "1001"}, stream=ChunkStream(b"x"))
    )
    with pytest.raises(ReferenceDownloadError) as caught:
        run_download("https://public.example/video", destination, declared, resolver, max_bytes=1000)
    assert caught.value.code == "reference_too_large"
    assert destination.read_bytes() == b"existing"

    streamed = httpx.MockTransport(
        lambda _request: httpx.Response(200, stream=ChunkStream(b"1234", b"5678"))
    )
    with pytest.raises(ReferenceDownloadError) as caught:
        run_download("https://public.example/video", destination, streamed, resolver, max_bytes=7)
    assert caught.value.code == "reference_too_large"
    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.glob(".source.mp4.*.tmp")) == []


@pytest.mark.parametrize("content_length", [b"\xb2", b"1" * 21, b"1, 1"])
def test_invalid_content_length_is_structured_and_never_published(tmp_path: Path, content_length: bytes):
    async def resolver(_hostname: str, _port: int):
        return [PUBLIC_V4]

    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers=[(b"Content-Length", content_length)], stream=ChunkStream(b"x"))
    )
    with pytest.raises(ReferenceDownloadError) as caught:
        run_download("https://public.example/video", tmp_path / "source.mp4", transport, resolver)
    assert caught.value.code == "reference_download_failed"
    assert not (tmp_path / "source.mp4").exists()


def test_compressed_response_is_rejected_before_decompression(tmp_path: Path):
    async def resolver(_hostname: str, _port: int):
        return [PUBLIC_V4]

    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=ChunkStream(b"compressed"))
    )
    with pytest.raises(ReferenceDownloadError) as caught:
        run_download("https://public.example/video", tmp_path / "source.mp4", transport, resolver)
    assert caught.value.code == "reference_content_encoding"


def test_redirect_limit_is_manual_and_bounded(tmp_path: Path):
    async def resolver(_hostname: str, _port: int):
        return [PUBLIC_V4]

    transport = httpx.MockTransport(
        lambda request: httpx.Response(302, headers={"Location": f"/next?from={request.url.path}"})
    )
    with pytest.raises(ReferenceDownloadError) as caught:
        run_download(
            "https://public.example/start",
            tmp_path / "source.mp4",
            transport,
            resolver,
            max_redirects=2,
        )
    assert caught.value.code == "reference_redirect_limit"
