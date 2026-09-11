import asyncio

import httpx
import pytest

from app.safe_download import ReferenceDownloadError, download_reference_url, is_tiktok_page_url

PAGE = "https://www.tiktok.com/@person/video/123456"


@pytest.mark.parametrize("url,expected", [
    (PAGE, True), ("https://tiktok.com/@p/video/1", True),
    ("https://tiktok.com.evil.example/video/1", False),
    ("https://eviltiktok.com/video/1", False), ("https://cdn.example/video.mp4", False),
    ("https://www.tiktok.com/media/file.mp4", False), ("https://vm.tiktok.com/abc", False),
    ("https://www.tiktok.com/@p/video/not-numeric", False), ("https://[::1/video", False),
])
def test_exact_tiktok_host_family(url, expected):
    assert is_tiktok_page_url(url) is expected


def download(tmp_path, handler, url=PAGE):
    async def resolver(hostname, port):
        return ["127.0.0.1", "93.184.216.34"]
    return asyncio.run(download_reference_url(url, tmp_path / "source.mp4", max_bytes=1024,
                      timeout_s=2, transport=httpx.MockTransport(handler), resolver=resolver))


def test_tiktok_protocol_then_pinned_media(tmp_path):
    seen = []
    def handler(request):
        seen.append(request)
        if request.headers["host"] == "www.tikwm.com":
            assert request.method == "GET"
            assert request.url.params["url"] == PAGE
            return httpx.Response(200, json={"code": 0, "data": {"play": "https://cdn.example/video.mp4?signature=test"}})
        return httpx.Response(200, content=b"video")
    assert download(tmp_path, handler) == 5
    assert [r.headers["host"] for r in seen] == ["www.tikwm.com", "cdn.example"]
    assert all(r.url.host == "93.184.216.34" for r in seen)
    assert [r.extensions["sni_hostname"] for r in seen] == ["www.tikwm.com", "cdn.example"]


@pytest.mark.parametrize("payload", [{"code": -1}, {"code": 0}, {"code": 0, "data": []}, [], {"code": 0, "data": {"play": ""}}])
def test_invalid_parser_results_are_clear_errors(tmp_path, payload):
    with pytest.raises(ReferenceDownloadError) as caught:
        download(tmp_path, lambda _: httpx.Response(200, json=payload))
    assert caught.value.code == "tiktok_resolution_failed"


@pytest.mark.parametrize("status,content", [(503, b"busy"), (200, b"<html>page</html>"), (302, b"")])
def test_parser_http_and_nonjson_errors(tmp_path, status, content):
    with pytest.raises(ReferenceDownloadError) as caught:
        download(tmp_path, lambda _: httpx.Response(status, content=content))
    assert caught.value.code == "tiktok_resolution_failed"


@pytest.mark.parametrize("redirect", [False, True])
def test_parser_media_private_target_and_redirect_are_rejected(tmp_path, redirect):
    seen = []
    def handler(request):
        seen.append(request)
        if request.headers["host"] == "www.tikwm.com":
            play = "https://cdn.example/video" if redirect else "https://127.0.0.1/secret"
            return httpx.Response(200, json={"code": 0, "data": {"play": play}})
        return httpx.Response(302, headers={"Location": "https://127.0.0.1/secret"})
    with pytest.raises(ReferenceDownloadError) as caught:
        download(tmp_path, handler)
    assert caught.value.code == "reference_url_forbidden"
    assert len(seen) == (2 if redirect else 1)


@pytest.mark.parametrize("url", ["https://user@tiktok.com/@p/video/1", "https://tiktok.com:444/@p/video/1", "https://[::1/video"])
def test_invalid_page_never_requests_parser(tmp_path, url):
    def forbidden(_request):
        raise AssertionError("invalid page reached parser")
    with pytest.raises(ReferenceDownloadError):
        download(tmp_path, forbidden, url)


@pytest.mark.parametrize("host,path", [("cdn.example", "/video.mp4"), ("www.tiktok.com", "/media/file.mp4"), ("vm.tiktok.com", "/abc")])
def test_non_tiktok_direct_url_never_calls_parser(tmp_path, host, path):
    def handler(request):
        assert request.headers["host"] == host
        return httpx.Response(200, content=b"video")
    assert download(tmp_path, handler, "https://" + host + path) == 5
