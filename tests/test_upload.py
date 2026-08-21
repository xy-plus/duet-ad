import json
import shutil
import subprocess

import httpx
import pytest
from conftest import AUTH, make_settings
from fastapi.testclient import TestClient

from app import downloader
from app.main import create_app


def test_upload_ok(client, video_1s, settings):
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations", headers=AUTH,
                        files={"file": ("clip.mp4", f, "video/mp4")},
                        data={"note": "清洁演示"})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "queued"
    cdir = settings.data_dir / body["id"]
    assert (cdir / "source.mp4").is_file()
    meta = json.loads((cdir / "meta.json").read_text())
    assert meta["note"] == "清洁演示"
    assert meta["status"] == "queued"


def test_upload_uppercase_ext_ok(client, video_1s):
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations", headers=AUTH,
                        files={"file": ("CLIP.MP4", f, "video/mp4")})
    assert r.status_code == 201


def test_upload_bad_ext_422(client):
    r = client.post("/api/conversations", headers=AUTH,
                    files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_upload_non_video_content_422_and_cleaned_up(client, settings):
    r = client.post("/api/conversations", headers=AUTH,
                    files={"file": ("fake.mp4", b"not a video at all", "video/mp4")})
    assert r.status_code == 422
    assert list(settings.data_dir.iterdir()) == [] if settings.data_dir.exists() else True


def test_upload_oversize_422(tmp_path, video_1s):
    settings = make_settings(tmp_path, max_upload_mb=0)
    with TestClient(create_app(settings)) as c:
        with open(video_1s, "rb") as f:
            r = c.post("/api/conversations", headers=AUTH,
                       files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 422
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_upload_accepts_arbitrary_positive_duration(client, video_1s, monkeypatch, settings):
    """项目不对视频总时长设上限。"""
    fake = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({
            "format": {"duration": "999.0"},
            "streams": [{"width": 320, "height": 240}],
        }), stderr="",
    )
    monkeypatch.setattr("app.storage.subprocess.run", lambda *a, **kw: fake)
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations", headers=AUTH,
                        files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 201
    meta = json.loads((settings.data_dir / r.json()["id"] / "meta.json").read_text())
    assert meta["duration_s"] == 999.0


def test_upload_requires_auth(client, video_1s):
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations",
                        files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 401


def test_upload_neither_file_nor_url_400(client):
    r = client.post("/api/conversations", headers=AUTH, data={"note": "x"})
    assert r.status_code == 400


def test_upload_both_file_and_url_400(client, video_1s):
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations", headers=AUTH,
                        files={"file": ("clip.mp4", f, "video/mp4")},
                        data={"reference_url": "https://example.com/a.mp4"})
    assert r.status_code == 400


class _FakeConn:
    def close(self):
        pass


class _FakeResponse:
    """钉住 _open_pinned 用的假 HTTP 响应：status + headers + 一次性 body。"""

    def __init__(self, status, headers=None, body=b""):
        self.status = status
        self._headers = headers or {}
        self._body = body

    def getheader(self, name):
        return self._headers.get(name.lower())

    def read(self, n=-1):
        data, self._body = self._body, b""
        return data


def test_url_resolves_to_private_ip_422(client, monkeypatch, settings):
    """注入假 resolver：域名解析到私网 IP，直接拒。"""
    monkeypatch.setattr("app.downloader._local_resolve", lambda host: ["10.0.0.1"])
    r = client.post("/api/conversations", headers=AUTH,
                    data={"reference_url": "http://internal.example.com/a.mp4"})
    assert r.status_code == 422
    assert "private" in r.json()["detail"]
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_url_redirect_to_private_ip_422(client, monkeypatch, settings):
    """每次跳转独立重校验：跳到 link-local 地址必须拒。"""
    monkeypatch.setattr("app.downloader._local_resolve", lambda host: ["93.184.216.34"])
    resp = _FakeResponse(302, {"location": "http://169.254.169.254/latest/meta-data"})
    monkeypatch.setattr("app.downloader._open_pinned", lambda *a, **kw: (_FakeConn(), resp))
    r = client.post("/api/conversations", headers=AUTH,
                    data={"reference_url": "http://cdn.example.com/a.mp4"})
    assert r.status_code == 422
    assert "private" in r.json()["detail"]
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_url_content_length_over_limit_422(client, monkeypatch, settings):
    monkeypatch.setattr("app.downloader._local_resolve", lambda host: ["93.184.216.34"])
    over = settings.max_upload_mb * 1024 * 1024 + 1
    resp = _FakeResponse(200, {"content-length": str(over)})
    monkeypatch.setattr("app.downloader._open_pinned", lambda *a, **kw: (_FakeConn(), resp))
    r = client.post("/api/conversations", headers=AUTH,
                    data={"reference_url": "http://cdn.example.com/big.mp4"})
    assert r.status_code == 422
    assert "exceeds" in r.json()["detail"]
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_url_connection_refused_422(client, monkeypatch, settings):
    """连接被拒（OSError）归一为 DownloadError → 422，不残留会话目录。"""
    monkeypatch.setattr("app.downloader._local_resolve", lambda host: ["93.184.216.34"])

    def boom(*a, **kw):
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("app.downloader._pinned_socket", boom)
    r = client.post("/api/conversations", headers=AUTH,
                    data={"reference_url": "http://cdn.example.com/a.mp4"})
    assert r.status_code == 422
    assert "refused" in r.json()["detail"]
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_doh_resolve_via_mock_transport():
    def handler(request):
        assert request.url.host == "1.1.1.1"
        assert request.url.path == "/dns-query"
        assert request.url.params["name"] == "cdn.example.com"
        assert request.url.params["type"] == "A"
        assert request.headers["accept"] == "application/dns-json"
        return httpx.Response(200, json={
            "Status": 0,
            "Answer": [
                {"name": "cdn.example.com", "type": 5, "data": "alias.example.com"},
                {"name": "cdn.example.com", "type": 1, "data": "93.184.216.34"},
            ],
        })

    addresses = downloader._doh_resolve(
        "cdn.example.com", proxy="http://127.0.0.1:7897", timeout=5,
        transport=httpx.MockTransport(handler),
    )
    assert addresses == ["93.184.216.34"]

    def no_answer(request):
        return httpx.Response(200, json={"Status": 0})

    with pytest.raises(downloader.DownloadError):
        downloader._doh_resolve(
            "cdn.example.com", proxy="http://127.0.0.1:7897", timeout=5,
            transport=httpx.MockTransport(no_answer),
        )


def test_proxy_download_uses_doh_not_local_dns(tmp_path, monkeypatch):
    """代理路径全程不碰本机 DNS：_local_resolve 被碰即炸，DoH 供 IP，假响应落盘。"""

    def no_local(host):
        raise AssertionError("local DNS must not be used when proxy is set")

    monkeypatch.setattr("app.downloader._local_resolve", no_local)
    monkeypatch.setattr("app.downloader._doh_resolve", lambda host, **kw: ["93.184.216.34"])
    resp = _FakeResponse(200, {"content-length": "4"}, b"data")
    monkeypatch.setattr("app.downloader._open_pinned", lambda *a, **kw: (_FakeConn(), resp))
    dest = tmp_path / "v.mp4"
    downloader.download_public_video(
        "http://cdn.example.com/v.mp4", dest,
        proxy="http://127.0.0.1:7897", max_bytes=1024, timeout=5,
    )
    assert dest.read_bytes() == b"data"


def test_tiktok_video_facts_via_mock_transport():
    def handler(request):
        assert request.url.host == "www.tikwm.com"
        return httpx.Response(200, json={
            "code": 0,
            "data": {"play": " https://cdn.example.com/v.mp4 ", "duration": 13},
        })

    facts = downloader.tiktok_video_facts(
        "https://www.tiktok.com/@someone/video/7664758988675878151",
        api_transport=httpx.MockTransport(handler),
    )
    assert facts == {"video_id": "7664758988675878151", "play": "https://cdn.example.com/v.mp4"}


def test_reference_download_retries_two_transient_failures(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cdir = tmp_path / "session"
    cdir.mkdir()
    calls = 0

    def flaky(_url, dest, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise downloader.DownloadError("temporary", retryable=True)
        dest.write_bytes(b"video")

    monkeypatch.setattr(downloader, "download_public_video", flaky)
    result = downloader.fetch_reference(
        "https://example.com/video.mp4", cdir, settings
    )

    assert calls == 3
    assert result.read_bytes() == b"video"


def test_reference_download_does_not_retry_permanent_failure(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cdir = tmp_path / "session"
    cdir.mkdir()
    calls = 0

    def rejected(_url, _dest, **_kwargs):
        nonlocal calls
        calls += 1
        raise downloader.DownloadError("private address")

    monkeypatch.setattr(downloader, "download_public_video", rejected)
    with pytest.raises(downloader.DownloadError, match="private address"):
        downloader.fetch_reference("https://example.com/video.mp4", cdir, settings)

    assert calls == 1


def test_tiktok_retry_refreshes_temporary_play_url(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cdir = tmp_path / "session"
    cdir.mkdir()
    facts_calls = 0
    play_urls = []

    def facts(_url, _proxy, *, timeout):
        nonlocal facts_calls
        facts_calls += 1
        return {"video_id": "123", "play": f"https://cdn.example.com/{facts_calls}.mp4"}

    def download(play_url, dest, **_kwargs):
        play_urls.append(play_url)
        if len(play_urls) == 1:
            raise downloader.DownloadError("expired", retryable=True)
        dest.write_bytes(b"video")

    monkeypatch.setattr(downloader, "tiktok_video_facts", facts)
    monkeypatch.setattr(downloader, "download_public_video", download)
    downloader.fetch_reference(
        "https://www.tiktok.com/@someone/video/123", cdir, settings
    )

    assert facts_calls == 2
    assert play_urls == [
        "https://cdn.example.com/1.mp4",
        "https://cdn.example.com/2.mp4",
    ]


def test_create_with_reference_url_queued(client, monkeypatch, settings, video_1s):
    seen = {}

    def fake_fetch(url, cdir, s):
        seen["url"] = url
        dest = cdir / "source.mp4"
        shutil.copyfile(video_1s, dest)
        return dest

    monkeypatch.setattr("app.downloader.fetch_reference", fake_fetch)
    r = client.post("/api/conversations", headers=AUTH,
                    data={"reference_url": "https://example.com/clip.mp4", "note": "链接"})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "queued"
    assert seen["url"] == "https://example.com/clip.mp4"
    cdir = settings.data_dir / body["id"]
    assert (cdir / "source.mp4").is_file()
    meta = json.loads((cdir / "meta.json").read_text())
    assert meta["note"] == "链接"
    assert meta["status"] == "queued"


def test_upload_rate_limit(client):
    # 每 IP 每分钟 10 次；第 11 次 429（用坏扩展名触发，不计 ffprobe 成本）
    last = None
    for _ in range(11):
        last = client.post("/api/conversations", headers=AUTH,
                           files={"file": ("a.txt", b"x", "text/plain")})
    assert last.status_code == 429
