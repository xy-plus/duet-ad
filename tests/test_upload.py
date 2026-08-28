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


@pytest.mark.parametrize(
    ("dialogue_mode", "expected_provenance"),
    [("edit", "asr+edited"), ("custom", "manual")],
)
def test_upload_freezes_manual_dialogue_before_pipeline(
    client, video_1s, settings, dialogue_mode, expected_provenance
):
    lines = [
        {"start_s": 0, "end_s": 0.4, "text": "  用户台词一  "},
        {"text": "用户台词二", "start_s": 0.5, "end_s": 0.9},
    ]
    with open(video_1s, "rb") as source:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            files={"file": ("clip.mp4", source, "video/mp4")},
            data={
                "dialogue_mode": dialogue_mode,
                "lines": json.dumps(lines, ensure_ascii=False),
            },
        )

    assert response.status_code == 201, response.json()
    cid = response.json()["id"]
    meta = json.loads(
        (settings.data_dir / cid / "meta.json").read_text(encoding="utf-8")
    )
    assert meta["status"] == "queued"
    assert meta["dialogue_mode"] == dialogue_mode
    assert meta["voice_lines"] == [
        {
            "text": "用户台词一",
            "start_s": 0.0,
            "end_s": 0.4,
            "classification": None,
            "provenance": expected_provenance,
        },
        {
            "text": "用户台词二",
            "start_s": 0.5,
            "end_s": 0.9,
            "classification": None,
            "provenance": expected_provenance,
        },
    ]
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
    assert detail.status_code == 200
    assert detail.json()["dialogue"] == {
        "mode": dialogue_mode,
        "lines": [
            {"text": "用户台词一", "start_s": 0.0, "end_s": 0.4},
            {"text": "用户台词二", "start_s": 0.5, "end_s": 0.9},
        ],
        "auto_lines": [],
    }


def test_upload_none_freezes_empty_dialogue_without_lines(
    client, video_1s, settings
):
    with open(video_1s, "rb") as source:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            files={"file": ("clip.mp4", source, "video/mp4")},
            data={"dialogue_mode": "none"},
        )

    assert response.status_code == 201, response.json()
    meta = json.loads(
        (settings.data_dir / response.json()["id"] / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["dialogue_mode"] == "none"
    assert meta["voice_lines"] == []


@pytest.mark.parametrize(
    ("data", "detail"),
    [
        ({"dialogue_mode": "edit"}, "invalid_dialogue"),
        ({"dialogue_mode": "custom", "lines": "not-json"}, "invalid_dialogue"),
        ({"dialogue_mode": "custom", "lines": "[]"}, "invalid_dialogue"),
        (
            {
                "dialogue_mode": "custom",
                "lines": json.dumps([
                    {"text": "越界", "start_s": 0.0, "end_s": 2.0}
                ]),
            },
            "invalid_dialogue",
        ),
        (
            {
                "dialogue_mode": "auto",
                "lines": json.dumps([
                    {"text": "迟到输入", "start_s": 0.0, "end_s": 0.5}
                ]),
            },
            "invalid_dialogue",
        ),
        ({"dialogue_mode": "none", "lines": "[]"}, "invalid_dialogue"),
    ],
)
def test_upload_rejects_noncanonical_dialogue_without_keeping_project(
    client, video_1s, settings, data, detail
):
    with open(video_1s, "rb") as source:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            files={"file": ("clip.mp4", source, "video/mp4")},
            data=data,
        )

    assert response.status_code == 422
    assert response.json() == {"detail": detail}
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_stale_none_voice_mode_requires_refresh_without_creating_upload(
    client, video_1s, settings
):
    with open(video_1s, "rb") as source:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            files={"file": ("clip.mp4", source, "video/mp4")},
            data={"voice_mode": "none"},
        )
    assert response.status_code == 409
    assert response.json() == {"detail": "页面版本已更新，请刷新页面后重试。"}
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_upload_rejects_unknown_multipart_field_without_creating_upload(
    client, video_1s, settings
):
    with open(video_1s, "rb") as source:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            files={"file": ("clip.mp4", source, "video/mp4")},
            data={"unexpected": "value"},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_create_request"}
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_stale_none_voice_mode_with_unknown_field_is_not_misclassified(
    client, video_1s, settings
):
    with open(video_1s, "rb") as source:
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            files={"file": ("clip.mp4", source, "video/mp4")},
            data={"voice_mode": "none", "unexpected": "value"},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_create_request"}
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


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


def test_upload_accepts_long_video_and_rejects_only_over_300_seconds(
    client, video_1s, monkeypatch, settings
):
    durations = iter((10.01, 15.0, 30.0, 300.0, 300.001))

    def fake_probe(*_args, **_kwargs):
        duration = next(durations)
        return subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({
                "format": {"duration": str(duration)},
                "streams": [{"width": 320, "height": 240, "duration": str(duration)}],
            }), stderr="",
        )

    monkeypatch.setattr("app.storage.subprocess.run", fake_probe)
    for duration in (10.01, 15.0, 30.0, 300.0):
        with open(video_1s, "rb") as f:
            response = client.post(
                "/api/conversations", headers=AUTH,
                files={"file": ("clip.mp4", f, "video/mp4")},
            )
        assert response.status_code == 201, (duration, response.text)

    with open(video_1s, "rb") as f:
        rejected = client.post(
            "/api/conversations", headers=AUTH,
            files={"file": ("clip.mp4", f, "video/mp4")},
        )
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["actual_duration_s"] == 300.001
    assert rejected.json()["detail"]["max_duration_s"] == 300.0


def test_long_upload_rejects_non_keep_audio_and_cleans_up(
    client, video_1s, monkeypatch, settings
):
    fake = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({
            "format": {"duration": "15.001"},
            "streams": [{"width": 320, "height": 240, "duration": "15.001"}],
        }), stderr="",
    )
    monkeypatch.setattr("app.storage.subprocess.run", lambda *_a, **_kw: fake)
    with open(video_1s, "rb") as f:
        response = client.post(
            "/api/conversations", headers=AUTH,
            files={"file": ("clip.mp4", f, "video/mp4")},
            data={"voice_mode": "rewrite"},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "long_video_audio_mode_unsupported"}
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_upload_accepts_exact_h3_limit(client, video_1s, monkeypatch, settings):
    fake = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({
            "format": {"duration": "10.0"},
            "streams": [{"width": 320, "height": 240, "duration": "10.0"}],
        }), stderr="",
    )
    monkeypatch.setattr("app.storage.subprocess.run", lambda *a, **kw: fake)
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations", headers=AUTH,
                        files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 201
    meta = json.loads((settings.data_dir / r.json()["id"] / "meta.json").read_text())
    assert meta["duration_s"] == 10.0


@pytest.mark.parametrize(
    "video_duration,format_duration,voice_mode,expected_status",
    [
        (10.0, 10.1, "rewrite", 201),
        (10.001, 9.9, "rewrite", 422),
        (300.0, 300.2, "keep", 201),
        (300.001, 299.9, "keep", 422),
    ],
)
def test_upload_gates_use_video_stream_duration_only(
    client, video_1s, monkeypatch, video_duration, format_duration,
    voice_mode, expected_status,
):
    fake = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({
            "format": {"duration": str(format_duration)},
            "streams": [{
                "width": 320, "height": 240, "duration": str(video_duration),
            }],
        }), stderr="",
    )
    monkeypatch.setattr("app.storage.subprocess.run", lambda *_a, **_kw: fake)
    with open(video_1s, "rb") as file:
        response = client.post(
            "/api/conversations", headers=AUTH,
            files={"file": ("clip.mp4", file, "video/mp4")},
            data={"voice_mode": voice_mode},
        )
    assert response.status_code == expected_status, response.json()


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
