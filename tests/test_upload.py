import json
import subprocess

from conftest import AUTH, make_settings
from fastapi.testclient import TestClient

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


def test_upload_too_long_422_via_mock(client, video_1s, monkeypatch, settings):
    """伪造 ffprobe 输出：时长 999s > 上限 2s。"""
    fake = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({"format": {"duration": "999.0"}}), stderr="",
    )
    monkeypatch.setattr("app.storage.subprocess.run", lambda *a, **kw: fake)
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations", headers=AUTH,
                        files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 422
    assert not settings.data_dir.exists() or list(settings.data_dir.iterdir()) == []


def test_upload_requires_auth(client, video_1s):
    with open(video_1s, "rb") as f:
        r = client.post("/api/conversations",
                        files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 401


def test_upload_missing_file_422(client):
    r = client.post("/api/conversations", headers=AUTH, data={"note": "x"})
    assert r.status_code == 422


def test_upload_rate_limit(client):
    # 每 IP 每分钟 10 次；第 11 次 429（用坏扩展名触发，不计 ffprobe 成本）
    last = None
    for _ in range(11):
        last = client.post("/api/conversations", headers=AUTH,
                           files={"file": ("a.txt", b"x", "text/plain")})
    assert last.status_code == 429
