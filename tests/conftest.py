import os
import subprocess

# app.main 在 import 时会读取环境变量建 app，测试导入前必须备好
os.environ.setdefault("ACCESS_TOKEN", "test-token")

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def make_settings(tmp_path, **overrides):
    base = dict(
        access_token=TOKEN,
        data_dir=tmp_path / "data",
        max_upload_mb=5,
        max_duration_s=2,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def video_1s(tmp_path):
    """用 ffmpeg 生成 1 秒真实视频。"""
    p = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-pix_fmt", "yuv420p", str(p),
        ],
        check=True, capture_output=True,
    )
    return p
