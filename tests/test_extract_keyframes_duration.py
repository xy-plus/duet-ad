import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills" / "video-maker" / "scripts" / "extract_keyframes.py"
)
SPEC = importlib.util.spec_from_file_location("extract_keyframes_duration_test", SCRIPT)
assert SPEC and SPEC.loader
extract = importlib.util.module_from_spec(SPEC)
_dont_write_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    SPEC.loader.exec_module(extract)
finally:
    sys.dont_write_bytecode = _dont_write_bytecode


def _probe(monkeypatch, stream):
    monkeypatch.setattr(extract.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        extract.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(
            stdout=json.dumps({"streams": [stream]}), returncode=0, stderr=""
        ),
    )
    return extract.probe_video_duration(Path("source.mp4"), decoded_duration=20.0)


def test_manifest_duration_prefers_video_stream_for_vfr(monkeypatch):
    assert _probe(monkeypatch, {"duration": "16.766667"}) == 16.766667


def test_manifest_duration_uses_duration_ts_time_base(monkeypatch):
    assert _probe(monkeypatch, {
        "duration": "N/A", "duration_ts": "503", "time_base": "1/30",
    }) == pytest.approx(503 / 30)


def test_manifest_duration_uses_decoded_fallback_not_format(monkeypatch):
    assert _probe(monkeypatch, {}) == 20.0
