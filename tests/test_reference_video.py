import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app import reference_video


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required",
)


def _run(*argv: str) -> None:
    subprocess.run(argv, check=True, capture_output=True)


def _make_source(path: Path, *, duration: float = 2.0, offset: float = 0.375) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-itsoffset",
        str(offset),
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x120:rate=24:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=48000:duration={duration}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-copyts",
        str(path),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stream_types(path: Path) -> list[str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [item["codec_type"] for item in json.loads(result.stdout)["streams"]]


def _derive(root: Path, source: Path) -> reference_video.ReferenceVideoResult:
    return reference_video.derive_reference_video(
        root=root,
        source_path=source.relative_to(root).as_posix(),
        expected_source_sha256=_sha256(source),
        output_path="artifacts/reference.mp4",
        receipt_path="artifacts/reference_video.json",
    )


def test_derives_canonical_receipt_and_preserves_the_complete_video_timeline(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen" / "segment.mp4"
    (tmp_path / "artifacts").mkdir()
    _make_source(source)

    result = _derive(tmp_path, source)

    assert result.output_path == tmp_path / "artifacts" / "reference.mp4"
    assert result.receipt_path == tmp_path / "artifacts" / "reference_video.json"
    assert _stream_types(source) == ["video", "audio"]
    assert _stream_types(result.output_path) == ["video"]
    receipt_bytes = result.receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt_bytes == json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert receipt["schema"] == "duet.reference-video"
    assert receipt["version"] == 1
    assert receipt["source"] == {
        "path": "frozen/segment.mp4",
        "sha256": _sha256(source),
        "size": source.stat().st_size,
        "streams": ["video", "audio"],
        "video": receipt["output"]["video"],
    }
    assert receipt["output"]["path"] == "artifacts/reference.mp4"
    assert receipt["output"]["sha256"] == _sha256(result.output_path)
    assert receipt["output"]["size"] == result.output_path.stat().st_size
    video = receipt["output"]["video"]
    assert video["codec_name"] == "h264"
    assert video["time_base"] == "1/12288"
    assert video["start_pts"] == 4608
    assert video["start_time"] == "0.375000"
    assert video["duration_ts"] == 24576
    assert video["duration"] == "2.000000"
    assert video["r_frame_rate"] == "24/1"
    assert video["avg_frame_rate"] == "24/1"
    assert video["frame_count"] == 48
    assert video["packet_timeline"] == {
        "basis": "stream_start_pts",
        "count": 48,
        "sha256": video["packet_timeline"]["sha256"],
    }
    assert len(video["packet_timeline"]["sha256"]) == 64
    assert video["frame_timeline"] == {
        "basis": "stream_start_pts",
        "count": 48,
        "sha256": video["frame_timeline"]["sha256"],
    }
    assert len(video["frame_timeline"]["sha256"]) == 64
    assert receipt["contract"] == {
        "audio_stream_count": 0,
        "container": "mp4",
        "max_duration_s": 10,
        "preserved_video_fields": [
            "codec_name",
            "codec_tag_string",
            "profile",
            "level",
            "width",
            "height",
            "pix_fmt",
            "time_base",
            "start_pts",
            "start_time",
            "duration_ts",
            "duration",
            "r_frame_rate",
            "avg_frame_rate",
            "frame_count",
            "packet_timeline",
            "frame_timeline",
        ],
        "timestamp_basis": "stream_start_pts",
        "video_mode": "stream_copy",
    }
    assert receipt["derivation"]["argv"] == list(reference_video.FFMPEG_COMMAND)
    assert result.source_sha256 == receipt["source"]["sha256"]
    assert result.output_sha256 == receipt["output"]["sha256"]
    assert result.receipt_sha256 == hashlib.sha256(receipt_bytes).hexdigest()


def test_repeated_runs_publish_identical_output_and_receipt_bytes(tmp_path: Path) -> None:
    source = tmp_path / "frozen" / "segment.mp4"
    (tmp_path / "artifacts").mkdir()
    _make_source(source)

    first = _derive(tmp_path, source)
    first_output = first.output_path.read_bytes()
    first_receipt = first.receipt_path.read_bytes()
    second = _derive(tmp_path, source)

    assert second.output_path.read_bytes() == first_output
    assert second.receipt_path.read_bytes() == first_receipt
    assert second.output_sha256 == first.output_sha256
    assert second.receipt_sha256 == first.receipt_sha256


@pytest.mark.parametrize(
    ("source_path", "output_path", "receipt_path"),
    [
        ("../segment.mp4", "artifacts/reference.mp4", "artifacts/reference_video.json"),
        ("frozen/segment.mp4", "../reference.mp4", "artifacts/reference_video.json"),
        ("frozen/segment.mp4", "artifacts/reference.mp4", "../reference_video.json"),
        ("/tmp/segment.mp4", "artifacts/reference.mp4", "artifacts/reference_video.json"),
    ],
)
def test_rejects_absolute_or_escaping_paths(
    tmp_path: Path, source_path: str, output_path: str, receipt_path: str
) -> None:
    source = tmp_path / "frozen" / "segment.mp4"
    (tmp_path / "artifacts").mkdir()
    _make_source(source)

    with pytest.raises(reference_video.ReferenceVideoError, match="path_invalid"):
        reference_video.derive_reference_video(
            root=tmp_path,
            source_path=source_path,
            expected_source_sha256=_sha256(source),
            output_path=output_path,
            receipt_path=receipt_path,
        )


def test_rejects_symlink_source_and_symlink_output_parent(tmp_path: Path) -> None:
    source = tmp_path / "frozen" / "segment.mp4"
    artifacts = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    artifacts.mkdir()
    outside.mkdir()
    _make_source(source)
    (tmp_path / "frozen" / "linked.mp4").symlink_to(source)

    with pytest.raises(reference_video.ReferenceVideoError, match="path_invalid"):
        reference_video.derive_reference_video(
            root=tmp_path,
            source_path="frozen/linked.mp4",
            expected_source_sha256=_sha256(source),
            output_path="artifacts/reference.mp4",
            receipt_path="artifacts/reference_video.json",
        )

    (tmp_path / "linked-artifacts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(reference_video.ReferenceVideoError, match="path_invalid"):
        reference_video.derive_reference_video(
            root=tmp_path,
            source_path="frozen/segment.mp4",
            expected_source_sha256=_sha256(source),
            output_path="linked-artifacts/reference.mp4",
            receipt_path="linked-artifacts/reference_video.json",
        )


def test_rejects_source_hash_mismatch_before_derivation(tmp_path: Path) -> None:
    source = tmp_path / "frozen" / "segment.mp4"
    (tmp_path / "artifacts").mkdir()
    _make_source(source)

    with pytest.raises(reference_video.ReferenceVideoError, match="source_hash_mismatch"):
        reference_video.derive_reference_video(
            root=tmp_path,
            source_path="frozen/segment.mp4",
            expected_source_sha256="0" * 64,
            output_path="artifacts/reference.mp4",
            receipt_path="artifacts/reference_video.json",
        )
    assert not (tmp_path / "artifacts" / "reference.mp4").exists()
    assert not (tmp_path / "artifacts" / "reference_video.json").exists()


def test_rejects_video_longer_than_ten_seconds(tmp_path: Path) -> None:
    source = tmp_path / "frozen" / "segment.mp4"
    (tmp_path / "artifacts").mkdir()
    _make_source(source, duration=10.125, offset=0.0)

    with pytest.raises(reference_video.ReferenceVideoError, match="duration_invalid"):
        _derive(tmp_path, source)
    assert not (tmp_path / "artifacts" / "reference.mp4").exists()
    assert not (tmp_path / "artifacts" / "reference_video.json").exists()


def test_derivation_failure_does_not_replace_published_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "frozen" / "segment.mp4"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _make_source(source)
    output = artifacts / "reference.mp4"
    receipt = artifacts / "reference_video.json"
    output.write_bytes(b"published-output")
    receipt.write_bytes(b"published-receipt")

    def fail_derivation(*_args: object, **_kwargs: object) -> None:
        raise reference_video.ReferenceVideoError("reference_video_ffmpeg_failed")

    monkeypatch.setattr(reference_video, "_run_ffmpeg", fail_derivation)

    with pytest.raises(reference_video.ReferenceVideoError, match="ffmpeg_failed"):
        _derive(tmp_path, source)
    assert output.read_bytes() == b"published-output"
    assert receipt.read_bytes() == b"published-receipt"
