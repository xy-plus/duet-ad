#!/usr/bin/env python3
"""Probe a video, export exact or evenly sampled PNG frames, and build a contact sheet."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment-dependent message
    raise SystemExit(
        "OpenCV is required. Install opencv-python-headless in a task-local environment."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export video keyframes and create contact_sheet.jpg plus manifest.json."
    )
    parser.add_argument("video", help="Input MP4, MOV, or WebM file.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--times", help="Comma-separated timestamps in seconds.")
    choice.add_argument("--sample-count", type=int, help="Number of evenly sampled frames.")
    choice.add_argument(
        "--fps",
        type=float,
        help="Sample this many frames per second; writes paged contact sheets.",
    )
    parser.add_argument("--prefix", default="frame", help="Filename role label.")
    parser.add_argument("--columns", type=int, default=4, help="Contact-sheet columns.")
    return parser.parse_args()


def parse_times(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit("--times must contain comma-separated numbers.") from exc
    if not values:
        raise SystemExit("--times did not contain any timestamps.")
    if values != sorted(values):
        raise SystemExit("--times must be chronological.")
    return values


def probe_audio(video: Path) -> bool | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
        data = json.loads(completed.stdout or "{}")
        return bool(data.get("streams"))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def probe_video_duration(video: Path, decoded_duration: float) -> float:
    """Return the v:0 visual timeline; container/audio/OpenCV metadata are excluded."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", (
                "stream=duration,duration_ts,time_base,avg_frame_rate,r_frame_rate"
            ),
            "-of", "json", str(video),
        ]
        try:
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=30
            )
            stream = (json.loads(completed.stdout or "{}").get("streams") or [])[0]
            try:
                raw_duration = stream.get("duration")
                if isinstance(raw_duration, bool):
                    raise TypeError
                candidate = float(raw_duration)
                if math.isfinite(candidate) and candidate > 0:
                    return candidate
            except (TypeError, ValueError):
                pass
            try:
                raw_duration_ts = stream.get("duration_ts")
                if isinstance(raw_duration_ts, bool):
                    raise TypeError
                ticks = float(raw_duration_ts)
                numerator, denominator = str(stream.get("time_base")).split("/", 1)
                candidate = ticks * float(numerator) / float(denominator)
                if math.isfinite(candidate) and candidate > 0:
                    return candidate
            except (TypeError, ValueError, ZeroDivisionError):
                pass
            packet_command = [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_packets", "-show_entries",
                "packet=pts_time,dts_time,duration_time", "-of", "json", str(video),
            ]
            packets_result = subprocess.run(
                packet_command, check=True, capture_output=True, text=True, timeout=30
            )
            packets = json.loads(packets_result.stdout or "{}").get("packets")
            if isinstance(packets, list):
                starts: list[float] = []
                ends: list[float] = []
                starts_with_duration: set[float] = set()
                for packet in packets:
                    if not isinstance(packet, dict):
                        continue
                    start = _finite_packet_number(packet.get("pts_time"))
                    if start is None:
                        start = _finite_packet_number(packet.get("dts_time"))
                    if start is None:
                        continue
                    starts.append(start)
                    packet_duration = _finite_packet_number(packet.get("duration_time"))
                    if packet_duration is not None and packet_duration > 0:
                        ends.append(start + packet_duration)
                        starts_with_duration.add(start)
                starts = sorted(set(starts))
                step = None
                if len(starts) > 1:
                    positive = [b - a for a, b in zip(starts, starts[1:]) if b > a]
                    if positive:
                        step = positive[-1]
                if step is None:
                    rate = _positive_rate(stream.get("avg_frame_rate")) or _positive_rate(
                        stream.get("r_frame_rate")
                    )
                    step = 1 / rate if rate else None
                if (
                    starts
                    and starts[-1] not in starts_with_duration
                    and step is not None
                    and step > 0
                ):
                    ends.append(starts[-1] + step)
                if starts and ends:
                    candidate = max(ends) - starts[0]
                    if math.isfinite(candidate) and candidate > 0:
                        return candidate
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError):
            pass
    raise SystemExit("Could not determine the video stream duration.")


def _finite_packet_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_rate(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        rate = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def write_image(path: Path, frame: np.ndarray, extension: str, params: list[int] | None = None) -> None:
    success, encoded = cv2.imencode(extension, frame, params or [])
    if not success:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def fit_thumbnail(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def build_sheet(
    contact_frames: list[tuple[np.ndarray, str]], columns: int, thumb_height: int
) -> np.ndarray:
    columns = min(columns, len(contact_frames))
    rows = math.ceil(len(contact_frames) / columns)
    thumb_width = 360
    label_height = 42
    gap = 8
    sheet_width = columns * thumb_width + (columns - 1) * gap
    sheet_height = rows * (thumb_height + label_height) + (rows - 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), 255, dtype=np.uint8)
    for zero_index, (frame, label) in enumerate(contact_frames):
        row = zero_index // columns
        column = zero_index % columns
        x = column * (thumb_width + gap)
        y = row * (thumb_height + label_height + gap)
        sheet[y : y + thumb_height, x : x + thumb_width] = fit_thumbnail(
            frame, thumb_width, thumb_height
        )
        cv2.putText(
            sheet,
            label,
            (x + 8, y + thumb_height + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return sheet


def write_contact_sheets(
    out_dir: Path,
    contact_frames: list[tuple[np.ndarray, str]],
    columns: int,
    thumb_height: int,
    paged: bool,
) -> None:
    """paged 时按页输出 contact_sheet_01.jpg…（每页 columns×6 帧），否则单张 contact_sheet.jpg。"""
    page_size = columns * 6
    if not paged or len(contact_frames) <= page_size:
        sheet = build_sheet(contact_frames, columns, thumb_height)
        write_image(out_dir / "contact_sheet.jpg", sheet, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 92])
        return
    for start in range(0, len(contact_frames), page_size):
        sheet = build_sheet(contact_frames[start : start + page_size], columns, thumb_height)
        write_image(
            out_dir / f"contact_sheet_{start // page_size + 1:02d}.jpg",
            sheet,
            ".jpg",
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )


def _run_frame_extract(command: list[str]) -> None:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Could not decode requested video frames: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise SystemExit(f"Could not decode requested video frames{suffix}")


def extract_frames(
    video: Path,
    out_dir: Path,
    times: list[float],
    *,
    uniform_rate: float | None,
) -> list[np.ndarray]:
    """Decode requested frames in one ffmpeg pass using presentation timestamps."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to extract video frames.")
    with tempfile.TemporaryDirectory(prefix=".frames-", dir=out_dir) as raw_tmp:
        tmp = Path(raw_tmp)
        temporary_paths = [tmp / f"{index:06d}.png" for index in range(1, len(times) + 1)]
        if uniform_rate is not None:
            pattern = tmp / "%06d.png"
            step = 1.0 / uniform_rate
            _run_frame_extract([
                ffmpeg, "-v", "error", "-y", "-i", str(video),
                "-map", "0:v:0", "-vf",
                f"setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={step:.9f},"
                f"fps=fps={uniform_rate:.12g}:start_time=0",
                "-frames:v", str(len(times)), str(pattern),
            ])
        else:
            branches = "".join(f"[source{index}]" for index in range(len(times)))
            filters = []
            if len(times) == 1:
                filters.append("[0:v:0]setpts=PTS-STARTPTS[normalized]")
                inputs = ["[normalized]"]
            else:
                filters.append(
                    f"[0:v:0]setpts=PTS-STARTPTS,split={len(times)}{branches}"
                )
                inputs = [f"[source{index}]" for index in range(len(times))]
            for index, (input_name, timestamp) in enumerate(zip(inputs, times)):
                filters.append(
                    f"{input_name}trim=start={timestamp:.9f},setpts=PTS-STARTPTS[out{index}]"
                )
            command = [
                ffmpeg, "-v", "error", "-y", "-i", str(video),
                "-filter_complex", ";".join(filters),
            ]
            for index, path in enumerate(temporary_paths):
                command.extend([
                    "-map", f"[out{index}]", "-frames:v", "1", "-c:v", "png", str(path),
                ])
            _run_frame_extract(command)
        frames = []
        for path in temporary_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise SystemExit("Could not decode all requested video frames.")
            frames.append(frame)
        return frames


def main() -> int:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video does not exist: {video}")
    if args.sample_count is not None and args.sample_count < 1:
        raise SystemExit("--sample-count must be positive.")
    if args.fps is not None and args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.columns < 1:
        raise SystemExit("--columns must be positive.")
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise SystemExit("Video metadata is incomplete or invalid.")

    decoded_duration = frame_count / fps
    duration = probe_video_duration(video, decoded_duration)
    last_safe_time = max(0.0, duration - (1.0 / fps))
    if args.times:
        times = parse_times(args.times)
    elif args.fps is not None:
        step = 1.0 / args.fps
        times = [k * step for k in range(int(duration * args.fps) + 1)]
        if not times or last_safe_time - times[-1] > step / 2:
            times.append(last_safe_time)
        else:
            times[-1] = last_safe_time
    elif args.sample_count == 1:
        times = [0.0]
    else:
        times = [
            last_safe_time * index / (args.sample_count - 1)
            for index in range(args.sample_count)
        ]

    for timestamp in times:
        if timestamp < 0 or timestamp > duration:
            raise SystemExit(
                f"Timestamp {timestamp:.3f}s is outside video duration {duration:.3f}s."
            )

    if args.fps is not None:
        uniform_rate = args.fps
    elif args.sample_count is not None and len(times) > 1 and last_safe_time > 0:
        uniform_rate = (len(times) - 1) / last_safe_time
    elif args.sample_count == 1:
        uniform_rate = 1.0
    else:
        uniform_rate = None
    decoded_frames = extract_frames(
        video,
        out_dir,
        [min(timestamp, last_safe_time) for timestamp in times],
        uniform_rate=uniform_rate,
    )

    records: list[dict[str, object]] = []
    contact_frames: list[tuple[np.ndarray, str]] = []
    digits = max(2, len(str(len(times))))
    for index, (requested_time, frame) in enumerate(zip(times, decoded_frames), start=1):
        filename = f"{index:0{digits}d}_{args.prefix}_{requested_time:07.3f}s.png"
        write_image(out_dir / filename, frame, ".png")
        records.append(
            {
                "index": index,
                "time_seconds": round(requested_time, 6),
                "file": filename,
            }
        )
        contact_frames.append((frame, f"{index:02d}  {requested_time:.3f}s"))

    write_contact_sheets(
        out_dir,
        contact_frames,
        args.columns,
        thumb_height=round(360 * height / width),
        paged=args.fps is not None,
    )

    manifest = {
        "source": str(video),
        "file_size_bytes": video.stat().st_size,
        "duration_seconds": round(duration, 6),
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "orientation": "portrait" if height > width else "landscape" if width > height else "square",
        "audio_present": probe_audio(video),
        "frames": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
