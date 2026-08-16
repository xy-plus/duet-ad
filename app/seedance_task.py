#!/usr/bin/env python3
"""Build, submit, poll, and download Volcengine Ark Seedance tasks safely."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seedance-2-0-260128"
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "expired"}
PASS_THROUGH_PREFIXES = ("https://", "http://", "asset://", "data:")


def add_shared_wait_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait", action="store_true", help="Poll until terminal status.")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--wait-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--state-file", help="Write the latest task JSON.")
    parser.add_argument("--download", help="Download the successful MP4 here.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage Ark Seedance generation tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Build or create one task.")
    prompt_group = create.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text.")
    prompt_group.add_argument("--prompt-file", help="UTF-8 prompt file.")
    create.add_argument("--ref-images", nargs="+", required=True, help="One to nine ordered images.")
    create.add_argument("--model", default=DEFAULT_MODEL)
    create.add_argument(
        "--ratio", default="9:16", choices=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    )
    create.add_argument("--duration", type=int, default=15)
    create.add_argument("--resolution", default="720p", choices=["480p", "720p", "1080p"])
    create.add_argument(
        "--generate-audio", action=argparse.BooleanOptionalAction, default=True
    )
    create.add_argument("--watermark", action=argparse.BooleanOptionalAction, default=False)
    create.add_argument("--payload-out", help="Write the exact request body as JSON.")
    create.add_argument("--dry-run", action="store_true", help="Validate without network or cost.")
    create.add_argument(
        "--confirm-submit",
        action="store_true",
        help="Mechanical gate required for a live creation after user confirmation.",
    )
    add_shared_wait_arguments(create)

    status = subparsers.add_parser("status", help="Query or resume an existing task.")
    status.add_argument("task_id")
    add_shared_wait_arguments(status)
    return parser.parse_args()


def write_json(path_value: str | None, data: dict[str, Any]) -> None:
    if not path_value:
        return
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        prompt = args.prompt.strip()
    else:
        path = Path(args.prompt_file).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Prompt file does not exist: {path}")
        prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit("Prompt is empty.")
    return prompt


def media_reference(value: str) -> str:
    if value.startswith(PASS_THROUGH_PREFIXES):
        return value
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Reference image does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise SystemExit(f"Reference is not recognized as an image: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= len(args.ref_images) <= 9:
        raise SystemExit("Seedance tasks require one to nine reference images.")
    if not 4 <= args.duration <= 15:
        raise SystemExit("Duration must be between 4 and 15 seconds.")
    content: list[dict[str, Any]] = [{"type": "text", "text": read_prompt(args)}]
    for value in args.ref_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": media_reference(value)},
                "role": "reference_image",
            }
        )
    return {
        "model": args.model,
        "content": content,
        "generate_audio": args.generate_audio,
        "ratio": args.ratio,
        "duration": args.duration,
        "resolution": args.resolution,
        "watermark": args.watermark,
    }


def api_key() -> str:
    value = os.environ.get("ARK_API_KEY", "").strip()
    if not value:
        raise SystemExit("ARK_API_KEY is not configured in the environment.")
    return value


def request_json(
    method: str,
    url: str,
    key: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ark HTTP {exc.code}: {message[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ark request failed: {exc.reason}") from exc


def task_url(task_id: str) -> str:
    return f"{BASE_URL}/contents/generations/tasks/{task_id}"


def summarize(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: task.get(key)
        for key in (
            "id",
            "model",
            "status",
            "resolution",
            "ratio",
            "duration",
            "framespersecond",
            "seed",
            "generate_audio",
        )
        if key in task
    }


def poll_task(
    initial: dict[str, Any],
    key: str,
    interval: float,
    wait_timeout: float,
    request_timeout: float,
    state_file: str | None,
) -> dict[str, Any]:
    task = initial
    task_id = str(task.get("id") or "")
    if not task_id:
        raise RuntimeError("Ark response did not include a task ID.")
    deadline = time.monotonic() + wait_timeout
    last_status = None
    while True:
        status = str(task.get("status") or "unknown")
        if status != last_status:
            print(f"Task {task_id}: {status}", flush=True)
            last_status = status
        write_json(state_file, task)
        if status in TERMINAL_STATUSES:
            return task
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Task {task_id} did not finish within {wait_timeout:.0f}s.")
        time.sleep(max(1.0, interval))
        task = request_json("GET", task_url(task_id), key, request_timeout)


def download_result(task: dict[str, Any], destination: str, timeout: float) -> Path:
    content = task.get("content") or {}
    url = content.get("video_url") if isinstance(content, dict) else None
    if not url:
        raise RuntimeError("Successful task response did not include content.video_url.")
    path = Path(destination).expanduser().resolve()
    if path.exists() and path.is_dir():
        path = path / "generated.mp4"
    elif not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        path = path / "generated.mp4"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        path.write_bytes(response.read())
    if path.stat().st_size == 0:
        raise RuntimeError("Downloaded video is empty.")
    print(f"Downloaded: {path}", flush=True)
    return path


def run_create(args: argparse.Namespace) -> int:
    payload = build_payload(args)
    write_json(args.payload_out, payload)
    if args.dry_run:
        summary = {
            "dry_run": True,
            "model": payload["model"],
            "content_items": len(payload["content"]),
            "reference_images": len(payload["content"]) - 1,
            "generate_audio": payload["generate_audio"],
            "ratio": payload["ratio"],
            "duration": payload["duration"],
            "resolution": payload["resolution"],
            "watermark": payload["watermark"],
            "payload_out": str(Path(args.payload_out).resolve()) if args.payload_out else None,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.confirm_submit:
        raise SystemExit(
            "Live creation blocked. Obtain a separate user confirmation, then add --confirm-submit."
        )
    key = api_key()
    task = request_json(
        "POST", f"{BASE_URL}/contents/generations/tasks", key, args.request_timeout, payload
    )
    write_json(args.state_file, task)
    task_id = str(task.get("id") or "")
    if not task_id:
        raise RuntimeError("Ark create response did not include a task ID.")
    print(f"Created task: {task_id}", flush=True)
    if args.wait:
        task = poll_task(
            task,
            key,
            args.interval,
            args.wait_timeout,
            args.request_timeout,
            args.state_file,
        )
    if args.download:
        if str(task.get("status")) != "succeeded":
            raise RuntimeError("Cannot download because the task has not succeeded.")
        download_result(task, args.download, args.request_timeout)
    print(json.dumps(summarize(task), ensure_ascii=False, indent=2))
    return 0 if str(task.get("status")) not in {"failed", "cancelled", "expired"} else 1


def run_status(args: argparse.Namespace) -> int:
    key = api_key()
    task = request_json("GET", task_url(args.task_id), key, args.request_timeout)
    write_json(args.state_file, task)
    if args.wait:
        task = poll_task(
            task,
            key,
            args.interval,
            args.wait_timeout,
            args.request_timeout,
            args.state_file,
        )
    if args.download:
        if str(task.get("status")) != "succeeded":
            raise RuntimeError("Cannot download because the task has not succeeded.")
        download_result(task, args.download, args.request_timeout)
    print(json.dumps(summarize(task), ensure_ascii=False, indent=2))
    return 0 if str(task.get("status")) not in {"failed", "cancelled", "expired"} else 1


def main() -> int:
    args = parse_args()
    if args.interval <= 0 or args.wait_timeout <= 0 or args.request_timeout <= 0:
        raise SystemExit("Timeout and interval values must be positive.")
    return run_create(args) if args.command == "create" else run_status(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # keep credentials out of tracebacks and logs
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
