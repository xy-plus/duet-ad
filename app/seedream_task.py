#!/usr/bin/env python3
"""Build, submit, poll, and save Volcengine Ark Seedream image edit tasks safely.

编辑端点（multipart 表单，OpenAI images API 兼容风格，与 seedance 的 /api/v3 不同）：

    POST {BASE_URL}/api/v1/images/edits
    异步任务查询：GET {EDIT_URL}/{request_id}

官方文档实现时未确证（https://www.volcengine.com/docs/82379/1541523 为 JS 渲染页，
镜像文档记该接口为同步返回 data[]），端点与查询路径均做成模块常量；脚本同时兼容
两种响应：提交即带图（data[]/content 内 b64_json 或 url）则直接写盘，否则按
request_id（或 id/task_id）轮询至 succeeded/failed 后取图。b64_json 优先于 url。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

BASE_URL = "https://ark.cn-beijing.volces.com"
# 编辑端点走 /api/v1（seedance 用 /api/v3；两者路径前缀不同，勿混用）
EDIT_URL = f"{BASE_URL}/api/v1/images/edits"
# 异步任务查询端点：官方文档未确证路径，如上游调整改此常量即可
TASK_URL_TEMPLATE = f"{EDIT_URL}/{{id}}"
DEFAULT_MODEL = "doubao-seedream-5-0-pro-260628"
TERMINAL_STATUSES = {"succeeded", "failed"}
RESPONSE_FORMAT = "b64_json"  # 官方支持则优先 b64（免二次下载），否则可改 "url"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage Ark Seedream image edit tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    edit = subparsers.add_parser("edit", help="Edit one PNG image.")
    edit.add_argument("--image", required=True, help="Source PNG image.")
    edit.add_argument("--prompt", required=True, help="Edit instruction.")
    edit.add_argument("--out", required=True, help="Output PNG path.")
    edit.add_argument("--model", default=DEFAULT_MODEL)
    edit.add_argument("--dry-run", action="store_true", help="Validate without network or cost.")
    edit.add_argument(
        "--confirm-submit",
        action="store_true",
        help="Mechanical gate required for a live edit after user confirmation.",
    )
    edit.add_argument("--state-file", help="Write the latest task JSON.")
    edit.add_argument("--poll-interval", type=float, default=5.0)
    edit.add_argument("--poll-timeout", type=float, default=600.0)
    edit.add_argument("--request-timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def write_json(path_value: str | None, data: dict[str, Any]) -> None:
    if not path_value:
        return
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def api_key() -> str:
    value = os.environ.get("ARK_API_KEY", "").strip()
    if not value:
        raise SystemExit("ARK_API_KEY is not configured in the environment.")
    return value


def read_image(value: str) -> tuple[bytes, str]:
    """读 PNG 字节并校验魔数（契约只收 PNG，避免把任意文件当图片上传）。"""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Image file does not exist: {path}")
    data = path.read_bytes()
    if not data.startswith(PNG_MAGIC):
        raise SystemExit(f"Image is not a PNG file: {path}")
    return data, path.name


def multipart_body(
    fields: dict[str, str], file_field: str, filename: str, data: bytes, mime: str
) -> tuple[bytes, str]:
    """手工构造 multipart/form-data body（不引入 requests）。"""
    boundary = "----ark-edit-" + uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode("utf-8")
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _urlopen_json(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ark HTTP {exc.code}: {message[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ark request failed: {exc.reason}") from exc


def submit_edit(
    key: str,
    model: str,
    prompt: str,
    image_bytes: bytes,
    filename: str,
    timeout: float,
) -> dict[str, Any]:
    body, content_type = multipart_body(
        {
            "model": model,
            "prompt": prompt,
            "response_format": RESPONSE_FORMAT,
            "watermark": "false",
        },
        "image",
        filename,
        image_bytes,
        "image/png",
    )
    request = urllib.request.Request(
        EDIT_URL,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": content_type},
    )
    return _urlopen_json(request, timeout)


def task_id_of(task: dict[str, Any]) -> str:
    """兼容多种命名：request_id / id / task_id。"""
    return str(task.get("request_id") or task.get("id") or task.get("task_id") or "")


def _find_first(obj: Any, key: str) -> Any:
    """递归查找第一个非空 key 值（兼容 data[]/content/... 多种结构）。"""
    if isinstance(obj, dict):
        if obj.get(key):
            return obj[key]
        for value in obj.values():
            found = _find_first(value, key)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_first(value, key)
            if found:
                return found
    return None


def has_image(task: dict[str, Any]) -> bool:
    return bool(_find_first(task, "b64_json") or _find_first(task, "url"))


def poll_task(
    initial: dict[str, Any],
    key: str,
    interval: float,
    wait_timeout: float,
    request_timeout: float,
    state_file: str | None,
) -> dict[str, Any]:
    task = initial
    task_id = task_id_of(task)
    if not task_id:
        raise RuntimeError("Ark response did not include a task ID.")
    deadline = time.monotonic() + wait_timeout
    last_status = None
    while True:
        status = str(task.get("status") or "unknown")
        if status != last_status:
            print(f"Edit task {task_id}: {status}", flush=True)
            last_status = status
        write_json(state_file, task)
        if status in TERMINAL_STATUSES:
            return task
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Edit task {task_id} did not finish within {wait_timeout:.0f}s.")
        time.sleep(max(1.0, interval))
        request = urllib.request.Request(
            TASK_URL_TEMPLATE.format(id=task_id),
            method="GET",
            headers={"Authorization": f"Bearer {key}"},
        )
        task = _urlopen_json(request, request_timeout)


def save_result(task: dict[str, Any], destination: str, timeout: float) -> Path:
    """b64_json 优先解码写盘；否则按 url 下载。写 --out（PNG 字节落盘）。"""
    b64 = _find_first(task, "b64_json")
    if b64:
        data = base64.b64decode(b64)
    else:
        url = _find_first(task, "url")
        if not url:
            raise RuntimeError("Successful edit response did not include image data (b64_json or url).")
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if path.stat().st_size == 0:
        raise RuntimeError("Downloaded image is empty.")
    print(f"Saved: {path}", flush=True)
    return path


def summarize(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: task.get(key)
        for key in ("request_id", "id", "model", "status")
        if key in task
    }


def run_edit(args: argparse.Namespace) -> int:
    image_bytes, filename = read_image(args.image)
    prompt = args.prompt.strip()
    if not prompt:
        raise SystemExit("Prompt is empty.")
    if args.dry_run:
        summary = {
            "dry_run": True,
            "model": args.model,
            "prompt": prompt,
            "image": str(Path(args.image).expanduser().resolve()),
            "out": str(Path(args.out).expanduser().resolve()),
            "response_format": RESPONSE_FORMAT,
            "watermark": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.confirm_submit:
        raise SystemExit(
            "Live edit blocked. Obtain a separate user confirmation, then add --confirm-submit."
        )
    key = api_key()
    task = submit_edit(key, args.model, prompt, image_bytes, filename, args.request_timeout)
    write_json(args.state_file, task)
    status = str(task.get("status") or "")
    if status in TERMINAL_STATUSES and status != "succeeded":
        print(json.dumps(summarize(task), ensure_ascii=False, indent=2))
        return 1
    if has_image(task):  # 同步响应：提交即带图，直接写盘
        save_result(task, args.out, args.request_timeout)
        print(json.dumps(summarize(task), ensure_ascii=False, indent=2))
        return 0
    task_id = task_id_of(task)
    if not task_id:
        raise RuntimeError("Ark edit response did not include a task ID or image data.")
    print(f"Created edit task: {task_id}", flush=True)
    task = poll_task(
        task, key, args.poll_interval, args.poll_timeout, args.request_timeout, args.state_file
    )
    status = str(task.get("status") or "unknown")
    print(json.dumps(summarize(task), ensure_ascii=False, indent=2))
    if status == "succeeded":
        save_result(task, args.out, args.request_timeout)
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_interval <= 0 or args.poll_timeout <= 0 or args.request_timeout <= 0:
        raise SystemExit("Timeout and interval values must be positive.")
    return run_edit(args)


def cli(argv: list[str] | None = None) -> int:
    """入口：一切异常转成非零退出码；报错不含密钥字面值。"""
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # keep credentials out of tracebacks and logs
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
