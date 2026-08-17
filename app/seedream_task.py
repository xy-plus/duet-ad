#!/usr/bin/env python3
"""Build, submit, and save Volcengine Ark Seedream image edit tasks safely.

编辑端点（实测契约，2026-08-18 首次真实调用验证）：Seedream 5.0 Pro 编辑 = 图生图，
走 images/generations（SeedEdit 专用的 /api/v1/images/edits 与本模型无关）：

    POST https://ark.cn-beijing.volces.com/api/v3/images/generations
    JSON: {"model", "prompt", "image": ["data:image/png;base64,<b64>"],
           "response_format": "b64_json", "watermark": false}

响应为同步 200（实测 60s 级返回）：{"model", "created", "data": [{"b64_json": ...}]}；
无 request_id、无异步轮询。data[0].b64_json 缺失、为空或非法即失败退出（契约恒 b64_json）。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://ark.cn-beijing.volces.com"
# Seedream 5.0 Pro 编辑 = 图生图（实测契约，2026-08-18）
EDIT_URL = f"{BASE_URL}/api/v3/images/generations"
DEFAULT_MODEL = "doubao-seedream-5-0-pro-260628"
RESPONSE_FORMAT = "b64_json"  # 实测契约：响应恒 b64_json
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
    edit.add_argument("--request-timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def api_key() -> str:
    value = os.environ.get("ARK_API_KEY", "").strip()
    if not value:
        raise SystemExit("ARK_API_KEY is not configured in the environment.")
    return value


def read_image(value: str) -> bytes:
    """读 PNG 字节并校验魔数（契约只收 PNG，避免把任意文件当图片上传）。"""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Image file does not exist: {path}")
    data = path.read_bytes()
    if not data.startswith(PNG_MAGIC):
        raise SystemExit(f"Image is not a PNG file: {path}")
    return data


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
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "image": [f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"],
        "response_format": RESPONSE_FORMAT,
        "watermark": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        EDIT_URL,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    return _urlopen_json(request, timeout)


def save_result(task: dict[str, Any], destination: str) -> Path:
    """data[0].b64_json 严格校验解码写 --out；缺失/为空/非法一律硬错误。"""
    data_list = task.get("data") or []
    first = data_list[0] if data_list else None
    b64 = (first or {}).get("b64_json")
    if not b64:
        raise RuntimeError("Successful edit response did not include b64_json image data.")
    data = base64.b64decode(b64, validate=True)
    if not data:
        raise RuntimeError("Empty image data in edit response.")
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"Saved: {path}", flush=True)
    return path


def summarize(task: dict[str, Any]) -> dict[str, Any]:
    """成功响应摘要：模型/时间 + 图片字节数（不把图本身打到 stdout）。"""
    summary = {key: task.get(key) for key in ("model", "created", "id") if key in task}
    data_list = task.get("data") or []
    first = data_list[0] if data_list else None
    if first and first.get("b64_json"):
        summary["image_bytes"] = len(base64.b64decode(first["b64_json"], validate=True))
    return summary


def run_edit(args: argparse.Namespace) -> int:
    image_bytes = read_image(args.image)
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
    task = submit_edit(key, args.model, prompt, image_bytes, args.request_timeout)
    save_result(task, args.out)
    print(json.dumps(summarize(task), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.request_timeout <= 0:
        raise SystemExit("Request timeout must be positive.")
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
