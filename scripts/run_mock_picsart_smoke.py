#!/usr/bin/env python3
"""Run a real-upstream smoke test with only the final Picsart stage mocked.

The input is selected at runtime from the first successful project exposed by
the 3213 HTTPS API.  DeepSeek and MediaKit use the same credentials as the
running duet-ad1 service.  No Picsart client is constructed; its adapter
copies each local muted segment through the submit/poll/download contract so
the real trim and concat stages also execute.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


REPOSITORY_ROOT = Path("/home/xy/duet-ad1-h3-max-r2v")
SERVICE_ENV = Path("/home/xy/.config/duet-ad1/service.env")
DEEPSEEK_ENV = Path("/home/xy/.config/claude/deepseek.env")
ORIGIN_3213 = "https://127.0.0.1:3213"
SMOKE_ROOT = REPOSITORY_ROOT / "data" / "smoke-runs"
MAX_SOURCE_BYTES = 500 * 1024 * 1024
MAX_REPLACEMENT_BYTES = 10 * 1024 * 1024

sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import Settings  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.providers import ProviderResult  # noqa: E402
from app.store import ProjectStore  # noqa: E402


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        values[key] = value.strip().strip('"').strip("'")
    return values


def require_secret(values: dict[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required credential {name} is absent")
    return value


def first_successful_project(client: httpx.Client) -> tuple[str, dict[str, Any]]:
    response = client.get(f"{ORIGIN_3213}/api/conversations")
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError("3213 conversations response is not a list")
    for row in rows:
        if not isinstance(row, dict):
            continue
        progress = row.get("project_progress")
        if (
            isinstance(progress, dict)
            and progress.get("status") == "succeeded"
            and progress.get("percent") == 100
            and row.get("has_video") is True
            and isinstance(row.get("id"), str)
        ):
            project_id = row["id"]
            detail_response = client.get(
                f"{ORIGIN_3213}/api/conversations/{project_id}"
            )
            detail_response.raise_for_status()
            detail = detail_response.json()
            if not isinstance(detail, dict):
                raise RuntimeError("3213 project detail is not an object")
            return project_id, detail
    raise RuntimeError("3213 has no successful project")


def download_bounded(
    client: httpx.Client,
    url: str,
    destination: Path,
    maximum: int,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
    total = 0
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    total += len(chunk)
                    if total > maximum:
                        raise RuntimeError(f"download exceeds {maximum} bytes")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        return total
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def without_sensitive_preview(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_sensitive_preview(item)
            for key, item in value.items()
            if key not in {"preview_url", "signed_url", "access_token"}
        }
    if isinstance(value, list):
        return [without_sensitive_preview(item) for item in value]
    return value


class MockPicsart:
    """Local implementation of the Picsart submit/poll/download contract."""

    def __init__(self, assets: dict[str, Path], assets_lock: threading.Lock) -> None:
        self.assets = assets
        self.assets_lock = assets_lock
        self.tasks: dict[str, Path] = {}
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def submit(
        self,
        *,
        prompt: str,
        video_url: str,
        image_url: str | None,
        duration: int,
        resolution: str,
        aspect_ratio: str,
    ) -> ProviderResult[str]:
        with self.assets_lock:
            source = self.assets.get(video_url)
        if source is None or not source.is_file():
            raise RuntimeError("mock Picsart received an unknown local asset")
        task_id = f"mock-picsart-{secrets.token_hex(10)}"
        with self.lock:
            self.tasks[task_id] = source
            self.requests.append(
                {
                    "task_id": task_id,
                    "prompt_chars": len(prompt),
                    "video_url": video_url,
                    "image_present": image_url is not None,
                    "duration": duration,
                    "resolution": resolution,
                    "aspect_ratio": aspect_ratio,
                    "prompt_expansion_mode": "quality",
                }
            )
        return ProviderResult(
            task_id,
            {
                "mock": True,
                "outcome": "accepted",
                "task_id": task_id,
                "provider": "picsart-mock",
            },
        )

    def poll(self, task_id: str) -> ProviderResult[str]:
        with self.lock:
            if task_id not in self.tasks:
                raise RuntimeError("mock Picsart poll received an unknown task")
        return ProviderResult(
            f"mock://picsart/{task_id}",
            {
                "mock": True,
                "outcome": "completed",
                "task_id": task_id,
                "provider": "picsart-mock",
            },
        )

    def download(self, url: str, destination: Path) -> ProviderResult[Path]:
        prefix = "mock://picsart/"
        if not url.startswith(prefix):
            raise RuntimeError("mock Picsart download URL is invalid")
        task_id = url.removeprefix(prefix)
        with self.lock:
            source = self.tasks.get(task_id)
        if source is None or not source.is_file():
            raise RuntimeError("mock Picsart download received an unknown task")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)
        return ProviderResult(
            destination,
            {
                "mock": True,
                "outcome": "downloaded",
                "task_id": task_id,
                "provider": "picsart-mock",
            },
            [destination],
        )


def trace_summary(project: Path) -> dict[str, Any]:
    trace = project / "logs" / "trace.jsonl"
    nodes: dict[str, dict[str, int]] = {}
    events: list[str] = []
    if trace.is_file():
        for line in trace.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = row.get("event")
            if isinstance(event, str):
                events.append(event)
            node = row.get("node")
            if isinstance(node, str) and event in {
                "node_started",
                "node_succeeded",
                "node_failed",
                "node_skipped",
            }:
                entry = nodes.setdefault(node, {})
                entry[event] = entry.get(event, 0) + 1
    return {"nodes": nodes, "event_count": len(events), "last_events": events[-5:]}


def run() -> int:
    service = read_env(SERVICE_ENV)
    deepseek_file = read_env(DEEPSEEK_ENV)
    access_token = require_secret(service, "ACCESS_TOKEN")
    mediakit_key = require_secret(service, "VOLC_MEDIAKIT_API_KEY")
    deepseek_key = (
        deepseek_file.get("DEEPSEEK_API_KEY", "").strip()
        or deepseek_file.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    )
    if not deepseek_key:
        raise RuntimeError("DeepSeek credential is absent from its source credential file")

    SMOKE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(SMOKE_ROOT, 0o700)

    with httpx.Client(
        verify=False,
        timeout=120.0,
        headers={"Authorization": f"Bearer {access_token}"},
    ) as client:
        source_project_id, detail = first_successful_project(client)
        effective_request = detail.get("effective_request")
        creation_input = detail.get("creation_input")
        if not isinstance(effective_request, dict) or not isinstance(creation_input, dict):
            raise RuntimeError("successful 3213 project has no complete frozen input")

        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = SMOKE_ROOT / f"{run_name}-{source_project_id}"
        input_root = run_root / "input"
        source_path = input_root / "source.mp4"
        replacement_path = input_root / "replacement-image.webp"
        source_url = f"{ORIGIN_3213}/api/conversations/{source_project_id}/files/source.mp4"
        replacement_meta = creation_input.get("replacement_image")
        replacement_url = (
            f"{ORIGIN_3213}/api/conversations/{source_project_id}/"
            "creation-input/replacement-image"
            if isinstance(replacement_meta, dict)
            else None
        )
        source_bytes = download_bounded(client, source_url, source_path, MAX_SOURCE_BYTES)
        replacement_bytes = 0
        if replacement_url is not None:
            replacement_bytes = download_bounded(
                client,
                replacement_url,
                replacement_path,
                MAX_REPLACEMENT_BYTES,
            )

    project_input = {
        "source_project_id": source_project_id,
        "effective_request": without_sensitive_preview(effective_request),
        "creation_input": without_sensitive_preview(creation_input),
        "source_endpoint": source_url,
        "replacement_endpoint": replacement_url,
        "downloaded_source_bytes": source_bytes,
        "downloaded_replacement_bytes": replacement_bytes,
    }
    run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run_root, 0o700)
    (run_root / "input_snapshot.json").write_text(
        json.dumps(project_input, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(run_root / "input_snapshot.json", 0o600)

    store = ProjectStore(run_root / "projects", acquire_process_lock=False)
    project_id, project, _state = store.create(
        effective_request=effective_request,
        client_request_id=f"mock-picsart-smoke-{source_project_id}",
    )
    inputs = project / "inputs"
    shutil.copyfile(source_path, inputs / "source.mp4")
    os.chmod(inputs / "source.mp4", 0o600)
    replacement_for_pipeline: Path | None = None
    if replacement_url is not None:
        replacement_for_pipeline = inputs / "replacement-image"
        shutil.copyfile(replacement_path, replacement_for_pipeline)
        os.chmod(replacement_for_pipeline, 0o600)
    source_meta = creation_input.get("source")
    if not isinstance(source_meta, dict):
        source_meta = {"mode": "upload", "filename": "source.mp4"}
    source_meta = {
        "mode": source_meta.get("mode", "upload"),
        "filename": source_meta.get("filename", "source.mp4"),
        "bytes": source_bytes,
    }
    replacement_snapshot = None
    if replacement_for_pipeline is not None:
        replacement_snapshot = {
            "filename": (
                replacement_meta.get("filename", "replacement-image.webp")
                if isinstance(replacement_meta, dict)
                else "replacement-image.webp"
            ),
            "bytes": replacement_bytes,
            "media_type": (
                replacement_meta.get("media_type", "image/webp")
                if isinstance(replacement_meta, dict)
                else "image/webp"
            ),
            "preview_url": f"/api/conversations/{project_id}/creation-input/replacement-image",
        }
    store.update(
        project_id,
        source_path="inputs/source.mp4",
        source_bytes=source_bytes,
        replacement_image_path=("inputs/replacement-image" if replacement_for_pipeline else None),
        creation_input={
            "version": 1,
            "source": source_meta,
            "replacement_image": replacement_snapshot,
        },
        has_source=True,
    )

    asset_map: dict[str, Path] = {}
    asset_lock = threading.Lock()

    def register_asset(_project_id: str, path: Path) -> str:
        token = secrets.token_urlsafe(18)
        url = f"https://mock-assets.invalid/{token}"
        with asset_lock:
            asset_map[url] = path.resolve(strict=True)
        return url

    settings = Settings(
        access_token=access_token,
        data_dir=run_root,
        public_base_url="https://smoke.invalid",
        deepseek_api_key=deepseek_key,
        mediakit_api_key=mediakit_key,
        enable_mediakit=service.get("ENABLE_MEDIAKIT_ERASE", "").lower()
        in {"1", "true", "yes", "on"},
        asr_cli=Path(
            service.get(
                "ASR_CLI",
                "/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli",
            )
        ),
        asr_model=Path(
            service.get(
                "ASR_MODEL", "/home/xy/.local/share/duet-asr/ggml-small.bin"
            )
        ),
        yamnet_model=REPOSITORY_ROOT / "models" / "yamnet.tflite",
        request_timeout_s=180.0,
        provider_poll_interval_s=0.1,
        provider_poll_timeout_s=10.0,
        retry_delay_s=1.0,
    )
    mock_picsart = MockPicsart(asset_map, asset_lock)
    pipeline = Pipeline(
        settings,
        store,
        register_asset,
        picsart=mock_picsart,
    )
    started = time.monotonic()
    pipeline.run(project_id)
    elapsed_s = time.monotonic() - started
    state = store.read(project_id)
    summary = {
        "source_project_id": source_project_id,
        "smoke_project_id": project_id,
        "run_root": str(run_root),
        "status": state.get("status"),
        "project_progress": state.get("project_progress"),
        "segment_count": state.get("segment_count"),
        "has_video": state.get("has_video"),
        "output_path": str(project / "output" / "generated.mp4"),
        "output_bytes": (
            (project / "output" / "generated.mp4").stat().st_size
            if (project / "output" / "generated.mp4").is_file()
            else 0
        ),
        "elapsed_s": round(elapsed_s, 3),
        "picsart_transport": "mock_only",
        "mock_picsart_submit_count": len(mock_picsart.requests),
        "mock_picsart_requests": mock_picsart.requests,
        "trace": trace_summary(project),
    }
    (run_root / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(run_root / "smoke_summary.json", 0o600)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    store.close()
    return 0 if state.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(run())
