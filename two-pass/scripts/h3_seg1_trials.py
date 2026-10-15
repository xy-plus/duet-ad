"""Independent experiment: resubmit project 8e124288 segment-1 H3 request 3x.

Reuses the exact params recorded in the project's picsart-submit log
(variable control: identical prompt, reference URLs, duration, resolution,
aspect ratio). Does not touch the pipeline or any project state.

Usage:
    PICSART_API_KEY=... python scripts/h3_seg1_trials.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

API_BASE = "https://api.picsart.com"
WORKFLOW = "minimax/h3-max/reference-to-video"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDED_INPUT = (
    PROJECT_ROOT
    / "data/projects/8e124288db2d43beb2c555f6762ce50f"
    / "logs/picsart-submit/segment-0001/attempt-1/input.json"
)
OUTPUT_DIR = PROJECT_ROOT / "data/smoke-runs/h3-seg1-trials"
TRIALS = 3
POLL_INTERVAL_S = 10
POLL_TIMEOUT_S = 1800


def submit(client: httpx.Client, params: dict) -> str:
    response = client.post(
        f"{API_BASE}/workflows/{WORKFLOW}/submit",
        json={"params": params},
    )
    response.raise_for_status()
    payload = response.json()
    task_id = payload["response"]["id"]
    print(f"submitted task_id={task_id}", flush=True)
    return task_id


def poll(client: httpx.Client, task_id: str) -> str:
    url = f"{API_BASE}/workflows/{WORKFLOW}/{task_id}/result"
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        response = client.get(url)
        response.raise_for_status()
        envelope = response.json()["response"]
        status = str(envelope.get("status") or "").upper()
        if status == "COMPLETED":
            video_url = envelope["result"]["video"]["url"]
            print(f"task {task_id} COMPLETED {video_url}", flush=True)
            return video_url
        if status in {"FAILED", "ERROR", "CANCELED", "CANCELLED"}:
            raise RuntimeError(f"task {task_id} ended with status {status}: {envelope}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"task {task_id} poll timeout")
        time.sleep(POLL_INTERVAL_S)


def download(client: httpx.Client, url: str, destination: Path) -> None:
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    print(f"downloaded {destination} ({destination.stat().st_size} bytes)", flush=True)


def main() -> int:
    api_key = os.environ.get("PICSART_API_KEY", "").strip()
    if not api_key:
        print("PICSART_API_KEY is required", file=sys.stderr)
        return 2
    params = json.loads(RECORDED_INPUT.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with httpx.Client(
        timeout=httpx.Timeout(120.0),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    ) as client:
        # --poll-only id1,id2,... : skip submission, poll existing tasks
        poll_only: list[str] = []
        if len(sys.argv) > 1 and sys.argv[1] == "--poll-only":
            poll_only = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
        task_ids = poll_only or [submit(client, params) for _ in range(TRIALS)]
        results: dict[str, str] = {}
        failures: dict[str, str] = {}
        for slot, task_id in enumerate(task_ids, 1):
            try:
                video_url = poll(client, task_id)
                download(client, video_url, OUTPUT_DIR / f"trial-{slot}.mp4")
                results[task_id] = video_url
            except Exception as exc:  # keep polling the remaining trials
                print(f"task {task_id} FAILED: {exc}", flush=True)
                failures[task_id] = str(exc)
        summary = {"task_ids": task_ids, "results": results, "failures": failures}
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
