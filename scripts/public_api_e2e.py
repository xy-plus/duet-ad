"""Public v1 API E2E: three real projects against the 3211/3212 release.

Cases (original-length source videos):
  - dog:      梳狗毛 16.95s, 换梳子, 中文, auto_rewrite
  - necklace: 项链 36.77s, 换项链, 保持原台词 (target_language omitted)
  - mouthwash: 漱口水 34.7s, 换漱口水(Listerine), English, auto_rewrite

Uses one plaintext test key from the duet-ad1 test-accounts CSV. Idempotent
submission keys are unique per invocation.

Usage:
    /home/xy/duet-ad1/.venv/bin/python scripts/public_api_e2e.py submit
    /home/xy/duet-ad1/.venv/bin/python scripts/public_api_e2e.py poll
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUET_DATA = Path("/home/xy/duet-ad1/data")
KEYS_CSV = Path(
    "/home/xy/.config/duet-ad1/public-api-test-accounts/test-001-to-test-020-api-keys.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data/smoke-runs/public-api-e2e"
STATE_FILE = OUTPUT_DIR / "jobs.json"
BASE = "https://8.166.140.227:3211/api/v1"
POLL_INTERVAL_S = 30

CASES = {
    "dog": {
        "source": DUET_DATA / "3b4c6ff51d384e4aa82170dc76e56748/source.mp4",
        "replacement": PROJECT_ROOT / "data/e2e-inputs/comb-clean.webp",
        "replacement_instruction": "将视频中给狗梳毛的开结梳替换为图片中的木质气垫梳",
        "target_language": "中文",
        "dialogue_mode": "auto_rewrite",
    },
    "necklace": {
        "source": PROJECT_ROOT
        / "data/projects/9ee796d756a64ba89acaab1395120b86/inputs/source.mp4",
        "replacement": PROJECT_ROOT
        / "data/projects/9ee796d756a64ba89acaab1395120b86/inputs/replacement-image",
        "replacement_instruction": "将视频中的项链换为图片中的项链",
        "target_language": None,  # 保持原台词
        "dialogue_mode": None,
    },
    "mouthwash": {
        "source": PROJECT_ROOT
        / "data/projects/8e124288db2d43beb2c555f6762ce50f/inputs/source.mp4",
        "replacement": PROJECT_ROOT
        / "data/projects/8e124288db2d43beb2c555f6762ce50f/inputs/replacement-image",
        "replacement_instruction": (
            "请将视频中人物手里所持的东西替换为图片中的漱口水，"
            "人物所持东西为牙刷、杯子、药片等\n\n"
            "漱口水商品介绍：Listerine Cool Mint 无酒精漱口水 1L 装，"
            "薄荷精油配方，30 秒漱口可杀灭 99.9% 导致口臭的细菌，"
            "温和不刺激，保持口气持久清新。"
        ),
        "target_language": "English",
        "dialogue_mode": "auto_rewrite",
    },
}


def _api_key() -> str:
    with KEYS_CSV.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["username"] == "test-001" and row["status"] == "active":
                return row["api_key"]
    raise SystemExit("test-001 key not found or inactive")


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(300.0),
        verify=False,
        headers={"Authorization": f"Bearer {_api_key()}"},
    )


def _submit(client: httpx.Client, name: str, case: dict) -> dict:
    fields: dict[str, str] = {
        "aspect_ratio": "9:16",
        "resolution": "768p",
        "replacement_instruction": case["replacement_instruction"],
    }
    if case["target_language"]:
        fields["target_language"] = case["target_language"]
    if case["dialogue_mode"]:
        fields["dialogue_mode"] = case["dialogue_mode"]
    with case["source"].open("rb") as source, case["replacement"].open("rb") as image:
        response = client.post(
            f"{BASE}/video-generations",
            data=fields,
            files={
                "source_video": ("source.mp4", source, "video/mp4"),
                "replacement_image": ("replacement.jpg", image, "image/jpeg"),
            },
            headers={"Idempotency-Key": f"pub-e2e-{name}-{int(time.time())}"},
        )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"{name}: submit {response.status_code} {response.text[:400]}")
    return response.json()


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with _client() as client:
        if command == "submit":
            jobs = {}
            for name, case in CASES.items():
                job = _submit(client, name, case)
                jobs[name] = {"id": job["id"], "status": job["status"]}
                print(f"{name}: {job['status']} id={job['id']}", flush=True)
            STATE_FILE.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
            return 0
        if command == "poll":
            jobs = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            pending = dict(jobs)
            while pending:
                for name, record in list(pending.items()):
                    job = client.get(f"{BASE}/video-generations/{record['id']}").json()
                    status = job["status"]
                    print(f"{name}: {status} {job.get('progress')}", flush=True)
                    if status == "succeeded":
                        video = job["result"]["video"]
                        content = client.get(f"{BASE}/video-generations/{record['id']}/content")
                        content.raise_for_status()
                        digest = hashlib.sha256(content.content).hexdigest()
                        assert digest == video["sha256"], f"{name}: sha256 mismatch"
                        destination = OUTPUT_DIR / f"{name}.mp4"
                        destination.write_bytes(content.content)
                        record["status"] = "succeeded"
                        record["file"] = str(destination)
                        print(f"{name}: downloaded {destination} sha256 ok", flush=True)
                        del pending[name]
                    elif status in {"failed", "submission_unknown"}:
                        record["status"] = status
                        record["error"] = job.get("error")
                        print(f"{name}: TERMINAL {status} {job.get('error')}", flush=True)
                        del pending[name]
                if pending:
                    time.sleep(POLL_INTERVAL_S)
            STATE_FILE.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
            failed = [n for n, r in jobs.items() if r.get("status") != "succeeded"]
            print("FAILED:", failed if failed else "none", flush=True)
            return 1 if failed else 0
    print("usage: public_api_e2e.py submit|poll", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
