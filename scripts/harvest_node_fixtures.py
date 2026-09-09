#!/usr/bin/env python3
"""Harvest real node inputs from historical projects into replay fixtures.

Scans ``data/projects/<id>/logs/<node>/<scope>/attempt-N/`` and copies the
latest successful attempt's ``input.json`` + ``input-media/`` (plus
``output.json`` as a reference) into
``tests/fixtures/node-replays/<project>/<node>/<scope>/``.  The latest failed
attempt per scope goes to the ``boundary/`` subset instead.  A run writes
``manifest.json`` recording, per fixture, the source project, attempt and the
sha256 of the historical input/output.

Fixtures are gitignored; rerun this script before the replay suite whenever
the fixture directory is missing or stale::

    /home/xy/duet-ad1/.venv/bin/python scripts/harvest_node_fixtures.py
    /home/xy/duet-ad1/.venv/bin/python scripts/harvest_node_fixtures.py --include-round2

Media files are deduplicated through content-addressed blobs under
``_blobs/`` (hardlinked into place).  Local model files (whisper.cpp weights,
YAMNet/YuNet models) are not copied; the replay tests resolve them from their
canonical host paths, and the manifest records them under ``skipped_media``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "data" / "projects"
FIXTURES_ROOT = ROOT / "tests" / "fixtures" / "node-replays"

# Complete historical projects plus the TTS-no-voice failure case.
DEFAULT_PROJECTS = [
    "2f47194ed7134701952813dc6234f1f9",  # 项链，保持台词，5 段
    "0934ab0330e94733af6b656665b9efbd",  # 漱口水，英文改编，5 段
    "5658196796f2448eb37a35b8577aebfb",  # generate 项链，台词生成模式
    "267c1f1fb23a4c889e551619373def45",  # 狗，中文改编，2 段
    "dfe6479538184759ba6cc953a0bda138",  # 狗，失败单：TTS 缺音色
]
ROUND2_PROJECT_PREFIXES = ["410a2403", "7f85e304", "be663f3c"]

# Every node the replay suite covers (H3 submit/poll/download/trim excluded).
NODES = [
    "probe",
    "scene-detection",
    "segment-planning",
    "segment-cut",
    "source-frames-4fps",
    "segment-frames-4fps",
    "keyframes-exact-nine",
    "keyframe-selection-exact-nine",
    "extract-audio",
    "whisper-asr",
    "yamnet",
    "spoken-dialogue-filter",
    "mask-reference-faces",
    "mute-reference-video",
    "assemble-dialogue-track",
    "replace-audio-track",
    "stitch",
    "replacement-sheet-inputs",
    "replacement-sheet-user-reference",
    "dialogue-localize",
    "deepseek-translation",
    "deepseek-dialogue-generation",
    "video-maker-project-index",
    "video-maker",
    "object-replace",
    "h3-max-r2v-prompt",
    "seedream-reference-submit",
    "dialogue-tts-line",
]

# Model weights resolve from canonical host paths at replay time.
_SKIP_MEDIA_BASENAMES = re.compile(r"(^ggml-.*\.bin$|\.tflite$|\.onnx$)")
_SKIP_MEDIA_MAX_BYTES = 96 * 1024 * 1024

_ATTEMPT_RE = re.compile(r"^attempt-(\d+)$")

# State keys the boundary subset needs to replay pre-TTS voice resolution.
_BOUNDARY_STATE_KEYS = (
    "effective_request",
    "translated_voice_lines",
    "enhanced_element_index",
    "voice_references",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attempts(scope_dir: Path) -> list[tuple[int, Path]]:
    numbered = []
    for child in scope_dir.iterdir():
        match = _ATTEMPT_RE.match(child.name)
        if match and child.is_dir():
            numbered.append((int(match.group(1)), child))
    return sorted(numbered)


def _pick_attempts(
    scope_dir: Path,
) -> tuple[tuple[int, Path] | None, tuple[int, Path] | None]:
    """Latest successful attempt and latest failed attempt for one scope."""
    success: tuple[int, Path] | None = None
    failed: tuple[int, Path] | None = None
    for number, directory in _attempts(scope_dir):
        has_output = (directory / "output.json").is_file()
        has_error = (directory / "error.json").is_file()
        if has_error:
            failed = (number, directory)
        elif has_output:
            success = (number, directory)
    return success, failed


class _BlobStore:
    """Content-addressed media store; fixtures hardlink blobs into place."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def place(self, source: Path, destination: Path) -> str:
        digest = _sha256_file(source)
        blob = self.root / digest
        if not blob.exists():
            temporary = self.root / f".{digest}.tmp"
            shutil.copyfile(source, temporary)
            os.replace(temporary, blob)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        try:
            os.link(blob, destination)
        except OSError:
            shutil.copyfile(blob, destination)
        return digest


def _harvest_attempt(
    attempt_dir: Path,
    destination: Path,
    blobs: _BlobStore,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    input_path = attempt_dir / "input.json"
    if not input_path.is_file():
        raise FileNotFoundError(f"attempt has no input.json: {attempt_dir}")
    input_bytes = input_path.read_bytes()
    (destination / "input.json").write_bytes(input_bytes)
    record: dict[str, Any] = {
        "input_sha256": _sha256_bytes(input_bytes),
        "media": [],
        "skipped_media": [],
    }
    output_path = attempt_dir / "output.json"
    if output_path.is_file():
        output_bytes = output_path.read_bytes()
        (destination / "output.json").write_bytes(output_bytes)
        record["output_sha256"] = _sha256_bytes(output_bytes)
    error_path = attempt_dir / "error.json"
    if error_path.is_file():
        shutil.copyfile(error_path, destination / "error.json")
    media_dir = attempt_dir / "input-media"
    if media_dir.is_dir():
        for source in sorted(media_dir.iterdir()):
            if not source.is_file():
                continue
            size = source.stat().st_size
            if (
                _SKIP_MEDIA_BASENAMES.search(source.name)
                or size > _SKIP_MEDIA_MAX_BYTES
            ):
                record["skipped_media"].append({
                    "name": source.name,
                    "size_bytes": size,
                    "sha256": _sha256_file(source),
                    "reason": "model_or_oversized",
                })
                continue
            digest = blobs.place(source, destination / "input-media" / source.name)
            record["media"].append({
                "name": source.name,
                "size_bytes": size,
                "sha256": digest,
            })
    return record


def _harvest_seedream_extras(
    project_dir: Path, destination: Path, record: dict[str, Any],
) -> None:
    """The Seedream node logs only the prompt sha; the prompt text itself
    lives at work/global-reference-sheet/prompt.txt.  Bind them here."""
    prompt_path = project_dir / "work" / "global-reference-sheet" / "prompt.txt"
    if not prompt_path.is_file():
        return
    prompt_bytes = prompt_path.read_bytes()
    digest = _sha256_bytes(prompt_bytes)
    extras = destination / "extras"
    extras.mkdir(exist_ok=True)
    (extras / "prompt.txt").write_bytes(prompt_bytes)
    record["extras"] = {
        "prompt.txt": {
            "sha256": digest,
            "matches_logged_prompt_sha256": digest == record.get("prompt_sha256"),
        }
    }


def harvest_project(
    project_id: str, blobs: _BlobStore, manifest: dict[str, Any],
) -> None:
    project_dir = PROJECTS_ROOT / project_id
    logs = project_dir / "logs"
    if not logs.is_dir():
        print(f"skip {project_id}: no logs", file=sys.stderr)
        return
    for node in NODES:
        node_dir = logs / node
        if not node_dir.is_dir():
            continue
        for scope_dir in sorted(node_dir.iterdir()):
            if not scope_dir.is_dir():
                continue
            success, failed = _pick_attempts(scope_dir)
            if success is not None:
                attempt, attempt_dir = success
                destination = (
                    FIXTURES_ROOT / project_id / node / scope_dir.name
                )
                record = _harvest_attempt(attempt_dir, destination, blobs)
                if node == "seedream-reference-submit":
                    try:
                        logged = json.loads(
                            (destination / "input.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        record["prompt_sha256"] = logged.get("prompt_sha256")
                    except json.JSONDecodeError:
                        pass
                    _harvest_seedream_extras(project_dir, destination, record)
                manifest["entries"].append({
                    "subset": "replays",
                    "project": project_id,
                    "node": node,
                    "scope": scope_dir.name,
                    "attempt": attempt,
                    **record,
                })
            if failed is not None:
                attempt, attempt_dir = failed
                destination = (
                    FIXTURES_ROOT / "boundary" / project_id / node / scope_dir.name
                )
                record = _harvest_attempt(attempt_dir, destination, blobs)
                manifest["entries"].append({
                    "subset": "boundary",
                    "project": project_id,
                    "node": node,
                    "scope": scope_dir.name,
                    "attempt": attempt,
                    **record,
                })
    state_path = project_dir / "state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = None
        if isinstance(state, dict) and state.get("status") == "failed":
            snapshot = {
                key: state.get(key) for key in _BOUNDARY_STATE_KEYS
            }
            snapshot["error"] = state.get("error")
            destination = FIXTURES_ROOT / "boundary" / project_id
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "state-snapshot.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["entries"].append({
                "subset": "boundary",
                "project": project_id,
                "node": "state-snapshot",
                "scope": "project",
                "attempt": None,
                "input_sha256": _sha256_bytes(
                    (destination / "state-snapshot.json").read_bytes()
                ),
                "media": [],
                "skipped_media": [],
            })


def main() -> None:
    global PROJECTS_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects-root", type=Path, default=PROJECTS_ROOT,
        help="absolute historical projects directory (read-only source)",
    )
    parser.add_argument(
        "--include-round2",
        action="store_true",
        help="also harvest round2 in-flight projects (read-only)",
    )
    parser.add_argument(
        "--projects",
        nargs="*",
        default=None,
        help="explicit project ids (default: the five historical projects)",
    )
    args = parser.parse_args()
    if not args.projects_root.is_absolute() or not args.projects_root.is_dir():
        parser.error("--projects-root must be an existing absolute directory")
    PROJECTS_ROOT = args.projects_root.resolve()

    project_ids = list(args.projects or DEFAULT_PROJECTS)
    if args.include_round2:
        for child in sorted(PROJECTS_ROOT.iterdir()):
            if child.is_dir() and any(
                child.name.startswith(prefix)
                for prefix in ROUND2_PROJECT_PREFIXES
            ):
                project_ids.append(child.name)

    blobs = _BlobStore(FIXTURES_ROOT / "_blobs")
    manifest: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "projects_root": str(PROJECTS_ROOT),
        "nodes": NODES,
        "entries": [],
    }
    for project_id in project_ids:
        harvest_project(project_id, blobs, manifest)
    FIXTURES_ROOT.mkdir(parents=True, exist_ok=True)
    (FIXTURES_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    harvested = {
        (entry["project"], entry["node"])
        for entry in manifest["entries"]
        if entry["subset"] == "replays"
    }
    print(f"fixtures: {len(harvested)} node/project pairs")
    for node in NODES:
        projects = sorted(pid for pid, n in harvested if n == node)
        print(f"  {node}: {len(projects)} project(s)")


if __name__ == "__main__":
    main()
