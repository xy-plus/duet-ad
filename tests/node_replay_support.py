"""Discovery and access helpers for harvested real-input node fixtures.

Fixtures live under ``tests/fixtures/node-replays/`` (gitignored) and are
produced by ``scripts/harvest_node_fixtures.py`` from historical project
logs.  Every fixture is a real ``input.json`` + ``input-media/`` a node
actually consumed, plus the historical ``output.json`` as reference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "node-replays"
MANIFEST_PATH = FIXTURES_ROOT / "manifest.json"

# Every node the suite must cover (H3 submit/poll/download/trim excluded).
REQUIRED_NODES = [
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

BOUNDARY_PROJECT = "dfe6479538184759ba6cc953a0bda138"  # TTS 缺音色失败单


@dataclass(frozen=True)
class Fixture:
    project: str
    node: str
    scope: str
    directory: Path
    subset: str = "replays"

    @property
    def input(self) -> dict[str, Any]:
        return json.loads(
            (self.directory / "input.json").read_text(encoding="utf-8")
        )

    @property
    def media(self) -> list[Path]:
        directory = self.directory / "input-media"
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.iterdir() if path.is_file())

    @property
    def expected_output(self) -> Any:
        """The historical output, unwrapping the ProviderResult envelope."""
        path = self.directory / "output.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value

    @property
    def test_id(self) -> str:
        return f"{self.project[:8]}/{self.scope}"


def discover(subset: str = "replays") -> list[Fixture]:
    root = FIXTURES_ROOT if subset == "replays" else FIXTURES_ROOT / "boundary"
    fixtures: list[Fixture] = []
    if not root.is_dir():
        return fixtures
    for project_dir in sorted(root.iterdir()):
        if not project_dir.is_dir() or project_dir.name in {"_blobs", "boundary"}:
            continue
        for node_dir in sorted(project_dir.iterdir()):
            if not node_dir.is_dir():
                continue
            for scope_dir in sorted(node_dir.iterdir()):
                if (scope_dir / "input.json").is_file():
                    fixtures.append(Fixture(
                        project=project_dir.name,
                        node=node_dir.name,
                        scope=scope_dir.name,
                        directory=scope_dir,
                        subset=subset,
                    ))
    return fixtures


def node_fixtures(node: str, subset: str = "replays") -> list[Fixture]:
    return [
        fixture for fixture in discover(subset) if fixture.node == node
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_by_suffix(fixture: Fixture, suffix: str) -> Path:
    """The input-media file whose historical name ends with ``suffix``."""
    for path in fixture.media:
        # copy_media prefixes "NNN-" to the original basename
        if path.name.split("-", 1)[-1] == suffix or path.name.endswith(suffix):
            return path
    raise AssertionError(
        f"fixture {fixture.node}/{fixture.scope} has no media *{suffix}"
    )


def sibling_fixture(
    fixture: Fixture, node: str, scope: str = "project",
) -> Fixture | None:
    directory = FIXTURES_ROOT / fixture.project / node / scope
    if not (directory / "input.json").is_file():
        return None
    return Fixture(
        project=fixture.project, node=node, scope=scope, directory=directory,
    )
