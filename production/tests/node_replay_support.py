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


def verify_historical_fixture(fixture: Fixture) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["entries"]
               if entry["subset"] == fixture.subset and entry["project"] == fixture.project
               and entry["node"] == fixture.node and entry["scope"] == fixture.scope]
    assert len(entries) == 1
    entry = entries[0]
    assert sha256_file(fixture.directory / "input.json") == entry["input_sha256"]
    if entry.get("output_sha256"):
        assert sha256_file(fixture.directory / "output.json") == entry["output_sha256"]
    expected_media = {item["name"]: item for item in entry.get("media", [])}
    actual_media = {path.name: path for path in fixture.media}
    assert set(actual_media) == set(expected_media)
    for name, path in actual_media.items():
        assert path.stat().st_size == expected_media[name]["size_bytes"]
        assert sha256_file(path) == expected_media[name]["sha256"]


def current_skill_replay(fixture: Fixture, skill_path: Path, directory: Path) -> str:
    verify_historical_fixture(fixture)
    skill = skill_path.read_text(encoding="utf-8")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "current-skill.md").write_text(skill, encoding="utf-8")
    (directory / "skill-provenance.json").write_text(json.dumps({
        "source_fixture": str(fixture.directory),
        "historical_input_sha256": sha256_file(fixture.directory / "input.json"),
        "current_skill_source": str(skill_path),
        "current_skill_sha256": hashlib.sha256(skill.encode()).hexdigest(),
    }, indent=2), encoding="utf-8")
    return skill


def current_sheet_replay(fixture: Fixture, directory: Path) -> dict[str, Any]:
    """Verify the frozen receipt, then build this version's prompt from its real spec.

    Historical artifacts remain immutable. The replay is a new invocation of
    the current node, not a resume of the historical paid request.
    """
    from app.replacement_sheet import materialize_replacement_sheet_inputs

    inputs = sibling_fixture(fixture, "replacement-sheet-inputs")
    seedream = sibling_fixture(fixture, "seedream-reference-submit")
    assert inputs is not None and seedream is not None
    verify_historical_fixture(inputs)
    verify_historical_fixture(seedream)
    historical_prompt = (seedream.directory / "extras" / "prompt.txt").read_bytes()
    old_sha = hashlib.sha256(historical_prompt).hexdigest()
    assert old_sha == seedream.input["prompt_sha256"]
    assert hashlib.sha256(inputs.expected_output["prompt"].encode()).hexdigest() == old_sha
    receipts = seedream.input["input_receipts"]
    assert len(seedream.media) == len(receipts)
    for path, receipt in zip(seedream.media, receipts):
        assert sha256_file(path) == receipt["sha256"]
    spec = inputs.input["spec"]
    result = materialize_replacement_sheet_inputs(
        directory, spec, user_image_present=len(receipts) > 1,
    )
    assert result["spec_sha256"] == spec["spec_sha256"]
    canvas = Path(result["canvas_path"])
    assert canvas.is_file()
    assert canvas.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result["prompt"].strip()
    # The historical receipt remains verified above. Its canvas is immutable;
    # this replay invokes the current local renderer, so renderer/version
    # drift must not require byte-identical pixels from that old receipt.
    # Bind every current tile to the same immutable source spec; no entity or
    # appearance may disappear while the node's global instruction evolves.
    for tile in spec["tiles"]:
        binding = (
            f'stable_key={json.dumps(tile["stable_key"], ensure_ascii=False)} | '
            f'SOURCE={json.dumps(tile["source"], ensure_ascii=False)} | '
            f'APPEARANCE={json.dumps(tile["appearance_description"], ensure_ascii=False)}'
        )
        assert binding in result["prompt"]
    (directory / "replay-provenance.json").write_text(json.dumps({
        "historical_prompt_sha256": old_sha,
        "current_prompt_sha256": hashlib.sha256(result["prompt"].encode()).hexdigest(),
        "historical_inputs_sha256": sha256_file(inputs.directory / "input.json"),
        "source_spec_sha256": spec["spec_sha256"],
        "source_fixture": str(inputs.directory),
    }, indent=2), encoding="utf-8")
    return result
