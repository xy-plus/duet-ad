"""Offline corpus freezer contract tests."""

from __future__ import annotations

import binascii
import hashlib
import json
import os
import shutil
import struct
import zlib
from pathlib import Path

import pytest

from scripts import freeze_skill_eval_corpus as freezer


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> bytes:
    data = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _png(width: int, height: int, red: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(
        b"\x00" + bytes((red, 20, 30)) * width for _row in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _jpeg(width: int, height: int) -> bytes:
    components = b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    sof = b"\x08" + struct.pack(">HH", height, width) + components
    return b"\xff\xd8\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof + b"\xff\xd9"


def _text_record(text: str) -> dict[str, str]:
    return {"text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def _make_project(base: Path, project_id: str = "a" * 32) -> dict[str, Path]:
    root = base / project_id
    root.mkdir(parents=True)
    reference = root / "inputs" / "replacement_image.jpg"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(_jpeg(7, 5))
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()

    topology = [
        {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
        {"index": 2, "chain_id": "chain-002", "join_mode": "hard_cut"},
    ]
    _write_json(root / "long_video_plan.json", {"version": 1, "segments": topology})
    _write_json(
        root / "work" / "generation-config.json",
        {
            "version": 1,
            "generation_config": {
                "optimize_image": True,
                "remove_subtitle": True,
                "remove_watermark": False,
            },
        },
    )
    _write_json(
        root / "work" / "element_index.json",
        {"people": {}, "entities": {}, "scenes": {}, "relations": {}},
    )

    fusion_segments = []
    source_frames: list[Path] = []
    proxy_frames: list[Path] = []
    for segment_index in (1, 2):
        source_bytes = _png(2 + segment_index, 3, 30 * segment_index)
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        source_frame = (
            root / "work" / "segments" / str(segment_index)
            / "work" / "keyframes" / "01.png"
        )
        source_frame.parent.mkdir(parents=True)
        source_frame.write_bytes(source_bytes)
        source_frames.append(source_frame)
        _write_json(
            source_frame.parent.parent / "keyframe_sampling.json",
            {
                "schema": "duet.backend-keyframe-sampling",
                "version": 2,
                "selection_method": "test",
                "keyframes": [{
                    "order": 1,
                    "path": "keyframes/01.png",
                    "sha256": source_sha,
                    "source_time_s": 0.0,
                    "source_scene_id": f"SCENE_{segment_index:02d}",
                    "transition": {"type": "start", "at_s": 0.0},
                }],
            },
        )

        proxy_bytes = _png(4, 6, 50 * segment_index)
        proxy_sha = hashlib.sha256(proxy_bytes).hexdigest()
        proxy_relative = f"work/prompt-fusion-proxies/{proxy_sha}.png"
        proxy = root / proxy_relative
        proxy.parent.mkdir(parents=True, exist_ok=True)
        proxy.write_bytes(proxy_bytes)
        proxy_frames.append(proxy)
        fusion_segments.append({
            "index": segment_index,
            "new_keyframes": [{
                "order": 1,
                "path": proxy_relative,
                "sha256": proxy_sha,
                "segment_time_s": 0.0,
                "source_scene_id": f"SCENE_{segment_index:02d}",
                "transition": {"type": "start", "at_segment_s": 0.0},
            }],
            "old_video_prompt": _text_record(f"old-{segment_index}"),
            "image_optimization_prompt": [
                {"order": 1, **_text_record(f"replacement-{segment_index}")},
            ],
            "relation_occurrences": [],
            "audio_content": {
                "lines_json": "[]",
                "lines_sha256": hashlib.sha256(b"[]").hexdigest(),
                "voice_references": [],
                "music_policy": "forbid",
            },
        })

    multimodal_data = _write_json(
        root / "work" / "multimodal_input.json",
        {
            "schema": "duet.video-prompt-fusion-input",
            "version": 2,
            "segments": fusion_segments,
        },
    )
    multimodal_sha = hashlib.sha256(multimodal_data).hexdigest()
    _write_json(
        root / "work" / "h3_prompt_plan.json",
        {
            "schema": "duet.video-prompt-fusion-output",
            "version": 2,
            "input_sha256": multimodal_sha,
            "segments": [
                {"index": 1, "visual": ["first"]},
                {"index": 2, "visual": ["second"]},
            ],
        },
    )
    _write_json(
        root / "meta.json",
        {
            "id": project_id,
            "status": "done",
            "_minimal_replacement_image_path": "inputs/replacement_image.jpg",
            "effective_request": {
                "replacement_guidance": {
                    "image_field": "replacement_image",
                    "instruction": "replace the object",
                },
            },
            "input_receipt": {
                "replacement_image": {
                    "sha256": reference_sha,
                    "bytes": reference.stat().st_size,
                },
            },
        },
    )
    return {
        "root": root,
        "reference": reference,
        "source_frame": source_frames[0],
        "proxy_frame": proxy_frames[0],
    }


def test_freeze_corpus_copies_minimal_closure_and_real_metadata(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "projects")
    output = tmp_path / "corpus" / "v1"

    report = freezer.freeze_corpus(
        [freezer.CaseSource(split="train", project_root=source["root"])],
        output,
    )

    manifest_data = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    assert report.manifest_sha256 == hashlib.sha256(manifest_data).hexdigest()
    assert (output / "manifest.sha256").read_text(encoding="ascii") == (
        f"{report.manifest_sha256}  manifest.json\n"
    )
    assert manifest["schema"] == freezer.CORPUS_SCHEMA
    assert manifest["stats"]["case_count"] == 1
    case = manifest["cases"][0]
    assert case["source_project_id"] == source["root"].name
    assert case["split"] == "train"
    assert case["image_postprocess"]["topology"] == [
        {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
        {"index": 2, "chain_id": "chain-002", "join_mode": "hard_cut"},
    ]
    assert case["request"]["generation_config"] == {
        "remove_subtitle": True,
        "remove_watermark": False,
    }
    assert str(source["root"]) not in manifest_data.decode("utf-8")
    assert "/home/" not in manifest_data.decode("utf-8")

    reference_sha = hashlib.sha256(source["reference"].read_bytes()).hexdigest()
    reference_metadata = manifest["blobs"][reference_sha]
    assert reference_metadata == {
        "bytes": source["reference"].stat().st_size,
        "media_type": "image/jpeg",
        "width": 7,
        "height": 5,
    }
    source_sha = hashlib.sha256(source["source_frame"].read_bytes()).hexdigest()
    assert manifest["blobs"][source_sha]["media_type"] == "image/png"
    assert manifest["blobs"][source_sha]["width"] == 3
    assert manifest["blobs"][source_sha]["height"] == 3

    for digest, metadata in manifest["blobs"].items():
        blob = output / "blobs" / "sha256" / digest
        assert blob.is_file()
        assert hashlib.sha256(blob.read_bytes()).hexdigest() == digest
        assert blob.stat().st_size == metadata["bytes"]
    frozen_reference = output / "blobs" / "sha256" / reference_sha
    assert (frozen_reference.stat().st_dev, frozen_reference.stat().st_ino) != (
        source["reference"].stat().st_dev,
        source["reference"].stat().st_ino,
    )
    assert not list(output.parent.glob(".v1.staging-*"))


def test_publish_uses_one_same_parent_rename(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _make_project(tmp_path / "projects")
    output = tmp_path / "corpus" / "v1"
    calls: list[tuple[Path, Path]] = []
    real_rename = os.rename

    def capture(source_path: os.PathLike[str], destination_path: os.PathLike[str]) -> None:
        calls.append((Path(source_path), Path(destination_path)))
        real_rename(source_path, destination_path)

    monkeypatch.setattr(freezer.os, "rename", capture)

    freezer.freeze_corpus(
        [freezer.CaseSource(split="regression", project_root=source["root"])],
        output,
    )

    assert len(calls) == 1
    assert calls[0][0].parent == output.parent
    assert calls[0][1] == output


def test_existing_output_is_rejected_without_touching_it(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "projects")
    output = tmp_path / "corpus" / "v1"
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        freezer.freeze_corpus(
            [freezer.CaseSource(split="train", project_root=source["root"])],
            output,
        )

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(output.parent.glob(".v1.staging-*"))


def test_relative_input_and_output_paths_are_rejected(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "projects")

    with pytest.raises(ValueError, match="project root must be absolute"):
        freezer.freeze_corpus(
            [freezer.CaseSource(split="train", project_root=Path(source["root"].name))],
            tmp_path / "absolute-output",
        )
    with pytest.raises(ValueError, match="output root must be absolute"):
        freezer.freeze_corpus(
            [freezer.CaseSource(split="train", project_root=source["root"])],
            Path("relative-output"),
        )


def test_symlink_input_fails_and_cleans_staging(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "projects")
    original = source["reference"].with_name("original.jpg")
    source["reference"].rename(original)
    source["reference"].symlink_to(original)
    output = tmp_path / "corpus" / "v1"

    with pytest.raises(ValueError, match="symlink"):
        freezer.freeze_corpus(
            [freezer.CaseSource(split="train", project_root=source["root"])],
            output,
        )

    assert not output.exists()
    assert not list(output.parent.glob(".v1.staging-*"))


def test_declared_frame_hash_mismatch_fails_before_publish(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "projects")
    sampling_path = (
        source["root"] / "work" / "segments" / "1" / "work" / "keyframe_sampling.json"
    )
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    sampling["keyframes"][0]["sha256"] = "0" * 64
    _write_json(sampling_path, sampling)
    output = tmp_path / "corpus" / "v1"

    with pytest.raises(ValueError, match="sampled frame SHA"):
        freezer.freeze_corpus(
            [freezer.CaseSource(split="train", project_root=source["root"])],
            output,
        )

    assert not output.exists()
    assert not list(output.parent.glob(".v1.staging-*"))


def test_materialize_is_independent_from_deleted_live_project(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "projects")
    source_id = source["root"].name
    expected_reference = source["reference"].read_bytes()
    expected_source_frame = source["source_frame"].read_bytes()
    corpus = tmp_path / "corpus" / "v1"
    frozen = freezer.freeze_corpus(
        [freezer.CaseSource(split="train", project_root=source["root"])],
        corpus,
    )
    shutil.rmtree(source["root"])
    output = tmp_path / "materialized" / source_id

    report = freezer.materialize_case(
        corpus / "manifest.json",
        source_id,
        output,
    )

    assert report.manifest_sha256 == frozen.manifest_sha256
    assert report.source_project_id == source_id
    assert (output / "inputs" / "replacement_image.jpg").read_bytes() == expected_reference
    assert (
        output / "work" / "segments" / "1" / "work" / "keyframes" / "01.png"
    ).read_bytes() == expected_source_frame
    assert (output / "work" / "multimodal_input.json").is_file()
    assert (output / "work" / "h3_prompt_plan.json").is_file()
    case = json.loads((output / "case.json").read_text(encoding="utf-8"))
    assert case == {
        "schema": "duet.skill-eval-case",
        "version": 1,
        "corpus_manifest_sha256": frozen.manifest_sha256,
        "source_project_id": source_id,
        "split": "train",
        "request": {
            "generation_config": {
                "remove_subtitle": True,
                "remove_watermark": False,
            },
            "user_replacement_prompt": "replace the object",
            "user_replacement_prompt_sha256": hashlib.sha256(
                b"replace the object"
            ).hexdigest(),
            "user_reference_image": {
                "path": "inputs/replacement_image.jpg",
                "sha256": hashlib.sha256(expected_reference).hexdigest(),
            },
        },
        "topology": [
            {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
            {"index": 2, "chain_id": "chain-002", "join_mode": "hard_cut"},
        ],
        "inputs": {
            "long_video_plan": "long_video_plan.json",
            "element_index": "work/element_index.json",
            "fusion_input": "work/multimodal_input.json",
            "baseline_h3_prompt_plan": "work/h3_prompt_plan.json",
        },
    }
    for path in output.rglob("*"):
        assert not path.is_symlink()
        if path.is_file():
            assert path.stat().st_nlink == 1
    reference_sha = hashlib.sha256(expected_reference).hexdigest()
    assert (
        output / "inputs" / "replacement_image.jpg"
    ).stat().st_ino != (
        corpus / "blobs" / "sha256" / reference_sha
    ).stat().st_ino
    assert not list(output.parent.glob(f".{source_id}.staging-*"))


def test_materialize_uses_one_same_parent_rename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = _make_project(tmp_path / "projects")
    corpus = tmp_path / "corpus" / "v1"
    freezer.freeze_corpus(
        [freezer.CaseSource(split="train", project_root=source["root"])],
        corpus,
    )
    output = tmp_path / "materialized" / source["root"].name
    calls: list[tuple[Path, Path]] = []
    real_rename = os.rename

    def capture(source_path: os.PathLike[str], destination_path: os.PathLike[str]) -> None:
        calls.append((Path(source_path), Path(destination_path)))
        real_rename(source_path, destination_path)

    monkeypatch.setattr(freezer.os, "rename", capture)

    freezer.materialize_case(
        corpus / "manifest.json",
        source["root"].name,
        output,
    )

    assert len(calls) == 1
    assert calls[0][0].parent == output.parent
    assert calls[0][1] == output


def test_materialize_rejects_manifest_digest_mismatch(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "projects")
    corpus = tmp_path / "corpus" / "v1"
    freezer.freeze_corpus(
        [freezer.CaseSource(split="train", project_root=source["root"])],
        corpus,
    )
    manifest_path = corpus / "manifest.json"
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    output = tmp_path / "materialized" / source["root"].name

    with pytest.raises(ValueError, match="manifest SHA-256"):
        freezer.materialize_case(manifest_path, source["root"].name, output)

    assert not output.exists()
    assert not list(output.parent.glob(f".{source['root'].name}.staging-*"))


def test_materialize_verifies_blob_hash_before_publish(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "projects")
    corpus = tmp_path / "corpus" / "v1"
    freezer.freeze_corpus(
        [freezer.CaseSource(split="train", project_root=source["root"])],
        corpus,
    )
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    digest = next(iter(manifest["blobs"]))
    blob = corpus / "blobs" / "sha256" / digest
    blob.chmod(0o600)
    blob.write_bytes(blob.read_bytes() + b"corrupt")
    output = tmp_path / "materialized" / source["root"].name

    with pytest.raises(ValueError, match="SHA-256 or byte count"):
        freezer.materialize_case(
            corpus / "manifest.json", source["root"].name, output,
        )

    assert not output.exists()


def test_materialize_rejects_traversal_even_with_matching_manifest_digest(
    tmp_path: Path,
) -> None:
    source = _make_project(tmp_path / "projects")
    corpus = tmp_path / "corpus" / "v1"
    freezer.freeze_corpus(
        [freezer.CaseSource(split="train", project_root=source["root"])],
        corpus,
    )
    manifest_path = corpus / "manifest.json"
    digest_path = corpus / "manifest.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["video_prompt_fusion"]["new_keyframes"][0][
        "logical_path"
    ] = "../../escape.png"
    manifest_data = _canonical_json(manifest)
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    manifest_path.chmod(0o600)
    digest_path.chmod(0o600)
    manifest_path.write_bytes(manifest_data)
    digest_path.write_text(f"{manifest_sha}  manifest.json\n", encoding="ascii")
    output = tmp_path / "materialized" / source["root"].name

    with pytest.raises(ValueError, match="Fusion keyframe path"):
        freezer.materialize_case(manifest_path, source["root"].name, output)

    assert not output.exists()
    assert not (tmp_path / "escape.png").exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_materialize_rejects_linked_corpus_blob(
    link_kind: str, tmp_path: Path,
) -> None:
    source = _make_project(tmp_path / "projects")
    corpus = tmp_path / "corpus" / "v1"
    freezer.freeze_corpus(
        [freezer.CaseSource(split="train", project_root=source["root"])],
        corpus,
    )
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    digest = next(iter(manifest["blobs"]))
    blob = corpus / "blobs" / "sha256" / digest
    outside = tmp_path / "outside-blob"
    if link_kind == "symlink":
        outside.write_bytes(blob.read_bytes())
        blob.unlink()
        blob.symlink_to(outside)
    else:
        os.link(blob, outside)
    output = tmp_path / "materialized" / source["root"].name

    with pytest.raises(ValueError, match="single-link regular files"):
        freezer.materialize_case(
            corpus / "manifest.json", source["root"].name, output,
        )

    assert not output.exists()


def test_materialize_cli_requires_absolute_paths(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as manifest_error:
        freezer.main([
            "materialize",
            "--manifest",
            "relative/manifest.json",
            "--source-project-id",
            "a" * 32,
            "--output-root",
            str(tmp_path / "absolute-output"),
        ])
    assert manifest_error.value.code == 2

    with pytest.raises(SystemExit) as output_error:
        freezer.main([
            "materialize",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--source-project-id",
            "a" * 32,
            "--output-root",
            "relative-output",
        ])
    assert output_error.value.code == 2
