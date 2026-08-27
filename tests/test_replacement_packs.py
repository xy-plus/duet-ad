"""Durable replacement-pack generation stays upstream of every frame POST."""

from __future__ import annotations

import asyncio
import binascii
import hashlib
import json
import struct
import sys
import types
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import replacement_packs as packs


def _png(width: int = 8, height: int = 6, rgb=(1, 2, 3)) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", binascii.crc32(kind + body) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    )


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "source").mkdir(parents=True)
    (root / "source" / "person.png").write_bytes(_png(rgb=(10, 20, 30)))
    (root / "source" / "scene.png").write_bytes(_png(rgb=(40, 50, 60)))
    return root.resolve()


def _plan(*, person_source: str = "source/person.png") -> packs.ProjectReplacementPlan:
    return packs.ProjectReplacementPlan(
        people=(packs.PersonPlan(
            "PERSON_01", (person_source,), "a wholly new adult presenter",
            {"age_group": "adult", "wardrobe": "linen jacket"},
        ),),
        scenes=(packs.ScenePlan(
            "SCENE_01", ("source/scene.png",), "a wholly new interview environment",
            {"lighting_style": "soft daylight"},
        ),),
        upstream_plan_sha256="a" * 64,
        upstream_source_inventory_sha256="b" * 64,
        execution_profile={"id": "dual-target-v2", "revision": 2},
    )


class FakeGenerator:
    def __init__(self, *, error: str | None = None) -> None:
        self.requests: list[packs.GenerationRequest] = []
        self.error = error

    async def generate(self, request: packs.GenerationRequest) -> packs.GenerationResult:
        receipt = json.loads((request.work_dir / "generation.json").read_text())
        assert receipt["status"] == "prepared"
        assert receipt["request"]["input_sha256"] == request.input_sha256
        self.requests.append(request)
        if self.error:
            raise packs.PackGenerationError(self.error)
        return packs.GenerationResult(
            images=(
                packs.GeneratedImage("primary", _png(9, 7, (1, 90, 2))),
                packs.GeneratedImage("alternate", _png(9, 7, (2, 90, 3))),
            ),
            producer_receipt={"adapter": "fake", "request": request.input_sha256},
        )


class FakeGate:
    def __init__(self, status: str = "pass", *, raises: bool = False) -> None:
        self.status = status
        self.raises = raises
        self.candidates: list[packs.ReplacementPackCandidateDTO] = []

    def evaluate(self, candidate, *, receipt_path):
        assert packs.load_replacement_pack_candidate(candidate.project_dir) == candidate
        assert receipt_path == candidate.project_dir / packs.QUALITY_RECEIPT_PATH
        self.candidates.append(candidate)
        if self.raises:
            raise RuntimeError("verifier unavailable")
        raw = {
            "schema": "duet.image-quality-receipt", "version": 1,
            "status": self.status, "publishable": self.status == "pass",
            "provider_retry_allowed": False,
            "plan_sha256": candidate.upstream_plan_sha256,
            "reference_pack_candidate_sha256": candidate.candidate_sha256,
            "failures": [] if self.status == "pass" else ["semantic_gate"],
        }
        raw["sha256"] = packs._hash(raw)
        return packs.PackQualityResult(self.status, self.status == "pass", raw)


def _prepare(root: Path, generator, gate, plan=None):
    return asyncio.run(packs.prepare_replacement_packs(
        root, plan or _plan(), model="seedream-test", revision=3,
        generator=generator, quality_gate=gate,
    ))


def test_builds_loads_and_reuses_exact_project_pack(tmp_path):
    root = _project(tmp_path)
    generator, gate = FakeGenerator(), FakeGate()
    first = _prepare(root, generator, gate)

    assert first.status == "ready" and first.pack is not None
    assert tuple(first.pack.people) == ("PERSON_01",)
    assert tuple(first.pack.scenes) == ("SCENE_01",)
    assert [image.role for image in first.pack.people["PERSON_01"].images] == [
        "primary", "alternate",
    ]
    assert first.pack.project_dir == root
    assert first.pack.upstream_plan_sha256 == "a" * 64
    assert first.pack.upstream_source_inventory_sha256 == "b" * 64
    assert first.pack.revision == 3
    for entity in (*first.pack.people.values(), *first.pack.scenes.values()):
        assert entity.producer_receipt_path.is_absolute()
        assert not Path(entity.producer_receipt_relative_path).is_absolute()
        assert all(source.path.is_absolute() for source in entity.sources)
        assert all(image.path.is_absolute() for image in entity.images)
        assert all(not Path(image.relative_path).is_absolute() for image in entity.images)
        assert not {source.path for source in entity.sources} & {
            image.path for image in entity.images
        }

    loaded = packs.load_replacement_pack(
        root,
        expected_upstream_plan_sha256="a" * 64,
        expected_upstream_source_inventory_sha256="b" * 64,
        expected_execution_profile_sha256=packs._hash({"id": "dual-target-v2", "revision": 2}),
        expected_model="seedream-test", expected_revision=3,
        expected_person_plan_ids=("PERSON_01",),
        expected_scene_plan_ids=("SCENE_01",),
    )
    assert loaded.candidate_sha256 == first.pack.candidate_sha256
    second = _prepare(root, generator, gate)
    assert second.status == "ready"
    assert len(generator.requests) == 2
    assert len(gate.candidates) == 1


def test_generation_request_is_fsynced_and_sources_are_only_negative_evidence(tmp_path):
    root = _project(tmp_path)
    generator = FakeGenerator()
    result = _prepare(root, generator, FakeGate())
    assert result.status == "ready"

    for request in generator.requests:
        assert request.roles == ("primary", "alternate")
        assert request.neutral_sha256 == hashlib.sha256(request.neutral_png).hexdigest()
        assert request.neutral_png not in [source.png_bytes for source in request.sources]
        assert "never use them as target identity or target scene references" in request.prompt
        frozen = json.loads((request.work_dir / "generation.json").read_text())["request"]
        assert frozen["model"] == "seedream-test"
        assert frozen["revision"] == 3
        assert frozen["upstream_plan_sha256"] == "a" * 64
        assert [item["role"] for item in frozen["role_prompts"]] == [
            "primary", "alternate",
        ]


@pytest.mark.parametrize(
    ("gate", "expected"),
    [(FakeGate("fail"), "failed"), (FakeGate("unknown"), "unknown"),
     (FakeGate(raises=True), "unknown")],
)
def test_quality_failure_or_unknown_never_publishes(tmp_path, gate, expected):
    root = _project(tmp_path)
    result = _prepare(root, FakeGenerator(), gate)
    assert result.status == expected and result.pack is None
    assert (root / packs.CANDIDATE_RECEIPT_PATH).is_file()
    assert not (root / packs.PACK_RECEIPT_PATH).exists()


def test_submission_unknown_stops_before_quality_gate(tmp_path):
    root = _project(tmp_path)
    gate = FakeGate()
    result = _prepare(root, FakeGenerator(error="submission_unknown"), gate)
    assert result.status == "submission_unknown" and result.pack is None
    assert not gate.candidates
    assert not (root / packs.CANDIDATE_RECEIPT_PATH).exists()


def test_changed_source_regenerates_only_affected_entity(tmp_path):
    root = _project(tmp_path)
    generator, gate = FakeGenerator(), FakeGate()
    assert _prepare(root, generator, gate).status == "ready"
    (root / "source" / "person2.png").write_bytes(_png(rgb=(70, 80, 90)))
    assert _prepare(root, generator, gate, _plan(person_source="source/person2.png")).status == "ready"
    assert [request.plan_id for request in generator.requests] == [
        "PERSON_01", "SCENE_01", "PERSON_01",
    ]
    assert len(gate.candidates) == 2


def test_loader_rejects_tamper_and_wrong_expected_ids(tmp_path):
    root = _project(tmp_path)
    pack = _prepare(root, FakeGenerator(), FakeGate()).pack
    assert pack is not None
    with pytest.raises(packs.ReplacementPackError, match="person plan ids"):
        packs.load_replacement_pack(root, expected_person_plan_ids=("PERSON_02",))
    pack.people["PERSON_01"].images[0].path.write_bytes(_png(rgb=(9, 9, 9)))
    with pytest.raises(packs.ReplacementPackError, match="binding mismatch"):
        packs.load_replacement_pack(root)


def test_traversal_symlink_and_reference_limit_fail_before_generation(tmp_path):
    root = _project(tmp_path)
    generator = FakeGenerator()
    escaped = _plan(person_source="../outside.png")
    with pytest.raises(packs.ReplacementPackError, match="project-relative"):
        _prepare(root, generator, FakeGate(), escaped)
    (root / "source" / "link.png").symlink_to(root / "source" / "person.png")
    with pytest.raises(packs.ReplacementPackError, match="symlink"):
        _prepare(root, generator, FakeGate(), _plan(person_source="source/link.png"))
    many = tuple(f"source/ref-{index}.png" for index in range(9))
    for path in many:
        (root / path).write_bytes(_png())
    plan = _plan()
    plan = packs.ProjectReplacementPlan(
        (packs.PersonPlan("PERSON_01", many, "new person", {}),), plan.scenes,
        plan.upstream_plan_sha256, plan.upstream_source_inventory_sha256,
        plan.execution_profile,
    )
    with pytest.raises(packs.ReplacementPackError, match="10-image limit"):
        _prepare(root, generator, FakeGate(), plan)
    assert not generator.requests


def test_canonical_source_inventory_formula_and_validation():
    frames = [
        {"segment_index": 0, "frame_index": 1, "frame_name": "0001.png",
         "source_sha256": "1" * 64, "ignored": True},
        {"segment_index": 0, "frame_index": 2, "frame_name": "0002.png",
         "source_sha256": "2" * 64},
    ]
    projected = [{key: frame[key] for key in (
        "segment_index", "frame_index", "frame_name", "source_sha256"
    )} for frame in frames]
    expected = hashlib.sha256(json.dumps(
        projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()
    assert packs.canonical_source_inventory_sha256(frames) == expected
    with pytest.raises(packs.ReplacementPackError):
        packs.canonical_source_inventory_sha256([frames[0], frames[0]])


def test_seedream_adapter_uses_neutral_then_primary_with_exact_v2_roles(tmp_path, monkeypatch):
    root = _project(tmp_path)
    calls = []

    async def fake_edit(settings, images, prompt, out, *, receipt_path, transport,
                        execution_binding):
        frozen = json.loads(receipt_path.with_name(out.stem + ".request.json").read_text())
        roles = execution_binding["input_roles"]
        assert [item["role"] for item in frozen["input_order"]] == roles
        assert images[0] not in {
            (root / "source" / "person.png").read_bytes(),
            (root / "source" / "scene.png").read_bytes(),
        }
        if out.stem == "alternate":
            assert roles[1].endswith(":primary")
            assert images[1] == (out.parent / "primary.png").read_bytes()
        output = _png(8, 6, (100 + len(calls), 2, 3))
        packs.seedream._atomic_bytes(out, output)
        raw = {
            "version": 2, "status": "succeeded",
            "plan_sha256": execution_binding["plan_sha256"],
            "profile": execution_binding["profile"],
            "revision": execution_binding["revision"],
            "input_order": [
                {"position": index, "role": role,
                 "sha256": hashlib.sha256(data).hexdigest()}
                for index, (role, data) in enumerate(zip(roles, images), 1)
            ],
        }
        packs.seedream._atomic_json(receipt_path, raw)
        calls.append((out.parent.name, out.stem, roles))
        return out

    monkeypatch.setattr(packs.seedream, "edit", fake_edit)
    settings = SimpleNamespace(seedream_model="seedream-v2", seedream_edit_mode="edit")
    result = asyncio.run(packs.prepare_replacement_packs_with_seedream(
        root, _plan(), settings=settings, revision=3, quality_gate=FakeGate(),
        transport=object(),
    ))
    assert result.status == "ready"
    assert [role for _, role, _ in calls] == [
        "primary", "alternate", "primary", "alternate",
    ]
    assert calls[0][2] == ["current_frame", "source_negative:person:PERSON_01:1"]
    assert calls[1][2] == [
        "current_frame", "target_reference:person:PERSON_01:primary",
        "source_negative:person:PERSON_01:1",
    ]


def test_seedream_submission_unknown_is_not_blindly_resubmitted(tmp_path, monkeypatch):
    root = _project(tmp_path)
    paid_posts = 0

    async def unknown_edit(settings, images, prompt, out, *, receipt_path, transport,
                           execution_binding):
        nonlocal paid_posts
        if receipt_path.exists():
            assert json.loads(receipt_path.read_text())["status"] == "submission_unknown"
            raise packs.seedream.SeedreamError("submission_unknown")
        paid_posts += 1
        packs.seedream._atomic_json(receipt_path, {
            "version": 2, "status": "submission_unknown",
            "plan_sha256": execution_binding["plan_sha256"],
        })
        raise packs.seedream.SeedreamError("submission_unknown")

    monkeypatch.setattr(packs.seedream, "edit", unknown_edit)
    settings = SimpleNamespace(seedream_model="seedream-v2", seedream_edit_mode="edit")
    gate = FakeGate()
    for _ in range(2):
        result = asyncio.run(packs.prepare_replacement_packs_with_seedream(
            root, _plan(), settings=settings, revision=3, quality_gate=gate,
        ))
        assert result.status == "submission_unknown"
    assert paid_posts == 1
    assert not gate.candidates


def test_image_quality_adapter_uses_loader_backed_frame_masks(tmp_path, monkeypatch):
    captured = {}

    class Receipt:
        status, publishable = "pass", True

        def to_dict(self):
            return {"status": "pass"}

    def evaluate(candidate, **kwargs):
        captured.update(kwargs)
        return Receipt()

    image_quality = types.ModuleType("app.image_quality")
    image_quality.evaluate_reference_packs = evaluate
    monkeypatch.setitem(sys.modules, "app.image_quality", image_quality)
    masks = (object(), object())
    gate = packs.ImageQualityPackGate(
        plan={"version": 2}, frame_masks=masks, profile={"quality": 1},
        semantic_verifier=None,
    )
    candidate = SimpleNamespace()
    gate.evaluate(candidate, receipt_path=tmp_path / "quality.json")
    assert captured["frame_masks"] == masks
    assert "mask_manifest" not in captured
