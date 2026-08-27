import asyncio
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from app import dual_target_poc
from app.config import Settings


def _png(value: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((8, 6, 3), value, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def _fixture(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    keyframes = project / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    names = []
    for index in range(1, 6):
        name = f"{index:02d}.png"
        (keyframes / name).write_bytes(_png(index * 20))
        names.append(name)
    plan = {
        "version": 2,
        "phase": "plan",
        "segment_indices": [0],
        "eligible": True,
        "reason": None,
        "person_plans": [{"id": "PERSON_01"}],
        "scene_plans": [{"id": "SCENE_01"}],
        "segments": [],
        "sha256": "a" * 64,
    }
    meta = {"schema_version": 2, "keyframes": names, "_image_continuity": plan}
    (project / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    refs = project / "work" / "replacement-packs" / "published"
    refs.mkdir(parents=True)

    def image(role: str, value: int):
        path = refs / f"{role}-{value}.png"
        path.write_bytes(_png(value))
        return SimpleNamespace(
            role=role, path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    profile = {"id": "poc-v1", "revision": 1}
    pack = SimpleNamespace(
        project_dir=project.resolve(),
        execution_profile=profile,
        candidate_sha256="b" * 64,
        receipt_sha256="c" * 64,
        people={"PERSON_01": SimpleNamespace(images=(
            image("primary", 101), image("alternate", 102),
        ))},
        scenes={"SCENE_01": SimpleNamespace(images=(
            image("primary", 111), image("alternate", 112),
        ))},
    )
    calls = []

    def load_pack(_project, **expected):
        calls.append(expected)
        assert expected["expected_upstream_plan_sha256"] == "a" * 64
        assert expected["expected_model"] == "doubao-seedream-5-0-pro-260628"
        assert expected["expected_revision"] == 1
        assert expected["expected_person_plan_ids"] == ("PERSON_01",)
        assert expected["expected_scene_plan_ids"] == ("SCENE_01",)
        if "expected_upstream_source_inventory_sha256" in expected:
            assert expected["expected_upstream_source_inventory_sha256"] == "d" * 64
            assert expected["expected_execution_profile_sha256"] == hashlib.sha256(
                json.dumps(
                    profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        return pack

    monkeypatch.setattr(
        dual_target_poc,
        "replacement_packs",
        SimpleNamespace(
            load_replacement_pack=load_pack,
            canonical_source_inventory_sha256=lambda _frames: "d" * 64,
        ),
    )
    monkeypatch.setattr(
        dual_target_poc.image_optimization,
        "dual_target_plan_receipt",
        lambda current: dict(current["_image_continuity"]),
        raising=False,
    )
    monkeypatch.setattr(
        dual_target_poc.image_optimization,
        "compile_segment_prompts",
        lambda _plan, mode: {0: f"{mode}: 同时替换人物和真实场景"},
        raising=False,
    )

    def freeze(_plan, *, revision, profile, model, frame_inventory):
        assert revision == 1
        return {
            "plan_sha256": "a" * 64,
            "profile": profile,
            "model": model,
            "revision": revision,
            "frames": [{
                **item,
                "observable_person_ids": (
                    [] if item["frame_index"] == 3 else ["PERSON_01"]
                ),
                "scene_id": "SCENE_01",
            } for item in frame_inventory],
        }

    monkeypatch.setattr(
        dual_target_poc.image_optimization,
        "freeze_execution_inputs",
        freeze,
        raising=False,
    )
    return project, pack, calls


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def test_three_frame_poc_is_receipt_bound_parallel_and_never_mutates_project(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    project, pack, loader_calls = _fixture(tmp_path, monkeypatch)
    evaluation = tmp_path / "evaluation"
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    project_before = _snapshot(project)
    payloads = []

    async def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        raw = base64.b64decode(payload["image"][0].split(",", 1)[1])
        return httpx.Response(200, json={
            "data": [{"b64_json": base64.b64encode(raw).decode()}]
        })

    result = asyncio.run(dual_target_poc.run_three_frame_seedream_poc(
        settings, project, evaluation,
        transport=httpx.MockTransport(handler),
    ))

    assert len(payloads) == 3
    assert len(result.frames) == 3
    assert [path.name for path in result.frames] == [
        "s0000-f0001.png", "s0000-f0003.png", "s0000-f0005.png",
    ]
    assert all(path.is_file() for path in result.frames)
    assert _snapshot(project) == project_before
    assert len(loader_calls) == 2
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["status"] == "done"
    assert receipt["replacement_pack_candidate_sha256"] == pack.candidate_sha256
    assert len(receipt["frames"]) == 3
    assert [item["input_order"][0]["role"] for item in receipt["frames"]] == [
        "current_frame", "current_frame", "current_frame",
    ]
    assert [item["role"] for item in receipt["frames"][0]["input_order"]] == [
        "current_frame",
        "identity:PERSON_01:primary", "identity:PERSON_01:alternate",
        "scene:SCENE_01:primary", "scene:SCENE_01:alternate",
    ]
    assert [item["role"] for item in receipt["frames"][1]["input_order"]] == [
        "current_frame", "scene:SCENE_01:primary", "scene:SCENE_01:alternate",
    ]
    attempts = sorted((evaluation / "work").glob("*.attempt.json"))
    assert len(attempts) == 3
    assert all(
        json.loads(path.read_text())["reference_pack_candidate_sha256"] == "b" * 64
        for path in attempts
    )

    asyncio.run(dual_target_poc.run_three_frame_seedream_poc(
        settings, project, evaluation,
        transport=httpx.MockTransport(handler),
    ))
    assert len(payloads) == 3


def test_three_frame_poc_submission_unknown_is_never_automatically_resent(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    project, _pack, _loader_calls = _fixture(tmp_path, monkeypatch)
    evaluation = tmp_path / "evaluation"
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("ambiguous", request=request)

    with pytest.raises(dual_target_poc.DualTargetPocError) as caught:
        asyncio.run(dual_target_poc.run_three_frame_seedream_poc(
            settings, project, evaluation,
            transport=httpx.MockTransport(handler),
        ))
    assert caught.value.code == "submission_unknown"
    assert calls == 3
    assert json.loads((evaluation / "run.json").read_text())["status"] == (
        "submission_unknown"
    )

    with pytest.raises(dual_target_poc.DualTargetPocError) as repeated:
        asyncio.run(dual_target_poc.run_three_frame_seedream_poc(
            settings, project, evaluation,
            transport=httpx.MockTransport(handler),
        ))
    assert repeated.value.code == "submission_unknown"
    assert calls == 3


def test_three_frame_poc_rejects_production_project_output_directory_before_post(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    project, _pack, _loader_calls = _fixture(tmp_path, monkeypatch)
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)

    with pytest.raises(dual_target_poc.DualTargetPocError) as caught:
        asyncio.run(dual_target_poc.run_three_frame_seedream_poc(
            settings, project, project / "work" / "poc-output",
        ))

    assert caught.value.code == "evaluation_dir_not_isolated"
    assert not (project / "work" / "poc-output").exists()
