from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.pipeline import Pipeline
from app.research_log import ResearchLog
from app.store import ProjectStore

WORKTREE = Path(__file__).resolve().parents[1]
HARVEST_PATH = WORKTREE / "scripts" / "harvest_node_fixtures.py"


def _load_harvest():
    spec = importlib.util.spec_from_file_location(
        "harvest_node_fixtures", HARVEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        access_token="test",
        data_dir=tmp_path,
        public_base_url="https://settings.test",
    )


def _pipeline(tmp_path: Path, store: ProjectStore) -> Pipeline:
    return Pipeline(
        _settings(tmp_path),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        seedance=object(),
    )


def test_mark_failed_writes_timestamped_boundary_snapshot(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "snapshot")

    store.mark_failed(project_id, code="first_error", message="first")
    store.mark_failed(project_id, code="second_error", message="second")

    snapshots = sorted((project / "logs" / "boundary-snapshots").glob("*.json"))
    assert len(snapshots) == 2
    assert "first_error" in snapshots[0].name
    assert "second_error" in snapshots[1].name
    # 快照是失败现场 state 的逐字节复制：错误码与失败态都在里面。
    latest = json.loads(snapshots[-1].read_text(encoding="utf-8"))
    assert latest["status"] == "failed"
    assert latest["error"] == {"code": "second_error", "message": "second"}
    assert latest["id"] == project_id


def test_persist_failure_writes_boundary_snapshot(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "persist-snapshot")
    pipeline = _pipeline(tmp_path, store)

    pipeline._persist_failure(
        project_id,
        ResearchLog(project),
        stage="pipeline_failed",
        code="seedance_resubmit_cap_exceeded",
        message="cap",
        min_percent=1,
    )

    snapshots = list((project / "logs" / "boundary-snapshots").glob("*.json"))
    assert len(snapshots) == 1
    assert "seedance_resubmit_cap_exceeded" in snapshots[0].name
    assert json.loads(snapshots[0].read_text(encoding="utf-8"))["error"] == {
        "code": "seedance_resubmit_cap_exceeded",
        "message": "cap",
    }


def test_boundary_snapshot_write_failure_never_breaks_business(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ProjectStore(tmp_path / "projects")
    project_id, project, _state = store.create({}, "snapshot-readonly")
    # 快照目录不可写：置 failed 照常成功，快照放弃（记 stderr）。
    (project / "logs").chmod(0o500)

    store.mark_failed(project_id, code="any_error", message="x")

    assert store.read(project_id)["status"] == "failed"
    (project / "logs").chmod(0o700)


def test_harvest_reads_frozen_snapshot_not_current_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harvest = _load_harvest()
    monkeypatch.setattr(
        harvest, "FIXTURES_ROOT", tmp_path / "fixtures"
    )

    projects_root = tmp_path / "projects"
    project_id = "a" * 32
    project = projects_root / project_id
    # 项目当前是 succeeded（续跑成功），但失败时刻固化过快照。
    (project / "logs" / "boundary-snapshots").mkdir(parents=True)
    frozen = {
        "id": project_id,
        "status": "failed",
        "effective_request": {"version": 1},
        "translated_voice_lines": [{"text": "一"}],
        "enhanced_element_index": {"people": {"person-01": {}}},
        "voice_references": [],
        "error": {"code": "pre_picsart_resume_failed", "message": "x"},
    }
    (project / "logs" / "boundary-snapshots" / "20260907T205656Z-x.json"
    ).write_text(json.dumps(frozen), encoding="utf-8")
    (project / "logs").mkdir(exist_ok=True)
    (project / "state.json").write_text(
        json.dumps({"id": project_id, "status": "succeeded"}), encoding="utf-8"
    )
    # 无任何节点 logs（boundary 快照采集不依赖它们）。
    monkeypatch.setattr(harvest, "NODES", [])
    monkeypatch.setattr(
        sys, "argv",
        [
            "harvest_node_fixtures.py",
            "--projects-root", str(projects_root),
            "--projects", project_id,
        ],
    )

    harvest.main()

    fixture = (
        tmp_path / "fixtures" / "boundary" / project_id / "state-snapshot.json"
    )
    snapshot = json.loads(fixture.read_text(encoding="utf-8"))
    assert snapshot["error"]["code"] == "pre_picsart_resume_failed"
    assert snapshot["voice_references"] == []
    assert snapshot["effective_request"] == {"version": 1}
    manifest = json.loads(
        (tmp_path / "fixtures" / "manifest.json").read_text(encoding="utf-8")
    )
    entries = [
        entry for entry in manifest["entries"]
        if entry["node"] == "state-snapshot"
    ]
    assert len(entries) == 1
    assert entries[0]["source_snapshot"] == "20260907T205656Z-x.json"


def test_harvest_without_frozen_snapshot_writes_no_boundary_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harvest = _load_harvest()
    monkeypatch.setattr(harvest, "FIXTURES_ROOT", tmp_path / "fixtures")

    projects_root = tmp_path / "projects"
    project_id = "b" * 32
    project = projects_root / project_id
    project.mkdir(parents=True)
    # 即使当前仍是 failed，没有固化快照就没有 boundary fixture（新语义：
    # 只认失败发生时固化的现场）。
    (project / "state.json").write_text(
        json.dumps({"id": project_id, "status": "failed"}), encoding="utf-8"
    )
    monkeypatch.setattr(harvest, "NODES", [])
    monkeypatch.setattr(
        sys, "argv",
        [
            "harvest_node_fixtures.py",
            "--projects-root", str(projects_root),
            "--projects", project_id,
        ],
    )

    harvest.main()

    assert not (
        tmp_path / "fixtures" / "boundary" / project_id / "state-snapshot.json"
    ).exists()
