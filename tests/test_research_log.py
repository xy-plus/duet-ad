from __future__ import annotations

import json
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.research_log import ResearchLog


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_research_log_secures_every_directory_and_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir(mode=0o755)
    source = tmp_path / "source.bin"
    source.write_bytes(b"raw-media")
    source.chmod(0o644)

    log = ResearchLog(project)
    attempt = log.attempt_dir("provider", "segment-0001", 1)
    log.write_json(attempt / "input.json", {"secret": "kept"})
    log.write_text(attempt / "stdout.log", "raw stdout")
    log.copy_media(attempt / "output-media", [source])
    log.event("node_succeeded", node="provider", scope="segment-0001")

    directories = [
        project / "logs",
        project / "logs" / "provider",
        project / "logs" / "provider" / "segment-0001",
        attempt,
        attempt / "output-media",
    ]
    files = [
        project / "logs" / "trace.jsonl",
        attempt / "input.json",
        attempt / "stdout.log",
        attempt / "output-media" / "001-source.bin",
    ]
    assert all(_mode(path) == 0o700 for path in directories)
    assert all(_mode(path) == 0o600 for path in files)
    assert (attempt / "output-media" / "001-source.bin").read_bytes() == b"raw-media"


def test_parallel_trace_writes_are_complete_json_lines(tmp_path: Path) -> None:
    log = ResearchLog(tmp_path / "project")
    total = 300
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda index: log.event("parallel", index=index), range(total)))

    rows = [
        json.loads(line)
        for line in log.trace.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == total
    assert {row["index"] for row in rows} == set(range(total))
    assert all(row["event"] == "parallel" for row in rows)
    assert _mode(log.trace) == 0o600


def test_write_failures_report_to_stderr_and_never_escape(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    log = ResearchLog(tmp_path / "project")

    def fail_write(path: Path, content: bytes, *, append: bool = False) -> None:
        raise OSError(f"disk unavailable for {path.name}")

    monkeypatch.setattr(log, "_write_bytes", fail_write)
    log.event("still-running")
    log.write_json(log.root / "value.json", {"value": 1})
    log.write_text(log.root / "stdout.log", "output")
    log.error(log.root / "error.json", RuntimeError("provider failed"))

    stderr = capsys.readouterr().err
    assert stderr.count("research log failure:") == 4
    assert "disk unavailable" in stderr


def test_directory_and_media_failures_are_best_effort(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    log = ResearchLog(tmp_path / "project")

    def fail_directory(path: Path) -> None:
        raise PermissionError(f"cannot create {path.name}")

    monkeypatch.setattr(log, "_ensure_directory", fail_directory)
    expected = log.root / "node" / "project" / "attempt-1"
    assert log.attempt_dir("node", "project", 1) == expected
    log.copy_media(log.root / "media", [tmp_path / "missing.bin"])

    stderr = capsys.readouterr().err
    assert stderr.count("research log failure:") == 2
    assert "cannot create" in stderr


def test_constructor_directory_failure_does_not_block(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def fail_directory(self: ResearchLog, path: Path) -> None:
        raise PermissionError("logs are read-only")

    monkeypatch.setattr(ResearchLog, "_ensure_directory", fail_directory)
    log = ResearchLog(tmp_path / "project")
    assert log.root == tmp_path / "project" / "logs"
    assert "research log failure: logs are read-only" in capsys.readouterr().err
