"""生产落盘裁剪（ARTIFACT_RETENTION=full|lean，开关见 app/config.py）。

落盘纪律（测试期）：每个节点的输入输出全部落盘、带指纹可复用、原始响应
字段一个不丢——为了省钱和调试。生产环境只保留续跑/对账/成片所需的产物：
成功项目完成后由 pipeline 钩子调用 prune_after_success 做一次幂等清理；
失败项目原样保留现场（prune_project 对非 succeeded 状态直接拒绝）。
未设置 ARTIFACT_RETENTION 时维持 full（现状），生产启动脚本显式写 lean。

lean 只删两类纯冗余（原文件仍在 inputs/work/output，JSON 证据一律不动）：

1. logs/ 下每个节点 attempt 的 input-media/ output-media/ 落盘副本；
2. work/ 下的抽帧与代理图目录（source-4fps、keyframes、
   analysis-keyframes），失败续跑会自动重抽。

绝不删除：state.json、logs 下所有 JSON（含 submit accepted 证据与
picsart/seedream 回执）、inputs/、output/ 成片、fal 指纹缓存
object-detect-response-*.json（续跑零 HTTP 复用），以及 state.json 里
以项目相对路径引用的任何文件（整个目录含被引用文件时跳过该目录）。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.research_log import ResearchLog

_MEDIA_EVIDENCE_DIRS = frozenset({"input-media", "output-media"})
_WORK_FRAME_DIRS = frozenset({"source-4fps", "keyframes", "analysis-keyframes"})


def _state_referenced_paths(state: Any) -> set[str]:
    """Every project-relative path string anywhere in the persisted state."""
    paths: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            candidate = value.strip().lstrip("/")
            if candidate and "://" not in candidate:
                paths.add(candidate)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(state)
    return paths


def _prune_targets(project_dir: Path) -> list[Path]:
    targets: list[Path] = []
    logs_root = project_dir / "logs"
    if logs_root.is_dir():
        targets.extend(
            path
            for path in sorted(logs_root.rglob("*"))
            if path.is_dir() and path.name in _MEDIA_EVIDENCE_DIRS
        )
    work_root = project_dir / "work"
    if work_root.is_dir():
        targets.extend(
            path
            for path in sorted(work_root.rglob("*"))
            if path.is_dir() and path.name in _WORK_FRAME_DIRS
        )
    return targets


def prune_project(
    project_dir: Path,
    *,
    state: dict[str, Any],
    log: ResearchLog | None = None,
) -> dict[str, Any]:
    """Delete lean-mode redundant artifacts of a SUCCEEDED project.

    Idempotent: re-running on a pruned project removes nothing.  Refuses any
    non-succeeded state so a failed project's debugging scene is untouchable.
    """
    if not isinstance(state, dict) or state.get("status") != "succeeded":
        raise ValueError("lean artifact pruning requires a succeeded project")
    project_dir = Path(project_dir)
    protected = _state_referenced_paths(state)
    summary: dict[str, Any] = {
        "removed_files": 0,
        "removed_bytes": 0,
        "removed_dirs": [],
        "skipped_referenced": [],
        "errors": [],
    }
    for target in _prune_targets(project_dir):
        if target.is_symlink():
            continue
        files = [
            path
            for path in target.rglob("*")
            if path.is_file() or path.is_symlink()
        ]
        if any(
            path.relative_to(project_dir).as_posix() in protected
            for path in files
        ):
            summary["skipped_referenced"].append(
                target.relative_to(project_dir).as_posix()
            )
            continue
        try:
            removed_bytes = sum(path.stat().st_size for path in files if path.is_file())
            shutil.rmtree(target)
        except FileNotFoundError:
            continue
        except OSError as exc:
            summary["errors"].append(f"{target}: {exc}")
            continue
        summary["removed_files"] += len(files)
        summary["removed_bytes"] += removed_bytes
        summary["removed_dirs"].append(target.relative_to(project_dir).as_posix())
    if log is not None:
        log.event("artifact_prune_lean", **summary)
    return summary


def prune_after_success(
    project_dir: Path,
    *,
    state: dict[str, Any],
    log: ResearchLog | None = None,
) -> dict[str, Any]:
    """Pipeline success hook entry: containment so pruning never breaks a run."""
    try:
        summary = prune_project(project_dir, state=state, log=log)
    except Exception as exc:
        if log is not None:
            log.event("artifact_prune_failed", error=str(exc))
        return {"pruned": False, "error": str(exc)}
    return {"pruned": True, **summary}
