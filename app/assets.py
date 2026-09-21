"""资产库：素材（替换参考图）与成片（生成视频）的只读聚合视图。

资产是引用不是拷贝：全部指向项目里已有文件，数据从 store.list() 的各项目
state（input_receipt / creation_input / generation_history / output 状态）
现算推导，零迁移、无缓存层（项目数 <200，每次请求聚合可接受）。文件已
不在磁盘的条目跳过（如 lean 裁剪后）。归属沿用 app/users.py 的
can_access_project：普通用户只见自己的项目产物，superuser 全见。

素材预览直接复用现有 creation-input 图片路由（自带归属鉴权），不新增
按 sha 取图的端点；成片 file_url 复用现有 files 路由（generated.mp4 与
history/vN 两种都已有鉴权与存在性校验）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

from app.store import ProjectStore, replacement_input_paths, replacement_receipts
from app.users import ResolvedUser, can_access_project


# 与 app/main.py REPLACEMENT_MEDIA_TYPES 同源语义：参考图只接受这三种。
MATERIAL_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
# used_in 只保留最近几次使用（列表已按 created_at 倒序遍历，取前 N 个）。
USED_IN_LIMIT = 5


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _project_materials(
    state: dict[str, Any],
) -> Iterator[tuple[int, str, int, str, dict[str, Any]]]:
    """展开一个项目的替换参考图：(序号, sha256, bytes, 相对路径, 快照)。

    state 结构不可信时（receipt 损坏、路径清单与 receipt 数量不一致）
    fail closed 跳过该项目，不炸整个聚合。
    """
    receipt = state.get("input_receipt")
    if not isinstance(receipt, dict):
        return
    try:
        receipts = replacement_receipts(receipt)
    except ValueError:
        return
    if not receipts:
        return
    expected_paths = replacement_input_paths(len(receipts))
    if state.get("replacement_image_paths") != expected_paths:
        return
    creation = state.get("creation_input")
    snapshots = creation.get("replacement_images") if isinstance(creation, dict) else None
    if not isinstance(snapshots, list):
        snapshot = creation.get("replacement_image") if isinstance(creation, dict) else None
        snapshots = [snapshot if isinstance(snapshot, dict) else {}]
    for index, entry in enumerate(receipts, 1):
        snapshot = snapshots[index - 1] if index - 1 < len(snapshots) else {}
        if not isinstance(snapshot, dict):
            snapshot = {}
        yield index, entry["sha256"], entry["bytes"], expected_paths[index - 1], snapshot


def _file_exists(store: ProjectStore, project_id: str, relative: str) -> bool:
    try:
        store.resolve_project_file(project_id, store.path(project_id) / relative)
    except (FileNotFoundError, OSError):
        return False
    return True


def collect_assets(store: ProjectStore, user: ResolvedUser) -> dict[str, Any]:
    """聚合当前用户可见的素材（按 sha256 去重）与成片（当前版+历史版）。"""
    materials: dict[str, dict[str, Any]] = {}
    videos: list[dict[str, Any]] = []
    for state in store.list():  # created_at 倒序：每组首个命中即最近一次使用
        if not can_access_project(state, user):
            continue
        project_id = state["id"]
        title = state.get("title") if isinstance(state.get("title"), str) else ""
        created_at = state.get("created_at") if isinstance(state.get("created_at"), str) else ""
        duration_s = state.get("duration_s")
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
            duration_s = None

        for index, sha256, size, relative, snapshot in _project_materials(state):
            if not _file_exists(store, project_id, relative):
                continue
            entry = materials.get(sha256)
            if entry is None:
                media_type = snapshot.get("media_type")
                entry = {
                    "sha256": sha256,
                    "bytes": size,
                    "media_type": (
                        media_type if media_type in MATERIAL_MEDIA_TYPES
                        else "application/octet-stream"
                    ),
                    "filename": snapshot.get("filename") or "参考图",
                    "preview_url": (
                        f"/api/conversations/{project_id}"
                        f"/creation-input/replacement-images/{index}"
                    ),
                    "use_count": 0,
                    "used_in": [],
                }
                materials[sha256] = entry
            entry["use_count"] += 1
            if len(entry["used_in"]) < USED_IN_LIMIT:
                entry["used_in"].append({
                    "project_id": project_id,
                    "title": title,
                    "created_at": created_at,
                })

        history = state.get("generation_history")
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, dict):
                    continue
                version = item.get("version")
                if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                    continue
                video = item.get("video")
                expected = f"output/history/v{version}/generated.mp4"
                if not isinstance(video, dict) or video.get("path") != expected:
                    continue
                if not _file_exists(store, project_id, expected):
                    continue
                completed_at = item.get("completed_at")
                videos.append({
                    "project_id": project_id,
                    "title": title,
                    "version": version,
                    "completed_at": completed_at if isinstance(completed_at, str) else None,
                    "file_url": (
                        f"/api/conversations/{project_id}/files/history/v{version}/generated.mp4"
                    ),
                    "duration_s": duration_s,
                })
        if (
            state.get("status") == "succeeded"
            and state.get("has_video") is True
            and _file_exists(store, project_id, "output/generated.mp4")
        ):
            videos.append({
                "project_id": project_id,
                "title": title,
                "version": None,  # null = 当前版
                "completed_at": state.get("updated_at"),
                "file_url": f"/api/conversations/{project_id}/files/generated.mp4",
                "duration_s": duration_s,
            })

    material_list = sorted(
        materials.values(),
        key=lambda entry: entry["used_in"][0]["created_at"] if entry["used_in"] else "",
        reverse=True,
    )
    videos.sort(key=lambda item: item.get("completed_at") or "", reverse=True)
    return {"materials": material_list, "videos": videos}


def find_material(
    store: ProjectStore, sha256: str, user: ResolvedUser
) -> tuple[Path, dict[str, Any]] | None:
    """按 sha256 在当前用户可见的项目里定位素材文件（用于创建项目时从库选图）。

    返回 (文件路径, {filename, bytes, media_type})；文件落盘内容与 receipt
    逐字节校验（大小+sha256 复核），不一致或无权访问都返回 None。
    """
    for state in store.list():
        if not can_access_project(state, user):
            continue
        project_id = state["id"]
        for _index, sha, size, relative, snapshot in _project_materials(state):
            if sha != sha256:
                continue
            try:
                path = store.resolve_project_file(
                    project_id, store.path(project_id) / relative
                )
            except (FileNotFoundError, OSError):
                continue
            media_type = snapshot.get("media_type")
            if media_type not in MATERIAL_MEDIA_TYPES:
                continue
            try:
                if path.stat().st_size != size or _sha256_file(path) != sha256:
                    continue
            except OSError:
                continue
            return path, {
                "filename": snapshot.get("filename") or "参考图",
                "bytes": size,
                "media_type": media_type,
            }
    return None
