"""Deterministic, provider-neutral replacement reference-sheet artifacts.

The enhanced project element index is the authority for every binding in this
module.  Model-authored prose may describe a design, but it cannot assign a
different stable key, tile, category, or source image.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPLACEMENT_SHEET_SPEC_NAME = "duet.replacement-reference-sheet"
REPLACEMENT_SHEET_SPEC_VERSION = 1
REPLACEMENT_SHEET_WIDTH = 2048
REPLACEMENT_SHEET_HEIGHT = 2048

_ELEMENT_CATEGORIES = ("people", "entities", "scenes")
_PNG_PARAMETERS = [cv2.IMWRITE_PNG_COMPRESSION, 3]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _png_bytes(image: np.ndarray) -> bytes:
    encoded, payload = cv2.imencode(".png", image, _PNG_PARAMETERS)
    if not encoded:
        raise ValueError("replacement sheet PNG encoding failed")
    return payload.tobytes()


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _project_frame_receipt(
    project: Path,
    path: Path,
    *,
    segment_index: int,
    frame_order: int,
) -> dict[str, Any] | None:
    try:
        if path.is_symlink():
            return None
        resolved = path.resolve(strict=True)
        project_root = project.resolve(strict=True)
        relative = resolved.relative_to(project_root)
        info = resolved.stat()
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return {
        "segment_index": segment_index,
        "frame_order": frame_order,
        "path": relative.as_posix(),
        "size_bytes": info.st_size,
        "sha256": _sha256_file(resolved),
    }


def _prepared_frames(
    project: Path,
    prepared: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[tuple[Path, dict[str, Any] | None], ...]]:
    collected: list[tuple[int, int, list[object]]] = []
    for prepared_order, item in enumerate(prepared):
        if not isinstance(item, Mapping):
            continue
        segment_index = _positive_int(item.get("index"))
        paths = item.get("analysis_paths")
        if segment_index is None or not isinstance(paths, (list, tuple)):
            continue
        collected.append((segment_index, prepared_order, list(paths)))

    frames_by_segment: dict[
        int, tuple[tuple[Path, dict[str, Any] | None], ...]
    ] = {}
    for segment_index, _prepared_order, paths in sorted(collected):
        segment_frames: list[tuple[Path, dict[str, Any] | None]] = []
        for frame_order, raw_path in enumerate(paths, 1):
            try:
                path = Path(raw_path)
            except TypeError:
                continue
            if not path.is_absolute():
                path = project / path
            receipt = _project_frame_receipt(
                project,
                path,
                segment_index=segment_index,
                frame_order=frame_order,
            )
            segment_frames.append((path, receipt))
        # The normal pipeline has one prepared record per segment.  Keeping the
        # first makes malformed duplicates deterministic without a new gate.
        frames_by_segment.setdefault(segment_index, tuple(segment_frames))
    return frames_by_segment


def _representative_receipt(
    record: Mapping[str, Any],
    frames_by_segment: Mapping[
        int, tuple[tuple[Path, dict[str, Any] | None], ...]
    ],
) -> dict[str, Any] | None:
    candidates: set[tuple[int, int]] = set()
    occurrences = record.get("occurrences")
    if isinstance(occurrences, list):
        for occurrence in occurrences:
            if not isinstance(occurrence, Mapping):
                continue
            segment_index = _positive_int(occurrence.get("segment_index"))
            frame_orders = occurrence.get("frame_orders")
            if segment_index is None or not isinstance(frame_orders, list):
                continue
            for raw_order in frame_orders:
                frame_order = _positive_int(raw_order)
                if frame_order is not None:
                    candidates.add((segment_index, frame_order))

    for segment_index, frame_order in sorted(candidates):
        frames = frames_by_segment.get(segment_index, ())
        if frame_order <= len(frames):
            receipt = frames[frame_order - 1][1]
            if receipt is not None:
                return dict(receipt)
    return None


def replacement_sheet_spec_sha256(spec: Mapping[str, Any]) -> str:
    """Return the digest of a spec without its self-referential digest field."""
    payload = dict(spec)
    payload.pop("spec_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def build_replacement_sheet_spec(
    project: Path,
    enhanced_element_index: Mapping[str, Any],
    prepared: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze stable-key-to-tile bindings and representative source receipts.

    Only elements with a dictionary ``replacement`` occupy a tile.  Missing or
    unusable occurrence metadata falls back to the first readable analysis
    frame; it never becomes a semantic quality gate.
    """
    frames_by_segment = _prepared_frames(project, prepared)
    records: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for category in _ELEMENT_CATEGORIES:
        category_records = enhanced_element_index.get(category)
        if not isinstance(category_records, Mapping):
            continue
        for stable_key in sorted(
            (key for key in category_records if isinstance(key, str)),
            key=lambda value: value.encode("ascii", "backslashreplace"),
        ):
            record = category_records[stable_key]
            if not isinstance(record, Mapping):
                continue
            replacement = record.get("replacement")
            if isinstance(replacement, dict):
                records.append((category, stable_key, record, replacement))

    count = len(records)
    columns = math.isqrt(count)
    if columns * columns < count:
        columns += 1
    rows = (count + columns - 1) // columns if columns else 0
    tiles: list[dict[str, Any]] = []
    for offset, (category, stable_key, record, replacement) in enumerate(records):
        tiles.append({
            "tile_id": f"TILE_{offset + 1:02d}",
            "category": category,
            "stable_key": stable_key,
            "row": offset // columns + 1,
            "column": offset % columns + 1,
            "source": replacement.get("source"),
            "target": replacement.get("target_visual_description"),
            "source_frame_receipt": _representative_receipt(
                record,
                frames_by_segment,
            ),
        })

    spec: dict[str, Any] = {
        "schema": REPLACEMENT_SHEET_SPEC_NAME,
        "version": REPLACEMENT_SHEET_SPEC_VERSION,
        "enhanced_element_index_sha256": hashlib.sha256(
            _canonical_json(enhanced_element_index)
        ).hexdigest(),
        "width": REPLACEMENT_SHEET_WIDTH,
        "height": REPLACEMENT_SHEET_HEIGHT,
        "columns": columns,
        "rows": rows,
        "tiles": tiles,
    }
    spec["spec_sha256"] = replacement_sheet_spec_sha256(spec)
    return spec


def _validate_spec(spec: Mapping[str, Any], *, allow_empty: bool) -> list[Mapping[str, Any]]:
    if (
        spec.get("schema") != REPLACEMENT_SHEET_SPEC_NAME
        or spec.get("version") != REPLACEMENT_SHEET_SPEC_VERSION
        or spec.get("width") != REPLACEMENT_SHEET_WIDTH
        or spec.get("height") != REPLACEMENT_SHEET_HEIGHT
        or spec.get("spec_sha256") != replacement_sheet_spec_sha256(spec)
    ):
        raise ValueError("replacement sheet spec binding is invalid")
    tiles = spec.get("tiles")
    if not isinstance(tiles, list):
        raise ValueError("replacement sheet spec tiles are invalid")
    if not tiles and not allow_empty:
        raise ValueError("replacement sheet has no replacement tiles")
    if any(not isinstance(tile, Mapping) for tile in tiles):
        raise ValueError("replacement sheet spec tiles are invalid")
    return tiles


def _tile_bounds(
    tile: Mapping[str, Any],
    *,
    columns: int,
    rows: int,
) -> tuple[int, int, int, int] | None:
    row = _positive_int(tile.get("row"))
    column = _positive_int(tile.get("column"))
    if row is None or column is None or rows < 1 or columns < 1:
        return None
    x0 = (column - 1) * REPLACEMENT_SHEET_WIDTH // columns
    x1 = column * REPLACEMENT_SHEET_WIDTH // columns
    y0 = (row - 1) * REPLACEMENT_SHEET_HEIGHT // rows
    y1 = row * REPLACEMENT_SHEET_HEIGHT // rows
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _ascii_label(value: object) -> str:
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _draw_layout(image: np.ndarray, spec: Mapping[str, Any]) -> None:
    tiles = spec["tiles"]
    columns = int(spec["columns"])
    rows = int(spec["rows"])
    for tile in tiles:
        bounds = _tile_bounds(tile, columns=columns, rows=rows)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        width = x1 - x0
        height = y1 - y0
        thickness = 2 if min(width, height) >= 128 else 1
        cv2.rectangle(
            image,
            (x0, y0),
            (max(x0, x1 - 1), max(y0, y1 - 1)),
            (24, 24, 24),
            thickness,
            cv2.LINE_AA,
        )
        if width < 8 or height < 8:
            continue
        first = _ascii_label(tile.get("tile_id", "TILE"))
        second = _ascii_label(tile.get("stable_key", ""))
        available_width = max(1, width - 12)
        scale = min(0.72, max(0.20, min(width, height) / 420.0))
        for text in (first, second):
            text_width = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
            )[0][0]
            if text_width > available_width:
                scale = max(0.12, scale * available_width / text_width)
        line_height = max(7, int(round(22 * scale)))
        band_height = min(height - 2, max(12, line_height * 2 + 8))
        cv2.rectangle(
            image,
            (x0 + thickness, y0 + thickness),
            (max(x0 + thickness, x1 - thickness - 1), y0 + band_height),
            (248, 248, 248),
            cv2.FILLED,
        )
        baseline_y = min(y1 - 2, y0 + line_height + 2)
        cv2.putText(
            image,
            first,
            (x0 + 6, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (12, 12, 12),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            second,
            (x0 + 6, min(y1 - 2, baseline_y + line_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (12, 12, 12),
            thickness,
            cv2.LINE_AA,
        )


def _fit_into_tile(
    destination: np.ndarray,
    source: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = bounds
    tile_width = x1 - x0
    tile_height = y1 - y0
    source_height, source_width = source.shape[:2]
    if min(tile_width, tile_height, source_width, source_height) < 1:
        return
    scale = min(tile_width / source_width, tile_height / source_height)
    width = max(1, min(tile_width, int(round(source_width * scale))))
    height = max(1, min(tile_height, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(source, (width, height), interpolation=interpolation)
    left = x0 + (tile_width - width) // 2
    top = y0 + (tile_height - height) // 2
    destination[top:top + height, left:left + width] = resized


def _receipt_image(
    project: Path,
    receipt: object,
) -> np.ndarray | None:
    if receipt is None:
        return None
    if not isinstance(receipt, Mapping):
        raise ValueError("replacement sheet source-frame receipt is invalid")
    relative = receipt.get("path")
    expected_sha256 = receipt.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_sha256, str):
        raise ValueError("replacement sheet source-frame receipt is invalid")
    try:
        project_root = project.resolve(strict=True)
        unresolved = project / relative
        if unresolved.is_symlink():
            raise ValueError("source-frame symlink is not allowed")
        candidate = unresolved.resolve(strict=True)
        candidate.relative_to(project_root)
        if _sha256_file(candidate) != expected_sha256:
            raise ValueError("source-frame SHA-256 does not match")
        payload = candidate.read_bytes()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(
            "replacement sheet source-frame receipt does not match"
        ) from exc
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError("replacement sheet source-frame is not an image")
    return decoded


def _build_seedream_prompt(
    spec: Mapping[str, Any],
    base_prompt: str,
    *,
    user_image_present: bool,
) -> str:
    lines = []
    normalized_base = base_prompt.strip()
    if normalized_base:
        lines.extend((normalized_base, ""))
    lines.extend([
        "CREATE ONE PROJECT-WIDE REPLACEMENT REFERENCE SHEET.",
        (
            f"The frozen global-index binding is {spec['spec_sha256']}. "
            "It is authoritative and must not be reinterpreted."
        ),
    ])
    if user_image_present:
        lines.extend([
            (
                "Input Image 1 is the user-reference image. It may influence "
                "only a tile whose SOURCE is exactly image_1. It must not "
                "preserve, alter, or constrain any other tile."
            ),
            "Input Image 2 is the blank 2048x2048 row-major layout canvas.",
            (
                "Input Image 3 is source-frame evidence in the identical "
                "frozen layout."
            ),
        ])
    else:
        lines.extend([
            "Input Image 1 is the blank 2048x2048 row-major layout canvas.",
            (
                "Input Image 2 is source-frame evidence in the identical "
                "frozen layout."
            ),
            (
                "No user-reference image is supplied. Do not treat either "
                "input as user-reference evidence."
            ),
        ])
    lines.extend([
        (
            "Render every listed tile once, inside its assigned cell. Do not "
            "rename, merge, omit, duplicate, reorder, or move stable keys. Do "
            "not add unlisted people, entities, or scenes."
        ),
        (
            "Keep the grid geometry and leave each label band readable. The "
            "finished result must be one 2048x2048 image, not separate images."
        ),
        "FROZEN TILE BINDINGS:",
    ])
    for tile in spec["tiles"]:
        lines.append(
            " | ".join([
                str(tile.get("tile_id")),
                f"row={tile.get('row')}",
                f"column={tile.get('column')}",
                f"category={tile.get('category')}",
                f"stable_key={json.dumps(tile.get('stable_key'), ensure_ascii=False)}",
                f"SOURCE={json.dumps(tile.get('source'), ensure_ascii=False)}",
                f"TARGET={json.dumps(tile.get('target'), ensure_ascii=False)}",
            ])
        )
    return "\n".join(lines).strip()


def materialize_replacement_sheet_inputs(
    project: Path,
    directory: Path,
    spec: Mapping[str, Any],
    base_prompt: str,
    *,
    user_image_present: bool = False,
) -> dict[str, Any]:
    """Atomically publish the frozen spec, prompt, and two Seedream inputs."""
    tiles = _validate_spec(spec, allow_empty=False)
    columns = int(spec["columns"])
    rows = int(spec["rows"])
    canvas = np.full(
        (REPLACEMENT_SHEET_HEIGHT, REPLACEMENT_SHEET_WIDTH, 3),
        255,
        dtype=np.uint8,
    )
    evidence = np.full_like(canvas, 232)
    for tile in tiles:
        bounds = _tile_bounds(tile, columns=columns, rows=rows)
        if bounds is None:
            continue
        frame = _receipt_image(project, tile.get("source_frame_receipt"))
        if frame is not None:
            _fit_into_tile(evidence, frame, bounds)
    _draw_layout(canvas, spec)
    _draw_layout(evidence, spec)
    prompt = _build_seedream_prompt(
        spec,
        base_prompt,
        user_image_present=user_image_present,
    )

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    canvas_path = directory / "canvas.png"
    evidence_path = directory / "source-evidence.png"
    prompt_path = directory / "prompt.txt"
    spec_path = directory / "spec.json"
    _atomic_write(canvas_path, _png_bytes(canvas))
    _atomic_write(evidence_path, _png_bytes(evidence))
    _atomic_write(prompt_path, prompt.encode("utf-8"))
    # Publishing the bound spec last makes it the commit marker for callers.
    _atomic_write(
        spec_path,
        (json.dumps(spec, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return {
        "spec": dict(spec),
        "spec_sha256": spec["spec_sha256"],
        "prompt": prompt,
        "canvas_path": canvas_path,
        "source_evidence_path": evidence_path,
    }


def finalize_replacement_sheet(
    provider_output: Path | bytes,
    spec: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Normalize one provider image and restore authoritative tile labels."""
    _validate_spec(spec, allow_empty=False)
    if isinstance(provider_output, Path):
        payload = provider_output.read_bytes()
    elif isinstance(provider_output, bytes):
        payload = provider_output
    else:
        raise TypeError("replacement sheet provider output must be Path or bytes")
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError("replacement sheet provider output is not an image")
    interpolation = (
        cv2.INTER_AREA
        if (
            decoded.shape[1] > REPLACEMENT_SHEET_WIDTH
            or decoded.shape[0] > REPLACEMENT_SHEET_HEIGHT
        )
        else cv2.INTER_LANCZOS4
    )
    normalized = cv2.resize(
        decoded,
        (REPLACEMENT_SHEET_WIDTH, REPLACEMENT_SHEET_HEIGHT),
        interpolation=interpolation,
    )
    _draw_layout(normalized, spec)
    output = _png_bytes(normalized)
    _atomic_write(destination, output)
    return {
        "path": destination.resolve().as_posix(),
        "size_bytes": len(output),
        "sha256": hashlib.sha256(output).hexdigest(),
        "width": REPLACEMENT_SHEET_WIDTH,
        "height": REPLACEMENT_SHEET_HEIGHT,
    }
