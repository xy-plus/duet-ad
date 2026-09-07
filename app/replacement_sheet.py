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
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPLACEMENT_SHEET_SPEC_NAME = "duet.replacement-reference-sheet"
REPLACEMENT_SHEET_SPEC_VERSION = 2
REPLACEMENT_SHEET_WIDTH = 2048
REPLACEMENT_SHEET_HEIGHT = 2048

_ELEMENT_CATEGORIES = ("people", "entities", "scenes")
_REPLACEMENT_SOURCES = {"image_1", "generated_variant"}
_SPEC_KEYS = {
    "schema",
    "version",
    "enhanced_element_index_sha256",
    "width",
    "height",
    "columns",
    "rows",
    "tiles",
    "spec_sha256",
}
_V2_TILE_KEYS = {
    "tile_id",
    "category",
    "stable_key",
    "row",
    "column",
    "source",
    "appearance_description",
}
_PNG_PARAMETERS = [cv2.IMWRITE_PNG_COMPRESSION, 3]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def replacement_sheet_spec_sha256(spec: Mapping[str, Any]) -> str:
    """Return the digest of a spec without its self-referential digest field."""
    payload = dict(spec)
    payload.pop("spec_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _replacement_records(
    enhanced_element_index: Mapping[str, Any],
) -> list[tuple[str, str, Mapping[str, Any]]]:
    records: list[tuple[str, str, Mapping[str, Any]]] = []
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
            if isinstance(replacement, Mapping) and replacement:
                records.append((category, stable_key, replacement))
    return records


def _grid_dimensions(count: int) -> tuple[int, int]:
    columns = math.isqrt(count)
    if columns * columns < count:
        columns += 1
    rows = (count + columns - 1) // columns if columns else 0
    return columns, rows


def build_replacement_sheet_spec(
    enhanced_element_index: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a v2 appearance atlas bound only by global stable keys."""
    records = _replacement_records(enhanced_element_index)
    columns, rows = _grid_dimensions(len(records))
    tiles: list[dict[str, Any]] = []
    for offset, (category, stable_key, replacement) in enumerate(records):
        source = replacement.get("source")
        appearance = replacement.get("appearance_description")
        if source not in _REPLACEMENT_SOURCES:
            raise ValueError("replacement sheet source is invalid")
        if not isinstance(appearance, str) or not appearance.strip():
            raise ValueError("replacement sheet appearance description is invalid")
        tiles.append({
            "tile_id": f"TILE_{offset + 1:02d}",
            "category": category,
            "stable_key": stable_key,
            "row": offset // columns + 1,
            "column": offset % columns + 1,
            "source": source,
            "appearance_description": appearance.strip(),
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


def validate_replacement_sheet_spec(
    spec: Mapping[str, Any],
    *,
    allow_empty: bool = False,
) -> list[Mapping[str, Any]]:
    """Validate the current v2 sheet binding."""
    if (
        set(spec) != _SPEC_KEYS
        or spec.get("schema") != REPLACEMENT_SHEET_SPEC_NAME
        or type(spec.get("version")) is not int
        or spec.get("version") != REPLACEMENT_SHEET_SPEC_VERSION
        or spec.get("width") != REPLACEMENT_SHEET_WIDTH
        or spec.get("height") != REPLACEMENT_SHEET_HEIGHT
        or not _is_sha256(spec.get("enhanced_element_index_sha256"))
        or not _is_sha256(spec.get("spec_sha256"))
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
    columns, rows = _grid_dimensions(len(tiles))
    if spec.get("columns") != columns or spec.get("rows") != rows:
        raise ValueError("replacement sheet v2 layout is invalid")
    stable_keys: set[str] = set()
    for offset, tile in enumerate(tiles):
        stable_key = tile.get("stable_key")
        appearance = tile.get("appearance_description")
        if (
            set(tile) != _V2_TILE_KEYS
            or tile.get("tile_id") != f"TILE_{offset + 1:02d}"
            or tile.get("category") not in _ELEMENT_CATEGORIES
            or not isinstance(stable_key, str)
            or not stable_key
            or stable_key in stable_keys
            or tile.get("row") != offset // columns + 1
            or tile.get("column") != offset % columns + 1
            or tile.get("source") not in _REPLACEMENT_SOURCES
            or not isinstance(appearance, str)
            or not appearance.strip()
            or appearance != appearance.strip()
        ):
            raise ValueError("replacement sheet v2 tile binding is invalid")
        stable_keys.add(stable_key)
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


def _build_seedream_prompt(
    spec: Mapping[str, Any],
    *,
    user_image_present: bool,
) -> str:
    image_bound_tiles = [
        tile for tile in spec["tiles"] if tile.get("source") == "image_1"
    ]
    if image_bound_tiles and not user_image_present:
        raise ValueError("replacement sheet user-reference image is missing")

    lines = [
        "CREATE ONE PROJECT-WIDE REPLACEMENT REFERENCE SHEET.",
        (
            f"The frozen global-index binding is {spec['spec_sha256']}. "
            "It is authoritative and must not be reinterpreted."
        ),
        "Input Image 1 is the blank 2048x2048 row-major layout canvas.",
    ]
    if user_image_present:
        lines.extend([
            (
                "Input Image 2 is the user-reference image. In the frozen "
                "Object Replace provenance, SOURCE=image_1 maps only to this "
                "Seedream Input Image 2."
            ),
            (
                "Input Image 2 may influence only tiles whose SOURCE is "
                "exactly image_1. Tiles with SOURCE=generated_variant must "
                "ignore it completely."
            ),
        ])
    else:
        lines.extend([
            (
                "No user-reference image is supplied. Input Image 1 is the "
                "only image input and is layout geometry, never appearance "
                "evidence."
            ),
        ])
    lines.extend([
        (
            "This is an appearance atlas, not a reenacted video frame. Render "
            "each listed stable key exactly once and only in its assigned "
            "cell. Never rename, merge, omit, duplicate, reorder, or move a "
            "stable key."
        ),
        (
            "Every cell is semantically isolated. Only that cell's stable key "
            "may appear in it; no other listed or unlisted person, entity, "
            "foreground object, animal, body part, or scene may leak into it."
        ),
        (
            "PEOPLE tiles: show exactly one replacement person as a "
            "head-and-shoulders face close-up portrait: front-facing, "
            "neutral expression, the face filling most of the cell, against "
            "a plain neutral background, so the new facial identity is "
            "unmistakable. When only a partial person is indexed, show "
            "exactly the named visible body portion instead. Show no props, "
            "held objects, other subjects, interactions, or environment."
        ),
        (
            "ENTITIES tiles: show exactly one isolated replacement object or "
            "animal against a plain neutral background. Show no person, hand, "
            "other entity, use/contact relationship, action, or environment."
        ),
        (
            "SCENES tiles: show one empty, unoccupied replacement environment. "
            "Include only its fixed architecture, surfaces, and fixed "
            "furnishings; show no indexed person, animal, movable entity, "
            "product, hand, or foreground subject."
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
                (
                    "APPEARANCE="
                    f"{json.dumps(tile.get('appearance_description'), ensure_ascii=False)}"
                ),
            ])
        )
    return "\n".join(lines).strip()


def materialize_replacement_sheet_inputs(
    directory: Path,
    spec: Mapping[str, Any],
    *,
    user_image_present: bool = False,
) -> dict[str, Any]:
    """Atomically publish a v2 spec, backend prompt, and layout input image."""
    validate_replacement_sheet_spec(spec, allow_empty=False)
    canvas = np.full(
        (REPLACEMENT_SHEET_HEIGHT, REPLACEMENT_SHEET_WIDTH, 3),
        255,
        dtype=np.uint8,
    )
    _draw_layout(canvas, spec)
    prompt = _build_seedream_prompt(
        spec,
        user_image_present=user_image_present,
    )

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    canvas_path = directory / "canvas.png"
    prompt_path = directory / "prompt.txt"
    spec_path = directory / "spec.json"
    _atomic_write(canvas_path, _png_bytes(canvas))
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
    }


def finalize_replacement_sheet(
    provider_output: Path | bytes,
    spec: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Normalize one provider image and restore authoritative tile labels."""
    validate_replacement_sheet_spec(spec, allow_empty=False)
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


def extract_tile_reference(
    sheet_path: Path,
    spec: Mapping[str, Any],
    stable_key: str,
    destination: Path,
) -> Path:
    """Crop one tile below its label band from the finished sheet image."""
    tiles = validate_replacement_sheet_spec(spec, allow_empty=False)
    matches = [tile for tile in tiles if tile.get("stable_key") == stable_key]
    if len(matches) != 1:
        raise ValueError("replacement sheet tile binding is invalid")
    bounds = _tile_bounds(
        matches[0],
        columns=int(spec["columns"]),
        rows=int(spec["rows"]),
    )
    if bounds is None:
        raise ValueError("replacement sheet tile bounds are invalid")
    x0, y0, x1, y1 = bounds
    width = x1 - x0
    height = y1 - y0
    # Mirror the _draw_layout label band so the crop excludes it.
    scale = min(0.72, max(0.20, min(width, height) / 420.0))
    line_height = max(7, int(round(22 * scale)))
    band_height = min(height - 2, max(12, line_height * 2 + 8))
    decoded = cv2.imdecode(
        np.frombuffer(sheet_path.read_bytes(), np.uint8),
        cv2.IMREAD_COLOR,
    )
    if (
        decoded is None
        or decoded.shape[0] != REPLACEMENT_SHEET_HEIGHT
        or decoded.shape[1] != REPLACEMENT_SHEET_WIDTH
    ):
        raise ValueError("replacement sheet image is invalid")
    crop = decoded[y0 + band_height:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("replacement sheet tile crop is empty")
    _atomic_write(destination, _png_bytes(crop))
    return destination
