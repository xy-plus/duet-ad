"""Standalone probe: run AutoDL wan2.2animate motion retargeting on one segment.

Uses project 8e124288 segment-1: ref_video = source-muted.mp4,
ref_image = TILE_01 crop from the replacement sheet (label band trimmed).

Usage (env: source service.env + .env first):
    python3 scripts/retarget_probe.py [project_dir] [segment]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402  (constants only)
from app.providers import AutodlClient  # noqa: E402
from app.replacement_sheet import extract_tile_reference  # noqa: E402

DEFAULT_PROJECT = PROJECT_ROOT / "data/projects/8e124288db2d43beb2c555f6762ce50f"
OUTPUT_DIR = PROJECT_ROOT / "data/smoke-runs/retarget-probe"


def main() -> int:
    project = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROJECT
    segment = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    state = json.loads((project / "state.json").read_text(encoding="utf-8"))
    sheet_state = state.get("replacement_sheet")
    if not isinstance(sheet_state, dict):
        print("project has no replacement sheet state", file=sys.stderr)
        return 2
    spec = sheet_state.get("spec")
    tiles = spec.get("tiles") if isinstance(spec, dict) else None
    person_tile = next(
        (
            tile
            for tile in tiles or []
            if str(tile.get("stable_key", "")).startswith("person-")
        ),
        None,
    )
    if person_tile is None:
        print("no person tile in sheet spec", file=sys.stderr)
        return 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = OUTPUT_DIR / f"seg{segment}-retarget-reference.png"
    extract_tile_reference(
        project / "work/global-reference-sheet/reference-sheet.png",
        spec,
        person_tile["stable_key"],
        reference,
    )
    muted = project / "work/segments" / f"{segment:04d}" / "source-muted.mp4"
    if not muted.is_file():
        print(f"missing {muted}", file=sys.stderr)
        return 2
    print(f"ref_image={reference} ref_video={muted}", flush=True)
    client = AutodlClient(get_settings())
    task_id = client.submit_motion_retarget(
        image_path=reference, video_path=muted,
    ).value
    print(f"task_id={task_id}", flush=True)
    video_url = client.poll(task_id).value
    print(f"video_url={video_url}", flush=True)
    destination = OUTPUT_DIR / f"seg{segment}-retargeted.mp4"
    client.download(video_url, destination)
    print(f"downloaded {destination} ({destination.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
