#!/usr/bin/env python3
"""Generate a current three-Skill manifest from the source bytes.

This command is a report/template generator only: it never writes a CID's
``work/skills`` freeze.  The pipeline owns that durable operation through
``app.skill_milestone.freeze``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app import skill_milestone  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a hash-derived three-Skill milestone manifest."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository containing skills/<name>/SKILL.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="manifest file to write",
    )
    args = parser.parse_args()
    manifest = skill_milestone.manifest_for_sources(args.repository_root)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(skill_milestone.canonical_manifest_bytes(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
