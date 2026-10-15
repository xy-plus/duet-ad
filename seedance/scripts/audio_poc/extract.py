"""步骤1：抽原音频、切行片段、拷贝生产对照 full-track。

用法: python extract.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/xy/duet-ad1-h3-max-r2v/scripts/audio_poc")
from common import (  # noqa: E402
    PROD_REF_DIR,
    PROJECTS,
    line_clip_path,
    project_dir,
    project_lines,
)

FFMPEG = "/home/xy/.local/bin/ffmpeg"


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {' '.join(cmd)}\n{result.stderr[-500:]}")


def main() -> None:
    for short_id, proj in PROJECTS.items():
        out_dir = project_dir(short_id)
        clips_dir = out_dir / "line-clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        source_mp4 = Path(proj) / "inputs" / "source.mp4"
        original = out_dir / "original-track.wav"
        if not original.exists():
            run([FFMPEG, "-y", "-i", str(source_mp4), "-ac", "1", "-ar", "16000",
                 "-vn", str(original)])
        for line in project_lines(short_id):
            clip = line_clip_path(short_id, line["index"])
            if clip.exists():
                continue
            run([FFMPEG, "-y", "-ss", f"{line['start_s']:.3f}", "-to", f"{line['end_s']:.3f}",
                 "-i", str(original), "-ac", "1", "-ar", "16000", str(clip)])
        prod_full = Path(proj) / "work" / "dialogue-audio" / "full-track.wav"
        ref_dir = PROD_REF_DIR / short_id
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref = ref_dir / "full-track.wav"
        if prod_full.exists() and not ref.exists():
            shutil.copy2(prod_full, ref)
        print(f"{short_id}: {len(project_lines(short_id))} 行, 原音频与对照就绪")


if __name__ == "__main__":
    main()
