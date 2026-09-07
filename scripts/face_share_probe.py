"""Probe: per-segment face share estimation by the project-index agent.

Replays the recorded project-index request of project 8e124288 verbatim
(same input JSON, same analysis keyframes, same skill prompt and suffix),
with one addition: the schema gains a top-level ``segment_metrics`` array
and the instruction asks for a per-segment max face-share estimate.

Usage (env: source service.env + deepseek.env + .env first):
    python3 scripts/face_share_probe.py [project_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.providers import DeepSeekClient  # noqa: E402
from app.video_maker_schemas import project_index_schema  # noqa: E402

DEFAULT_PROJECT = PROJECT_ROOT / "data/projects/8e124288db2d43beb2c555f6762ce50f"
VIDEO_SKILL = PROJECT_ROOT / "skills/video-maker/SKILL.md"

FACE_SHARE_INSTRUCTION = (
    "\n\n额外任务：为每个输入 segment 输出一条 segment_metrics 记录。"
    "max_face_share_percent：逐帧估计该段画面中占比最大的单张人脸"
    "（完整面部区域，含额头至下巴、双耳之间）面积占整帧面积的整数百分比，"
    "取该段全部帧中的最大值；该段无任何可见人脸时记 0。"
    "只依据像素证据，不按 visual_prompt 推测。"
    "segment_metrics 必须逐项覆盖全部输入 segment，不重复、不遗漏。"
)


def extended_schema(segment_indices: list[int]) -> dict:
    schema = project_index_schema(segment_indices)
    schema["properties"]["segment_metrics"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "enum": segment_indices},
                "max_face_share_percent": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "required": ["segment_index", "max_face_share_percent"],
            "additionalProperties": False,
        },
        "minItems": len(segment_indices),
        "maxItems": len(segment_indices),
    }
    schema["required"].append("segment_metrics")
    return schema


def main() -> int:
    project = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROJECT
    request = json.loads(
        (project / "work/project_index_request.json").read_text(encoding="utf-8")
    )
    images: list[Path] = []
    for segment in request["segments"]:
        for frame in segment["frames"]:
            images.append(project / frame["path"])
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        print(f"missing analysis frames: {missing}", file=sys.stderr)
        return 2
    indices = [int(s["segment_index"]) for s in request["segments"]]
    instruction = (
        VIDEO_SKILL.read_text(encoding="utf-8")
        + (
            "\n\n严格执行项目级索引阶段；像素帧是视觉事实的唯一依据；"
            "segments[].visual_prompt 只用于汇总逐段已发现对象并检查跨段覆盖，"
            "不得用它替代或推翻像素证据；最终填写本次注入的 Schema。"
        )
        + FACE_SHARE_INSTRUCTION
    )
    settings = get_settings()
    client = DeepSeekClient(settings)
    result = client.complete_structured(
        instruction, request, extended_schema(indices), images,
    )
    output = result.value
    metrics = {
        item["segment_index"]: item["max_face_share_percent"]
        for item in output.get("segment_metrics", [])
    }
    print("segment_index  max_face_share_percent")
    for index in indices:
        print(f"{index:>13}  {metrics.get(index)}")
    out_path = project / "work" / "face_share_probe.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"full output saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
