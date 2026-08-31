import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import h3, image_optimization, long_generation, long_video
from conftest import make_settings


def _png(*, width: int, height: int, bgr: tuple[int, int, int]) -> bytes:
    image = np.full((height, width, 3), bgr, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    assert ok
    return encoded.tobytes()


def _fusion_fixture(root: Path, *, first_frame: bytes):
    keyframes_dir = root / "work" / "segments" / "1" / "work" / "postprocessed"
    keyframes_dir.mkdir(parents=True)
    frames = []
    sources = []
    optimization_frames = []
    for order in range(1, 10):
        data = first_frame if order == 1 else _png(
            width=7, height=5, bgr=(order, order * 2, order * 3),
        )
        path = keyframes_dir / f"{order:02d}.png"
        path.write_bytes(data)
        frames.append((path, data))
        sources.append({
            "order": order,
            "source_time_s": float(order - 1),
            "source_scene_id": "SCENE_01",
            "transition": (
                {"type": "start", "at_segment_s": 0.0}
                if order == 1 else
                {"type": "continuous", "at_segment_s": None}
            ),
        })
        text = f"optimized frame {order}"
        optimization_frames.append({
            "segment_index": 1,
            "frame_index": order,
            "current": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
    segment = long_generation.FrozenSegment(
        index=1,
        start_s=0.0,
        end_s=10.0,
        chain_id="chain-001",
        join_mode="hard_cut",
        workdir=root / "work" / "segments" / "1",
        first_frame=frames[0][0],
        first_frame_data=frames[0][1],
        last_frame=frames[-1][0],
        last_frame_data=frames[-1][1],
        prompt="fused visual",
        keyframes=tuple(frames),
        keyframe_sources=tuple(sources),
        dialogue=(),
        dialogue_sha256=hashlib.sha256(b"[]\n").hexdigest(),
    )
    plan = long_generation.FrozenPlan(
        root=root,
        source=root / "source.mp4",
        receipt="a" * 64,
        segments=(segment,),
        receipt_version=long_video.VISUAL_PLAN_RECEIPT_VERSION,
    )
    meta = {
        "segments": [{"index": 1, "visual_prompt": "source action"}],
        "_image_optimization": {"frames": optimization_frames},
    }
    return plan, meta


def test_prompt_fusion_proxy_is_ceil_half_content_addressed_and_immutable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    original = _png(width=7, height=5, bgr=(10, 20, 30))
    plan, meta = _fusion_fixture(root, first_frame=original)

    first_payload = json.loads(long_generation.build_prompt_fusion_input(
        root=root,
        meta=meta,
        plan=plan,
        dialogue_mode="none",
        dialogue_delivery="auto",
    ))
    first_binding = first_payload["segments"][0]["new_keyframes"][0]
    first_proxy = root / first_binding["path"]
    first_proxy_bytes = first_proxy.read_bytes()
    decoded = cv2.imdecode(np.frombuffer(first_proxy_bytes, np.uint8), cv2.IMREAD_COLOR)

    assert decoded.shape[:2] == (3, 4)
    assert first_binding["sha256"] == hashlib.sha256(first_proxy_bytes).hexdigest()
    assert first_proxy == (
        root / "work" / long_generation.PROMPT_FUSION_PROXY_DIR
        / f"{first_binding['sha256']}.png"
    )

    changed = _png(width=7, height=5, bgr=(90, 80, 70))
    first_path = plan.segments[0].keyframes[0][0]
    first_path.write_bytes(changed)
    changed_segment = replace(
        plan.segments[0],
        first_frame_data=changed,
        keyframes=((first_path, changed), *plan.segments[0].keyframes[1:]),
    )
    changed_plan = replace(plan, segments=(changed_segment,))
    second_payload = json.loads(long_generation.build_prompt_fusion_input(
        root=root,
        meta=meta,
        plan=changed_plan,
        dialogue_mode="none",
        dialogue_delivery="auto",
    ))
    second_binding = second_payload["segments"][0]["new_keyframes"][0]

    assert second_binding["path"] != first_binding["path"]
    assert first_proxy.read_bytes() == first_proxy_bytes


def test_prompt_fusion_rejects_tampered_content_addressed_proxy(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    plan, meta = _fusion_fixture(
        root, first_frame=_png(width=7, height=5, bgr=(10, 20, 30)),
    )
    payload = json.loads(long_generation.build_prompt_fusion_input(
        root=root,
        meta=meta,
        plan=plan,
        dialogue_mode="none",
        dialogue_delivery="auto",
    ))
    binding = payload["segments"][0]["new_keyframes"][0]
    (root / binding["path"]).write_bytes(b"tampered")

    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_input_invalid",
    ):
        long_generation.build_prompt_fusion_input(
            root=root,
            meta=meta,
            plan=plan,
            dialogue_mode="none",
            dialogue_delivery="auto",
        )


def test_fusion_to_h3_binds_frame_positions_not_image_bytes() -> None:
    frames = [{"order": order} for order in range(1, 10)]

    assert long_generation._frozen_fusion_frame_orders(
        {"new_keyframes": frames}, expected_count=9,
    ) == tuple(range(1, 10))

    frames[4] = {"order": 9}
    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_input_invalid",
    ):
        long_generation._frozen_fusion_frame_orders(
            {"new_keyframes": frames}, expected_count=9,
        )


def test_h3_request_keeps_full_resolution_frames_after_fusion_proxy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = _png(width=7, height=5, bgr=(10, 20, 30))
    plan, meta = _fusion_fixture(root, first_frame=source)
    input_data = long_generation.build_prompt_fusion_input(
        root=root,
        meta=meta,
        plan=plan,
        dialogue_mode="none",
        dialogue_delivery="auto",
    )
    input_payload = json.loads(input_data)
    fusion = long_generation.FrozenPromptFusion(
        version=long_generation.PROMPT_FUSION_VERSION,
        input_path=root / "work" / "multimodal_input.json",
        input_data=input_data,
        input_sha256=hashlib.sha256(input_data).hexdigest(),
        output_path=root / "work" / "h3_prompt_plan.json",
        output_data=b"{}\n",
        output_sha256=hashlib.sha256(b"{}\n").hexdigest(),
        segments=tuple(input_payload["segments"]),
        final_prompts=("fused visual",),
    )
    fused_plan = replace(plan, prompt_fusion=fusion)

    request = long_generation._request(
        make_settings(tmp_path, autodl_art_token="art"),
        "cid",
        fused_plan,
        fused_plan.segments[0],
        "parent-request",
        "none",
        prepare_inputs=False,
    )

    assert request.keyframes == fused_plan.segments[0].keyframes
    decoded = cv2.imdecode(
        np.frombuffer(request.keyframes[0][1], np.uint8), cv2.IMREAD_COLOR,
    )
    assert decoded.shape[:2] == (5, 7)
    assert request.keyframes[0][0] != root / input_payload["segments"][0][
        "new_keyframes"
    ][0]["path"]


def test_context_ir_reuses_frozen_fusion_proxies_in_full_h3_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = _png(width=7, height=5, bgr=(10, 20, 30))
    plan, meta = _fusion_fixture(root, first_frame=source)
    input_data = long_generation.build_prompt_fusion_input(
        root=root,
        meta=meta,
        plan=plan,
        dialogue_mode="none",
        dialogue_delivery="auto",
    )
    input_payload = json.loads(input_data)
    fusion = long_generation.FrozenPromptFusion(
        version=long_generation.PROMPT_FUSION_VERSION,
        input_path=root / "work" / "multimodal_input.json",
        input_data=input_data,
        input_sha256=hashlib.sha256(input_data).hexdigest(),
        output_path=root / "work" / "h3_prompt_plan.json",
        output_data=b"{}\n",
        output_sha256=hashlib.sha256(b"{}\n").hexdigest(),
        segments=tuple(input_payload["segments"]),
        final_prompts=("fused visual",),
    )
    fused_plan = replace(plan, prompt_fusion=fusion)
    request = long_generation._request(
        make_settings(tmp_path, autodl_art_token="art"),
        "cid",
        fused_plan,
        fused_plan.segments[0],
        "parent-request",
        "none",
        prepare_inputs=False,
    )
    monkeypatch.setattr(
        image_optimization,
        "half_resolution_png",
        lambda _data: pytest.fail("Context IR must not resize Fusion proxies again"),
    )

    context_frames = long_generation._context_ir_fusion_keyframes(
        fused_plan, fused_plan.segments[0], request,
    )

    assert tuple(path for path, _data in context_frames) == tuple(
        path for path, _data in request.keyframes
    )
    assert tuple(data for _path, data in context_frames) == tuple(
        (root / frame["path"]).read_bytes()
        for frame in input_payload["segments"][0]["new_keyframes"]
    )
    assert request.keyframes[0][1] == source
    assert context_frames[0][1] != source

    swapped = list(fusion.segments[0]["new_keyframes"])
    swapped[0], swapped[1] = swapped[1], swapped[0]
    broken_segment = dict(fusion.segments[0])
    broken_segment["new_keyframes"] = swapped
    broken = replace(
        fused_plan,
        prompt_fusion=replace(fusion, segments=(broken_segment,)),
    )
    with pytest.raises(
        long_generation.LongGenerationError,
        match="prompt_fusion_input_invalid",
    ):
        long_generation._context_ir_fusion_keyframes(
            broken, broken.segments[0], request,
        )
