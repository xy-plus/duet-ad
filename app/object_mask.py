"""Deterministic SAM gray-block masking for H3 reference videos.

fal SAM-3 receives all requested objects and, when a person is replaced,
``face`` in one comma-separated prompt.  Its calibrated multi-target response
is a single unlabelled per-frame union; blocks are grouped by detection call:
the person group is painted as a fully opaque solid black (0,0,0) block (any
translucency leaks source identity) and the object group as a translucent
blurred white (255,255,255) block (motion anchor); sparse/legacy anchors
without a mask use their bbox rectangle, and frames whose mask area collapses
below
``OBJECT_MASK_DEGRADED_AREA_RATIO`` of the interpolated bbox area render the
bbox alongside the fragmentary mask (degraded-frame fallback, see
``_degraded_labels``). Sparse anchors first discard local area
glitches, then are filled by local linear interpolation; a track that loses its box is held for
``OBJECT_MASK_HOLD_FRAMES`` frames.  Short runs of consecutive block-less
frames (≤ ``OBJECT_MASK_FLOW_GAP_MAX_FRAMES``) between two blocked frames are
filled by bidirectional dense-optical-flow mask propagation (see
``_flow_compensated_masks``).  The output uses the usual mp4v -> ffmpeg
H.264/yuv420p pipeline.

Pipeline integration (app/pipeline.py): ``mask-replaceable-objects`` consumes
the muted source directly.  It runs only when there are targets and FAL_KEY is
configured.  A persistent failure degrades to the unmasked video plus an
``object_mask_degraded`` log event; no new gate is added.  H3 receives one
frozen instruction that the black/white blocks conceal people and replaceable
objects, and may reconstruct them only from the descriptions/reference images.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app import media
from app.config import get_settings
from app.object_detect import ObjectDetection, select_provider


_LOGGER = logging.getLogger(__name__)

OBJECT_BLOCK_FILL = (255, 255, 255)  # 物品组纯白（BGR）块：99%+高斯模糊半透明
PERSON_BLOCK_FILL = (0, 0, 0)  # 人物组纯黑（BGR）块：100% 不透明实心（防身份泄漏）
OBJECT_MASK_HOLD_FRAMES = 12  # frames a lost object keeps its last known box
OBJECT_MASK_MAX_ATTEMPTS = 3  # retry budget for the pipeline node
OBJECT_MASK_GLITCH_WINDOW_FRAMES = 15
OBJECT_MASK_GLITCH_AREA_RATIO = 0.6
OBJECT_MASK_RLE_FEATHER_KERNEL = 7
# 短时空洞补偿：连续无块帧 ≤ 此值且前后都有有块帧时，用双向稠密光流把
# 前后边界帧的遮盖 mask warp 过来（算法选型见 _flow_compensated_masks 头
# 注释）。只补中间空洞，不向首检出帧之前/末检出帧之后延伸。
OBJECT_MASK_FLOW_GAP_MAX_FRAMES = 3
# 退化帧回退：同 label 的 rle mask 面积 < 同帧插值 bbox 面积 × 此比值时，
# 该帧视为"目标丢失的退化帧"（真实案例 bb5f36f6 seg0001 帧 200-210：
# SAM 丢失 person 只剩 hair 碎片 mask，面积比约 17%；正常帧 mask/bbox 面积
# 比 ≈99%——0.5 远在两侧分布之外，不会误判正常帧）。回退后该 label 的
# 插值 bbox 不再被 mask 抑制，与残缺 mask 并集渲染（见 _degraded_labels
# 头注释的取舍说明）。
OBJECT_MASK_DEGRADED_AREA_RATIO = 0.5


class ObjectMaskError(RuntimeError):
    """Object detection or masking could not be completed deterministically."""


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _center_distance(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    return ((a[0] + a[2] - b[0] - b[2]) ** 2 + (a[1] + a[3] - b[1] - b[3]) ** 2) / 4.0


def _without_glitch_anchors(
    track: list[tuple[int, tuple[float, float, float, float]]],
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Remove short-lived area collapses before interpolating a track."""
    retained = []
    for frame_index, bbox in track:
        nearby_areas = [
            (nearby[1][2] - nearby[1][0]) * (nearby[1][3] - nearby[1][1])
            for nearby in track
            if abs(nearby[0] - frame_index) <= OBJECT_MASK_GLITCH_WINDOW_FRAMES
        ]
        median_area = float(np.median(nearby_areas))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # A tracked object's projection cannot collapse below 60% of its
        # local median in one second and rebound; real recession is gradual,
        # so the local median follows it instead of treating it as a glitch.
        if area >= median_area * OBJECT_MASK_GLITCH_AREA_RATIO:
            retained.append((frame_index, bbox))
    return retained


def _frame_boxes(
    detections: list[ObjectDetection],
    fps: float,
    *,
    hold_frames: set[int] | None = None,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Expand sparse timestamped detections into a per-frame box list.

    Timestamps convert to frame indices via the video fps. Detections sharing
    a label are chained into tracks: each detection joins the track whose last
    box overlaps it most (IoU), ties and non-overlaps broken by nearest
    center; a detection that fits no open track starts a new one (a second
    simultaneous instance). Within a track, boxes between anchor frames are
    linearly interpolated; after the last anchor the final box is held for
    OBJECT_MASK_HOLD_FRAMES frames, then the object is treated as gone.
    ``hold_frames``（可选 out 参数）收集由续盖逻辑产生的帧号。
    """
    points = sorted(
        (
            (round(detection.timestamp_s * fps), detection.bbox_xyxy_normalized, detection.label)
            for detection in detections
        ),
        key=lambda point: point[0],
    )
    tracks: dict[str, list[list[tuple[int, tuple[float, float, float, float]]]]] = {}
    for frame_index, bbox, label in points:
        label_tracks = tracks.setdefault(label, [])
        open_tracks = [track for track in label_tracks if track[-1][0] < frame_index]
        if not open_tracks:
            label_tracks.append([(frame_index, bbox)])
            continue
        best = max(
            open_tracks,
            key=lambda track: (
                _iou(track[-1][1], bbox),
                -_center_distance(track[-1][1], bbox),
            ),
        )
        best.append((frame_index, bbox))
    boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for label_tracks in tracks.values():
        for track in label_tracks:
            track = _without_glitch_anchors(track)
            if not track:
                continue
            for (f0, b0), (f1, b1) in zip(track, track[1:]):
                # The next interval re-owns its left anchor; ending each
                # interval at f1 - 1 keeps every anchor frame painted once.
                for frame_index in range(f0, f1):
                    ratio = (frame_index - f0) / (f1 - f0)
                    box = tuple(
                        b0[axis] + (b1[axis] - b0[axis]) * ratio for axis in range(4)
                    )
                    boxes_by_frame.setdefault(frame_index, []).append(box)  # type: ignore[arg-type]
            last_frame, last_box = track[-1]
            boxes_by_frame.setdefault(last_frame, []).append(last_box)
            for frame_index in range(
                last_frame + 1, last_frame + OBJECT_MASK_HOLD_FRAMES + 1
            ):
                boxes_by_frame.setdefault(frame_index, []).append(last_box)
                if hold_frames is not None:
                    hold_frames.add(frame_index)
    return boxes_by_frame


def _pixel_box(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Convert a 0-1 normalized box to pixel coordinates, clipped to the frame."""
    x0 = max(0, min(width, int(round(bbox[0] * width))))
    y0 = max(0, min(height, int(round(bbox[1] * height))))
    x1 = max(0, min(width, int(round(bbox[2] * width))))
    y1 = max(0, min(height, int(round(bbox[3] * height))))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _apply_object_block(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    opacity: float,
    blurred_frame: np.ndarray | None = None,
    fill: tuple[int, int, int] = OBJECT_BLOCK_FILL,
) -> None:
    """Mix the group's block (object white / person black) into the box in place.

    ``opacity=1.0`` deliberately retains the direct assignment so the default
    behavior remains byte-for-byte identical to the original solid block.
    """
    x0, y0, x1, y1 = box
    region = frame[y0:y1, x0:x1]
    if opacity == 1.0:
        region[:] = fill
        return
    if blurred_frame is None:
        raise ObjectMaskError("blurred frame is required for a translucent object block")
    fill_values = np.asarray(fill, dtype=np.float64)
    region[:] = np.rint(
        fill_values * opacity + blurred_frame[y0:y1, x0:x1].astype(np.float64) * (1.0 - opacity)
    ).astype(np.uint8)


def _decode_rle_mask(rle: str, width: int, height: int) -> np.ndarray:
    """Decode fal's row-major ``(start, length)`` RLE into a bool mask.

    The calibrated fixture's first frame yields bbox IoU 0.9958 with metadata
    under C order, versus 0.2437 under Fortran order; fal is not COCO counts.
    """
    try:
        counts = [int(value) for value in rle.split()]
    except (AttributeError, ValueError) as exc:
        raise ObjectMaskError("object mask RLE counts are invalid") from exc
    if len(counts) % 2 or any(count < 0 for count in counts):
        raise ObjectMaskError("object mask RLE pairs are invalid")
    pairs = list(zip(counts[::2], counts[1::2]))
    pixel_count = width * height
    # Real fal cache sample: frame 104 ends at 589825 for a 576*1024 frame.
    # Accept that documented one-pixel overrun and clamp it explicitly.
    if any(length == 0 or start >= pixel_count + 1 for start, length in pairs):
        raise ObjectMaskError("object mask RLE dimensions do not match the frame")
    if any(next_start < start + length for (start, length), (next_start, _) in zip(pairs, pairs[1:])):
        raise ObjectMaskError("object mask RLE pairs overlap or are unordered")
    flat = np.zeros(width * height, dtype=np.uint8)
    for start, length in pairs:
        end = min(start + length, pixel_count)
        flat[start:end] = 1
    return flat.reshape((height, width)).astype(bool)


def _apply_object_rle_mask(frame: np.ndarray, mask: np.ndarray, *, opacity: float,
                           blurred_frame: np.ndarray | None = None,
                           fill: tuple[int, int, int] = OBJECT_BLOCK_FILL) -> None:
    """Feather an RLE mask and apply the group's block blend in place."""
    alpha = cv2.GaussianBlur(mask.astype(np.float32),
                             (OBJECT_MASK_RLE_FEATHER_KERNEL,) * 2, 0)[..., None]
    if opacity == 1.0:
        replacement: np.ndarray | tuple[int, int, int] = fill
    else:
        if blurred_frame is None:
            raise ObjectMaskError("blurred frame is required for a translucent object block")
        replacement = (np.asarray(fill, dtype=np.float64) * opacity
                       + blurred_frame.astype(np.float64) * (1.0 - opacity))
    frame[:] = np.rint(frame.astype(np.float64) * (1.0 - alpha)
                       + np.asarray(replacement, dtype=np.float64) * alpha).astype(np.uint8)


# ==== 短时空洞的双向光流补偿（mask propagation） ====
# 算法选型（2026-09-20 调研）：视频目标分割（VOS）对"关键帧之间 mask 空洞"
# 的标准解法是 mask propagation——用稠密光流把已有 mask 沿运动场 warp 到
# 空洞帧（VS-ReID, arXiv:1708.00197；DAVIS 2018 Mask Propagation Network；
# NeurIPS 2023 双向光流对照实验；CVAT 等标注工具的 track/interpolate 同族）。
# 复刻其最简确定性形态：空洞帧 mask = 前向 warp（前一有块帧 mask 经光流
# 拉到空洞帧）与后向 warp（后一有块帧 mask 同理）按距离线性混合后阈值化。
# 光流用 OpenCV 现成的 cv2.DISOpticalFlow（PRESET_MEDIUM，实时稠密、逆查
# 稳定），warp 用 cv2.remap 反向采样，零新依赖。光流一律在**原始画面**（未
# 画灰块）上计算——灰块会破坏运动场估计。


def _union_block_mask(
    masks: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> np.ndarray:
    """一帧的实际遮盖区域（rle mask 与 bbox 矩形的并集），uint8 0/1。"""
    union = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        union |= mask.astype(np.uint8)
    for x0, y0, x1, y1 in boxes:
        union[y0:y1, x0:x1] = 1
    return union


def _flow_warp_mask(
    source_gray: np.ndarray,
    source_mask: np.ndarray,
    target_gray: np.ndarray,
    dis: cv2.DISOpticalFlow,
) -> np.ndarray:
    """把 source 帧的 mask 经稠密光流 warp 到 target 帧坐标（反向 warping）。

    光流方向 target→source（dis.calc(target, source)），remap 时对 target
    的每个像素回采 source 中对应位置的 mask 值，得到 float 掩码图（[0,1]）。
    """
    flow = dis.calc(target_gray, source_gray, None)
    height, width = source_mask.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return cv2.remap(
        source_mask.astype(np.float32),
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _flow_compensated_masks(
    prev_gray: np.ndarray,
    prev_mask: np.ndarray,
    next_gray: np.ndarray,
    next_mask: np.ndarray,
    gap_grays: list[np.ndarray],
    dis: cv2.DISOpticalFlow,
) -> list[np.ndarray]:
    """逐空洞帧生成补偿 mask：双向 warp 按到边界的距离线性混合，≥0.5 阈值化。

    第 i 个空洞帧（0 基，共 n 个）的前向权重 (n-i)/(n+1)、后向权重
    (i+1)/(n+1)：离哪个边界近就多信哪个。返回与 gap_grays 等长的 bool mask
    列表；调用方保证空洞非空且两侧边界都存在（中间空洞才补偿）。
    """
    count = len(gap_grays)
    compensated: list[np.ndarray] = []
    for index, gap_gray in enumerate(gap_grays):
        ratio = (index + 1) / (count + 1)
        warped_prev = _flow_warp_mask(prev_gray, prev_mask, gap_gray, dis)
        warped_next = _flow_warp_mask(next_gray, next_mask, gap_gray, dis)
        blended = (1.0 - ratio) * warped_prev + ratio * warped_next
        compensated.append(blended >= 0.5)
    return compensated


# 与光流空洞补偿的分工：光流只补"整帧无块"（无 mask 且无 bbox，通常是整帧
# 目标丢失且 track 断开/超续盖窗口）；本节补"有残缺 mask 的退化帧"（mask
# 在但面积骤降），两者互斥——退化帧有 mask 所以不是空洞帧，空洞帧没有
# mask 所以无从判定退化。
def _degraded_labels(
    masks_by_label: dict[str, list[np.ndarray]],
    boxes_by_label: dict[str, list[tuple[int, int, int, int]]],
    width: int,
    height: int,
) -> set[str]:
    """逐 label 判定退化帧：该 label 的 mask 并集面积 < 同帧插值 bbox 并集
    面积 × OBJECT_MASK_DEGRADED_AREA_RATIO 时，该 label 在本帧退化。

    bbox 来源不限于两锚点插值——续盖（hold）框同样参与：它仍是该目标最近
    的已知位置，比残缺 mask 更可能覆盖全身。取舍：回退采用"bbox 与残缺
    mask 并集"而非"纯 bbox 替换"——并集同样确定（不引入 mask 是否可信的
    二次判断），且保留残缺 mask 里的真实检测细节（如头发轮廓的羽化边）。
    逐 label 独立判定：一个目标退化不影响其他目标的 mask 优先语义。
    """
    degraded: set[str] = set()
    for label, label_masks in masks_by_label.items():
        label_boxes = boxes_by_label.get(label)
        if not label_boxes:
            continue
        mask_area = int(_union_block_mask(label_masks, [], width, height).sum())
        box_area = int(_union_block_mask([], label_boxes, width, height).sum())
        if box_area > 0 and mask_area < box_area * OBJECT_MASK_DEGRADED_AREA_RATIO:
            degraded.add(label)
    return degraded


def _production_detector() -> Any:
    client = select_provider(get_settings())
    if client is None:
        raise ObjectMaskError(
            "no object detection provider is available; "
            "the object mask node must not run"
        )
    return client


def _detect_all(
    source: Path,
    target_groups: list[list[str]],
    detector: Any,
    artifacts_dir: Path | None,
) -> list[ObjectDetection]:
    """每组 targets 一次 SAM 请求（分组调用）。

    fal/wavespeed 的逗号多目标返回是逐帧合并 mask，无法归因到具体词；
    人物组与物品组必须分组渲染不同透明度（人物 100% 防身份泄漏、物品
    99%+模糊保动作锚点），所以检测也按组分开调用，detection label（=
    该组的完整逗号 prompt）天然携带组归属。每组内部的多个词仍合并为一
    次调用；每组的原始响应缓存按各自完整 prompt 指纹自然分开。空结果
    只对该组有意义，逐组记日志。
    """
    client = detector
    if artifacts_dir is not None and hasattr(detector, "with_cache"):
        client = detector.with_cache(artifacts_dir / "object-detect-response.json")
    detections: list[ObjectDetection] = []
    for group_targets in target_groups:
        group_detections = client.detect(source, group_targets)
        if not group_detections:
            _LOGGER.warning(
                "object mask: target group %r got no detections",
                ", ".join(target.strip() for target in group_targets),
            )
        detections.extend(group_detections)
    return detections


def mask_objects(
    source: Path,
    destination: Path,
    targets: list[str],
    *,
    person_targets: list[str] | None = None,
    detector: Any | None = None,
    detect_artifacts_dir: Path | None = None,
    block_opacity: float = 1.0,
    block_blur: int = 31,
    person_block_opacity: float = 1.0,
) -> dict[str, Any]:
    """Cover every detected replaceable object with its group's block
    (person: solid black, objects: translucent blurred white),
    keeping fps/resolution. ``detector`` follows the app/object_detect.py
    contract (detect(video_path, targets) -> list[ObjectDetection]); when
    omitted, the production SAM-3 client is used (wavespeed → sandbase →
    fal 链路顺序，见 object_detect.select_provider，需对应平台 key)。
    检测分组：``targets`` 是物品组（OR 合并检测词），``person_targets``
    是人物组（换人时 pipeline 显式传入 ["person", "hair"]）；两组非空时
    各发一次 detect，组归属由此显式传入，不按 label 字符串内容猜测。
    渲染按组分类：人物组 label 的 mask/bbox 用 ``person_block_opacity``
    （默认 1.0 纯填充不模糊——2% 残影即身份泄漏实测），物品组用
    ``block_opacity``/``block_blur``（99%+ 模糊保动作锚点）。
    ``detect_artifacts_dir`` 是检测产物落盘目录（项目 work 目录内）：
    解析后的契约 detections 写 object-detections.json（见 _detect_all）；
    检测原始响应的指纹缓存写实例级共享目录
    <DATA_DIR>/shared/object-detect-cache/（跨项目命中即零 HTTP，见
    object_detect 的落盘纪律注释），with_cache 传入的本目录
    object-detect-response.json 仅作项目级缓存的回退读。多目标的 SAM 结果
    无目标归属；其 detection label 是完整 prompt，所有框仍统一遮盖。
    ``block_opacity`` controls the fill contribution and defaults to the
    original fully opaque behavior.  For translucent blocks, ``block_blur``
    controls the Gaussian kernel applied once per source frame before its
    blurred residual is mixed into every block on that frame. """
    if not isinstance(source, Path) or not source.is_file():
        raise ObjectMaskError("object mask source video is invalid")
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise ObjectMaskError("object mask targets must be non-empty strings")
    person_targets = list(person_targets) if person_targets is not None else []
    if any(
        not isinstance(target, str) or not target.strip()
        for target in person_targets
    ):
        raise ObjectMaskError("object mask person targets must be non-empty strings")
    # 人物组 label = 该组完整逗号 prompt（client 端 join 的确定性结果，
    # 与 detections 里的 label 逐字一致）。
    person_label = ", ".join(target.strip() for target in person_targets) or None
    if (
        isinstance(block_opacity, bool)
        or not isinstance(block_opacity, (int, float))
        or not np.isfinite(float(block_opacity))
        or not 0.0 < float(block_opacity) <= 1.0
    ):
        raise ObjectMaskError("object mask block opacity must be a number in (0, 1]")
    block_opacity = float(block_opacity)
    if (
        isinstance(person_block_opacity, bool)
        or not isinstance(person_block_opacity, (int, float))
        or not np.isfinite(float(person_block_opacity))
        or not 0.0 < float(person_block_opacity) <= 1.0
    ):
        raise ObjectMaskError(
            "object mask person block opacity must be a number in (0, 1]"
        )
    person_block_opacity = float(person_block_opacity)
    if isinstance(block_blur, bool) or not isinstance(block_blur, int) or block_blur < 1 or block_blur % 2 != 1:
        raise ObjectMaskError("object mask block blur must be a positive odd integer")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ObjectMaskError(f"could not open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if width < 1 or height < 1 or fps <= 0:
        capture.release()
        raise ObjectMaskError("object mask source metadata is incomplete")
    target_groups = [group for group in (targets, person_targets) if group]
    if detector is None and target_groups:
        detector = _production_detector()
    detections = (
        _detect_all(source, target_groups, detector, detect_artifacts_dir)
        if target_groups
        else []
    )
    if detect_artifacts_dir is not None and target_groups:
        detect_artifacts_dir.mkdir(parents=True, exist_ok=True)
        (detect_artifacts_dir / "object-detections.json").write_text(
            json.dumps(
                {
                    "targets": list(targets),
                    "person_targets": list(person_targets),
                    "detections": [
                        {
                            "label": detection.label,
                            "timestamp_s": detection.timestamp_s,
                            "bbox_xyxy_normalized": list(
                                detection.bbox_xyxy_normalized
                            ),
                            "rle": detection.rle,
                        }
                        for detection in detections
                    ],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    hold_frame_set: set[int] = set()
    # 逐 label 维护 track 结构（_frame_boxes 内部本就按 label 分组，逐 label
    # 调用结果与合并调用一致）：退化帧回退要按 label 独立比较 mask 与 bbox
    # 面积，渲染时按 label 决定是否保留该 label 的 bbox。
    detection_labels = list(dict.fromkeys(d.label for d in detections))
    boxes_by_frame_per_label = {
        label: _frame_boxes(
            [d for d in detections if d.label == label],
            fps,
            hold_frames=hold_frame_set,
        )
        for label in detection_labels
    }
    rles_by_frame_label: dict[int, dict[str, list[str]]] = {}
    for detection in detections:
        if detection.rle is not None:
            frame_key = round(detection.timestamp_s * fps)
            rles_by_frame_label.setdefault(frame_key, {}).setdefault(
                detection.label, []
            ).append(detection.rle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".objects-masked-",
        suffix=".mp4",
        dir=destination.parent,
        delete=False,
    ) as handle:
        raw = Path(handle.name)
    stats: dict[str, Any] = {
        "frames": 0, "frames_with_block": 0, "blocks_total": 0,
        "flow_compensated_frames": 0, "degraded_bbox_fallback_frames": 0,
        "person_frames_with_block": 0, "object_frames_with_block": 0,
    }
    try:
        writer = cv2.VideoWriter(
            str(raw),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ObjectMaskError("could not open the object mask writer")
        # 无块帧先缓冲不写：等到下一个有块帧（或视频结束）时，若空洞 ≤
        # OBJECT_MASK_FLOW_GAP_MAX_FRAMES 且两侧都有有块帧，用双向光流把
        # 边界帧的遮盖 mask warp 过来补上；否则原样写出（首尾不延伸）。
        frame_index = 0
        dis_flow: cv2.DISOpticalFlow | None = None
        pending_gap: list[np.ndarray] = []
        # 最近有块帧的（原始灰度图, 遮盖 union mask）；光流必须用画灰块之前
        # 的原图，灰块会破坏运动场估计。第三个元素是该帧是否画了人物组块
        # （光流补偿帧的分类参数按边界帧组归属决定，见循环内注释）。
        last_blocked: tuple[np.ndarray, np.ndarray, bool] | None = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            masks_by_label: dict[str, list[np.ndarray]] = {}
            for label, label_rles in (
                rles_by_frame_label.get(frame_index) or {}
            ).items():
                masks_by_label[label] = [
                    _decode_rle_mask(rle, width, height) for rle in label_rles
                ]
            boxes_by_label: dict[str, list[tuple[int, int, int, int]]] = {}
            for label, per_label_boxes in boxes_by_frame_per_label.items():
                pixel_boxes = [
                    pixel_box
                    for bbox in per_label_boxes.get(frame_index, [])
                    if (pixel_box := _pixel_box(bbox, width, height)) is not None
                ]
                if pixel_boxes:
                    boxes_by_label[label] = pixel_boxes
            # 退化帧回退（_degraded_labels 头注释）：mask 面积骤降的 label
            # 不再"mask 抛弃 bbox"，插值 bbox 与残缺 mask 并集渲染兜底全身；
            # 其余 label 维持 mask 优先（real anchors supersede bbox）。
            fallback_labels = _degraded_labels(
                masks_by_label, boxes_by_label, width, height
            )
            if fallback_labels:
                stats["degraded_bbox_fallback_frames"] += 1
            # 逐 label 分类渲染：人物组 label 用 person_block_opacity +
            # 纯黑实心块（默认 1.0 不模糊，防身份泄漏），物品组 label 用
            # block_opacity/block_blur + 纯白半透明模糊块（保动作锚点）。
            label_runs: list[
                tuple[
                    str, list[np.ndarray], list[tuple[int, int, int, int]],
                    float, tuple[int, int, int],
                ]
            ] = []
            for label in detection_labels:
                label_masks = masks_by_label.get(label, [])
                label_boxes = boxes_by_label.get(label, [])
                if label_masks and label not in fallback_labels:
                    label_boxes = []
                if not label_masks and not label_boxes:
                    continue
                if label == person_label:
                    label_runs.append((
                        label, label_masks, label_boxes,
                        person_block_opacity, PERSON_BLOCK_FILL,
                    ))
                else:
                    label_runs.append((
                        label, label_masks, label_boxes,
                        block_opacity, OBJECT_BLOCK_FILL,
                    ))
            masks = [mask for _l, ms, _b, _o, _f in label_runs for mask in ms]
            boxes = [box for _l, _m, bs, _o, _f in label_runs for box in bs]
            painted_person = any(
                label == person_label for label, _m, _b, _o, _f in label_runs
            )
            painted_object = any(
                label != person_label for label, _m, _b, _o, _f in label_runs
            )
            stats["frames"] += 1
            frame_index += 1
            if not label_runs:
                pending_gap.append(frame)
                continue
            if pending_gap:
                gap_frames, pending_gap = pending_gap, []
                if (
                    last_blocked is not None
                    and len(gap_frames) <= OBJECT_MASK_FLOW_GAP_MAX_FRAMES
                ):
                    if dis_flow is None:
                        dis_flow = cv2.DISOpticalFlow.create(
                            cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
                        )
                    prev_gray, prev_mask, prev_had_person = last_blocked
                    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    next_mask = _union_block_mask(masks, boxes, width, height)
                    gap_grays = [
                        cv2.cvtColor(gap_frame, cv2.COLOR_BGR2GRAY)
                        for gap_frame in gap_frames
                    ]
                    # 补偿 mask 是边界帧的并集（可能混两类 label）：任一
                    # 边界含人物组块即按人物参数渲染（身份泄漏宁可从严）。
                    compensated_person = prev_had_person or painted_person
                    compensated_opacity = (
                        person_block_opacity if compensated_person else block_opacity
                    )
                    compensated_fill = (
                        PERSON_BLOCK_FILL if compensated_person else OBJECT_BLOCK_FILL
                    )
                    for gap_frame, compensated_mask in zip(
                        gap_frames,
                        _flow_compensated_masks(
                            prev_gray, prev_mask,
                            next_gray, next_mask,
                            gap_grays, dis_flow,
                        ),
                    ):
                        # 补偿帧与检测帧走同一渲染路径（feather + 透明度混合）
                        blurred_gap = (
                            cv2.GaussianBlur(gap_frame, (block_blur, block_blur), 0)
                            if compensated_opacity < 1.0
                            else None
                        )
                        _apply_object_rle_mask(
                            gap_frame, compensated_mask,
                            opacity=compensated_opacity, blurred_frame=blurred_gap,
                            fill=compensated_fill,
                        )
                        stats["blocks_total"] += 1
                        stats["frames_with_block"] += 1
                        stats["flow_compensated_frames"] += 1
                        stats[
                            "person_frames_with_block"
                            if compensated_person
                            else "object_frames_with_block"
                        ] += 1
                        writer.write(gap_frame)
                else:
                    for gap_frame in gap_frames:
                        writer.write(gap_frame)
            stats["blocks_total"] += len(boxes) + len(masks)
            stats["frames_with_block"] += 1
            if painted_person:
                stats["person_frames_with_block"] += 1
            if painted_object:
                stats["object_frames_with_block"] += 1
            blurred_frame = (
                cv2.GaussianBlur(frame, (block_blur, block_blur), 0)
                if any(opacity < 1.0 for _l, _m, _b, opacity, _f in label_runs)
                else None
            )
            source_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            union_mask = _union_block_mask(masks, boxes, width, height)
            for _label, run_masks, run_boxes, opacity, fill in label_runs:
                for mask in run_masks:
                    _apply_object_rle_mask(frame, mask, opacity=opacity,
                                           blurred_frame=blurred_frame, fill=fill)
                for box in run_boxes:
                    _apply_object_block(
                        frame, box, opacity=opacity, blurred_frame=blurred_frame,
                        fill=fill,
                    )
            writer.write(frame)
            last_blocked = (source_gray, union_mask, painted_person)
        # 视频以空洞结尾（无后边界）：不向后延伸，原样写出。
        for gap_frame in pending_gap:
            writer.write(gap_frame)
        capture.release()
        writer.release()
        if stats["frames"] < 1:
            raise ObjectMaskError("object mask source has no frames")
        # Re-encode to H.264/yuv420p for provider compatibility.
        media.run_command(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(raw),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-an", "-movflags", "+faststart",
                str(destination),
            ],
            timeout=300,
        )
    finally:
        raw.unlink(missing_ok=True)
    painted = set(range(stats["frames"]))
    anchor_frame_set = {
        round(detection.timestamp_s * fps) for detection in detections
    }
    stats["anchor_frames"] = len(anchor_frame_set & painted)
    stats["hold_frames"] = len(hold_frame_set & painted)
    stats["interpolated_frames"] = (
        stats["frames_with_block"] - stats["anchor_frames"] - stats["hold_frames"]
    )
    labels = list(dict.fromkeys(detection.label for detection in detections))
    if not labels:
        labels = [", ".join(target.strip() for target in targets)] if targets else []
    stats["targets"] = {
        label: {
            "detections": sum(1 for detection in detections if detection.label == label),
            "detected_frames": len({
                round(detection.timestamp_s * fps)
                for detection in detections
                if detection.label == label
            } & painted),
        }
        for label in labels
    }
    return stats
