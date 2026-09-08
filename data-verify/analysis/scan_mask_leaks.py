"""Scan masked segment videos for leaked faces (YuNet re-detect + gray coverage)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/xy/duet-ad1-h3-max-r2v-release/data-verify")
PROJECTS = ROOT / "projects"
OUT = ROOT / "analysis"
EVID = OUT / "evidence"
MODEL = Path("/home/xy/duet-ad1-h3-max-r2v/models/face_detection_yunet_2023mar.onnx")

SCORE = 0.6
NMS = 0.3
STEP = 1  # every frame


def gray_ratio(frame: np.ndarray) -> float:
    b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))
    m = (np.abs(r - 128) <= 14) & (np.abs(g - 128) <= 14) & (np.abs(b - 128) <= 14)
    # low chroma spread to exclude colored content that happens near 128
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    m &= spread <= 10
    return float(m.mean())


def main() -> None:
    EVID.mkdir(parents=True, exist_ok=True)
    detector = None
    report = {}
    for proj_dir in sorted(PROJECTS.iterdir()):
        pid = proj_dir.name
        segs = sorted((proj_dir / "work" / "segments").glob("*/source-faces-masked.mp4"))
        preport = {}
        for seg in segs:
            sname = seg.parent.name
            cap = cv2.VideoCapture(str(seg))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            idx = 0
            frames = []
            leak_frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % STEP == 0:
                    if detector is None:
                        detector = cv2.FaceDetectorYN.create(
                            str(MODEL), "", (frame.shape[1], frame.shape[0]),
                            SCORE, NMS, 5000)
                    detector.setInputSize((frame.shape[1], frame.shape[0]))
                    _, faces = detector.detect(frame)
                    gr = gray_ratio(frame)
                    n = 0 if faces is None else len(faces)
                    rec = {"f": idx, "t": round(idx / fps, 3), "faces": n,
                           "gray": round(gr, 4)}
                    frames.append(rec)
                    if n > 0:
                        vis = frame.copy()
                        for fc in faces:
                            x, y, w, h = (int(v) for v in fc[:4])
                            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        img = EVID / f"leak_{pid[:8]}_{sname}_f{idx:04d}.jpg"
                        cv2.imwrite(str(img), vis)
                        leak_frames.append(rec | {"img": img.name})
                idx += 1
            cap.release()
            total = len(frames)
            grays = np.array([r["gray"] for r in frames])
            # sudden drop: frame gray < 40% of rolling median (window 15), faces present elsewhere
            med = np.median(grays) if total else 0
            drops = [r for r, m in zip(frames,
                     [np.median(grays[max(0, i - 7):i + 8]) for i in range(total)])
                     if m > 0.005 and r["gray"] < 0.4 * m]
            preport[sname] = {
                "fps": fps, "frames_scanned": total,
                "median_gray": round(float(med), 4),
                "leak_count": len(leak_frames),
                "leak_ratio": round(len(leak_frames) / max(total, 1), 4),
                "leak_frames": leak_frames,
                "gray_drop_frames": drops,
            }
            print(f"{pid[:8]} {sname}: scanned={total} leaks={len(leak_frames)} "
                  f"gray_med={med:.4f} drops={len(drops)}", flush=True)
        report[pid] = preport
    (OUT / "mask_scan.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "mask_scan.json")


if __name__ == "__main__":
    sys.exit(main())
