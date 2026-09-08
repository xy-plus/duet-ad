"""Detect faces on source.mp4, verify the masked video has gray coverage there."""
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
# same padding as face_mask.py so we test the area that should be covered
PAD_RATIO, HAIR_PAD, CHIN_PAD = 0.45, 0.75, 0.5


def boxes_for(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    out = []
    if faces is None:
        return out
    for f in faces:
        x, y, fw, fh = (float(v) for v in f[:4])
        x0 = max(0, int(round(x - fw * PAD_RATIO)))
        y0 = max(0, int(round(y - fh * HAIR_PAD)))
        x1 = min(w, int(round(x + fw + fw * PAD_RATIO)))
        y1 = min(h, int(round(y + fh + fh * CHIN_PAD)))
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1))
    return out


def gray_frac(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    b, g, r = (region[:, :, i].astype(np.int16) for i in range(3))
    m = (np.abs(r - 128) <= 16) & (np.abs(g - 128) <= 16) & (np.abs(b - 128) <= 16)
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    m &= spread <= 12
    return float(m.mean())


def main():
    EVID.mkdir(parents=True, exist_ok=True)
    detector = None
    report = {}
    for proj_dir in sorted(p for p in PROJECTS.iterdir() if p.is_dir()):
        pid = proj_dir.name
        preport = {}
        for seg_dir in sorted((proj_dir / "work" / "segments").glob("*")):
            src = seg_dir / "source.mp4"
            masked = seg_dir / "source-faces-masked.mp4"
            if not src.is_file() or not masked.is_file():
                continue
            sname = seg_dir.name
            cap_s = cv2.VideoCapture(str(src))
            cap_m = cv2.VideoCapture(str(masked))
            fps = cap_s.get(cv2.CAP_PROP_FPS) or 30.0
            idx = 0
            leaks = []
            n_face_frames = 0
            while True:
                ok_s, fs = cap_s.read()
                ok_m, fm = cap_m.read()
                if not ok_s or not ok_m:
                    break
                if detector is None:
                    detector = cv2.FaceDetectorYN.create(
                        str(MODEL), "", (fs.shape[1], fs.shape[0]), SCORE, NMS, 5000)
                boxes = boxes_for(detector, fs)
                if boxes:
                    n_face_frames += 1
                for (x0, y0, x1, y1) in boxes:
                    gf = gray_frac(fm[y0:y1, x0:x1])
                    if gf < 0.5:
                        vis_s = fs.copy()
                        cv2.rectangle(vis_s, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        vis_m = fm.copy()
                        cv2.rectangle(vis_m, (x0, y0), (x1, y1), (0, 0, 255), 2)
                        combo = np.hstack([vis_s, vis_m])
                        img = EVID / f"srcleak_{pid[:8]}_{sname}_f{idx:04d}.jpg"
                        cv2.imwrite(str(img), combo)
                        leaks.append({"f": idx, "t": round(idx / fps, 3),
                                      "gray_in_box": round(gf, 3),
                                      "box": [x0, y0, x1, y1], "img": img.name})
                idx += 1
            cap_s.release(); cap_m.release()
            preport[sname] = {"frames": idx, "face_frames_src": n_face_frames,
                              "true_leaks": leaks}
            print(f"{pid[:8]} {sname}: frames={idx} src_face_frames={n_face_frames} "
                  f"true_leaks={len(leaks)}", flush=True)
        report[pid] = preport
    (OUT / "mask_true_leaks.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "mask_true_leaks.json")


if __name__ == "__main__":
    sys.exit(main())
