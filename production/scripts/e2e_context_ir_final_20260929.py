"""Context IR 最终版全量 E2E（2026-09-29 第二轮，五块补全+机械追加+轮询修复）：睫毛夹项目新建跑全链，h3-context-ir 已启用。

收集：各段 context_ir.json（status/usage/清洗计数）、h3-context-ir 节点 attempt 数、
context_ir_fallback 事件、增强后 prompt 长度、Picsart 回执、成片。
产物到 data-verify/e2e-context-ir-final-20260929/。
"""
import json
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path("/home/xy/duet-ad1-h3-max-r2v")
OUT = ROOT / "data-verify/e2e-context-ir-final-20260929"
NAME = "电热睫毛夹"
OLD_ID = "b6240a721a244447abc697c55f6f9e1f"
DEADLINE = time.time() + 3 * 3600

token = None
for line in Path.home().joinpath(".config/duet-ad1/service.env").read_text().splitlines():
    if line.startswith("ACCESS_TOKEN="):
        token = line.split("=", 1)[1].strip()
assert token

def multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields:
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    for name, path, filename in files:
        ctype = "video/mp4" if filename.endswith(".mp4") else ("image/png" if filename.endswith(".png") else "image/jpeg")
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
        body += path.read_bytes()
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), boundary

def get(pid):
    r = urllib.request.Request(
        f"http://127.0.0.1:3212/api/conversations/{pid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read())

OUT.mkdir(exist_ok=True)

old_state = json.loads((ROOT / "data/projects" / OLD_ID / "state.json").read_text())
req = dict(old_state["effective_request"])
req["prompt_review"] = False
req.pop("fusion", None)
inputs = ROOT / "data/projects" / OLD_ID / "inputs"
imgs = sorted(inputs.glob("replacement-image*"))
files = [("file", inputs / "source.mp4", "source.mp4")]
for i, img in enumerate(imgs):
    fn = old_state["creation_input"]["replacement_images"][i]["filename"]
    files.append(("replacement_image", img, fn))
body, boundary = multipart(
    [("client_request_id", str(uuid.uuid4())), ("generation_request", json.dumps(req, ensure_ascii=False))],
    files,
)
r = urllib.request.Request(
    "http://127.0.0.1:3212/api/conversations",
    data=body,
    headers={"Authorization": f"Bearer {token}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
try:
    with urllib.request.urlopen(r, timeout=120) as resp:
        state = json.loads(resp.read())
except urllib.error.HTTPError as e:
    print("CREATE FAILED HTTP", e.code, e.read().decode()[:2000], flush=True)
    sys.exit(1)
pid = state["id"]
(OUT / "projects.json").write_text(json.dumps({NAME: {"new_id": pid, "old_id": OLD_ID}}, ensure_ascii=False, indent=1))
print(f"[{time.strftime('%H:%M:%S')}] created {NAME} -> {pid} ({state['status']})", flush=True)

def dump_context_ir_evidence():
    proj = ROOT / "data/projects" / pid
    report = {"segments": {}, "node_attempts": {}, "fallback_events": []}
    for seg_dir in sorted((proj / "work/segments").glob("*")):
        cir = seg_dir / "context_ir.json"
        hp = seg_dir / "h3_prompt.txt"
        entry = {}
        if cir.exists():
            d = json.loads(cir.read_text())
            entry = {
                "status": d.get("status"),
                "task_id": d.get("task_id"),
                "distilled_chars": d.get("distilled_input_chars"),
                "usage": d.get("usage"),
                "sanitized_scenetrans_removed": d.get("sanitized_scenetrans_removed"),
                "optimized_chars": len(hp.read_text()) if hp.exists() else None,
            }
            for key in ("enhanced_prompt_raw", "raw_enhanced_prompt"):
                if d.get(key):
                    (OUT / f"seg{seg_dir.name}-enhanced-raw.txt").write_text(d[key])
        if hp.exists():
            entry["h3_prompt_chars"] = len(hp.read_text())
            shutil.copy2(hp, OUT / f"seg{seg_dir.name}-final-h3-prompt.txt")
        origin = seg_dir / "h3_prompt_origin.txt"
        if origin.exists():
            entry["origin_chars"] = len(origin.read_text())
            shutil.copy2(origin, OUT / f"seg{seg_dir.name}-origin-prompt.txt")
        report["segments"][seg_dir.name] = entry
    node_logs = proj / "logs/h3-context-ir"
    if node_logs.exists():
        for scope in sorted(node_logs.glob("*")):
            report["node_attempts"][scope.name] = len(list(scope.glob("attempt-*")))
    st = json.loads((proj / "state.json").read_text())
    for ev in st.get("events", []):
        if "context_ir" in json.dumps(ev, ensure_ascii=False):
            report["fallback_events"].append(ev)
    recs = st.get("picsart_submission_receipts", {}).get("segments", {})
    report["picsart_receipts"] = {k: [r["status"] for r in v] for k, v in recs.items()}
    (OUT / "context-ir-evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] context-ir evidence: {json.dumps(report['segments'], ensure_ascii=False)}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] node attempts: {report['node_attempts']} fallback_events: {len(report['fallback_events'])} receipts: {report['picsart_receipts']}", flush=True)
    return report

def collect():
    st = json.loads((ROOT / "data/projects" / pid / "state.json").read_text())
    out_path = st.get("output_path")
    src = Path(out_path) if out_path else None
    if not src or not src.exists():
        cands = sorted((ROOT / "data/projects" / pid / "output").glob("*.mp4"))
        src = cands[-1] if cands else None
    if src and src.exists():
        dst = OUT / f"{NAME}.mp4"
        shutil.copy2(src, dst)
        print(f"[{time.strftime('%H:%M:%S')}] collected -> {dst}", flush=True)
        return True
    return False

last = None
history = []
while time.time() < DEADLINE:
    try:
        st = get(pid)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] poll error: {e}", flush=True)
        time.sleep(45)
        continue
    status = st.get("status")
    prog = st.get("project_progress") or {}
    stage = prog.get("stage") or prog.get("node") or ""
    snap = f"{status}|{stage}"
    if snap != last:
        last = snap
        history.append({"t": time.strftime("%H:%M:%S"), "status": status, "stage": stage})
        print(f"[{time.strftime('%H:%M:%S')}] {snap}", flush=True)
    if status == "failed":
        err = st.get("error")
        print(f"[{time.strftime('%H:%M:%S')}] FAILED error={json.dumps(err, ensure_ascii=False)[:1500]}", flush=True)
        history.append({"t": time.strftime("%H:%M:%S"), "error": err})
        dump_context_ir_evidence()
        break
    if status == "succeeded":
        dump_context_ir_evidence()
        collect()
        break
    time.sleep(45)

(OUT / "monitor-history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1))
print("MONITOR DONE", get(pid).get("status"), flush=True)
