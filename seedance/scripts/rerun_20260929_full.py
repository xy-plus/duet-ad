"""睫毛夹全量重跑（2026-09-29 第二轮）：按用户原项目 b6240a72 参数新建，完整链路从头跑。

背景：验证 e0d87fd 部分可见人物兜底在生产全链路的表现（上一轮 0fb88d02 跑在旧 skill 上，
其 person-03 眼部极特写在新 skill 下需走兜底条款）。prompt_review=false。
成片收集到 data-verify/rerun-20260929-full/。
"""
import json
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path("/home/xy/duet-ad1-h3-max-r2v")
OUT = ROOT / "data-verify/rerun-20260929-full"
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

# ---- create ----
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

# ---- monitor ----
def record_prompt_lengths():
    lens = {}
    for f in sorted((ROOT / "data/projects" / pid / "work/segments").glob("*/h3_prompt.txt")):
        lens[f.parent.name] = len(f.read_text())
    if lens:
        (OUT / f"prompt-lengths-{NAME}.json").write_text(json.dumps(lens, indent=1))
        print(f"[{time.strftime('%H:%M:%S')}] prompt lengths: {lens}", flush=True)

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
measured = False
collected = False
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
    if not measured and stage in ("h3-max-r2v", "prompt-review", "stitch"):
        try:
            record_prompt_lengths()
            measured = True
        except Exception:
            pass
    if status == "failed":
        err = st.get("error")
        print(f"[{time.strftime('%H:%M:%S')}] FAILED error={json.dumps(err, ensure_ascii=False)[:1500]}", flush=True)
        history.append({"t": time.strftime("%H:%M:%S"), "error": err})
        break
    if status == "succeeded":
        if not measured:
            try:
                record_prompt_lengths()
            except Exception:
                pass
        collected = collect()
        break
    time.sleep(45)

(OUT / "monitor-history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1))
print("MONITOR DONE", get(pid).get("status"), "collected:", collected, flush=True)
