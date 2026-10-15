"""两次出片复刻监控：轮询 3262 实例上的复刻项目到终态，成片收进 data-verify/twopass-a564dbb9-20260929/。"""
import json
import shutil
import time
import urllib.request
from pathlib import Path

WORKTREE = Path("/home/xy/duet-ad1-h3-max-r2v/.worktree/two-pass-h3")
OUT = Path("/home/xy/duet-ad1-h3-max-r2v/data-verify/twopass-a564dbb9-20260929")
BASE = "http://127.0.0.1:3262"
DEADLINE = time.time() + 3 * 3600

token = None
for line in Path.home().joinpath(".config/duet-ad1/service.env").read_text().splitlines():
    if line.startswith("ACCESS_TOKEN="):
        token = line.split("=", 1)[1].strip()

PID = json.loads((OUT / "create.json").read_text())["new_id"]

def get():
    r = urllib.request.Request(
        f"{BASE}/api/conversations/{PID}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read())

def collect():
    state = json.loads((WORKTREE / "data/projects" / PID / "state.json").read_text())
    out_path = state.get("output_path")
    src = Path(out_path) if out_path else None
    if not src or not src.exists():
        cands = sorted((WORKTREE / "data/projects" / PID / "output").glob("*.mp4"))
        src = cands[-1] if cands else None
    if src and src.exists():
        dst = OUT / "a564dbb9-twopass.mp4"
        shutil.copy2(src, dst)
        print(f"[{time.strftime('%H:%M:%S')}] collected -> {dst}", flush=True)
        return True
    print(f"[{time.strftime('%H:%M:%S')}] succeeded but no mp4 found", flush=True)
    return False

last = None
history = []
collected = False
while time.time() < DEADLINE:
    try:
        st = get()
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
        history.append({"error": err})
        break
    if status == "succeeded":
        collected = collect()
        break
    time.sleep(45)

(OUT / "monitor-history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1))
print("MONITOR DONE", status, "collected:", collected, flush=True)
