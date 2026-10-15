"""E2E 监控：轮询两个新项目到终态，记录节点进度与耗时。"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/xy/duet-ad1-h3-max-r2v")
IDS = {
    "eyelash-curler": "bb723c04397e439a91cac4b2f95edbee",
    "sunscreen": "13583ffa259b4cf7a82319efb7040ea3",
}
DEADLINE = time.time() + 90 * 60
OUT = ROOT / "data-verify/e2e-fresh-20260928"

token = None
for line in Path.home().joinpath(".config/duet-ad1/service.env").read_text().splitlines():
    if line.startswith("ACCESS_TOKEN="):
        token = line.split("=", 1)[1].strip()

def get(pid):
    r = urllib.request.Request(
        f"http://127.0.0.1:3212/api/conversations/{pid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read())

last = {}
history = {k: [] for k in IDS}
while time.time() < DEADLINE:
    done = True
    for name, pid in IDS.items():
        try:
            st = get(pid)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] {name} poll error: {e}", flush=True)
            done = False
            continue
        status = st.get("status")
        prog = st.get("project_progress") or {}
        stage = prog.get("stage") or prog.get("node") or ""
        snap = f"{status}|{stage}"
        if snap != last.get(name):
            last[name] = snap
            history[name].append({"t": time.strftime("%H:%M:%S"), "status": status, "stage": stage})
            print(f"[{time.strftime('%H:%M:%S')}] {name} {snap}", flush=True)
        if status not in ("succeeded", "failed"):
            done = False
        elif status == "failed" and not history[name][-1].get("error_dumped"):
            err = st.get("error")
            print(f"[{time.strftime('%H:%M:%S')}] {name} FAILED error={json.dumps(err, ensure_ascii=False)[:1500]}", flush=True)
            history[name][-1]["error_dumped"] = True
            history[name][-1]["error"] = err
    if done:
        break
    time.sleep(45)

OUT.mkdir(exist_ok=True)
(OUT / "monitor-history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1))
print("MONITOR DONE", {k: get(v).get("status") for k, v in IDS.items()}, flush=True)
