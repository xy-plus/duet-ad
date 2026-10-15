"""重跑监控（2026-09-29）：两个项目到终态；H3 冻结时记录各段 prompt 长度，成功后收集成片到 data-verify/rerun-20260929/。"""
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/xy/duet-ad1-h3-max-r2v")
OUT = ROOT / "data-verify/rerun-20260929"
IDS = {k: v["new_id"] for k, v in json.loads((OUT / "projects.json").read_text()).items()}
DEADLINE = time.time() + 3 * 3600

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

def record_prompt_lengths(name, pid):
    lens = {}
    for f in sorted((ROOT / "data/projects" / pid / "work/segments").glob("*/h3_prompt.txt")):
        lens[f.parent.name] = len(f.read_text())
    if lens:
        (OUT / f"prompt-lengths-{name}.json").write_text(json.dumps(lens, indent=1))
        print(f"[{time.strftime('%H:%M:%S')}] {name} prompt lengths: {lens}", flush=True)

def collect(name, pid):
    state = json.loads((ROOT / "data/projects" / pid / "state.json").read_text())
    out_path = state.get("output_path")
    src = Path(out_path) if out_path else None
    if not src or not src.exists():
        cands = sorted((ROOT / "data/projects" / pid / "output").glob("*.mp4"))
        src = cands[-1] if cands else None
    if src and src.exists():
        dst = OUT / f"{name}.mp4"
        shutil.copy2(src, dst)
        print(f"[{time.strftime('%H:%M:%S')}] {name} collected -> {dst}", flush=True)
        return True
    return False

last = {}
history = {k: [] for k in IDS}
collected, measured = set(), set()
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
        if name not in measured and stage in ("h3-max-r2v", "prompt-review", "stitch"):
            try:
                record_prompt_lengths(name, pid)
                measured.add(name)
            except Exception:
                pass
        if status not in ("succeeded", "failed"):
            done = False
        elif status == "failed" and not history[name][-1].get("error_dumped"):
            err = st.get("error")
            print(f"[{time.strftime('%H:%M:%S')}] {name} FAILED error={json.dumps(err, ensure_ascii=False)[:1500]}", flush=True)
            history[name][-1]["error_dumped"] = True
            history[name][-1]["error"] = err
        elif status == "succeeded" and name not in collected:
            if name not in measured:
                try:
                    record_prompt_lengths(name, pid)
                    measured.add(name)
                except Exception:
                    pass
            if collect(name, pid):
                collected.add(name)
    if done:
        break
    time.sleep(45)

(OUT / "monitor-history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1))
print("MONITOR DONE", {k: get(v).get("status") for k, v in IDS.items()}, "collected:", sorted(collected), flush=True)
