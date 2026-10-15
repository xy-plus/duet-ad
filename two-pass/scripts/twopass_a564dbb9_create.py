"""一次性脚本：按 a564dbb9 的原始参数在两次出片实验实例（3262）新建复刻项目。"""
import json
import sys
import urllib.request
import uuid
from pathlib import Path

ROOT = Path("/home/xy/duet-ad1-h3-max-r2v")
OLD_ID = "a564dbb97e3148988f3df4d8d48571da"
BASE = "http://127.0.0.1:3262"

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

old_state = json.loads((ROOT / "data/projects" / OLD_ID / "state.json").read_text())
req = dict(old_state["effective_request"])
req["prompt_review"] = False
req.pop("fusion", None)  # 旧版（7abb4d6）创建白名单不含 fusion（app/main.py:274-277）
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
    f"{BASE}/api/conversations",
    data=body,
    headers={"Authorization": f"Bearer {token}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
try:
    with urllib.request.urlopen(r, timeout=120) as resp:
        state = json.loads(resp.read())
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:2000])
    sys.exit(1)
out = {"new_id": state["id"], "status": state["status"], "old_id": OLD_ID}
print(json.dumps(out, ensure_ascii=False))
(ROOT / "data-verify/twopass-a564dbb9-20260929").mkdir(exist_ok=True)
(ROOT / "data-verify/twopass-a564dbb9-20260929/create.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
