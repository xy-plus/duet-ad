"""一次性展示复刻脚本：按五个老项目原始参数在 3212 新建项目（prompt_review=false）。

五个非重复项目（2026-09-25 audio-poc 试听同一批，md5 对账确认）：
  b6240a72 电热睫毛夹 / 6ecc60ca 防晒霜 / c57d9d6c 户外灯串
  e46bf03d 庭院灯串(无替换) / 1cc3adb4 车载湿巾(无替换)
"""
import json
import sys
import urllib.request
import uuid
from pathlib import Path

ROOT = Path("/home/xy/duet-ad1-h3-max-r2v")
OLD = {
    "电热睫毛夹": "b6240a721a244447abc697c55f6f9e1f",
    "防晒霜": "6ecc60ca953147978c6926fdf3887e38",
    "户外灯串": "c57d9d6c4c924c1a90b7fff35bdc6d64",
    "庭院灯串无替换": "e46bf03dcc514d06a6d56846ddb382ab",
    "车载湿巾": "1cc3adb4e58442ea926242bc3ecb3063",
}

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

results = {}
for name, old_id in OLD.items():
    old_state = json.loads((ROOT / "data/projects" / old_id / "state.json").read_text())
    req = dict(old_state["effective_request"])
    req["prompt_review"] = False  # 新建不开 prompt_review，让管线一路跑完
    req.pop("fusion", None)  # 现行 API 白名单已不含 fusion（422 invalid_generation_request）
    inputs = ROOT / "data/projects" / old_id / "inputs"
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
        print(name, "HTTP", e.code, e.read().decode()[:2000])
        sys.exit(1)
    results[name] = {"new_id": state["id"], "status": state["status"], "old_id": old_id}
    print(name, "->", state["id"], state["status"])

out = ROOT / "data-verify/showcase-20260928"
out.mkdir(exist_ok=True)
(out / "projects.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
