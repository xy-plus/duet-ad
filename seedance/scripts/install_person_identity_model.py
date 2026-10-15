"""Explicit SFace installation; never invoked by request handling."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
LICENSE_URL = "https://raw.githubusercontent.com/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/LICENSE"
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Optional existing local model; avoids download")
    parser.add_argument("--license-source", type=Path, help="Existing official Apache license for offline installation")
    args = parser.parse_args()
    if any(path and not path.is_absolute() for path in (args.destination, args.source, args.license_source)):
        parser.error("all filesystem paths must be literal absolute paths")
    if args.source and not args.license_source:
        parser.error("offline installation requires --license-source")
    payload = args.source.read_bytes() if args.source else urllib.request.urlopen(URL, timeout=120).read()
    license_payload = args.license_source.read_bytes() if args.license_source else urllib.request.urlopen(LICENSE_URL, timeout=60).read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        parser.error("model SHA256 mismatch; destination unchanged")
    if hashlib.sha256(license_payload).hexdigest() != LICENSE_SHA256:
        parser.error("license SHA256 mismatch; destination unchanged")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        args.destination.with_suffix(".LICENSE").write_bytes(license_payload)
        os.replace(temporary, args.destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Installed SHA256 {SHA256}: {args.destination}")


if __name__ == "__main__":
    main()
