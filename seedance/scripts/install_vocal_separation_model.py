"""Explicit vocal-separation installation; never invoked by request handling.

两件事（都是部署步骤，按需分步或一次跑完）：

1. 创建独立 venv（固定绝对路径 /home/xy/duet-ad1-h3-max-r2v/.venv-separation）
   并安装 audio-separator[cpu] + CPU-only torch（pytorch cpu index）。
   audio-separator 硬依赖 torch，不进生产共享 venv；vocal-separation 节点
   以 CLI 子进程调用本脚本产出的可执行文件。
2. 下载人声分离 ONNX 模型（UVR-MDX-NET-Inst_HQ_3，POC 定稿选型，见
   data-verify/audio-poc-20260925/separation-validation）到 models/，
   带 SHA256 校验与原子落盘。

结束后打印应写入配置的 VOCAL_SEPARATION_CLI / VOCAL_SEPARATION_MODEL_PATH。
用法（路径必须逐字写绝对路径）：

    python /home/xy/duet-ad1-h3-max-r2v/scripts/install_vocal_separation_model.py \
        [--destination /home/xy/duet-ad1-h3-max-r2v/models/UVR-MDX-NET-Inst_HQ_3.onnx] \
        [--source /abs/local/model.onnx] [--url <https-url>] [--sha256 <hex>] \
        [--skip-venv] [--skip-model]
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

# POC 定稿（data-verify/audio-poc-20260925/separation-validation，5/5 项目
# whisper 复扫伴奏无残留人声）；SHA256 实测自 POC 已下载文件。
DEFAULT_MODEL_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx"
DEFAULT_MODEL_SHA256 = "317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc"
DEFAULT_DESTINATION = Path(
    "/home/xy/duet-ad1-h3-max-r2v/models/UVR-MDX-NET-Inst_HQ_3.onnx"
)
SEPARATION_VENV = Path("/home/xy/duet-ad1-h3-max-r2v/.venv-separation")
AUDIO_SEPARATOR_VERSION = "0.47.0"
# POC 实测可用版本（data-verify/audio-poc-20260925）：pin 死避免 pip 解析
# 到未验证的新版本。torchvision 不由本脚本安装（audio-separator[cpu] 不
# 依赖它），无需 pin。
TORCH_VERSION = "2.14.0+cpu"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def install_venv(venv: Path) -> Path:
    """创建分离 venv 并装 CPU-only torch + audio-separator；返回 CLI 路径。"""
    python = venv / "bin" / "python"
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    # 先装 CPU-only torch（pytorch cpu index，pin POC 实测版本），再装
    # audio-separator[cpu]（其 torch 依赖已满足，不会拉回 CUDA 轮）。
    subprocess.run(
        [str(python), "-m", "pip", "install",
         "--index-url", PYTORCH_CPU_INDEX, f"torch=={TORCH_VERSION}"],
        check=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install",
         f"audio-separator[cpu]=={AUDIO_SEPARATOR_VERSION}"],
        check=True,
    )
    cli = venv / "bin" / "audio-separator"
    if not cli.is_file():
        raise RuntimeError(f"audio-separator CLI missing after install: {cli}")
    return cli


def install_model(destination: Path, source: Path | None, url: str, sha256: str) -> None:
    payload = (
        source.read_bytes()
        if source
        else urllib.request.urlopen(url, timeout=600).read()
    )
    if hashlib.sha256(payload).hexdigest() != sha256.lower().strip():
        raise SystemExit("model SHA256 mismatch; destination unchanged")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Installed SHA256 {sha256}: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=DEFAULT_DESTINATION,
        help="Model install path (default: POC-pinned constant)",
    )
    parser.add_argument(
        "--source", type=Path, help="Optional existing local model; avoids download"
    )
    parser.add_argument(
        "--url", default=DEFAULT_MODEL_URL, help="Model download URL (default: POC-pinned constant)"
    )
    parser.add_argument(
        "--sha256",
        default=DEFAULT_MODEL_SHA256,
        help="Expected model SHA256 hex digest (default: POC-pinned constant)",
    )
    parser.add_argument("--skip-venv", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    if any(
        path is not None and not Path(path).is_absolute()
        for path in (args.destination, args.source)
    ):
        parser.error("all filesystem paths must be literal absolute paths")
    cli = SEPARATION_VENV / "bin" / "audio-separator"
    if not args.skip_venv:
        cli = install_venv(SEPARATION_VENV)
        print(f"Installed audio-separator CLI: {cli}")
    if not args.skip_model:
        install_model(args.destination, args.source, args.url, args.sha256)
    print("\n应写入配置（.env / service.env）：")
    print(f"VOCAL_SEPARATION_CLI={cli}")
    print(f"VOCAL_SEPARATION_MODEL_PATH={args.destination}")


if __name__ == "__main__":
    main()
