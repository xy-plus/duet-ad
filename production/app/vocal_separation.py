"""Vocal separation for the vocal-separation node (audio-separator CLI 子进程).

把 work/source.wav 分离成非人声轨（BGM/音效/环境音，进混音）与人声轨
（仅调试用）。模型为 MDX-Net 类 ONNX（选型见
data-verify/audio-poc-20260925/separation-validation 的 POC 结论）。
audio-separator 硬依赖 torch，为不污染生产共享 venv，按 whisper.cpp/ASR_CLI
先例做成独立 venv 的 CLI 子进程调用（venv 与模型由
scripts/install_vocal_separation_model.py 落位）。VOCAL_SEPARATION_CLI /
VOCAL_SEPARATION_MODEL_PATH 缺失即本模块明确报错（部署错误应当暴露，
不静默降级为无音效复刻）。CLI 超时的 subprocess.TimeoutExpired 原样上抛，
节点重试体系按暂时性失败处理；非零退出是确定性失败，报错不重试。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


# POC 实测 UVR-MDX-NET-Inst_HQ_3 约 1× 实时；留足长视频余量。
CLI_TIMEOUT_S = 1800.0


class VocalSeparationError(RuntimeError):
    """人声分离失败（配置缺失、输入无效、CLI 非零退出、输出不符合两 stem 预期）。"""


def separate_vocals(
    source_wav: Path,
    accompaniment_dest: Path,
    vocals_dest: Path,
    *,
    cli_path: Path | None,
    model_path: Path | None,
) -> dict[str, Any]:
    """Split source audio into accompaniment (non-vocal) and vocals stems."""
    if cli_path is None:
        raise VocalSeparationError(
            "VOCAL_SEPARATION_CLI must be set to the audio-separator "
            "executable (scripts/install_vocal_separation_model.py)"
        )
    cli = Path(cli_path)
    if cli.is_symlink() or not cli.is_file():
        raise VocalSeparationError(f"vocal separation CLI is missing: {cli}")
    if model_path is None:
        raise VocalSeparationError(
            "VOCAL_SEPARATION_MODEL_PATH must be set to the vocal "
            "separation model file (scripts/install_vocal_separation_model.py)"
        )
    model = Path(model_path)
    if model.is_symlink() or not model.is_file():
        raise VocalSeparationError(
            f"vocal separation model is missing: {model}"
        )
    if (
        not isinstance(source_wav, Path)
        or source_wav.is_symlink()
        or not source_wav.is_file()
    ):
        raise VocalSeparationError(
            f"vocal separation source audio is invalid: {source_wav}"
        )
    accompaniment_dest.parent.mkdir(parents=True, exist_ok=True)
    vocals_dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".vocal-separation-", dir=accompaniment_dest.parent
    ) as work:
        outputs = _run_cli(cli, model, source_wav, Path(work))
        vocals_output, accompaniment_output = _classify_outputs(outputs)
        os.replace(accompaniment_output, accompaniment_dest)
        os.replace(vocals_output, vocals_dest)
    return {
        "accompaniment": accompaniment_dest,
        "vocals": vocals_dest,
        "model": model.name,
    }


def _run_cli(
    cli: Path, model: Path, source_wav: Path, work_dir: Path
) -> list[Path]:
    """Run the audio-separator CLI; returns the stem files in work_dir."""
    argv = [
        str(cli),
        str(source_wav),
        "-m", model.name,
        "--model_file_dir", str(model.parent),
        "--output_dir", str(work_dir),
        "--output_format", "WAV",
        "--sample_rate", "48000",
        "--log_level", "warning",
    ]
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=CLI_TIMEOUT_S,
    )
    if completed.returncode != 0:
        raise VocalSeparationError(
            f"vocal separation CLI failed (exit {completed.returncode}): "
            f"{completed.stderr.strip()[-512:]}"
        )
    return sorted(work_dir.glob("*.wav"))


def _classify_outputs(outputs: list[Path]) -> tuple[Path, Path]:
    """Pick (vocals, accompaniment) out of the separator's two stems.

    分离器输出文件名带 stem 标签（如 source_(Vocals)_model.wav /
    source_(Instrumental)_model.wav）；两 stem 中按 "vocal" 字样认人声轨，
    其余一轨即非人声。输出数或标签不符预期一律明确报错，不猜。
    """
    files = [output for output in outputs if Path(output).is_file()]
    if len(files) != 2:
        raise VocalSeparationError(
            f"vocal separation produced unexpected outputs: "
            f"{[str(output) for output in outputs]}"
        )
    vocals = [output for output in files if "vocal" in output.name.lower()]
    if len(vocals) != 1:
        raise VocalSeparationError(
            f"vocal separation vocals stem is ambiguous: "
            f"{[output.name for output in files]}"
        )
    (accompaniment,) = [output for output in files if output is not vocals[0]]
    return vocals[0], accompaniment
