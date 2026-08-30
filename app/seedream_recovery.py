"""One-shot Codex neutralization after an explicit Seedream policy rejection.

This is an error-recovery middleware around the existing Seedream call.  It does
not add a pipeline node: the same image-edit node either returns its normal
output or closes with the original provider rejection and durable diagnostics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
from pathlib import Path

from app import error_trace, seedream
from app.codex_runner import CodexRunner
from app.config import Settings


MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "max"
CONTENT_REJECTION_CODES = frozenset({
    "InputTextSensitiveContentDetected",
    "OutputImageSensitiveContentDetected",
})
_MAX_OUTPUT_BYTES = 128 * 1024
_FIDELITY_FIELDS = frozenset({
    "subject_stable_keys",
    "subject_object_relations",
    "action_phases_and_causality",
    "composition_and_camera",
    "environment",
    "chronology",
    "dialogue_boundaries",
    "material_bindings",
})
_JSON_BINDING_RE = re.compile(
    r'"(?:person_id|entity_id|scene_id|frame_id|source_sha256|frame_name|asset_id|'
    r'material_id|stable_key|binding_key)"\s*:\s*"([^"\\]{1,256})"'
)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_.:/-]{1,256}")
_SHA_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _protected_literals(prompt: str) -> list[str]:
    values = [*(_JSON_BINDING_RE.findall(prompt)), *(_MENTION_RE.findall(prompt))]
    values.extend(_SHA_RE.findall(prompt))
    return sorted(set(values))


def _validate_output(raw: bytes, *, original_prompt: str) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("neutralization output must be UTF-8 JSON") from None
    if not isinstance(value, dict) or set(value) != {
        "version",
        "original_prompt_sha256",
        "neutralized_prompt",
        "protected_literals",
        "semantic_fidelity",
        "changes",
    }:
        raise ValueError("neutralization output has unexpected fields")
    if value["version"] != 1 or value["original_prompt_sha256"] != _sha256(original_prompt):
        raise ValueError("neutralization output is not bound to its input")
    neutralized = value["neutralized_prompt"]
    if (
        not isinstance(neutralized, str)
        or not neutralized.strip()
        or len(neutralized.encode("utf-8")) > 64 * 1024
        or neutralized == original_prompt
    ):
        raise ValueError("neutralized_prompt violates the string contract")
    protected = _protected_literals(original_prompt)
    if value["protected_literals"] != protected:
        raise ValueError("protected literal ledger does not match the input")
    if any(literal not in neutralized for literal in protected):
        raise ValueError("neutralized_prompt changed a stable binding")
    fidelity = value["semantic_fidelity"]
    if (
        not isinstance(fidelity, dict)
        or set(fidelity) != _FIDELITY_FIELDS
        or any(item is not True for item in fidelity.values())
    ):
        raise ValueError("semantic fidelity contract is incomplete")
    changes = value["changes"]
    if not isinstance(changes, list) or not 1 <= len(changes) <= 64:
        raise ValueError("neutralization change ledger is invalid")
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"original", "neutralized"}:
            raise ValueError("neutralization change entry is invalid")
        before = change["original"]
        after = change["neutralized"]
        if (
            not isinstance(before, str)
            or not isinstance(after, str)
            or not before
            or not after
            or before == after
            or before not in original_prompt
            or after not in neutralized
        ):
            raise ValueError("neutralization change entry is not evidenced")
    return value


def _neutralization_prompt(input_name: str, protected: list[str]) -> str:
    fidelity = ", ".join(sorted(_FIDELITY_FIELDS))
    return f"""你是供应商内容审核拒绝后的提示词中性化器。只处理 {input_name} 中的 original_prompt。

目标：只把可能触发供应商内容审核的措辞改成客观、中性、非煽动的视觉表达，并最大限度逐字保留原语义。不得删除关键语义，不得新增人物、实体、动作、因果、镜头、环境、时序、台词或故事。主体 stable key、关系的主客体与方向、动作阶段和因果、构图、机位、环境、时间顺序、台词边界、素材绑定必须保持。素材引用和 protected_literals 必须逐字保留。

输出且仅输出一个 JSON 对象，禁止 Markdown 或解释。字段必须恰好是：
- version: 1
- original_prompt_sha256: 原输入中的值
- neutralized_prompt: 完整的中性化提示词；不得与原文相同
- protected_literals: 原样返回输入数组（当前为 {json.dumps(protected, ensure_ascii=False)}）
- semantic_fidelity: 恰好包含 {fidelity}，每项仅当确实保持时写 true
- changes: 1 到 64 个对象，每项字段恰为 original 和 neutralized；两段都必须逐字出现在各自完整提示词中，仅记录必要措辞替换

如果无法在不改变关键语义的前提下中性化，也不要编造或删减；进程应失败而不是发布不忠实输出。"""


def _diagnostic_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f"{receipt_path.stem}.neutralization.json")


def _retry_receipt_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f"{receipt_path.stem}.neutralized{receipt_path.suffix}")


def _write_recovery(path: Path, payload: dict) -> None:
    seedream._atomic_json(path, payload)


def _try_write_recovery(path: Path, payload: dict) -> bool:
    try:
        _write_recovery(path, payload)
        return True
    except Exception as write_error:
        try:
            error_trace.record(
                path.with_name(f"{path.stem}.write-error.json"),
                call_path=["postprocess", "seedream", path.stem, "neutralization_receipt"],
                error=write_error,
            )
        except BaseException:
            pass
        return False


def _load_recovery(path: Path, original_sha: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("neutralization receipt is unreadable") from None
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or value.get("original_prompt_sha256") != original_sha
    ):
        raise ValueError("neutralization receipt is not bound to the prompt")
    return value


def _validated_recovery_output(recovery: dict, original_prompt: str) -> dict:
    contract = {
        key: recovery.get(key)
        for key in (
            "version",
            "original_prompt_sha256",
            "neutralized_prompt",
            "protected_literals",
            "semantic_fidelity",
            "changes",
        )
    }
    return _validate_output(
        json.dumps(contract, ensure_ascii=False).encode("utf-8"),
        original_prompt=original_prompt,
    )


def _run_codex(
    settings: Settings,
    session_dir: Path,
    original_prompt: str,
) -> dict:
    protected = _protected_literals(original_prompt)
    with tempfile.TemporaryDirectory(
        prefix="seedream-neutralize-", dir="/tmp"
    ) as temporary:
        stage = Path(temporary).resolve()
        input_path = stage / "input.json"
        input_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "original_prompt_sha256": _sha256(original_prompt),
                    "original_prompt": original_prompt,
                    "protected_literals": protected,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        runner = CodexRunner(
            timeout_s=settings.codex_timeout_s,
            concurrency=1,
            model=MODEL,
            reasoning_effort=REASONING_EFFORT,
        )
        return runner.run_isolated_until_output(
            stage,
            _neutralization_prompt(input_path.name, protected),
            session_dir=session_dir,
            output_path=stage / "output.json",
            max_output_bytes=_MAX_OUTPUT_BYTES,
            validate_output=lambda raw: _validate_output(
                raw, original_prompt=original_prompt,
            ),
        )


async def edit_with_content_recovery(
    settings: Settings,
    images: list[bytes],
    prompt: str,
    out: Path,
    *,
    receipt_path: Path,
    session_dir: Path,
    transport=None,
) -> Path:
    """Run Seedream and permit exactly one policy-neutralized resubmission."""
    original_error: seedream.SeedreamError | None = None
    first_kwargs = {"receipt_path": receipt_path}
    if transport is not None:
        first_kwargs["transport"] = transport
    try:
        return await seedream.edit(
            settings, images, prompt, out,
            **first_kwargs,
        )
    except seedream.SeedreamError as caught:
        if (
            caught.code != "provider_rejected"
            or caught.provider_error_code not in CONTENT_REJECTION_CODES
        ):
            raise
        original_error = caught

    assert original_error is not None

    try:
        session = session_dir.resolve(strict=True)
        receipt = receipt_path.resolve(strict=True)
        receipt.relative_to(session)
        if not session.is_dir() or not receipt.is_file():
            raise OSError
    except (OSError, ValueError):
        raise original_error

    original_sha = _sha256(prompt)
    diagnostic = _diagnostic_path(receipt)
    retry_receipt = _retry_receipt_path(receipt)
    try:
        recovery = _load_recovery(diagnostic, original_sha)
    except ValueError:
        raise original_error
    if recovery is None:
        recovery = {
            "version": 1,
            "status": "codex_running",
            "original_prompt_sha256": original_sha,
            "neutralized_prompt_sha256": None,
            "provider_error_codes": [original_error.provider_error_code],
            "provider_error_traces": [str(
                receipt.with_suffix(".error.json").relative_to(session)
            )],
            "codex_call": {
                "call_path": ["postprocess", "seedream", receipt.stem, "neutralize"],
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "attempt": 1,
            },
        }
        if not _try_write_recovery(diagnostic, recovery):
            raise original_error
        try:
            result = await asyncio.to_thread(_run_codex, settings, session, prompt)
        except Exception as codex_error:
            recovery.update(
                status="codex_failed",
                codex_error=error_trace.exception_tree(codex_error),
            )
            _try_write_recovery(diagnostic, recovery)
            raise original_error
        neutralized = result["neutralized_prompt"]
        recovery.update(
            status="neutralized_prompt_ready",
            neutralized_prompt=neutralized,
            neutralized_prompt_sha256=_sha256(neutralized),
            semantic_fidelity=result["semantic_fidelity"],
            protected_literals=result["protected_literals"],
            changes=result["changes"],
        )
        if not _try_write_recovery(diagnostic, recovery):
            raise original_error
    elif recovery.get("status") in {
        "codex_running", "codex_failed", "provider_rejected",
    }:
        raise original_error
    try:
        neutralized = _validated_recovery_output(recovery, prompt)["neutralized_prompt"]
    except (TypeError, ValueError, json.JSONDecodeError):
        raise original_error
    if recovery.get("neutralized_prompt_sha256") != _sha256(neutralized):
        raise original_error

    retry_kwargs = {
        "receipt_path": retry_receipt,
        "max_post_attempts": 1,
    }
    if transport is not None:
        retry_kwargs["transport"] = transport
    try:
        result_path = await seedream.edit(
            settings, images, neutralized, out,
            **retry_kwargs,
        )
    except seedream.SeedreamError as retry_error:
        if retry_error.code == "provider_rejected":
            codes = list(recovery.get("provider_error_codes") or [])
            codes.append(retry_error.provider_error_code)
            traces = list(recovery.get("provider_error_traces") or [])
            traces.append(str(
                retry_receipt.with_suffix(".error.json").relative_to(session)
            ))
            recovery.update(
                status="provider_rejected",
                provider_error_codes=codes,
                provider_error_traces=traces,
            )
            _try_write_recovery(diagnostic, recovery)
        raise
    recovery["status"] = "succeeded"
    _try_write_recovery(diagnostic, recovery)
    return result_path
