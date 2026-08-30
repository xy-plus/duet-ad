"""One-shot, receipt-bound neutralization after a Seedream policy rejection.

The Codex process may rewrite only free text.  Structural semantics are frozen
by the backend, SHA-bound to the Codex response, and injected by the backend
after validation.  Durable exclusive claims make both Codex and provider
budgets single-winner across threads and processes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from app import error_trace, seedream
from app import codex_output_schemas
from app.codex_runner import CodexRunner
from app.config import Settings


MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "max"
CONTENT_REJECTION_CODES = frozenset({
    "InputTextSensitiveContentDetected",
    "OutputImageSensitiveContentDetected",
})
_MAX_OUTPUT_BYTES = 128 * 1024
_JSON_BINDING_RE = re.compile(
    r'"(?:person_id|entity_id|scene_id|frame_id|source_sha256|frame_name|asset_id|'
    r'material_id|stable_key|binding_key)"\s*:\s*"([^"\\]{1,256})"'
)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_.:/-]{1,256}")
_SHA_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_REJECTED_TEXT_KEYS = frozenset({
    "current", "default", "prompt", "text", "free_text",
    "neutralized_free_text", "description", "caption", "narrative",
    "instruction", "original_prompt", "replacement_description",
})
_STRUCTURAL_SCALAR_KEY_RE = re.compile(
    r"^(?:"
    r"(?:stable|person|entity|scene|frame|asset|material|binding|tile|subject|object|relation)_id|"
    r"(?:stable|person|entity|scene|frame|asset|material|binding|tile|subject|object|relation)_key|"
    r"(?:source|input|output|plan|continuity|execution_input|evidence|receipt)_sha256|"
    r"frame_name|predicate|state|geometry|preserve|visibility|count|quantity|number|"
    r"order|sequence|frame_index|segment_index|source_interval_index|"
    r"start|end|start_s|end_s|duration_s|timestamp|time|timing|chronology|causality|"
    r"action|phase|transition|source_transition_from_previous|"
    r"camera|camera_motion|composition|layout|position|environment|background|setting|"
    r"input_role|purpose|kind|role|owner_id|observable_person_ids"
    r")$",
    re.IGNORECASE,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _protected_literals(prompt: str) -> list[str]:
    values = [*(_JSON_BINDING_RE.findall(prompt)), *(_MENTION_RE.findall(prompt))]
    values.extend(_SHA_RE.findall(prompt))
    return sorted(set(values))


def _semantic_projection(
    value: object, *, path: str = "$", allow_scalar: bool = False,
) -> object | None:
    """Recursively retain only explicitly structural fields.

    No parent/container key can authorize an entire child object.  In
    particular, frozen frame records also contain ``default``/``current``
    provider prose; those values and all other free-form text fields are
    excluded at every depth.
    """
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for raw_key in sorted(value, key=str):
            if not isinstance(raw_key, str):
                continue
            normalized_key = raw_key.casefold()
            if (
                normalized_key in _REJECTED_TEXT_KEYS
                or "prompt" in normalized_key
                or normalized_key.endswith("_text")
            ):
                continue
            child = value[raw_key]
            child_path = f"{path}.{raw_key}"
            scalar_allowed = _STRUCTURAL_SCALAR_KEY_RE.fullmatch(raw_key) is not None
            nested = _semantic_projection(
                child, path=child_path, allow_scalar=scalar_allowed,
            )
            if nested not in (None, {}, []):
                projected[raw_key] = nested
        return projected or None
    if isinstance(value, (list, tuple)):
        projected_items = [
            item
            for index, raw in enumerate(value)
            if (
                item := _semantic_projection(
                    raw, path=f"{path}[{index}]", allow_scalar=allow_scalar,
                )
            ) is not None
        ]
        return projected_items or None
    if allow_scalar and (
        value is None or isinstance(value, (str, int, float, bool))
    ):
        return value
    return None


def freeze_semantic_contract(
    original_prompt: str,
    semantic_context: Mapping[str, Any] | None,
) -> dict:
    """Build the immutable backend authority injected after neutralization."""
    if not isinstance(original_prompt, str) or not original_prompt:
        raise ValueError("invalid original prompt")
    if semantic_context is not None and not isinstance(semantic_context, Mapping):
        raise ValueError("invalid semantic context")
    projection = _semantic_projection(dict(semantic_context or {})) or {}
    payload = {
        "version": 1,
        "original_prompt_sha256": _sha256(original_prompt),
        "protected_literals": _protected_literals(original_prompt),
        "structured_semantics": projection,
    }
    encoded = _canonical_json(payload).encode("utf-8")
    if len(encoded) > 256 * 1024:
        raise ValueError("semantic contract is too large")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def _validate_output(
    raw: bytes,
    *,
    original_prompt: str,
    semantic_contract: Mapping[str, Any],
) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("neutralization output must be UTF-8 JSON") from None
    if not isinstance(value, dict) or set(value) != {
        "version",
        "original_prompt_sha256",
        "semantic_contract_sha256",
        "neutralized_free_text",
    }:
        raise ValueError("neutralization output has unexpected fields")
    if (
        value["version"] != 1
        or value["original_prompt_sha256"] != _sha256(original_prompt)
        or value["semantic_contract_sha256"] != semantic_contract.get("sha256")
    ):
        raise ValueError("neutralization output is not bound to immutable input")
    neutralized = value["neutralized_free_text"]
    if (
        not isinstance(neutralized, str)
        or not neutralized.strip()
        or len(neutralized.encode("utf-8")) > 64 * 1024
        or neutralized == original_prompt
    ):
        raise ValueError("neutralized_free_text violates the string contract")
    return value


def _inject_semantic_contract(free_text: str, contract: Mapping[str, Any]) -> str:
    # The model never authors this block.  It is always reconstructed from the
    # frozen backend value, so deletion/reversal/count drift is mechanically
    # corrected before the one authorized provider retry.
    return (
        f"{free_text.strip()}\n\n"
        "[BACKEND_IMMUTABLE_SEMANTIC_CONTRACT; authoritative; preserve exactly]\n"
        f"semantic_contract_sha256={contract['sha256']}\n"
        f"semantic_contract={_canonical_json(contract)}"
    )


def _neutralization_prompt(input_name: str) -> str:
    return f"""你是供应商内容审核拒绝后的提示词中性化器。读取 {input_name}。

你只能修改 original_free_text 中可能触发审核的自由措辞，使其客观、中性、非煽动。semantic_contract 是后端冻结的只读权威语义，禁止改写、删减、概括或自行判断是否保持；后端会独立注入并校验它。不得新增故事或元素。

按注入的输出 Schema 返回；两个 SHA 字段原样绑定输入，neutralized_free_text 是中性化后的完整自由文本且不得与原文相同。

无法只改自由措辞时进程失败，不得伪造合同或自报语义一致。"""


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


def _load_recovery(path: Path, original_sha: str, contract_sha: str) -> dict | None:
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
        or value.get("semantic_contract_sha256") != contract_sha
    ):
        raise ValueError("neutralization receipt is not bound to the prompt")
    return value


def _validated_recovery_output(
    recovery: dict,
    original_prompt: str,
    semantic_contract: Mapping[str, Any],
) -> dict:
    contract = {
        key: recovery.get(key)
        for key in (
            "version", "original_prompt_sha256", "semantic_contract_sha256",
            "neutralized_free_text",
        )
    }
    return _validate_output(
        _canonical_json(contract).encode("utf-8"),
        original_prompt=original_prompt,
        semantic_contract=semantic_contract,
    )


def _run_codex(
    settings: Settings,
    session_dir: Path,
    original_prompt: str,
    semantic_contract: Mapping[str, Any],
) -> dict:
    with tempfile.TemporaryDirectory(prefix="seedream-neutralize-", dir="/tmp") as temporary:
        stage = Path(temporary).resolve()
        input_path = stage / "input.json"
        input_path.write_text(
            json.dumps({
                "version": 1,
                "original_prompt_sha256": _sha256(original_prompt),
                "semantic_contract_sha256": semantic_contract["sha256"],
                "original_free_text": original_prompt,
                "semantic_contract": semantic_contract,
            }, ensure_ascii=False, indent=2),
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
            _neutralization_prompt(input_path.name),
            session_dir=session_dir,
            output_path=stage / "output.json",
            max_output_bytes=_MAX_OUTPUT_BYTES,
            validate_output=lambda raw: _validate_output(
                raw,
                original_prompt=original_prompt,
                semantic_contract=semantic_contract,
            ),
            output_schema=codex_output_schemas.neutralization_schema(
                original_prompt_sha256=_sha256(original_prompt),
                semantic_contract_sha256=str(semantic_contract["sha256"]),
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
    semantic_context: Mapping[str, Any] | None = None,
    transport=None,
) -> Path:
    """Run Seedream and permit exactly one contract-bound neutralized POST."""
    original_error: seedream.SeedreamError | None = None
    try:
        return await seedream.edit(
            settings, images, prompt, out,
            receipt_path=receipt_path,
            **({"transport": transport} if transport is not None else {}),
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
        semantic_contract = freeze_semantic_contract(prompt, semantic_context)
    except (OSError, ValueError):
        raise original_error

    original_sha = _sha256(prompt)
    contract_sha = semantic_contract["sha256"]
    diagnostic = _diagnostic_path(receipt)
    retry_receipt = _retry_receipt_path(receipt)
    initial = {
        "version": 1,
        # Persist uncertainty before starting Codex.  A process crash can
        # therefore never reopen its one-shot budget.
        "status": "submission_unknown",
        "stage": "codex",
        "original_prompt_sha256": original_sha,
        "semantic_contract_sha256": contract_sha,
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
        "claimed_at_unix_ns": time.time_ns(),
    }
    try:
        recovery = _load_recovery(diagnostic, original_sha, contract_sha)
    except ValueError:
        raise seedream.SeedreamError("submission_unknown") from None
    if recovery is None:
        try:
            won_codex = seedream._exclusive_json(diagnostic, initial)
        except Exception:
            raise original_error
        if not won_codex:
            try:
                recovery = _load_recovery(diagnostic, original_sha, contract_sha)
            except ValueError:
                raise seedream.SeedreamError("submission_unknown") from None
        else:
            recovery = initial
            try:
                result = await asyncio.to_thread(
                    _run_codex, settings, session, prompt, semantic_contract,
                )
            except Exception as codex_error:
                recovery.update(
                    status="failed",
                    stage="codex",
                    codex_error=error_trace.exception_tree(codex_error),
                )
                _try_write_recovery(diagnostic, recovery)
                raise original_error
            free_text = result["neutralized_free_text"]
            neutralized = _inject_semantic_contract(free_text, semantic_contract)
            recovery.update(
                status="neutralized_prompt_ready",
                stage="provider_retry",
                neutralized_free_text=free_text,
                neutralized_prompt=neutralized,
                neutralized_prompt_sha256=_sha256(neutralized),
            )
            if not _try_write_recovery(diagnostic, recovery):
                raise original_error

    assert recovery is not None
    if (
        recovery.get("status") == "submission_unknown"
        and recovery.get("stage") != "provider_retry"
    ):
        raise seedream.SeedreamError("submission_unknown")
    if recovery.get("status") == "failed":
        raise original_error
    if recovery.get("status") == "provider_rejected":
        codes = recovery.get("provider_error_codes")
        provider_code = codes[-1] if isinstance(codes, list) and codes else None
        raise seedream.SeedreamError(
            "provider_rejected",
            provider_error_code=provider_code if isinstance(provider_code, str) else None,
        )
    try:
        validated = _validated_recovery_output(recovery, prompt, semantic_contract)
        neutralized = _inject_semantic_contract(
            validated["neutralized_free_text"], semantic_contract,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise original_error
    if (
        recovery.get("neutralized_prompt") != neutralized
        or recovery.get("neutralized_prompt_sha256") != _sha256(neutralized)
    ):
        raise original_error

    try:
        result_path = await seedream.edit(
            settings, images, neutralized, out,
            receipt_path=retry_receipt,
            max_post_attempts=1,
            **({"transport": transport} if transport is not None else {}),
        )
    except seedream.SeedreamError as retry_error:
        codes = list(recovery.get("provider_error_codes") or [])
        if retry_error.provider_error_code is not None:
            codes.append(retry_error.provider_error_code)
        traces = list(recovery.get("provider_error_traces") or [])
        traces.append(str(retry_receipt.with_suffix(".error.json").relative_to(session)))
        recovery.update(
            status=(
                "provider_rejected"
                if retry_error.code == "provider_rejected"
                else "submission_unknown"
                if retry_error.code == "submission_unknown"
                else "failed"
            ),
            stage="provider_retry",
            provider_error_codes=codes,
            provider_error_traces=traces,
            retry_error_code=retry_error.code,
        )
        _try_write_recovery(diagnostic, recovery)
        raise
    recovery.update(status="succeeded", stage="done")
    _try_write_recovery(diagnostic, recovery)
    return result_path
