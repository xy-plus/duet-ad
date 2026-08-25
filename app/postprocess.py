"""Fail-closed, segment-scoped MediaKit -> Seedream postprocessing."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path

from app import image_optimization, mediakit, seedream, storage
from app.config import Settings
from app.sanitize import sanitize

OPTION_KEYS = ("remove_subtitle", "remove_brand", "optimize_image")
_OLD_OPTION_KEYS = frozenset({"remove_subtitle", "remove_brand"})
_OPTION_SET = frozenset(OPTION_KEYS)
_STALE_KEYS = frozenset({"change_bg", "face_hold"})
_PUBLIC_STATUSES = frozenset({"running", "done", "failed"})
_PUBLIC_STAGES = frozenset({"queued", "text", "brand", "seedream", "publishing", "done"})
_PUBLIC_ERROR_CODES = frozenset({
    "cancelled", "submission_unknown", "provider_rejected",
    "provider_protocol_error", "postprocess_artifacts_invalid",
    "postprocess_canonical_conflict", "image_optimization_prompt_invalid",
    "postprocess_receipt_invalid", "segment_failed",
})
_FRAME_ERROR_RE = re.compile(r"^frame ([A-Za-z0-9_.-]{1,128}) failed(?:$|:)")
_PUBLIC_FRAME_RE = re.compile(
    r"^(?:[A-Za-z0-9_.-]{1,128}|segments/[1-9]\d*/work/postprocessed/[A-Za-z0-9_.-]{1,128})$"
)


class PostprocessError(Exception):
    def __init__(self, status: int, detail: str | dict[str, str]) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def _structured(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _public_error(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value in _PUBLIC_ERROR_CODES:
        return value
    if isinstance(value, str):
        matched = _FRAME_ERROR_RE.match(value)
        if matched and matched.group(1) not in {".", ".."}:
            return f"frame {matched.group(1)} failed"
    return "postprocess_failed"


def public_state(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    status = value.get("status")
    raw_options = value.get("options")
    options = {
        key: raw_options.get(key) if isinstance(raw_options, dict)
        and isinstance(raw_options.get(key), bool) else False
        for key in OPTION_KEYS
    }
    raw_frames = value.get("frames")
    result = {
        "status": (
            status if isinstance(status, str) and status in _PUBLIC_STATUSES else "failed"
        ),
        "options": options,
        "frames": [
            item for item in raw_frames
            if isinstance(item, str) and _PUBLIC_FRAME_RE.match(item)
            and item.rsplit("/", 1)[-1] not in {".", ".."}
        ] if isinstance(raw_frames, list) else [],
        "error": _public_error(value.get("error")),
    }
    raw_segments = value.get("segments")
    if raw_segments is None:
        raw_segments = []
        valid_segments = True
    else:
        valid_segments = isinstance(raw_segments, list)
    indices = [
        item.get("index") for item in raw_segments
        if isinstance(item, dict)
    ] if isinstance(raw_segments, list) else []
    valid_segments = (
        valid_segments and len(indices) == len(raw_segments)
        and all(isinstance(index, int) and not isinstance(index, bool) for index in indices)
        and (
        indices == [0] or indices == list(range(1, len(indices) + 1))
        )
    )
    if valid_segments:
        for item in raw_segments:
            completed, total, revision = (
                item.get("completed_frames"), item.get("total_frames"), item.get("revision")
            )
            if (
                isinstance(completed, bool) or not isinstance(completed, int)
                or isinstance(total, bool) or not isinstance(total, int)
                or isinstance(revision, bool) or not isinstance(revision, int)
                or completed < 0 or total < 0 or completed > total or revision < 1
            ):
                valid_segments = False
                break
    result["segments"] = [
        {
            "index": item.get("index"),
            "status": (
                item.get("status")
                if isinstance(item.get("status"), str)
                and item.get("status") in _PUBLIC_STATUSES else "failed"
            ),
            "stage": (
                item.get("stage")
                if isinstance(item.get("stage"), str)
                and item.get("stage") in _PUBLIC_STAGES else "unknown"
            ),
            "completed_frames": item["completed_frames"],
            "total_frames": item["total_frames"],
            "revision": item["revision"],
            "error": _public_error(item.get("error")),
        }
        for item in raw_segments
    ] if valid_segments else []
    if not valid_segments:
        result.update(status="failed", frames=[], error="postprocess_receipt_invalid")
    return result


def _private_receipt(meta: dict) -> dict:
    raw = meta.get("_postprocess_receipt")
    expected_keys = {
        "version", "options", "model", "edit_mode", "prompt_template",
        "timeout_s", "prompts",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys or raw.get("version") != 1:
        raise PostprocessError(409, "postprocess_receipt_invalid")
    options = raw.get("options")
    post = meta.get("postprocess")
    if not isinstance(post, dict):
        raise PostprocessError(409, "postprocess_receipt_invalid")
    public_options = post.get("options")
    if (
        not isinstance(options, dict) or set(options) != _OPTION_SET
        or any(not isinstance(options[key], bool) for key in OPTION_KEYS)
        or not any(options.values())
        or not isinstance(public_options, dict) or set(public_options) != _OPTION_SET
        or any(not isinstance(public_options[key], bool) for key in OPTION_KEYS)
        or any(public_options[key] != options[key] for key in OPTION_KEYS)
    ):
        raise PostprocessError(409, "postprocess_receipt_invalid")
    timeout = raw.get("timeout_s")
    if (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout)) or timeout <= 0
    ):
        raise PostprocessError(409, "postprocess_receipt_invalid")
    private_frozen = {
        "version": 1, "model": raw.get("model"), "edit_mode": raw.get("edit_mode"),
        "prompt_template": raw.get("prompt_template"), "segments": raw.get("prompts"),
    }
    project_frozen = image_optimization.receipt(meta)
    if project_frozen is None or private_frozen != project_frozen:
        raise PostprocessError(409, "postprocess_receipt_invalid")
    return {
        **raw,
        "options": {key: options[key] for key in OPTION_KEYS},
        "prompts": [dict(item) for item in project_frozen["segments"]],
    }


def _parse_options(payload: dict) -> dict[str, bool]:
    raw = payload.get("options")
    if not isinstance(raw, dict):
        raise PostprocessError(422, "at least one option required")
    if not raw:
        raise PostprocessError(422, "at least one option required")
    if any(not isinstance(value, bool) for value in raw.values()):
        raise PostprocessError(422, "options must be booleans")
    keys = frozenset(raw)
    stale = keys - _OPTION_SET
    if stale and stale <= _STALE_KEYS:
        raise PostprocessError(409, "页面版本已更新，请刷新页面后重试。")
    if keys not in {_OLD_OPTION_KEYS, _OPTION_SET}:
        unknown = sorted(keys - _OPTION_SET)
        if unknown:
            raise PostprocessError(422, f"unknown options: {', '.join(unknown)}")
        raise PostprocessError(422, "invalid_postprocess_options")
    options = {key: bool(raw.get(key, False)) for key in OPTION_KEYS}
    if not any(options.values()):
        raise PostprocessError(422, "at least one option required")
    return options


def _capability_gate(settings: Settings, options: dict[str, bool]) -> None:
    if (options["remove_subtitle"] or options["remove_brand"]) and not settings.enable_mediakit_erase:
        raise PostprocessError(501, "MediaKit erase is disabled.")
    if options["optimize_image"] and not os.environ.get("ARK_API_KEY", "").strip():
        raise PostprocessError(501, "Seedream image optimization is disabled.")


def _options_match(previous: object, current: dict[str, bool]) -> bool:
    if not isinstance(previous, dict):
        return False
    if set(previous) and not any(previous.get(key) is True for key in OPTION_KEYS):
        return True
    return all(bool(previous.get(key, False)) == value for key, value in current.items())


def _pure_legacy(previous: object) -> bool:
    return (
        isinstance(previous, dict)
        and bool(set(previous) & _STALE_KEYS)
        and not any(previous.get(key) is True for key in OPTION_KEYS)
    )


def _clear_canonical(cdir: Path, grouped: dict[int, list[tuple[Path, Path]]]) -> None:
    for targets in grouped.values():
        destination = targets[0][1].parent
        if destination.is_dir():
            shutil.rmtree(destination)


def _group_targets(cdir: Path, meta: dict) -> dict[int, list[tuple[Path, Path]]]:
    segments = meta.get("segments")
    grouped: dict[int, list[tuple[Path, Path]]] = {}
    if isinstance(segments, list) and segments:
        for segment in segments:
            index = segment.get("index") if isinstance(segment, dict) else None
            if not isinstance(index, int) or isinstance(index, bool) or index < 1:
                raise PostprocessError(409, "artifacts not ready")
            src_dir = cdir / "work" / "segments" / str(index) / "work" / "keyframes"
            dst_dir = cdir / "work" / "segments" / str(index) / "work" / "postprocessed"
            files = sorted(path for path in src_dir.glob("*.png") if path.is_file())
            if not files:
                raise PostprocessError(409, "artifacts not ready")
            grouped[index] = [(path, dst_dir / path.name) for path in files]
    else:
        src_dir = cdir / "work" / "keyframes"
        files = sorted(path for path in src_dir.glob("*.png") if path.is_file())
        if not files:
            raise PostprocessError(409, "artifacts not ready")
        grouped[0] = [(path, cdir / "work" / "postprocessed" / path.name) for path in files]
    return grouped


def _segment_state(index: int, total: int, revision: int = 1) -> dict:
    return {
        "index": index, "status": "running", "stage": "queued",
        "completed_frames": 0, "total_frames": total,
        "revision": revision, "error": None,
    }


async def start(settings: Settings, cid: str, payload: dict,
                locks: dict[str, asyncio.Lock]) -> None:
    if set(payload) != {"confirm", "options"}:
        raise PostprocessError(422, "invalid_postprocess_request")
    if payload.get("confirm") is not True:
        raise PostprocessError(409, "confirmation required")
    options = _parse_options(payload)
    _capability_gate(settings, options)
    lock = locks.setdefault(cid, asyncio.Lock())
    async with lock:
        cdir = (settings.data_dir / cid).resolve()
        def mutate(meta: dict) -> None:
            if meta.get("schema_version") != 2:
                raise PostprocessError(409, "read_only")
            if meta.get("status") != "done":
                raise PostprocessError(409, "artifacts not ready")
            if isinstance(meta.get("generation"), dict) or meta.get("_input_owner"):
                raise PostprocessError(409, "generation_already_started")
            previous = meta.get("postprocess")
            if isinstance(previous, dict) and previous.get("status") == "running":
                raise PostprocessError(409, "already running")
            if (
                isinstance(previous, dict) and previous.get("status") in {"done", "failed"}
                and not _options_match(previous.get("options"), options)
            ):
                raise PostprocessError(409, _structured(
                    "postprocess_options_locked", "后处理选项已锁定，请刷新页面后按原选项重试。"
                ))
            grouped = _group_targets(cdir, meta)
            for frames in grouped.values():
                destination = frames[0][1].parent
                if destination.is_dir():
                    existing = sorted(
                        path.name for path in destination.glob("*.png") if path.is_file()
                    )
                    expected = sorted(canonical.name for _, canonical in frames)
                    if existing != expected and not (
                        isinstance(previous, dict)
                        and _pure_legacy(previous.get("options"))
                    ):
                        raise PostprocessError(409, "postprocess_canonical_conflict")
            if isinstance(previous, dict) and _pure_legacy(previous.get("options")):
                _clear_canonical(cdir, grouped)
            optimization = image_optimization.receipt(meta, settings)
            if optimization is None:
                raise PostprocessError(409, "image_optimization_prompt_invalid")
            private = {
                "version": 1, "options": options,
                "model": optimization["model"],
                "edit_mode": optimization["edit_mode"],
                "prompt_template": optimization["prompt_template"],
                "timeout_s": settings.seedream_timeout_s,
                "prompts": optimization["segments"],
            }
            states = [
                _segment_state(index, len(frames)) for index, frames in grouped.items()
            ]
            reuse_done = (
                isinstance(previous, dict) and previous.get("status") == "done"
                and _options_match(previous.get("options"), options)
                and all(all(canonical.is_file() for _, canonical in frames)
                        for frames in grouped.values())
            )
            if reuse_done:
                for item in states:
                    item.update(
                        status="done", stage="done",
                        completed_frames=item["total_frames"],
                    )
            meta["_image_optimization"] = optimization
            meta["_postprocess_receipt"] = private
            meta["postprocess"] = {
                "status": "done" if reuse_done else "running",
                "options": options,
                "frames": sorted(
                    _frame_ref(index, canonical.name)
                    for index, frames in grouped.items()
                    for _, canonical in frames if reuse_done
                ),
                "segments": states, "error": None,
            }

        if storage.mutate_meta(settings.data_dir, cid, mutate) is None:
            raise PostprocessError(404, "not found")
def _prompt(private: dict, index: int) -> str:
    for item in private.get("prompts", []):
        if item.get("segment_index") == index:
            return item["current"]
    raise PostprocessError(409, "image_optimization_prompt_invalid")


def _frame_ref(index: int, name: str) -> str:
    return name if index == 0 else f"segments/{index}/work/postprocessed/{name}"


def _private_dir(cdir: Path, index: int) -> Path:
    return cdir / "work" / ".postprocess-private" / str(index)


def _mutate_postprocess(settings: Settings, cid: str, mutator) -> dict | None:
    def mutate(meta: dict) -> None:
        raw = meta.get("postprocess")
        post = dict(raw) if isinstance(raw, dict) else {}
        mutator(meta, post)
        meta["postprocess"] = post

    return storage.mutate_meta(settings.data_dir, cid, mutate)


def _update_segment(settings: Settings, cid: str, index: int, **changes) -> None:
    def mutate(_meta: dict, post: dict) -> None:
        segments = [dict(item) for item in post.get("segments", [])]
        for item in segments:
            if item.get("index") == index:
                item.update(changes)
                break
        post["segments"] = segments
        post["frames"] = sorted(
            _frame_ref(item["index"], path.name)
            for item in segments if item.get("status") == "done"
            for path in _canonical_files(settings.data_dir / cid, item["index"])
        )

    _mutate_postprocess(settings, cid, mutate)


def _canonical_files(cdir: Path, index: int) -> list[Path]:
    root = (
        cdir / "work" / "postprocessed" if index == 0
        else cdir / "work" / "segments" / str(index) / "work" / "postprocessed"
    )
    return sorted(path for path in root.glob("*.png") if path.is_file()) if root.is_dir() else []


async def _mediakit_stage(settings: Settings, cdir: Path, index: int,
                          inputs: list[Path], stage: str, scene: str,
                          sem: asyncio.Semaphore) -> list[Path]:
    root = _private_dir(cdir, index) / stage
    root.mkdir(parents=True, exist_ok=True)
    outputs = [root / path.name for path in inputs]

    async def one(source: Path, output: Path) -> None:
        if output.is_file():
            return
        async with sem:
            await mediakit.erase_image(settings, cdir, source, output, True, (scene,))

    results = await asyncio.gather(
        *(one(source, output) for source, output in zip(inputs, outputs)),
        return_exceptions=True,
    )
    errors = [
        f"frame {output.name} failed: {sanitize(str(result))}"
        for output, result in zip(outputs, results) if isinstance(result, BaseException)
    ]
    if errors:
        raise PostprocessError(502, errors[0])
    return outputs


async def _seedream_stage(settings: Settings, cdir: Path, cid: str, index: int,
                          inputs: list[Path], prompt: str, model: str, mode: str,
                          timeout_s: float,
                          sem: asyncio.Semaphore) -> list[Path]:
    root = _private_dir(cdir, index) / "seedream"
    attempts = _private_dir(cdir, index) / "attempts"
    root.mkdir(parents=True, exist_ok=True)
    attempts.mkdir(parents=True, exist_ok=True)
    outputs = [root / path.name for path in inputs]
    latest = storage.load_meta(settings.data_dir, cid) or {}
    revision = next(
        (item.get("revision", 1) for item in (latest.get("postprocess") or {}).get("segments", [])
         if item.get("index") == index),
        1,
    )
    task_settings = replace(
        settings, seedream_model=model, seedream_edit_mode=mode,
        seedream_timeout_s=timeout_s,
    )

    async def call(position: int, image_inputs: list[Path], output: Path) -> None:
        if output.is_file():
            return
        async with sem:
            await seedream.edit(
                task_settings, [path.read_bytes() for path in image_inputs], prompt, output,
                receipt_path=attempts / f"{position:04d}-r{revision}.json",
            )

    if mode == "independent_parallel":
        results = await asyncio.gather(
            *(call(i, [source], output) for i, (source, output) in enumerate(zip(inputs, outputs), 1)),
            return_exceptions=True,
        )
    else:
        await call(1, inputs, outputs[0])
        results = await asyncio.gather(
            *(call(i, [source, outputs[0]], output)
              for i, (source, output) in enumerate(zip(inputs[1:], outputs[1:]), 2)),
            return_exceptions=True,
        )
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        error = errors[0]
        if isinstance(error, seedream.SeedreamError):
            raise PostprocessError(502, error.code)
        raise PostprocessError(502, sanitize(str(error)))
    return outputs


def _publish_segment(outputs: list[Path], targets: list[tuple[Path, Path]]) -> None:
    # Publish the complete directory in one rename; no partial canonical set is observable.
    if len(outputs) != len(targets) or any(not path.is_file() for path in outputs):
        raise PostprocessError(502, "postprocess_artifacts_invalid")
    destination = targets[0][1].parent
    if destination.is_dir():
        existing = sorted(path.name for path in destination.glob("*.png") if path.is_file())
        expected = sorted(canonical.name for _, canonical in targets)
        if existing == expected:
            return
        raise PostprocessError(409, "postprocess_canonical_conflict")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.publishing")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        for output, (_, canonical) in zip(outputs, targets):
            shutil.copyfile(output, temporary / canonical.name)
        os.replace(temporary, destination)
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)


async def _run_segment(settings: Settings, cid: str, cdir: Path, index: int,
                       targets: list[tuple[Path, Path]], options: dict[str, bool],
                       private: dict, mediakit_sem: asyncio.Semaphore,
                       seedream_sem: asyncio.Semaphore) -> None:
    inputs = [source for source, _ in targets]
    try:
        if options["remove_subtitle"]:
            _update_segment(settings, cid, index, stage="text")
            inputs = await _mediakit_stage(
                settings, cdir, index, inputs, "text", mediakit.TEXT_SCENE, mediakit_sem
            )
            _update_segment(settings, cid, index, completed_frames=len(inputs))
        if options["remove_brand"]:
            _update_segment(settings, cid, index, stage="brand")
            inputs = await _mediakit_stage(
                settings, cdir, index, inputs, "brand", mediakit.ICON_SCENE, mediakit_sem
            )
            _update_segment(settings, cid, index, completed_frames=len(inputs))
        if options["optimize_image"]:
            _update_segment(settings, cid, index, stage="seedream", completed_frames=0)
            inputs = await _seedream_stage(
                settings, cdir, cid, index, inputs, _prompt(private, index),
                private["model"], private["edit_mode"], private["timeout_s"], seedream_sem,
            )
            _update_segment(settings, cid, index, completed_frames=len(inputs))
        _update_segment(settings, cid, index, stage="publishing")
        _publish_segment(inputs, targets)
        _update_segment(
            settings, cid, index, status="done", stage="done",
            completed_frames=len(targets), error=None,
        )
    except asyncio.CancelledError:
        latest = storage.load_meta(settings.data_dir, cid) or {}
        ambiguous = index in _ambiguous_segments(cdir, latest.get("postprocess"))
        _update_segment(
            settings, cid, index, status="failed",
            error="submission_unknown" if ambiguous else "cancelled",
        )
        raise
    except Exception as exc:
        detail = exc.detail if isinstance(exc, PostprocessError) else sanitize(str(exc))
        _update_segment(settings, cid, index, status="failed", error=detail)


async def run_task(settings: Settings, cid: str, mediakit_sem: asyncio.Semaphore,
                   seedream_sem: asyncio.Semaphore | None = None,
                   only_segments: set[int] | None = None) -> None:
    cdir = (settings.data_dir / cid).resolve()
    seedream_sem = seedream_sem or asyncio.Semaphore(settings.seedream_concurrency)
    meta = storage.load_meta(settings.data_dir, cid)
    if meta is None:
        return
    try:
        private = _private_receipt(meta)
    except PostprocessError:
        _mutate_postprocess(
            settings, cid,
            lambda _meta, post: post.update(
                status="failed", error="postprocess_receipt_invalid"
            ),
        )
        return
    options = private["options"]
    grouped = _group_targets(cdir, meta)
    if only_segments is not None:
        grouped = {index: targets for index, targets in grouped.items() if index in only_segments}
    states = {
        item.get("index"): item.get("status")
        for item in (meta.get("postprocess") or {}).get("segments", [])
        if isinstance(item, dict)
    }
    grouped = {
        index: targets for index, targets in grouped.items()
        if states.get(index) != "done"
    }
    try:
        await asyncio.gather(*(
            _run_segment(settings, cid, cdir, index, targets, options, private,
                         mediakit_sem, seedream_sem)
            for index, targets in grouped.items()
        ))
    except asyncio.CancelledError:
        def cancel(_meta: dict, post: dict) -> None:
            unknown = any(
                item.get("error") == "submission_unknown"
                for item in post.get("segments", []) if isinstance(item, dict)
            )
            post.update(
                status="failed",
                error="submission_unknown" if unknown else "cancelled",
            )

        _mutate_postprocess(settings, cid, cancel)
        raise

    def finalize(_meta: dict, post: dict) -> None:
        segments = post.get("segments") or []
        failed = [item for item in segments if item.get("status") == "failed"]
        running = [item for item in segments if item.get("status") == "running"]
        post["status"] = "running" if running else ("failed" if failed else "done")
        post["error"] = (
            "submission_unknown"
            if any(item.get("error") == "submission_unknown" for item in failed)
            else ("segment_failed" if failed else None)
        )
        post["frames"] = sorted(
            _frame_ref(item["index"], path.name)
            for item in segments if item.get("status") == "done"
            for path in _canonical_files(cdir, item["index"])
        )

    _mutate_postprocess(settings, cid, finalize)


async def retry_segment(settings: Settings, cid: str, index: int, payload: dict,
                        locks: dict[str, asyncio.Lock]) -> None:
    if set(payload) != {"confirm", "expected_revision"}:
        raise PostprocessError(422, "invalid_postprocess_retry_request")
    if payload.get("confirm") is not True:
        raise PostprocessError(409, "confirmation required")
    expected = payload.get("expected_revision")
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise PostprocessError(422, "invalid_postprocess_retry_request")
    lock = locks.setdefault(cid, asyncio.Lock())
    async with lock:
        def mutate(meta: dict) -> None:
            private = _private_receipt(meta)
            post = dict(meta["postprocess"])
            segments = [dict(item) for item in post.get("segments", [])]
            canonical = private["options"]
            _capability_gate(settings, canonical)
            target = next((item for item in segments if item.get("index") == index), None)
            if target is None or target.get("status") != "failed":
                raise PostprocessError(409, "segment_not_retryable")
            if target.get("revision") != expected:
                raise PostprocessError(409, _structured(
                    "postprocess_revision_changed", "分段状态已更新，请刷新页面后重试。"
                ))
            target.update(
                status="running", error=None, revision=expected + 1,
                stage=target.get("stage") or "queued",
            )
            post.update(status="running", error=None, segments=segments)
            meta["postprocess"] = post

        updated = storage.mutate_meta(settings.data_dir, cid, mutate)
        if updated is None:
            raise PostprocessError(404, "not found")


def generation_keyframes(cdir: Path, meta: dict, originals: list[Path]) -> list[Path]:
    state = meta.get("postprocess")
    if state is None:
        return originals
    if not isinstance(state, dict) or state.get("status") != "done":
        raise PostprocessError(409, "postprocess_not_ready")
    frame_refs = state.get("frames")
    if not isinstance(frame_refs, list) or any(not isinstance(item, str) for item in frame_refs):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    available = set(frame_refs)
    selected = []
    work = cdir.resolve() / "work"
    for original in originals:
        source = original.resolve()
        try:
            relative = source.relative_to(work)
        except ValueError:
            raise PostprocessError(409, "postprocess_artifacts_invalid") from None
        parts = relative.parts
        if len(parts) == 2 and parts[0] == "keyframes":
            output, ref = work / "postprocessed" / source.name, source.name
        elif len(parts) == 5 and parts[0] == "segments" and parts[1].isdigit() \
                and parts[2:4] == ("work", "keyframes"):
            output = work / "segments" / parts[1] / "work" / "postprocessed" / source.name
            ref = f"segments/{parts[1]}/work/postprocessed/{source.name}"
        else:
            raise PostprocessError(409, "postprocess_artifacts_invalid")
        if ref not in available or not output.is_file():
            raise PostprocessError(409, "postprocess_artifacts_invalid")
        selected.append(output)
    if len({path.resolve() for path in selected}) != len(selected):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    return selected


_ATTEMPT_RE = re.compile(r"^\d+-r([1-9]\d*)\.json$")


def _ambiguous_segments(cdir: Path, post: object) -> set[int]:
    ambiguous: set[int] = set()
    if not isinstance(post, dict):
        return ambiguous
    current = {
        item.get("index"): (item.get("revision"), item.get("status"))
        for item in post.get("segments", []) if isinstance(item, dict)
    }
    attempts_root = cdir / "work" / ".postprocess-private"
    for attempt in attempts_root.glob("*/attempts/*.json") if attempts_root.is_dir() else ():
        matched = _ATTEMPT_RE.match(attempt.name)
        try:
            index = int(attempt.parents[1].name)
        except ValueError:
            continue
        revision, status = current.get(index, (None, None))
        if matched is None or status == "done" or int(matched.group(1)) != revision:
            continue
        try:
            payload = json.loads(attempt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {"status": "submission_unknown"}
        if payload.get("status") in {"submitting", "submission_unknown"}:
            ambiguous.add(index)
    return ambiguous


def recover_running(settings: Settings) -> list[str]:
    """Fail ambiguous Seedream submissions; return only locally safe jobs to resume."""
    jobs = []
    for meta in storage.list_conversations(settings.data_dir):
        post = meta.get("postprocess")
        if not isinstance(post, dict) or post.get("status") not in {"running", "failed"}:
            continue
        cid = meta["id"]
        try:
            private = _private_receipt(meta)
        except PostprocessError:
            _mutate_postprocess(
                settings, cid,
                lambda _meta, current: current.update(
                    status="failed", error="postprocess_receipt_invalid"
                ),
            )
            continue
        ambiguous = _ambiguous_segments(settings.data_dir / cid, post)
        if ambiguous:
            for index in ambiguous:
                _update_segment(
                    settings, cid, index, status="failed", error="submission_unknown"
                )
            _mutate_postprocess(
                settings, cid,
                lambda _meta, current: current.update(
                    status="failed", error="submission_unknown"
                ),
            )
            continue
        if post.get("status") == "failed":
            continue
        jobs.append(cid)
    return jobs
