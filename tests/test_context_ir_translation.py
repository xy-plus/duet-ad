import hashlib
import json

import httpx

from app import context_ir_translation


def test_translation_is_cached_by_source_hash_and_never_mutates_h3_state(tmp_path):
    prompt = "A woman opens the box.\nThe camera moves closer."
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    attempt = tmp_path / ".h3" / "attempts" / "000001" / "attempt.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text('{"status":"ready_for_h3"}\n', encoding="utf-8")
    before = attempt.read_bytes()
    receipt = tmp_path / "prepared_input.json"
    receipt.write_text('{"binding":"immutable"}\n', encoding="utf-8")
    receipt_before = receipt.read_bytes()
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": "一名女子打开盒子。\n镜头逐渐靠近。"},
                }],
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = context_ir_translation.translate(
            root=tmp_path,
            prompt=prompt,
            source_sha256=digest,
            api_key="secret",
            model="MiniMax-M2.7",
            timeout_s=30,
            client=client,
        )
        second = context_ir_translation.translate(
            root=tmp_path,
            prompt=prompt,
            source_sha256=digest,
            api_key="secret",
            model="MiniMax-M2.7",
            timeout_s=30,
            client=client,
        )

    assert first == second
    assert first.language == "zh-CN"
    assert first.translation == "一名女子打开盒子。\n镜头逐渐靠近。"
    assert len(requests) == 1
    assert requests[0]["model"] == "MiniMax-M2.7"
    assert prompt in requests[0]["messages"][1]["content"]
    assert attempt.read_bytes() == before
    assert receipt.read_bytes() == receipt_before
    assert not list((tmp_path / ".h3").rglob("*translation*"))
    assert list((tmp_path / "work" / "context_ir_translations").glob("*.json"))


def test_translation_rejects_source_hash_drift_before_provider_call(tmp_path):
    called = False

    def handler(_request):
        nonlocal called
        called = True
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        try:
            context_ir_translation.translate(
                root=tmp_path,
                prompt="source",
                source_sha256="0" * 64,
                api_key="secret",
                model="MiniMax-M2.7",
                timeout_s=30,
                client=client,
            )
        except context_ir_translation.TranslationError as exc:
            assert exc.code == "context_ir_translation_source_mismatch"
        else:
            raise AssertionError("hash drift must fail closed")
    assert called is False
