import hashlib
import json

import pytest

from app import minimal_creation


def _request(**overrides):
    request = {
        "version": 1,
        "output": {
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "fit_mode": "auto",
        },
        "processing": {
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {
            "mode": "auto_rewrite",
            "target_language": "  日语  ",
        },
        "replacement_guidance": None,
    }
    request.update(overrides)
    return request


def _parse(value):
    return minimal_creation.parse_generation_request(
        json.dumps(value, ensure_ascii=False)
    )


def _assert_error(code, action):
    with pytest.raises(minimal_creation.MinimalCreationError) as captured:
        action()
    assert captured.value.code == code
    assert captured.value.status_code == 422
    return captured.value


def test_capability_is_the_complete_fixed_v1_object_and_is_fresh():
    expected = {
        "supported": True,
        "version": 1,
        "endpoint": "/api/conversations",
        "encoding": "multipart/form-data",
        "request_field": "generation_request",
        "replacement_image_field": "replacement_image",
        "aspect_ratios": ["16:9", "9:16"],
        "resolutions": ["480p", "768p"],
        "defaults": {
            "fit_mode": "auto",
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {
            "mode": "auto_rewrite",
            "translation": True,
        },
        "replacement": {
            "supported": True,
            "accept": ["image/jpeg", "image/png", "image/webp"],
            "max_bytes": 10485760,
            "max_instruction_chars": 1000,
        },
    }
    first = minimal_creation.capability()
    assert first == expected
    first["defaults"]["remove_logo"] = False
    assert minimal_creation.capability() == expected


def test_parse_normalizes_public_request_and_maps_logo_only_after_boundary():
    parsed = _parse(_request())

    assert parsed.effective_request == {
        "version": 1,
        "output": {
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "fit_mode": "auto",
        },
        "processing": {
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {
            "mode": "auto_rewrite",
            "target_language": "日语",
        },
        "replacement_guidance": None,
    }
    assert parsed.internal_processing == {
        "optimize_image": True,
        "remove_subtitle": True,
        "remove_watermark": True,
    }
    assert "remove_watermark" not in parsed.effective_request["processing"]


def test_target_language_and_replacement_guidance_keep_the_public_shape():
    request = _request()
    request["dialogue"]["target_language"] = "  英语  "
    request["replacement_guidance"] = {
        "instruction": "  把白色水杯替换成参考图中的产品杯  ",
        "image_field": "replacement_image",
    }
    parsed = _parse(request)

    assert parsed.effective_request["dialogue"] == {
        "mode": "auto_rewrite",
        "target_language": "英语",
    }
    assert parsed.effective_request["replacement_guidance"] == {
        "instruction": "把白色水杯替换成参考图中的产品杯",
        "image_field": "replacement_image",
    }
    minimal_creation.validate_replacement_pair(
        parsed, replacement_image_present=True
    )


def test_canonical_hash_uses_normalized_sorted_compact_utf8_json():
    first = _request()
    second = {
        "replacement_guidance": None,
        "dialogue": {
            "target_language": "日语",
            "mode": "auto_rewrite",
        },
        "processing": {
            "remove_logo": True,
            "remove_subtitle": True,
            "optimize_image": True,
        },
        "output": {
            "fit_mode": "auto",
            "resolution": "768p",
            "aspect_ratio": "9:16",
        },
        "version": 1,
    }
    parsed_first = _parse(first)
    parsed_second = minimal_creation.parse_generation_request(
        json.dumps(second, indent=4, ensure_ascii=False).encode("utf-8")
    )
    expected_bytes = json.dumps(
        parsed_first.effective_request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert parsed_first.effective_request == parsed_second.effective_request
    assert parsed_first.generation_request_sha256 == hashlib.sha256(
        expected_bytes
    ).hexdigest()
    assert (
        parsed_first.generation_request_sha256
        == parsed_second.generation_request_sha256
    )
    assert len(parsed_first.generation_request_sha256) == 64
    assert parsed_first.generation_request_sha256.islower()


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("not json", "invalid_generation_request_json"),
        ("[]", "invalid_generation_request_json"),
        (b'\xff', "invalid_generation_request_json"),
        ('{"version":NaN}', "invalid_generation_request_json"),
        (
            '{"version":1,"version":1,"output":{},"processing":{},'
            '"dialogue":{},"replacement_guidance":null}',
            "invalid_generation_request",
        ),
        (
            '{"version":1,"output":{"aspect_ratio":"9:16",'
            '"aspect_ratio":"9:16","resolution":"768p","fit_mode":"auto"},'
            '"processing":{"optimize_image":true,"remove_subtitle":true,'
            '"remove_logo":true},"dialogue":{"mode":"auto_rewrite",'
            '"target_language":"日语"},'
            '"replacement_guidance":null}',
            "invalid_generation_request",
        ),
    ],
)
def test_json_is_utf8_object_without_non_json_constants_or_duplicate_keys(raw, code):
    _assert_error(code, lambda: minimal_creation.parse_generation_request(raw))


@pytest.mark.parametrize("version", [True, False, 1.0, "1", 2, None])
def test_version_is_exact_integer_one(version):
    _assert_error(
        "unsupported_generation_request_version",
        lambda: _parse(_request(version=version)),
    )


@pytest.mark.parametrize(
    "output",
    [
        {"aspect_ratio": "1:1", "resolution": "768p", "fit_mode": "auto"},
        {"aspect_ratio": "9:16", "resolution": "1080p", "fit_mode": "auto"},
        {"aspect_ratio": "9:16", "resolution": "768p", "fit_mode": "cover"},
    ],
)
def test_output_values_are_restricted_to_capability(output):
    _assert_error(
        "invalid_output_config",
        lambda: _parse(_request(output=output)),
    )


@pytest.mark.parametrize(
    ("processing", "code"),
    [
        (
            {
                "optimize_image": False,
                "remove_subtitle": True,
                "remove_logo": True,
            },
            "processing_must_be_enabled",
        ),
        (
            {"optimize_image": True, "remove_subtitle": True},
            "processing_must_be_enabled",
        ),
        (
            {
                "optimize_image": True,
                "remove_subtitle": True,
                "remove_watermark": True,
            },
            "invalid_generation_request",
        ),
    ],
)
def test_processing_requires_exact_public_true_fields(processing, code):
    _assert_error(code, lambda: _parse(_request(processing=processing)))


def test_all_objects_reject_unknown_or_missing_keys():
    extra = _request(unexpected=True)
    _assert_error("invalid_generation_request", lambda: _parse(extra))

    missing_output_key = _request()
    del missing_output_key["output"]["fit_mode"]
    _assert_error(
        "invalid_generation_request", lambda: _parse(missing_output_key)
    )

    extra_dialogue_key = _request()
    extra_dialogue_key["dialogue"]["lines"] = []
    _assert_error(
        "invalid_generation_request", lambda: _parse(extra_dialogue_key)
    )


def test_dialogue_is_exact_auto_rewrite_with_target_language():
    legacy_script = _request()
    legacy_script["dialogue"] = {
        "mode": "rewrite",
        "script": "用户预写台词",
        "language": {"mode": "translate", "target": "日语"},
    }
    _assert_error("invalid_generation_request", lambda: _parse(legacy_script))

    missing_target = _request()
    missing_target["dialogue"] = {"mode": "auto_rewrite"}
    _assert_error("invalid_generation_request", lambda: _parse(missing_target))

    empty_target = _request()
    empty_target["dialogue"]["target_language"] = "  "
    _assert_error("target_language_required", lambda: _parse(empty_target))

    unsupported = _request()
    unsupported["dialogue"]["mode"] = "rewrite"
    _assert_error("invalid_generation_request", lambda: _parse(unsupported))


def test_target_language_length_is_trimmed_utf16_code_units():
    at_limit = _request()
    at_limit["dialogue"]["target_language"] = "  " + ("😀" * 40) + "  "
    parsed = _parse(at_limit)
    assert minimal_creation.utf16_code_units(
        parsed.effective_request["dialogue"]["target_language"]
    ) == minimal_creation.TARGET_LANGUAGE_MAX_CHARS

    over_limit = _request()
    over_limit["dialogue"]["target_language"] = "😀" * 41
    _assert_error("target_language_too_long", lambda: _parse(over_limit))

    empty = _request()
    empty["dialogue"]["target_language"] = " \n\t "
    error = _assert_error("target_language_required", lambda: _parse(empty))
    assert error.detail() == {
        "code": "target_language_required",
        "message": "请填写目标语言",
        "field": "generation_request.dialogue.target_language",
    }
    assert minimal_creation.public_error_detail(error) == error.detail()


def test_replacement_instruction_uses_trimmed_utf16_length():
    at_limit = _request(
        replacement_guidance={
            "instruction": "  " + ("😀" * 500) + "  ",
            "image_field": "replacement_image",
        }
    )
    assert minimal_creation.utf16_code_units(
        _parse(at_limit).effective_request["replacement_guidance"]["instruction"]
    ) == minimal_creation.REPLACEMENT_MAX_INSTRUCTION_CHARS

    over_limit = _request(
        replacement_guidance={
            "instruction": "😀" * 501,
            "image_field": "replacement_image",
        }
    )
    _assert_error("replacement_instruction_too_long", lambda: _parse(over_limit))

    empty = _request(
        replacement_guidance={
            "instruction": "  ",
            "image_field": "replacement_image",
        }
    )
    _assert_error("replacement_instruction_required", lambda: _parse(empty))



def test_replacement_image_field_is_an_optional_ignored_transport_hint():
    variants = [
        {"instruction": "替换产品"},
        {"instruction": "替换产品", "image_field": "replacement_image"},
        {"instruction": "替换产品", "image_field": "other_image"},
        {"instruction": "替换产品", "image_field": None},
    ]
    parsed = [
        _parse(_request(replacement_guidance=guidance))
        for guidance in variants
    ]

    assert all(
        item.effective_request["replacement_guidance"] == {
            "instruction": "替换产品",
            "image_field": "replacement_image",
        }
        for item in parsed
    )
    assert len({item.generation_request_sha256 for item in parsed}) == 1


def test_replacement_guidance_and_image_must_be_present_as_a_pair():
    without_guidance = _parse(_request())
    minimal_creation.validate_replacement_pair(
        without_guidance, replacement_image_present=False
    )
    _assert_error(
        "replacement_guidance_required",
        lambda: minimal_creation.validate_replacement_pair(
            without_guidance, replacement_image_present=True
        ),
    )

    with_guidance = _parse(
        _request(
            replacement_guidance={
                "instruction": "替换产品",
                "image_field": "replacement_image",
            }
        )
    )
    _assert_error(
        "replacement_image_required",
        lambda: minimal_creation.validate_replacement_pair(
            with_guidance, replacement_image_present=False
        ),
    )
    minimal_creation.validate_replacement_pair(
        with_guidance, replacement_image_present=True
    )
