import json
import re

from test_minimal_frontend_contract import (
    _capability as _minimal_creation_capability,
    _function_source,
)
from test_web_h3_contract import APP_JS, _run_contract


INDEX_HTML = APP_JS.parent / "index.html"
STYLES = APP_JS.parent / "styles.css"


def _capability(defaults=None, **overrides):
    capability = {
        "supported": True,
        "create_field": "generation_config",
        "encoding": "multipart_json",
        "fields": {
            "optimize_image": "boolean",
            "remove_subtitle": "boolean",
            "remove_watermark": "boolean",
        },
        "defaults": defaults or {
            "optimize_image": True,
            "remove_subtitle": False,
            "remove_watermark": False,
        },
    }
    capability.update(overrides)
    return {"generation_config": capability}


def test_capability_requires_the_exact_safe_create_contract():
    good = _capability()
    result = _run_contract(
        "(()=>{const good=" + json.dumps(good) + ";return {"
        "good:contract.normalizeGenerationConfigCapability(good),"
        "unsupported:contract.normalizeGenerationConfigCapability({generation_config:{...good.generation_config,supported:false}}),"
        "wrongField:contract.normalizeGenerationConfigCapability({generation_config:{...good.generation_config,create_field:'other'}}),"
        "extra:contract.normalizeGenerationConfigCapability({generation_config:{...good.generation_config,fields:{...good.generation_config.fields,extra:'boolean'}}}),"
        "driftedDefault:contract.normalizeGenerationConfigCapability({generation_config:{...good.generation_config,defaults:{...good.generation_config.defaults,optimize_image:false}}})}})()"
    )
    assert result["good"] == good["generation_config"]
    assert result["unsupported"] is None
    assert result["wrongField"] is None
    assert result["extra"] is None
    assert result["driftedDefault"] is None


def test_create_field_is_complete_json_only_when_capability_and_values_are_valid():
    capability = _capability()["generation_config"]
    config = {
        "optimize_image": False,
        "remove_subtitle": True,
        "remove_watermark": True,
    }
    result = _run_contract(
        "(()=>{const capability=" + json.dumps(capability) + ";const config=" + json.dumps(config) + ";"
        "return {supported:contract.buildGenerationConfigCreateField(capability,config),"
        "missing:contract.buildGenerationConfigCreateField(capability,{optimize_image:true,remove_subtitle:false}),"
        "unknown:contract.buildGenerationConfigCreateField(capability,{...config,remove_brand:true}),"
        "unsupported:contract.buildGenerationConfigCreateField({...capability,supported:false},config)}})()"
    )
    assert result["supported"] == {
        "name": "generation_config",
        "value": json.dumps(config, separators=(",", ":")),
    }
    assert result["missing"] is None
    assert result["unknown"] is None
    assert result["unsupported"] is None


def test_legacy_precreate_controls_are_hidden_with_fixed_true_compatibility_values():
    html = INDEX_HTML.read_text(encoding="utf-8")
    composer = html.split('<form id="composer"', 1)[1].split("</form>", 1)[0]
    visible, legacy = composer.split(
        '<div class="legacy-contract-controls" hidden aria-hidden="true">', 1
    )

    for element_id in (
        "generation-config", "generation-config-summary-text", "generation-config-fields",
        "generation-optimize-image", "generation-remove-subtitle",
        "generation-remove-watermark", "generation-config-status",
    ):
        assert f'id="{element_id}"' not in visible
        assert f'id="{element_id}"' in legacy
    assert '<fieldset id="generation-config-fields" disabled>' in legacy
    for element_id in (
        "generation-optimize-image",
        "generation-remove-subtitle",
        "generation-remove-watermark",
    ):
        attrs = legacy.split(f'id="{element_id}"', 1)[1].split(">", 1)[0]
        assert 'type="checkbox"' in attrs
        assert "checked" in attrs


def test_minimal_creation_fixes_processing_true_and_capability_mismatch_fails_closed():
    capability = _minimal_creation_capability(version=1)
    result = _run_contract(
        "(()=>{const capability="
        + json.dumps(capability)
        + ";const input={aspectRatio:'16:9',resolution:'480p',"
        "targetLanguage:' 日语 ',hasReplacementImage:false,replacementInstruction:''};"
        "const capture=(value)=>{try{return contract.buildMinimalGenerationRequest(input,value)}"
        "catch(error){return error.message}};return {"
        "valid:capture(capability),"
        "futureV2:capture({...capability,version:2}),"
        "wrongField:capture({...capability,request_field:'generation_config'}),"
        "driftedDefault:capture({...capability,defaults:{...capability.defaults,remove_logo:false}})}})()"
    )
    assert result["valid"]["version"] == 1
    assert result["valid"]["processing"] == {
        "optimize_image": True,
        "remove_subtitle": True,
        "remove_logo": True,
    }
    assert result["valid"]["dialogue"] == {
        "mode": "auto_rewrite",
        "target_language": "日语",
    }
    assert "script" not in result["valid"]["dialogue"]
    assert result["futureV2"] == "生成服务尚未支持当前创建方式"
    assert result["wrongField"] == "生成服务尚未支持当前创建方式"
    assert result["driftedDefault"] == "生成服务尚未支持当前创建方式"


def test_active_upload_sends_minimal_request_and_optional_replacement_without_legacy_fields():
    source = APP_JS.read_text(encoding="utf-8")
    upload = _function_source(source, "uploadConversation")
    send = source.split("async function handleSend", 1)[1].split(
        "/* ===== 事件绑定与启动", 1
    )[0]
    append_arguments = re.findall(r"fd\.append\((.+)\);", upload)

    assert append_arguments == [
        '"file", file, file.name',
        '"reference_url", url',
        '"client_request_id", requestId',
        "capability.request_field, JSON.stringify(generationRequest)",
        "capability.replacement_image_field, replacementImage, replacementImage.name",
    ]
    assert "if (replacementImage)" in upload
    assert "generationRequest = buildMinimalGenerationRequest(" in send
    assert "replacementImage: state.replacementImage" in send
    for legacy_field in (
        '"voice_mode"',
        '"dialogue_mode"',
        '"dialogue_review_policy"',
        '"generation_config"',
    ):
        assert legacy_field not in upload
    assert "buildGenerationConfigCreateField(" not in upload
    assert "generationConfigCapability" not in upload
    assert "generationConfigCapability" not in send


def test_frozen_config_requires_exact_fields_and_sha_and_suppresses_midway_confirm():
    config = {
        "optimize_image": True,
        "remove_subtitle": False,
        "remove_watermark": True,
    }
    digest = "a" * 64
    result = _run_contract(
        "(()=>{const config=" + json.dumps(config) + f";const sha='{digest}';return {{"
        "valid:contract.frozenGenerationConfig({generation_config:config,generation_config_sha256:sha}),"
        "missingSha:contract.frozenGenerationConfig({generation_config:config}),"
        "extra:contract.frozenGenerationConfig({generation_config:{...config,remove_brand:false},generation_config_sha256:sha}),"
        "labels:contract.generationConfigLabels(config),"
        "poll:contract.shouldPollDetail({status:'done',has_video:false,"
        "project_progress:{percent:60,status:'running'},"
        "generation_config:config,generation_config_sha256:sha})}})()"
    )
    assert result["valid"] == {"config": config, "sha256": digest}
    assert result["missingSha"] is None
    assert result["extra"] is None
    assert result["labels"] == ["图片优化", "保留字幕", "去水印"]
    assert result["poll"] is True

    source = APP_JS.read_text(encoding="utf-8")
    ask = source.split("function shouldRenderPostprocessAsk", 1)[1].split(
        "/* postprocess 存在", 1
    )[0]
    final = source.split("function renderFinalSection", 1)[1].split(
        "function setDisclosureState", 1
    )[0]
    active_result_renderers = "\n".join(
        _function_source(source, name)
        for name in (
            "renderResults",
            "renderStable",
            "renderPpDynamic",
            "renderGenerationDynamic",
        )
    )
    assert "if (frozenGenerationConfig(detail)) return false" in ask
    assert "服务端会自动运行至成片，无需再次确认或提交" in final
    assert "renderFrozenGenerationConfig(" not in active_result_renderers


def test_generation_config_layout_is_discoverable_and_mobile_safe():
    css = STYLES.read_text(encoding="utf-8")
    summary = css.split(".generation-config-summary {", 1)[1].split("}", 1)[0]
    fields = css.split(".generation-config-fields {", 1)[1].split("}", 1)[0]
    assert "min-height: 42px" in summary
    assert "display: flex" in fields
    assert "flex-wrap: wrap" in fields
