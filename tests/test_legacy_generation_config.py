import json

from test_web_h3_contract import APP_JS, _run_contract
from test_legacy_phase2_frame_viewer import _run_jsdom_contract


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


def test_precreate_controls_expose_defaults_and_conservative_fallback_copy():
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")
    for element_id in (
        "generation-config", "generation-config-summary-text", "generation-config-fields",
        "generation-optimize-image", "generation-remove-subtitle",
        "generation-remove-watermark", "generation-config-status",
    ):
        assert f'id="{element_id}"' in html
    assert 'id="generation-optimize-image" type="checkbox"' in html
    optimize_tail = html.split('id="generation-optimize-image"', 1)[1].split(">", 1)[0]
    assert "checked" in optimize_tail
    assert "checked" not in html.split('id="generation-remove-subtitle"', 1)[1].split(">", 1)[0]
    assert "checked" not in html.split('id="generation-remove-watermark"', 1)[1].split(">", 1)[0]
    assert 'apiJSON("/api/capabilities")' in source
    assert '"使用服务器默认配置"' in source
    assert '"当前服务器未声明可选配置；页面不会发送未知字段。"' in source
    assert '"提交一次后将按此配置自动运行至成片，中途无需确认。"' in source


def test_generation_config_disclosure_is_open_on_first_render_and_creation_reset():
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")
    assert '<details id="generation-config" class="generation-config" open>' in html

    enter = source.split("function enterApp()", 1)[1].split(
        "/* ===== 侧栏会话列表", 1
    )[0]
    new_chat = source.split('$("new-chat-btn").addEventListener', 1)[1].split(
        '$("attach-btn")', 1
    )[0]
    assert "resetGenerationConfigDisclosure()" in enter
    assert "resetGenerationConfigDisclosure()" in new_chat

    result = _run_jsdom_contract(
        "(()=>{const details=document.createElement('details');details.id='generation-config';"
        "details.open=false;document.body.appendChild(details);"
        "contract.resetGenerationConfigDisclosure();return {open:details.open,connected:details.isConnected}})()"
    )
    assert result == {"open": True, "connected": True}


def test_upload_wires_capability_gated_multipart_json_without_provider_calls():
    source = APP_JS.read_text(encoding="utf-8")
    upload = source.split("function uploadConversation", 1)[1].split(
        "async function handleSend", 1
    )[0]
    send = source.split("async function handleSend", 1)[1].split(
        "/* ===== 事件绑定与启动", 1
    )[0]
    assert "buildGenerationConfigCreateField(" in upload
    assert "if (configField) fd.append(configField.name, configField.value)" in upload
    assert 'fd.append("generation_config"' not in upload
    assert "state.generationConfigCapability" in send
    assert "generationConfigCapability: state.generationConfigCapability" in send


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
        "poll:contract.shouldPollDetail({status:'done',has_video:false,generation_config:config,generation_config_sha256:sha})}})()"
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
    stable = source.split("function renderStable", 1)[1].split(
        "/* 动态区", 1
    )[0]
    assert "if (frozenGenerationConfig(detail)) return false" in ask
    assert "服务端会自动运行至成片，无需再次确认或提交" in final
    assert "renderFrozenGenerationConfig(detail)" in stable


def test_generation_config_layout_is_discoverable_and_mobile_safe():
    css = STYLES.read_text(encoding="utf-8")
    summary = css.split(".generation-config-summary {", 1)[1].split("}", 1)[0]
    fields = css.split(".generation-config-fields {", 1)[1].split("}", 1)[0]
    assert "min-height: 42px" in summary
    assert "display: flex" in fields
    assert "flex-wrap: wrap" in fields
