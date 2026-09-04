import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".deploy" / "caddy" / "config.json"


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _assert_single_reverse_proxy(server, *, listen, upstream):
    assert server["listen"] == [listen]
    assert server["protocols"] == ["h1", "h2"]
    assert server["automatic_https"] == {"disable_redirects": True}
    assert server["tls_connection_policies"] == [
        {"certificate_selection": {"any_tag": ["duet-ad1-cert"]}}
    ]
    assert server["routes"] == [
        {
            "handle": [
                {
                    "handler": "reverse_proxy",
                    "upstreams": [{"dial": upstream}],
                }
            ]
        }
    ]


def test_each_origin_uses_one_fastapi_frontend_without_static_fallback():
    servers = _config()["apps"]["http"]["servers"]
    _assert_single_reverse_proxy(
        servers["srv3211"], listen=":3211", upstream="127.0.0.1:3212"
    )
    _assert_single_reverse_proxy(
        servers["srv3213"], listen=":3213", upstream="127.0.0.1:3214"
    )


def test_both_origins_reuse_the_single_loaded_certificate():
    config = _config()
    load_files = config["apps"]["tls"]["certificates"]["load_files"]
    assert len(load_files) == 1
    assert load_files[0]["tags"] == ["duet-ad1-cert"]
