import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".deploy" / "caddy" / "config.json"
RUNBOOK_PATH = ROOT / ".deploy" / "runbook.md"


EXPECTED_SRV3211 = {
    "listen": [":3211"],
    "protocols": ["h1", "h2"],
    "automatic_https": {"disable_redirects": True},
    "routes": [
        {
            "handle": [
                {
                    "handler": "reverse_proxy",
                    "upstreams": [{"dial": "127.0.0.1:3212"}],
                }
            ]
        }
    ],
    "tls_connection_policies": [
        {"certificate_selection": {"any_tag": ["duet-ad1-cert"]}}
    ],
}


def _config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_legacy_3211_server_contract_is_unchanged():
    servers = _config()["apps"]["http"]["servers"]
    assert servers["srv3211"] == EXPECTED_SRV3211


def test_3213_serves_api_without_rewrite_and_spa_with_cache_contract():
    config = _config()
    server = config["apps"]["http"]["servers"]["srv3213"]
    assert server["listen"] == [":3213"]
    assert server["protocols"] == ["h1", "h2"]
    assert server["automatic_https"] == {"disable_redirects": True}
    assert server["tls_connection_policies"] == [
        {"certificate_selection": {"any_tag": ["duet-ad1-cert"]}}
    ]

    api_route, static_route = server["routes"]
    assert api_route == {
        "match": [{"path": ["/api/*"]}],
        "handle": [
            {
                "handler": "reverse_proxy",
                "upstreams": [{"dial": "127.0.0.1:3212"}],
            }
        ],
        "terminal": True,
    }

    assert static_route.keys() == {"handle"}
    (subroute,) = static_route["handle"]
    assert subroute["handler"] == "subroute"
    root_route, try_files_route, assets_route, index_route, file_route = (
        subroute["routes"]
    )
    assert root_route == {
        "handle": [
            {
                "handler": "vars",
                "root": "/home/xy/duet-ad1/web-next/dist",
            }
        ]
    }
    assert try_files_route == {
        "match": [
            {
                "file": {
                    "try_files": ["{http.request.uri.path}", "/index.html"]
                }
            }
        ],
        "handle": [
            {
                "handler": "rewrite",
                "uri": "{http.matchers.file.relative}",
            }
        ],
    }
    assert assets_route == {
        "match": [{"path": ["/assets/*"]}],
        "handle": [
            {
                "handler": "headers",
                "response": {
                    "set": {
                        "Cache-Control": [
                            "public, max-age=31536000, immutable"
                        ]
                    }
                },
            }
        ],
    }
    assert index_route == {
        "match": [{"path": ["/", "/index.html"]}],
        "handle": [
            {
                "handler": "headers",
                "response": {"set": {"Cache-Control": ["no-store"]}},
            }
        ],
    }
    assert file_route == {"handle": [{"handler": "file_server"}]}


def test_3213_reuses_the_single_loaded_certificate():
    load_files = _config()["apps"]["tls"]["certificates"]["load_files"]
    assert len(load_files) == 1
    assert load_files[0]["tags"] == ["duet-ad1-cert"]


def test_runbook_has_safe_single_service_rollout_and_read_only_smoke():
    runbook = RUNBOOK_PATH.read_text(encoding="utf-8")
    for required in (
        "/home/xy/duet-ad1/.worktree/antd-x-prototype/web-next",
        "node ./node_modules/.bin/vite preview --host 0.0.0.0 --port 3213",
        "/usr/local/libexec/duet/caddy validate --config",
        "systemctl --user restart duet-ad1-caddy.service",
        "https://127.0.0.1:3213/",
        "https://127.0.0.1:3213/api/health",
        "生产 smoke 只允许 GET/HEAD",
        "禁止 POST/PUT/PATCH/DELETE",
        "3213 回滚",
    ):
        assert required in runbook
    assert "duet-ad1-frontend.service" not in runbook
