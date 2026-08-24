import pytest

from conftest import AUTH


def test_health_no_auth(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_protected_route_no_token_401(client):
    assert client.get("/api/conversations").status_code == 401


def test_protected_route_wrong_token_401(client):
    r = client.get("/api/conversations", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_protected_route_malformed_header_401(client):
    r = client.get("/api/conversations", headers={"Authorization": "not-a-bearer"})
    assert r.status_code == 401


def test_protected_route_ok(client):
    r = client.get("/api/conversations", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == []


def test_login_ok(client):
    r = client.post("/api/login", json={"token": "test-token"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_login_wrong_token_401(client):
    r = client.post("/api/login", json={"token": "nope"})
    assert r.status_code == 401


def test_login_missing_token_401(client):
    assert client.post("/api/login", json={}).status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {"token": "test-token", "unexpected": True},
        {"token": 123},
        {"token": None},
    ],
)
def test_login_rejects_extra_keys_and_non_string_tokens(client, payload):
    response = client.post("/api/login", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_login_request"}
