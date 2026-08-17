"""app/sanitize 公共脱敏函数（seedance/seedream 共用）的测试。"""

from app.sanitize import sanitize


def test_sanitize_strips_leak_lines_and_key(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "sk-live-abcdef123456")
    text = (
        "Authorization: Bearer sk-live-abcdef123456\n"
        "api_key=sk-live-abcdef123456\n"
        "plain line\n"
    )
    out = sanitize(text)
    assert "sk-live-abcdef123456" not in out
    assert "Authorization" not in out
    assert "api_key" not in out
    assert "plain line" in out


def test_sanitize_masks_key_literal(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "sk-abc")
    assert sanitize("error: sk-abc happened") == "error: *** happened"


def test_sanitize_limits_length(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    out = sanitize("x" * 1000, limit=10)
    assert len(out) <= 10
