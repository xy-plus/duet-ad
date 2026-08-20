from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / ".deploy" / "systemd" / "duet-ad1.service"


def test_user_service_preserves_codex_bwrap_sandbox_compatibility():
    """Host hardening must not pre-empt the agent's own user/mount/net sandbox."""
    source = UNIT.read_text(encoding="utf-8")
    for required in (
        "NoNewPrivileges=true",
        "LockPersonality=true",
        "RestrictSUIDSGID=true",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK",
        "UMask=0077",
    ):
        assert required in source

    for incompatible in (
        "PrivateTmp=",
        "ProtectSystem=",
        "ProtectControlGroups=",
        "ProtectKernelTunables=",
    ):
        assert incompatible not in source
