"""Local-only administration CLI for public API keys and credit balances."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app import public_api_auth, public_credits


_DEFAULT_REGISTRY = Path(
    "/home/xy/.config/duet-ad1/public-api-clients.json"
)


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage Duet public API clients and append-only credits"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create-key")
    create.add_argument("--registry", type=_absolute_path, default=_DEFAULT_REGISTRY)
    create.add_argument("--owner", required=True)
    create.add_argument("--key-id")
    create.add_argument("--key-output", type=_absolute_path)

    revoke = commands.add_parser("revoke-key")
    revoke.add_argument("--registry", type=_absolute_path, default=_DEFAULT_REGISTRY)
    revoke.add_argument("--key-id", required=True)

    adjust = commands.add_parser("credits-adjust")
    adjust.add_argument("--data-dir", type=_absolute_path, required=True)
    adjust.add_argument("--owner", required=True)
    adjust.add_argument("--credits", type=int, required=True)
    adjust.add_argument("--reason", required=True)
    adjust.add_argument("--idempotency-key", required=True)

    show = commands.add_parser("credits-show")
    show.add_argument("--data-dir", type=_absolute_path, required=True)
    show.add_argument("--owner", required=True)
    show.add_argument("--transactions", type=int, default=20)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "create-key":
        api_key = public_api_auth.create_client_key(
            args.registry, args.owner, key_id=args.key_id
        )
        payload = {
            "owner_id": args.owner,
            "api_key": api_key,
            "warning": "This plaintext API key is shown only once.",
        }
        if args.key_output is None:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            args.key_output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                args.key_output,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps({
                "owner_id": args.owner,
                "key_output": str(args.key_output),
            }, ensure_ascii=False))
        return 0
    if args.command == "revoke-key":
        public_api_auth.revoke_client_key(args.registry, args.key_id)
        print(json.dumps({"key_id": args.key_id, "state": "revoked"}))
        return 0
    if args.command == "credits-adjust":
        created = public_credits.adjust(
            args.data_dir,
            args.owner,
            args.credits,
            reason=args.reason,
            idempotency_key=args.idempotency_key,
        )
        current = public_credits.balance(args.data_dir, args.owner)
        print(json.dumps({
            "created": created,
            "owner_id": args.owner,
            **current,
        }, ensure_ascii=False))
        return 0
    current = public_credits.balance(args.data_dir, args.owner)
    events = public_credits.recent_events(
        args.data_dir, args.owner, limit=args.transactions
    )
    print(json.dumps({
        "owner_id": args.owner,
        **current,
        "transactions": events,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
