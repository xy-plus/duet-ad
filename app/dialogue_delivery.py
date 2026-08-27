"""Independent authority for where frozen dialogue is delivered."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping, Sequence


class DialogueDelivery(StrEnum):
    AUTO = "auto"
    ON_SCREEN = "on_screen"
    OFF_SCREEN = "off_screen"


def parse(value: object) -> DialogueDelivery:
    try:
        return DialogueDelivery(value)
    except (TypeError, ValueError):
        raise ValueError("invalid_dialogue_delivery") from None


def resolve(
    requested: DialogueDelivery,
    authoritative_lines: Sequence[Mapping[str, object]],
) -> DialogueDelivery:
    """Resolve project-level auto without guessing a speaker from pictures."""
    if requested is not DialogueDelivery.AUTO:
        return requested
    lines = tuple(authoritative_lines)
    if lines and all(line.get("classification") == "sung" for line in lines):
        return DialogueDelivery.OFF_SCREEN
    return DialogueDelivery.ON_SCREEN

