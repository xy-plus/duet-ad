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
    _authoritative_lines: Sequence[Mapping[str, object]],
) -> DialogueDelivery:
    """Resolve unbound auto dialogue without inventing on-screen authority."""
    if requested is not DialogueDelivery.AUTO:
        return requested
    return DialogueDelivery.OFF_SCREEN
