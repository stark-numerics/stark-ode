from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class InverterSafety(Protocol):
    """Safety controls consumed directly by inverter support."""

    block_sizes: bool


@dataclass(frozen=True, slots=True)
class InverterSafetyDefault:
    """Default inverter safety controls."""

    block_sizes: bool = True


__all__ = ["InverterSafety", "InverterSafetyDefault"]
