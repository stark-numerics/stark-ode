from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ResolventSafety(Protocol):
    """Safety controls consumed directly by resolvent support."""

    block_sizes: bool


@dataclass(frozen=True, slots=True)
class ResolventSafetyDefault:
    """Default resolvent safety controls."""

    block_sizes: bool = True


__all__ = ["ResolventSafety", "ResolventSafetyDefault"]
