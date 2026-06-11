from __future__ import annotations

from typing import Protocol

from stark.core.tolerance import Tolerance


class ResolventConfiguration(Protocol):
    resolvent_tolerance: Tolerance
    resolvent_maximum_steps: int


__all__ = ["ResolventConfiguration"]
