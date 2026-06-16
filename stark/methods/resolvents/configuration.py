from __future__ import annotations

from typing import Protocol

from stark.core.tolerance import Tolerance
from dataclasses import dataclass, field

@dataclass(frozen=False, slots=True)

class ResolventConfiguration(Protocol):
    resolvent_tolerance: Tolerance
    resolvent_maximum_steps: int

@dataclass(frozen=False, slots=True)
class ResolventConfigurationDefault:
    resolvent_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(atol=1.0e-9, rtol=1.0e-9)
    )
    resolvent_maximum_steps: int = 16

__all__ = ["ResolventConfiguration", "ResolventConfigurationDefault"]
