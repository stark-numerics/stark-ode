"""Configuration protocol and defaults consumed by resolvents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from stark.core.tolerance import Tolerance


class ResolventConfiguration(Protocol):
    """Configuration shape shared by iterative resolvent implementations."""

    @property
    def resolvent_tolerance(self) -> Tolerance:
        ...

    @property
    def resolvent_maximum_steps(self) -> int:
        ...


@dataclass(frozen=False, slots=True)
class ResolventConfigurationDefault:
    """Default resolvent configuration for standalone resolvent use."""

    resolvent_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(atol=1.0e-9, rtol=1.0e-9)
    )
    resolvent_maximum_steps: int = 16


__all__ = ["ResolventConfiguration", "ResolventConfigurationDefault"]
