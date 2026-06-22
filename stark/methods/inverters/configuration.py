from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from stark.core.tolerance import Tolerance


class InverterConfiguration(Protocol):
    """Configuration shape shared by iterative inverter implementations."""

    @property
    def inverter_tolerance(self) -> Tolerance:
        ...

    @property
    def inverter_maximum_steps(self) -> int:
        ...


@dataclass(frozen=False, slots=True)
class InverterConfigurationDefault:
    """Default inverter configuration used when callers do not provide one."""

    inverter_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(atol=1.0e-9, rtol=1.0e-9)
    )
    inverter_maximum_steps: int = 16


__all__ = ["InverterConfiguration", "InverterConfigurationDefault"]
