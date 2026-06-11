from __future__ import annotations

from typing import Protocol

from stark.core.tolerance import Tolerance


class InverterConfiguration(Protocol):
    inverter_tolerance: Tolerance
    inverter_maximum_steps: int


__all__ = ["InverterConfiguration"]
