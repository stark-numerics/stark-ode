from __future__ import annotations

from dataclasses import dataclass

from stark.tolerance import Tolerance


@dataclass(slots=True)
class InverterTolerance(Tolerance):
    """Tolerance object for linear inverse actions."""


__all__ = ["InverterTolerance"]
