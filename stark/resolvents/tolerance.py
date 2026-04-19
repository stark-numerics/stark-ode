from __future__ import annotations

from dataclasses import dataclass

from stark.execution.tolerance import Tolerance


@dataclass(slots=True)
class ResolventTolerance(Tolerance):
    """Tolerance object for nonlinear implicit resolution."""


__all__ = ["ResolventTolerance"]









