from __future__ import annotations

from dataclasses import dataclass

from stark.executor.tolerance import ExecutorTolerance


@dataclass(frozen=True, slots=True)
class InverterTolerance(ExecutorTolerance):
    """ExecutorTolerance object for linear inverse actions."""


__all__ = ["InverterTolerance"]









