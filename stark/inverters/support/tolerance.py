from __future__ import annotations

from dataclasses import dataclass

from stark.executor import ExecutorTolerance


@dataclass(frozen=True, slots=True)
class InverterTolerance(ExecutorTolerance):
    """Residual tolerance for an inverter call."""


__all__ = ["InverterTolerance"]
