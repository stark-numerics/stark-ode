from __future__ import annotations

from dataclasses import dataclass

from stark.executor.tolerance import ExecutorTolerance


@dataclass(frozen=True, slots=True)
class ResolventTolerance(ExecutorTolerance):
    """ExecutorTolerance object for nonlinear implicit resolution."""


__all__ = ["ResolventTolerance"]









