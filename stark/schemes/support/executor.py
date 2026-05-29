"""Scheme-facing SchemeExecutor protocol."""

from __future__ import annotations

from typing import Protocol

from stark.executor.adaptivity import ExecutorAdaptivity
from stark.schemes.support.safety import SchemeSafety
from stark.schemes.support.tolerance import SchemeTolerance


class SchemeExecutor(Protocol):
    """Runtime behaviour consumed directly by one-step schemes."""

    tolerance: SchemeTolerance
    safety: SchemeSafety
    adaptivity: ExecutorAdaptivity | None

    def bound(self, scale: float) -> float:
        ...

    def ratio(self, error: float, scale: float) -> float:
        ...

    def accepts(self, error: float, scale: float) -> bool:
        ...

    def adaptivity_or(self, fallback: ExecutorAdaptivity) -> ExecutorAdaptivity:
        ...


__all__ = ["SchemeExecutor"]
