"""Scheme-facing ExecutorTolerance protocols."""

from __future__ import annotations

from typing import Protocol


class SchemeTolerance(Protocol):
    """ExecutorTolerance behaviour consumed by adaptive schemes."""

    def bound(self, scale: float) -> float:
        ...

    def ratio(self, error: float, scale: float) -> float:
        ...

    def accepts(self, error: float, scale: float) -> bool:
        ...


__all__ = ["SchemeTolerance"]
