"""Scheme-facing tolerance protocols."""

from __future__ import annotations

from typing import Protocol


class SchemeTolerance(Protocol):
    """Tolerance behaviour consumed by adaptive schemes."""

    def bound(self, scale: float) -> float:
        ...

    def ratio(self, error: float, scale: float) -> float:
        ...

    def accepts(self, error: float, scale: float) -> bool:
        ...


__all__ = ["SchemeTolerance"]
