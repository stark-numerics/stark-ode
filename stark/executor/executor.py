from __future__ import annotations

from dataclasses import dataclass, field

from stark.executor.adaptivity import ExecutorAdaptivity
from stark.executor.safety import ExecutorSafety
from stark.executor.tolerance import ExecutorTolerance


@dataclass(frozen=True, slots=True)
class Executor:
    """
    Runtime execution worker for STARK.

    The executor carries cross-cutting runtime policy such as tolerance, safety,
    and adaptive-step regulation.
    """

    tolerance: ExecutorTolerance = field(default_factory=ExecutorTolerance)
    safety: ExecutorSafety = field(default_factory=ExecutorSafety)
    adaptivity: ExecutorAdaptivity | None = None

    def __repr__(self) -> str:
        return (
            "Executor("
            f"tolerance={self.tolerance!r}, "
            f"safety={self.safety!r}, "
            f"adaptivity={self.adaptivity!r})"
        )

    def __str__(self) -> str:
        adaptivity = "scheme default" if self.adaptivity is None else str(self.adaptivity)
        return f"{self.tolerance}, {self.safety}, adaptivity={adaptivity}"

    def bound(self, scale: float) -> float:
        return self.tolerance.bound(scale)

    def ratio(self, error: float, scale: float) -> float:
        return self.tolerance.ratio(error, scale)

    def accepts(self, error: float, scale: float) -> bool:
        return self.tolerance.accepts(error, scale)

    def adaptivity_or(self, fallback: ExecutorAdaptivity) -> ExecutorAdaptivity:
        return self.adaptivity if self.adaptivity is not None else fallback


__all__ = ["Executor"]









