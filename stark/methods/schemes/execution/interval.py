from __future__ import annotations

from typing import Protocol, Self
from stark.core.contracts.interval import IntervalLike

class SchemeInterval(Protocol):
    present: float
    step: float
    stop: float
    def copy(self) -> Self:
        ...

class SchemeShiftedInterval:
    """Reusable interval view shifted within the current scheme step."""

    __slots__ = ("interval",)

    def __init__(self) -> None:
        self.interval: IntervalLike | None = None

    def __call__(self, interval: IntervalLike, step: float, shift: float) -> IntervalLike:
        shifted = self.interval
        if shifted is None:
            shifted = interval.copy()
            self.interval = shifted
        shifted.present = interval.present + shift
        shifted.step = step
        shifted.stop = interval.stop
        return shifted


__all__ = ["SchemeShiftedInterval"]
