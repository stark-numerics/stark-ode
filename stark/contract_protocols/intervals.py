from __future__ import annotations

from typing import Protocol, Self


class IntervalLike(Protocol):
    """
    Protocol for a rolling integration interval.

    User-defined intervals are fine as long as they expose the same attributes
    and behavior as STARK's primitive `Interval`.
    """

    present: float
    step: float
    stop: float

    def copy(self) -> Self:
        ...

    def increment(self, dt: float) -> None:
        ...


__all__ = ["IntervalLike"]
