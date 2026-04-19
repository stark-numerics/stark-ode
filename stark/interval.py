from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Interval:
    """
    Concrete rolling integration interval.
    """

    present: float
    step: float
    stop: float

    def __repr__(self) -> str:
        return f"Interval(present={self.present!r}, step={self.step!r}, stop={self.stop!r})"

    def __str__(self) -> str:
        return f"[{self.present:g}, {self.stop:g}] step={self.step:g}"

    def copy(self) -> "Interval":
        return Interval(present=self.present, step=self.step, stop=self.stop)

    def increment(self, dt: float) -> None:
        self.present += dt


__all__ = ["Interval"]









