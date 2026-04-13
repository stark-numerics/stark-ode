from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Tolerance:
    """
    General STARK tolerance object for normalized error control.

    Any object providing `bound(scale)`, `ratio(error, scale)`, and
    `accepts(error, scale)` can be used in its place, but this class is the
    common duck-typed default for scheme, resolver, and inverter tolerances.
    """

    atol: float = 1.0e-6
    rtol: float = 1.0e-6

    def __repr__(self) -> str:
        return f"{type(self).__name__}(atol={self.atol!r}, rtol={self.rtol!r})"

    def __str__(self) -> str:
        return f"atol={self.atol:g}, rtol={self.rtol:g}"

    def bound(self, scale: float) -> float:
        return self.atol + self.rtol * scale

    def ratio(self, error: float, scale: float) -> float:
        return error / self.bound(scale)

    def accepts(self, error: float, scale: float) -> bool:
        return self.ratio(error, scale) <= 1.0


__all__ = ["Tolerance"]
