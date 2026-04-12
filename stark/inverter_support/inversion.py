from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Inversion:
    """User-facing stopping controls for linear inverse actions."""

    atol: float = 1.0e-9
    rtol: float = 1.0e-9
    max_iterations: int = 32
    restart: int = 16

    def __repr__(self) -> str:
        return (
            "Inversion("
            f"atol={self.atol!r}, "
            f"rtol={self.rtol!r}, "
            f"max_iterations={self.max_iterations!r}, "
            f"restart={self.restart!r})"
        )

    def __str__(self) -> str:
        return (
            f"atol={self.atol:g}, "
            f"rtol={self.rtol:g}, "
            f"max_iterations={self.max_iterations}, "
            f"restart={self.restart}"
        )

    def bound(self, scale: float) -> float:
        return self.atol + self.rtol * scale

    def ratio(self, error: float, scale: float) -> float:
        return error / self.bound(scale)

    def accepts(self, error: float, scale: float) -> bool:
        return self.ratio(error, scale) <= 1.0


__all__ = ["Inversion"]
