from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InverterPolicy:
    """Iteration and breakdown controls for linear inverters."""

    max_iterations: int = 32
    restart: int = 16
    breakdown_tol: float = 1.0e-30

    def __repr__(self) -> str:
        return (
            "InverterPolicy("
            f"max_iterations={self.max_iterations!r}, "
            f"restart={self.restart!r}, "
            f"breakdown_tol={self.breakdown_tol!r})"
        )

    def __str__(self) -> str:
        return (
            f"max_iterations={self.max_iterations}, "
            f"restart={self.restart}, "
            f"breakdown_tol={self.breakdown_tol:g}"
        )


__all__ = ["InverterPolicy"]









