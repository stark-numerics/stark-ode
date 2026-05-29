from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutorAdaptivity:
    """Adaptive step-size policy carried by an `Executor`."""

    safety: float = 0.8
    min_factor: float = 0.1
    max_factor: float = 5.0
    error_exponent: float = 0.2

    def __str__(self) -> str:
        return (
            f"safety={self.safety:g}, "
            f"min_factor={self.min_factor:g}, "
            f"max_factor={self.max_factor:g}, "
            f"error_exponent={self.error_exponent:g}"
        )

    def factor(self, error_ratio: float) -> float:
        if error_ratio == 0.0:
            return self.max_factor
        factor = self.safety * (1.0 / error_ratio) ** self.error_exponent
        return min(self.max_factor, max(self.min_factor, factor))

    def rejected_step(self, dt: float, error_ratio: float, remaining: float, label: str) -> float:
        dt *= self.factor(error_ratio)
        if dt <= 0.0:
            raise RuntimeError(f"{label} step size underflowed to zero.")
        return remaining if dt > remaining else dt

    def accepted_next_step(self, accepted_dt: float, error_ratio: float, remaining_after: float) -> float:
        if remaining_after <= 0.0:
            return 0.0
        next_step = accepted_dt * self.factor(error_ratio)
        return remaining_after if next_step > remaining_after else next_step


__all__ = ["ExecutorAdaptivity"]
