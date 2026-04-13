from __future__ import annotations

from stark.regulator import Regulator


class AdaptiveController:
    """Reusable adaptive step-size control driven by a `Regulator`."""

    __slots__ = ("safety", "min_factor", "max_factor", "error_exponent")

    def __init__(self, regulator: Regulator) -> None:
        self.safety = regulator.safety
        self.min_factor = regulator.min_factor
        self.max_factor = regulator.max_factor
        self.error_exponent = regulator.error_exponent

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


__all__ = ["AdaptiveController"]

