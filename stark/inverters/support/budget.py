from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InverterBudget:
    """Step budget for an inverter call."""

    maximum_steps: int = 32

    def __post_init__(self) -> None:
        if self.maximum_steps < 1:
            raise ValueError("InverterBudget.maximum_steps must be at least 1.")


@dataclass(frozen=True, slots=True)
class InverterBudgetRestarted(InverterBudget):
    """Step and restart budget for a restarted inverter."""

    restart_dimension: int = 16

    def __post_init__(self) -> None:
        InverterBudget.__post_init__(self)
        if self.restart_dimension < 1:
            raise ValueError("InverterBudgetRestarted.restart_dimension must be at least 1.")


__all__ = ["InverterBudget", "InverterBudgetRestarted"]
