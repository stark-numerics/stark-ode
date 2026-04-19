from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MonitorStep:
    scheme: str
    t_start: float
    t_end: float
    proposed_dt: float
    accepted_dt: float
    next_dt: float
    error_ratio: float
    rejection_count: int = 0


@dataclass(slots=True)
class Monitor:
    steps: list[MonitorStep] = field(default_factory=list)

    def __call__(self, step: MonitorStep) -> None:
        self.steps.append(step)

    def clear(self) -> None:
        self.steps.clear()


__all__ = ["Monitor", "MonitorStep"]
