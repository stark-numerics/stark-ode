from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass(frozen=True, slots=True)
class MonitorSchemeSummary:
    step_count: int
    fixed_step_count: int
    adaptive_step_count: int
    accepted_dt_min: float | None
    accepted_dt_median: float | None
    accepted_dt_max: float | None
    adaptive_rejection_count: int | None
    adaptive_rejection_max: int | None
    adaptive_error_ratio_min: float | None
    adaptive_error_ratio_median: float | None
    adaptive_error_ratio_max: float | None

    def __str__(self) -> str:
        return (
            "scheme: "
            f"steps={self.step_count}, "
            f"fixed={self.fixed_step_count}, "
            f"adaptive={self.adaptive_step_count}, "
            "dt min/median/max="
            f"{self.accepted_dt_min}/{self.accepted_dt_median}/{self.accepted_dt_max}"
        )


@dataclass(slots=True)
class MonitorSchemeStepFixed:
    scheme: str
    t_start: float
    t_end: float
    accepted_dt: float

    def __str__(self) -> str:
        return (
            f"{self.scheme}: "
            f"{self.t_start:.6g} -> {self.t_end:.6g}, "
            f"dt={self.accepted_dt:.6g}"
        )


@dataclass(slots=True)
class MonitorSchemeStepAdaptive:
    scheme: str
    t_start: float
    t_end: float
    proposed_dt: float
    accepted_dt: float
    next_dt: float
    error_ratio: float
    rejection_count: int = 0

    def __str__(self) -> str:
        return (
            f"{self.scheme}: "
            f"{self.t_start:.6g} -> {self.t_end:.6g}, "
            f"dt={self.accepted_dt:.6g}, "
            f"next={self.next_dt:.6g}, "
            f"error ratio={self.error_ratio:.6g}, "
            f"rejections={self.rejection_count}"
        )


def _min_median_max(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    return min(values), float(median(values)), max(values)


@dataclass(slots=True)
class MonitorScheme:
    fixed_steps: list[MonitorSchemeStepFixed] = field(default_factory=list)
    adaptive_steps: list[MonitorSchemeStepAdaptive] = field(default_factory=list)

    def record_fixed_step(
        self,
        scheme: str,
        t_start: float,
        accepted_dt: float,
    ) -> None:
        self.fixed_steps.append(
            MonitorSchemeStepFixed(
                scheme=scheme,
                t_start=t_start,
                t_end=t_start + accepted_dt,
                accepted_dt=accepted_dt,
            )
        )

    def record_adaptive_step(
        self,
        scheme: str,
        t_start: float,
        proposed_dt: float,
        accepted_dt: float,
        next_dt: float,
        error_ratio: float,
        rejection_count: int,
    ) -> None:
        self.adaptive_steps.append(
            MonitorSchemeStepAdaptive(
                scheme=scheme,
                t_start=t_start,
                t_end=t_start + accepted_dt,
                proposed_dt=proposed_dt,
                accepted_dt=accepted_dt,
                next_dt=next_dt,
                error_ratio=error_ratio,
                rejection_count=rejection_count,
            )
        )

    def summary(self) -> MonitorSchemeSummary:
        fixed_count = len(self.fixed_steps)
        adaptive_count = len(self.adaptive_steps)
        accepted_dts = [
            step.accepted_dt
            for step in self.fixed_steps
        ]
        accepted_dts.extend(
            step.accepted_dt
            for step in self.adaptive_steps
        )
        accepted_dt_min, accepted_dt_median, accepted_dt_max = _min_median_max(accepted_dts)

        if adaptive_count == 0:
            return MonitorSchemeSummary(
                step_count=fixed_count,
                fixed_step_count=fixed_count,
                adaptive_step_count=0,
                accepted_dt_min=accepted_dt_min,
                accepted_dt_median=accepted_dt_median,
                accepted_dt_max=accepted_dt_max,
                adaptive_rejection_count=None,
                adaptive_rejection_max=None,
                adaptive_error_ratio_min=None,
                adaptive_error_ratio_median=None,
                adaptive_error_ratio_max=None,
            )

        rejections = [
            step.rejection_count
            for step in self.adaptive_steps
        ]
        error_ratios = [
            step.error_ratio
            for step in self.adaptive_steps
        ]
        error_ratio_min, error_ratio_median, error_ratio_max = _min_median_max(error_ratios)

        return MonitorSchemeSummary(
            step_count=fixed_count + adaptive_count,
            fixed_step_count=fixed_count,
            adaptive_step_count=adaptive_count,
            accepted_dt_min=accepted_dt_min,
            accepted_dt_median=accepted_dt_median,
            accepted_dt_max=accepted_dt_max,
            adaptive_rejection_count=sum(rejections),
            adaptive_rejection_max=max(rejections),
            adaptive_error_ratio_min=error_ratio_min,
            adaptive_error_ratio_median=error_ratio_median,
            adaptive_error_ratio_max=error_ratio_max,
        )

    def clear(self) -> None:
        self.fixed_steps.clear()
        self.adaptive_steps.clear()


__all__ = [
    "MonitorScheme",
    "MonitorSchemeStepAdaptive",
    "MonitorSchemeStepFixed",
    "MonitorSchemeSummary",
]
