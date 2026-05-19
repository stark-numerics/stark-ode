from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass(frozen=True, slots=True)
class MonitorSummaryInverter:
    solve_count: int
    failure_count: int
    iteration_min: int | None
    iteration_median: float | None
    iteration_max: int | None
    initial_residual_min: float | None
    initial_residual_median: float | None
    initial_residual_max: float | None
    final_residual_min: float | None
    final_residual_median: float | None
    final_residual_max: float | None


@dataclass(slots=True)
class MonitorInverterSolve:
    inverter: str
    converged: bool
    iteration_count: int | None
    initial_residual: float | None
    final_residual: float | None
    failure_reason: str | None


def _min_median_max_float(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    return min(values), float(median(values)), max(values)


def _min_median_max_int(values: list[int]) -> tuple[int | None, float | None, int | None]:
    if not values:
        return None, None, None
    return min(values), float(median(values)), max(values)


@dataclass(slots=True)
class MonitorInverter:
    solves: list[MonitorInverterSolve] = field(default_factory=list)

    def record_solve(
        self,
        inverter: str,
        converged: bool,
        iteration_count: int | None,
        initial_residual: float | None,
        final_residual: float | None,
        failure_reason: str | None,
    ) -> None:
        self.solves.append(
            MonitorInverterSolve(
                inverter=inverter,
                converged=converged,
                iteration_count=iteration_count,
                initial_residual=initial_residual,
                final_residual=final_residual,
                failure_reason=failure_reason,
            )
        )

    def summary(self) -> MonitorSummaryInverter:
        iterations = [
            solve.iteration_count
            for solve in self.solves
            if solve.iteration_count is not None
        ]
        initial_residuals = [
            solve.initial_residual
            for solve in self.solves
            if solve.initial_residual is not None
        ]
        final_residuals = [
            solve.final_residual
            for solve in self.solves
            if solve.final_residual is not None
        ]
        iteration_min, iteration_median, iteration_max = _min_median_max_int(iterations)
        initial_min, initial_median, initial_max = _min_median_max_float(initial_residuals)
        final_min, final_median, final_max = _min_median_max_float(final_residuals)
        return MonitorSummaryInverter(
            solve_count=len(self.solves),
            failure_count=sum(
                1
                for solve in self.solves
                if not solve.converged
            ),
            iteration_min=iteration_min,
            iteration_median=iteration_median,
            iteration_max=iteration_max,
            initial_residual_min=initial_min,
            initial_residual_median=initial_median,
            initial_residual_max=initial_max,
            final_residual_min=final_min,
            final_residual_median=final_median,
            final_residual_max=final_max,
        )

    def clear(self) -> None:
        self.solves.clear()


__all__ = [
    "MonitorInverter",
    "MonitorInverterSolve",
    "MonitorSummaryInverter",
]
