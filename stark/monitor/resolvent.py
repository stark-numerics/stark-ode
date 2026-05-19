from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass(frozen=True, slots=True)
class MonitorSummaryResolvent:
    solve_count: int
    failure_count: int
    iteration_min: int | None
    iteration_median: float | None
    iteration_max: int | None
    error_min: float | None
    error_median: float | None
    error_max: float | None


@dataclass(slots=True)
class MonitorResolventSolve:
    resolvent: str
    alpha: float
    block_size: int
    iteration_count: int
    error: float
    scale: float
    converged: bool


def _min_median_max_float(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    return min(values), float(median(values)), max(values)


def _min_median_max_int(values: list[int]) -> tuple[int | None, float | None, int | None]:
    if not values:
        return None, None, None
    return min(values), float(median(values)), max(values)


@dataclass(slots=True)
class MonitorResolvent:
    solves: list[MonitorResolventSolve] = field(default_factory=list)

    def record_solve(
        self,
        resolvent: str,
        alpha: float,
        block_size: int,
        iteration_count: int,
        error: float,
        scale: float,
        converged: bool,
    ) -> None:
        self.solves.append(
            MonitorResolventSolve(
                resolvent=resolvent,
                alpha=alpha,
                block_size=block_size,
                iteration_count=iteration_count,
                error=error,
                scale=scale,
                converged=converged,
            )
        )

    def summary(self) -> MonitorSummaryResolvent:
        iterations = [
            solve.iteration_count
            for solve in self.solves
        ]
        errors = [
            solve.error
            for solve in self.solves
        ]
        iteration_min, iteration_median, iteration_max = _min_median_max_int(iterations)
        error_min, error_median, error_max = _min_median_max_float(errors)
        return MonitorSummaryResolvent(
            solve_count=len(self.solves),
            failure_count=sum(
                1
                for solve in self.solves
                if not solve.converged
            ),
            iteration_min=iteration_min,
            iteration_median=iteration_median,
            iteration_max=iteration_max,
            error_min=error_min,
            error_median=error_median,
            error_max=error_max,
        )

    def clear(self) -> None:
        self.solves.clear()


__all__ = [
    "MonitorResolvent",
    "MonitorResolventSolve",
    "MonitorSummaryResolvent",
]
