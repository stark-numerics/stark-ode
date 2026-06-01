from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from stark.contracts import IntervalLike, State
from stark.executor.adaptivity import ExecutorAdaptivity
from stark.schemes.execution.executor import SchemeExecutor

ErrorRatio = Callable[[float, float], float]
ErrorBound = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class SchemeStepAdaptiveAdvanceReport:
    """Compact accepted-advance report for adaptive schemes."""

    accepted_dt: float
    t_start: float
    proposed_dt: float
    next_dt: float
    error_ratio: float
    rejection_count: int

    @property
    def t_end(self) -> float:
        return self.t_start + self.accepted_dt


@dataclass(frozen=True, slots=True)
class SchemeStepAdaptiveProposal:
    """Initial step proposal data for one adaptive advance attempt."""

    remaining: float
    dt: float
    proposed_dt: float
    t_start: float


class SchemeStepControl:
    """Small adaptive step-control support object.

    Concrete schemes own their call routing. This object keeps adaptive policy,
    cached executor-derived helpers, and the latest advance report.
    """

    __slots__ = (
        "adaptivity",
        "_active_adaptivity",
        "_bound",
        "_ratio",
        "_report",
    )

    def __init__(self, adaptivity: ExecutorAdaptivity | None = None) -> None:
        self.adaptivity = adaptivity if adaptivity is not None else ExecutorAdaptivity()
        self._report = SchemeStepAdaptiveAdvanceReport(
            accepted_dt=0.0,
            t_start=0.0,
            proposed_dt=0.0,
            next_dt=0.0,
            error_ratio=0.0,
            rejection_count=0,
        )
        self._active_adaptivity = self.adaptivity
        self._ratio = None
        self._bound = None

    @classmethod
    def from_adaptivity(cls, adaptivity: ExecutorAdaptivity) -> SchemeStepControl:
        return cls(adaptivity)

    @property
    def active_adaptivity(self):
        return self._active_adaptivity

    @property
    def ratio(self) -> ErrorRatio:
        ratio = self._ratio
        if ratio is None:
            raise RuntimeError("Adaptive SchemeExecutor has not been prepared.")
        return ratio

    @property
    def bound(self) -> ErrorBound:
        bound = self._bound
        if bound is None:
            raise RuntimeError("Adaptive SchemeExecutor has not been prepared.")
        return bound

    def cache_executor(self, executor: SchemeExecutor) -> None:
        ratio = executor.ratio
        bound = executor.bound
        if not callable(ratio):
            raise TypeError("Adaptive SchemeExecutor must provide ratio(error, scale).")
        if not callable(bound):
            raise TypeError("Adaptive SchemeExecutor must provide bound(scale).")

        self._active_adaptivity = executor.adaptivity_or(self.adaptivity)
        self._ratio = ratio
        self._bound = bound

    def propose_step(self, interval: IntervalLike) -> SchemeStepAdaptiveProposal:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return SchemeStepAdaptiveProposal(
                remaining=remaining,
                dt=0.0,
                proposed_dt=0.0,
                t_start=interval.present,
            )

        dt = interval.step if interval.step <= remaining else remaining
        return SchemeStepAdaptiveProposal(
            remaining=remaining,
            dt=dt,
            proposed_dt=dt,
            t_start=interval.present,
        )

    def rejected_step(
        self,
        dt: float,
        error_ratio: float,
        remaining: float,
        label: str,
    ) -> float:
        return self._active_adaptivity.rejected_step(dt, error_ratio, remaining, label)

    def accepted_next_step(
        self,
        accepted_dt: float,
        error_ratio: float,
        remaining_after: float,
    ) -> float:
        return self._active_adaptivity.accepted_next_step(
            accepted_dt, error_ratio, remaining_after
        )

    def record_stopped(self, interval: IntervalLike) -> SchemeStepAdaptiveAdvanceReport:
        self._report = SchemeStepAdaptiveAdvanceReport(
            accepted_dt=0.0,
            t_start=interval.present,
            proposed_dt=0.0,
            next_dt=0.0,
            error_ratio=0.0,
            rejection_count=0,
        )
        return self._report

    def record_accepted(
        self,
        *,
        accepted_dt: float,
        t_start: float,
        proposed_dt: float,
        next_dt: float,
        error_ratio: float,
        rejection_count: int,
    ) -> SchemeStepAdaptiveAdvanceReport:
        self._report = SchemeStepAdaptiveAdvanceReport(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return self._report

    def report(self) -> SchemeStepAdaptiveAdvanceReport:
        return self._report


def default_adaptivity(scheme) -> ExecutorAdaptivity:
    default_adaptivity = getattr(type(scheme), "default_adaptivity", None)
    if default_adaptivity is None:
        return ExecutorAdaptivity()

    return default_adaptivity()


def initialise_adaptive_runtime(
    scheme,
    adaptivity: ExecutorAdaptivity | None = None,
) -> SchemeStepControl:
    support = SchemeStepControl(
        adaptivity if adaptivity is not None else default_adaptivity(scheme)
    )
    scheme.step_control = support
    return support



def adaptive_adaptivity(self):
    """Return the scheme's adaptive controller policy."""

    return self.step_control.adaptivity


__all__ = [
    "ErrorBound",
    "ErrorRatio",
    "SchemeStepAdaptiveProposal",
    "SchemeStepAdaptiveAdvanceReport",
    "SchemeStepControl",
    "default_adaptivity",
    "initialise_adaptive_runtime",
    "adaptive_adaptivity",
]
