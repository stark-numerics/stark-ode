from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from stark.contracts import IntervalLike, State
from stark.executor.adaptivity import ExecutorAdaptivity
from stark.schemes.support.executor import SchemeExecutor
from stark.schemes.support.monitoring import MonitorSchemeLike

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
    """Step-control support object for adaptive schemes.

    This object owns adaptivity binding, monitor binding, step-size
    calculation delegation, and the latest typed adaptive advance report.

    Concrete schemes own their public call routing and their accept/reject
    algorithm.
    """

    __slots__ = (
        "adaptivity",
        "_bound",
        "_adaptivity",
        "_monitor",
        "_ratio",
        "_report",
        "_runtime_bound",
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
        self._adaptivity = None
        self._monitor = None
        self._ratio = None
        self._bound = None
        self._runtime_bound = False

    @classmethod
    def from_adaptivity(cls, adaptivity: ExecutorAdaptivity) -> SchemeStepControl:
        return cls(adaptivity)

    @property
    def runtime_bound(self) -> bool:
        return self._runtime_bound

    @property
    def monitor(self) -> MonitorSchemeLike | None:
        return self._monitor

    @property
    def active_adaptivity(self):
        return self._adaptivity

    @property
    def ratio(self) -> ErrorRatio:
        return self._ratio

    @property
    def bound(self) -> ErrorBound:
        return self._bound

    def assign_executor(self, executor: SchemeExecutor) -> None:
        ratio = executor.ratio
        bound = executor.bound
        if not callable(ratio):
            raise TypeError("Adaptive SchemeExecutor must provide ratio(error, scale).")
        if not callable(bound):
            raise TypeError("Adaptive SchemeExecutor must provide bound(scale).")

        self._adaptivity = executor.adaptivity_or(self.adaptivity)
        self._ratio = ratio
        self._bound = bound
        self._runtime_bound = True

    def unassign_executor(self) -> None:
        self._runtime_bound = False
        self._adaptivity = None
        self._ratio = None
        self._bound = None

    def assign_monitor(self, monitor: MonitorSchemeLike) -> None:
        self._monitor = monitor

    def unassign_monitor(self) -> None:
        self._monitor = None

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
        adaptivity = self._adaptivity
        if adaptivity is None:
            raise RuntimeError("Adaptive SchemeExecutor has not been assigned.")

        return adaptivity.rejected_step(dt, error_ratio, remaining, label)

    def accepted_next_step(
        self,
        accepted_dt: float,
        error_ratio: float,
        remaining_after: float,
    ) -> float:
        adaptivity = self._adaptivity
        if adaptivity is None:
            raise RuntimeError("Adaptive SchemeExecutor has not been assigned.")

        return adaptivity.accepted_next_step(accepted_dt, error_ratio, remaining_after)

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
    refresh_adaptive_call(scheme)
    return support


def refresh_adaptive_call(scheme) -> None:
    if not scheme.step_control.runtime_bound:
        scheme.redirect_call = scheme.call_bind
        return

    scheme.redirect_call = (
        scheme.call_monitored
        if scheme.step_control.monitor is not None
        else scheme.call_pure
    )


def with_adaptive_runtime_methods(cls):
    """Install standard adaptive SchemeExecutor and monitor routing methods."""

    @property
    def adaptivity(self):
        return self.step_control.adaptivity

    def assign_executor(self, executor: SchemeExecutor) -> None:
        self.step_control.assign_executor(executor)
        refresh_adaptive_call(self)

    def unassign_executor(self) -> None:
        self.step_control.unassign_executor()
        refresh_adaptive_call(self)

    def assign_monitor(self, monitor: MonitorSchemeLike) -> None:
        self.step_control.assign_monitor(monitor)
        refresh_adaptive_call(self)

    def unassign_monitor(self) -> None:
        self.step_control.unassign_monitor()
        refresh_adaptive_call(self)

    def call_bind(
        self,
        interval: IntervalLike,
        state: State,
        executor: SchemeExecutor,
    ) -> float:
        self.assign_executor(executor)
        return self.redirect_call(interval, state, executor)

    def call_monitored(
        self,
        interval: IntervalLike,
        state: State,
        executor: SchemeExecutor,
    ) -> float:
        accepted_dt = self.call_pure(interval, state, executor)
        report = self.step_control.report()
        monitor = self.step_control.monitor

        if monitor is not None:
            monitor.record_adaptive_step(
                self.short_name,
                report.t_start,
                report.proposed_dt,
                report.accepted_dt,
                report.next_dt,
                report.error_ratio,
                report.rejection_count,
            )

        return accepted_dt

    cls.adaptivity = adaptivity
    cls.assign_executor = assign_executor
    cls.unassign_executor = unassign_executor
    cls.assign_monitor = assign_monitor
    cls.unassign_monitor = unassign_monitor
    cls.call_bind = call_bind
    cls.call_monitored = call_monitored
    return cls


__all__ = [
    "ErrorBound",
    "ErrorRatio",
    "SchemeStepAdaptiveProposal",
    "SchemeStepAdaptiveAdvanceReport",
    "SchemeStepControl",
    "default_adaptivity",
    "initialise_adaptive_runtime",
    "refresh_adaptive_call",
    "with_adaptive_runtime_methods",
]
