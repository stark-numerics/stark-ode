from __future__ import annotations

from dataclasses import dataclass

from stark.contracts import IntervalLike
from stark.execution.adaptive_controller import AdaptiveController
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator


@dataclass(frozen=True, slots=True)
class ReportAdaptiveAdvance:
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
class ProposedAdaptiveStep:
    """Initial step proposal data for one adaptive advance attempt."""

    remaining: float
    dt: float
    proposed_dt: float
    t_start: float


class SchemeSupportAdaptive:
    """Runtime support object for adaptive schemes.

    This object owns adaptive-controller binding, monitor binding, step-size
    controller delegation, and the latest typed adaptive advance report.

    Concrete schemes own their public call routing and their accept/reject
    algorithm.
    """

    __slots__ = (
        "controller",
        "regulator",
        "_bound",
        "_controller",
        "_monitor",
        "_ratio",
        "_report",
        "_runtime_bound",
    )

    def __init__(self, regulator: Regulator | None = None) -> None:
        self.regulator = regulator if regulator is not None else Regulator()
        self.controller = AdaptiveController(self.regulator)
        self._report = ReportAdaptiveAdvance(
            accepted_dt=0.0,
            t_start=0.0,
            proposed_dt=0.0,
            next_dt=0.0,
            error_ratio=0.0,
            rejection_count=0,
        )
        self._controller = None
        self._monitor = None
        self._ratio = None
        self._bound = None
        self._runtime_bound = False

    @classmethod
    def from_regulator(cls, regulator: Regulator) -> SchemeSupportAdaptive:
        return cls(regulator)

    @property
    def runtime_bound(self) -> bool:
        return self._runtime_bound

    @property
    def monitor(self):
        return self._monitor

    @property
    def active_controller(self):
        return self._controller

    @property
    def ratio(self):
        return self._ratio

    @property
    def bound(self):
        return self._bound

    def assign_executor(self, executor: Executor) -> None:
        self._controller = executor.adaptive_controller()
        self._ratio = executor.ratio
        self._bound = executor.bound
        self._runtime_bound = True

    def unassign_executor(self) -> None:
        self._runtime_bound = False
        self._controller = None
        self._ratio = None
        self._bound = None

    def assign_monitor(self, monitor) -> None:
        self._monitor = monitor

    def unassign_monitor(self) -> None:
        self._monitor = None

    def propose_step(self, interval: IntervalLike) -> ProposedAdaptiveStep:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return ProposedAdaptiveStep(
                remaining=remaining,
                dt=0.0,
                proposed_dt=0.0,
                t_start=interval.present,
            )

        dt = interval.step if interval.step <= remaining else remaining
        return ProposedAdaptiveStep(
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
        controller = self._controller
        if controller is None:
            raise RuntimeError("Adaptive executor has not been assigned.")

        return controller.rejected_step(dt, error_ratio, remaining, label)

    def accepted_next_step(
        self,
        accepted_dt: float,
        error_ratio: float,
        remaining_after: float,
    ) -> float:
        controller = self._controller
        if controller is None:
            raise RuntimeError("Adaptive executor has not been assigned.")

        return controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)

    def record_stopped(self, interval: IntervalLike) -> ReportAdaptiveAdvance:
        self._report = ReportAdaptiveAdvance(
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
    ) -> ReportAdaptiveAdvance:
        self._report = ReportAdaptiveAdvance(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return self._report

    def report(self) -> ReportAdaptiveAdvance:
        return self._report


__all__ = [
    "ProposedAdaptiveStep",
    "ReportAdaptiveAdvance",
    "SchemeSupportAdaptive",
]