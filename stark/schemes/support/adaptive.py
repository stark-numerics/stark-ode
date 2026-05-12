from __future__ import annotations

from dataclasses import dataclass

from stark.contracts import IntervalLike
from stark.execution.adaptive_controller import AdaptiveController
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator


_ADVANCE_ACCEPTED_DT = 0
_ADVANCE_T_START = 1
_ADVANCE_PROPOSED_DT = 2
_ADVANCE_NEXT_DT = 3
_ADVANCE_ERROR_RATIO = 4
_ADVANCE_REJECTION_COUNT = 5


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

    @classmethod
    def from_advance_report(cls, advance_report: list[float | int]) -> ReportAdaptiveAdvance:
        return cls(
            accepted_dt=float(advance_report[_ADVANCE_ACCEPTED_DT]),
            t_start=float(advance_report[_ADVANCE_T_START]),
            proposed_dt=float(advance_report[_ADVANCE_PROPOSED_DT]),
            next_dt=float(advance_report[_ADVANCE_NEXT_DT]),
            error_ratio=float(advance_report[_ADVANCE_ERROR_RATIO]),
            rejection_count=int(advance_report[_ADVANCE_REJECTION_COUNT]),
        )


@dataclass(frozen=True, slots=True)
class ProposedAdaptiveStep:
    """Initial step proposal data for one adaptive advance attempt."""

    remaining: float
    dt: float
    proposed_dt: float
    t_start: float


class SchemeSupportAdaptive:
    """Runtime support object for adaptive schemes.

    This owns adaptive-controller state that used to live directly on
    `SchemeBaseAdaptive`, while leaving concrete schemes in control of their
    stage algebra and rejection loops.
    """

    __slots__ = (
        "advance_report",
        "controller",
        "regulator",
        "_bound",
        "_controller",
        "_monitor",
        "_ratio",
        "_runtime_bound",
    )

    def __init__(self, regulator: Regulator | None = None) -> None:
        self.regulator = regulator if regulator is not None else Regulator()
        self.controller = AdaptiveController(self.regulator)
        self.advance_report: list[float | int] = [0.0, 0.0, 0.0, 0.0, 0.0, 0]

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

    def record_stopped(self, interval: IntervalLike) -> None:
        report = self.advance_report
        report[_ADVANCE_ACCEPTED_DT] = 0.0
        report[_ADVANCE_T_START] = interval.present
        report[_ADVANCE_PROPOSED_DT] = 0.0
        report[_ADVANCE_NEXT_DT] = 0.0
        report[_ADVANCE_ERROR_RATIO] = 0.0
        report[_ADVANCE_REJECTION_COUNT] = 0

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
        report = self.advance_report
        report[_ADVANCE_ACCEPTED_DT] = accepted_dt
        report[_ADVANCE_T_START] = t_start
        report[_ADVANCE_PROPOSED_DT] = proposed_dt
        report[_ADVANCE_NEXT_DT] = next_dt
        report[_ADVANCE_ERROR_RATIO] = error_ratio
        report[_ADVANCE_REJECTION_COUNT] = rejection_count
        return ReportAdaptiveAdvance.from_advance_report(report)

    def report(self) -> ReportAdaptiveAdvance:
        return ReportAdaptiveAdvance.from_advance_report(self.advance_report)


__all__ = [
    "ProposedAdaptiveStep",
    "ReportAdaptiveAdvance",
    "SchemeSupportAdaptive",
    "_ADVANCE_ACCEPTED_DT",
    "_ADVANCE_ERROR_RATIO",
    "_ADVANCE_NEXT_DT",
    "_ADVANCE_PROPOSED_DT",
    "_ADVANCE_REJECTION_COUNT",
    "_ADVANCE_T_START",
]