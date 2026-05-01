from __future__ import annotations

from abc import ABC, abstractmethod

from stark.accelerators.binding import BoundDerivative
from stark.auditor import Auditor
from stark.contracts import Derivative, ImExDerivative, IntervalLike, State, Workbench
from stark.execution.adaptive_controller import AdaptiveController
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.monitor import Monitor, MonitorStep
from stark.schemes.display import display_imex_resolvent_problem, display_implicit_resolvent_problem

_ADVANCE_ACCEPTED_DT = 0
_ADVANCE_T_START = 1
_ADVANCE_PROPOSED_DT = 2
_ADVANCE_NEXT_DT = 3
_ADVANCE_ERROR_RATIO = 4
_ADVANCE_REJECTION_COUNT = 5


class SchemeBase(ABC):
    descriptor: object
    tableau: object

    @classmethod
    def display_tableau(cls) -> str:
        return cls.descriptor.display_tableau(cls.tableau)

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return self.descriptor.repr_for(type(self).__name__, self.tableau)

    def __str__(self) -> str:
        return self.display_tableau()

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)


class SchemeBaseFixed(SchemeBase):
    def assign_monitor(self, monitor: Monitor) -> None:
        del monitor

    def unassign_monitor(self) -> None:
        pass


class SchemeBaseAdaptive(SchemeBase):
    __slots__ = (
        "regulator",
        "controller",
        "advance_report",
        "redirect_call",
        "redirect_advance_body",
        "_controller",
        "_monitor",
        "_ratio",
        "_bound",
        "_runtime_bound",
    )

    def initialise_runtime(self, regulator: Regulator | None = None) -> None:
        self.regulator = regulator if regulator is not None else self.default_regulator()
        self.controller = AdaptiveController(self.regulator)
        self.advance_report = [0.0, 0.0, 0.0, 0.0, 0.0, 0]
        self._controller = None
        self._monitor = None
        self._ratio = None
        self._bound = None
        self._runtime_bound = False
        self.redirect_advance_body = self.advance_body
        self.refresh_call()

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator()

    @abstractmethod
    def advance_body(self, interval: IntervalLike, state: State) -> None:
        """Advance one accepted step and overwrite `advance_report`."""

    def assign_executor(self, executor: Executor) -> None:
        self._controller = executor.adaptive_controller()
        self._ratio = executor.ratio
        self._bound = executor.bound
        self._runtime_bound = True
        self.refresh_call()

    def unassign_executor(self) -> None:
        self._runtime_bound = False
        self._controller = None
        self._ratio = None
        self._bound = None
        self.refresh_call()

    def assign_monitor(self, monitor: Monitor) -> None:
        self._monitor = monitor
        self.refresh_call()

    def unassign_monitor(self) -> None:
        self._monitor = None
        self.refresh_call()

    def refresh_call(self) -> None:
        if not self._runtime_bound:
            self.redirect_call = self.bind_and_call
            return
        self.redirect_call = self.monitored_call if self._monitor is not None else self.pure_call

    def bind_advance_body(self, advance_body) -> None:
        self.redirect_advance_body = advance_body

    def bind_and_call(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        self.assign_executor(executor)
        return self.redirect_call(interval, state, executor)

    def pure_call(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        self.redirect_advance_body(interval, state)
        return self.advance_report[_ADVANCE_ACCEPTED_DT]

    def monitored_call(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        self.redirect_advance_body(interval, state)
        advance_report = self.advance_report
        accepted_dt = advance_report[_ADVANCE_ACCEPTED_DT]
        monitor = self._monitor
        if monitor is not None:
            monitor(
                MonitorStep(
                    scheme=self.short_name,
                    t_start=advance_report[_ADVANCE_T_START],
                    t_end=advance_report[_ADVANCE_T_START] + accepted_dt,
                    proposed_dt=advance_report[_ADVANCE_PROPOSED_DT],
                    accepted_dt=accepted_dt,
                    next_dt=advance_report[_ADVANCE_NEXT_DT],
                    error_ratio=advance_report[_ADVANCE_ERROR_RATIO],
                    rejection_count=advance_report[_ADVANCE_REJECTION_COUNT],
                )
            )
        return accepted_dt


class SchemeBaseExplicit(SchemeBase):
    def initialise_explicit(self, derivative: Derivative, workbench: Workbench) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.derivative = BoundDerivative(derivative)
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.k1 = translation_probe

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)


class SchemeBaseImEx(SchemeBase):
    def initialise_imex(self, derivative: ImExDerivative, workbench: Workbench) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_imex_scheme_inputs(derivative, workbench, translation_probe)
        self.workspace = SchemeWorkspace(workbench, translation_probe)

    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_imex_resolvent_problem(cls.tableau, cls.descriptor.short_name, cls.descriptor.full_name)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)


class SchemeBaseImplicit(SchemeBase):
    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_implicit_resolvent_problem(cls.tableau, cls.descriptor.short_name, cls.descriptor.full_name)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.stepper.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.stepper.snapshot_state(state)


class SchemeBaseExplicitFixed(SchemeBaseExplicit, SchemeBaseFixed):
    __slots__ = ("derivative", "k1", "workspace", "redirect_call")

    def __init__(self, derivative: Derivative, workbench: Workbench) -> None:
        self.initialise_explicit(derivative, workbench)
        self.initialise_buffers()
        self.redirect_call = self.generic_call

    @abstractmethod
    def initialise_buffers(self) -> None:
        """Allocate stage-specific scratch storage beyond the shared `k1`."""

    @abstractmethod
    def generic_call(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        """Advance one fixed step using the generic translation operations."""

    def bind_fixed_call(self, call) -> None:
        self.redirect_call = call

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        return self.redirect_call(interval, state, executor)

class SchemeBaseExplicitAdaptive(SchemeBaseExplicit, SchemeBaseAdaptive):
    __slots__ = ("derivative", "k1", "workspace")

    def __init__(self, derivative: Derivative, workbench: Workbench, regulator: Regulator | None = None) -> None:
        self.initialise_explicit(derivative, workbench)
        self.initialise_runtime(regulator)
        self.initialise_buffers()

    @abstractmethod
    def initialise_buffers(self) -> None:
        """Allocate stage-specific scratch storage beyond the shared `k1`."""


class SchemeBaseImplicitFixed(SchemeBaseImplicit, SchemeBaseFixed):
    pass


class SchemeBaseImplicitAdaptive(SchemeBaseImplicit, SchemeBaseAdaptive):
    pass


class SchemeBaseImExFixed(SchemeBaseImEx, SchemeBaseFixed):
    __slots__ = ("workspace",)

    def __init__(self, derivative: ImExDerivative, workbench: Workbench) -> None:
        self.initialise_imex(derivative, workbench)


class SchemeBaseImExAdaptive(SchemeBaseImEx, SchemeBaseAdaptive):
    __slots__ = ("workspace",)

    def __init__(self, derivative: ImExDerivative, workbench: Workbench, regulator: Regulator | None = None) -> None:
        self.initialise_imex(derivative, workbench)
        self.initialise_runtime(regulator)


__all__ = [
    "SchemeBase",
    "SchemeBaseAdaptive",
    "SchemeBaseExplicit",
    "SchemeBaseExplicitAdaptive",
    "SchemeBaseExplicitFixed",
    "SchemeBaseFixed",
    "SchemeBaseImEx",
    "SchemeBaseImExAdaptive",
    "SchemeBaseImExFixed",
    "SchemeBaseImplicit",
    "SchemeBaseImplicitAdaptive",
    "SchemeBaseImplicitFixed",
]
