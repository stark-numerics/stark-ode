from __future__ import annotations

from abc import ABC, abstractmethod

from stark.auditor import Auditor
from stark.contracts import Derivative, ImExDerivative, IntervalLike, State, Workbench
from stark.execution.adaptive_controller import AdaptiveController
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.monitor import Monitor, MonitorStep
from stark.schemes.display import display_imex_resolvent_problem, display_implicit_resolvent_problem
from stark.schemes.support.display import SchemeDisplay
from stark.schemes.support.explicit import SchemeSupportExplicit
from stark.execution.adaptive_controller import AdaptiveController
from stark.schemes.support.adaptive import (
    _ADVANCE_ACCEPTED_DT,
    _ADVANCE_ERROR_RATIO,
    _ADVANCE_NEXT_DT,
    _ADVANCE_PROPOSED_DT,
    _ADVANCE_REJECTION_COUNT,
    _ADVANCE_T_START,
    SchemeSupportAdaptive,
)

class SchemeBase(ABC):
    descriptor: object
    tableau: object

    @classmethod
    def scheme_display(cls) -> SchemeDisplay:
        return SchemeDisplay(cls.descriptor, cls.tableau)

    @classmethod
    def display_tableau(cls) -> str:
        return cls.scheme_display().display_tableau()

    @property
    def short_name(self) -> str:
        return type(self).scheme_display().short_name

    @property
    def full_name(self) -> str:
        return type(self).scheme_display().full_name

    def __repr__(self) -> str:
        return type(self).scheme_display().repr_for(type(self).__name__)

    def __str__(self) -> str:
        return type(self).scheme_display().str_for()

    def __format__(self, format_spec: str) -> str:
        return type(self).scheme_display().format_for(format_spec)

class SchemeBaseFixed(SchemeBase):
    def assign_monitor(self, monitor: Monitor) -> None:
        del monitor

    def unassign_monitor(self) -> None:
        pass


class SchemeBaseAdaptive(SchemeBase):
    __slots__ = (
        "adaptive",
        "redirect_call",
        "redirect_advance_body",
    )

    def initialise_runtime(self, regulator: Regulator | None = None) -> None:
        self.adaptive = SchemeSupportAdaptive(
            regulator if regulator is not None else self.default_regulator()
        )
        self.redirect_advance_body = self.advance_body
        self.refresh_call()

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator()

    @property
    def regulator(self) -> Regulator:
        return self.adaptive.regulator

    @property
    def controller(self):
        return self.adaptive.controller

    @property
    def advance_report(self) -> list[float | int]:
        return self.adaptive.advance_report

    @property
    def _controller(self):
        return self.adaptive.active_controller

    @property
    def _monitor(self):
        return self.adaptive.monitor

    @property
    def _ratio(self):
        return self.adaptive.ratio

    @property
    def _bound(self):
        return self.adaptive.bound

    @property
    def _runtime_bound(self) -> bool:
        return self.adaptive.runtime_bound

    @abstractmethod
    def advance_body(self, interval: IntervalLike, state: State) -> None:
        """Advance one accepted step and overwrite `advance_report`."""

    def assign_executor(self, executor: Executor) -> None:
        self.adaptive.assign_executor(executor)
        self.refresh_call()

    def unassign_executor(self) -> None:
        self.adaptive.unassign_executor()
        self.refresh_call()

    def assign_monitor(self, monitor: Monitor) -> None:
        self.adaptive.assign_monitor(monitor)
        self.refresh_call()

    def unassign_monitor(self) -> None:
        self.adaptive.unassign_monitor()
        self.refresh_call()

    def refresh_call(self) -> None:
        if not self.adaptive.runtime_bound:
            self.redirect_call = self.bind_and_call
            return

        self.redirect_call = (
            self.monitored_call
            if self.adaptive.monitor is not None
            else self.pure_call
        )

    def bind_advance_body(self, advance_body) -> None:
        self.redirect_advance_body = advance_body

    def bind_and_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        self.assign_executor(executor)
        return self.redirect_call(interval, state, executor)

    def pure_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        self.redirect_advance_body(interval, state)
        return self.advance_report[_ADVANCE_ACCEPTED_DT]

    def monitored_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        self.redirect_advance_body(interval, state)
        report = self.adaptive.report()
        monitor = self.adaptive.monitor

        if monitor is not None:
            monitor(
                MonitorStep(
                    scheme=self.short_name,
                    t_start=report.t_start,
                    t_end=report.t_end,
                    proposed_dt=report.proposed_dt,
                    accepted_dt=report.accepted_dt,
                    next_dt=report.next_dt,
                    error_ratio=report.error_ratio,
                    rejection_count=report.rejection_count,
                )
            )

        return report.accepted_dt

class SchemeBaseExplicit(SchemeBase):
    def initialise_explicit(self, derivative: Derivative, workbench: Workbench) -> None:
        self.explicit = SchemeSupportExplicit.from_inputs(derivative, workbench)
        self.derivative = self.explicit.derivative
        self.workspace = self.explicit.workspace
        self.k1 = self.explicit.k1

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.explicit.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.explicit.snapshot_state(state)


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
