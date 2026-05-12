from __future__ import annotations

"""Transitional support classes for built-in scheme implementations.

STARK schemes are public structural contracts: a scheme is usable when it
provides `__call__(interval, state, executor)`, `snapshot_state(state)`, and
`set_apply_delta_safety(enabled)`.

Historically, many built-in schemes inherited algorithmic call routing from
these base classes. New and refactored built-in schemes should instead own
their public `__call__` directly and use these bases only for temporary shared
setup while the scheme family is migrated.

Do not add new scheme algorithms that depend on inherited `__call__` routing.
"""

from abc import ABC, abstractmethod

from stark.contracts import Derivative, ImExDerivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.monitor import Monitor, MonitorStep
from stark.schemes.display import (
    display_imex_resolvent_problem,
    display_implicit_resolvent_problem,
)
from stark.schemes.support.adaptive import (
    _ADVANCE_ACCEPTED_DT,
    _ADVANCE_ERROR_RATIO,
    _ADVANCE_NEXT_DT,
    _ADVANCE_PROPOSED_DT,
    _ADVANCE_REJECTION_COUNT,
    _ADVANCE_T_START,
    SchemeSupportAdaptive,
)
from stark.schemes.support.display import SchemeDisplay
from stark.schemes.support.explicit import SchemeSupportExplicit


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
    """Temporary fixed-step monitor surface for unconverted fixed schemes.

    Fixed schemes currently do not emit monitor records. Converted fixed schemes
    should own their own call routing so later monitor support can redirect at
    the concrete-scheme boundary.
    """

    def assign_monitor(self, monitor: Monitor) -> None:
        del monitor

    def unassign_monitor(self) -> None:
        pass


class SchemeBaseAdaptive(SchemeBase):
    """Transitional adaptive runtime binder.

    This class still owns executor/monitor redirection for unconverted adaptive
    schemes. Converted adaptive schemes should define their own `__call__`,
    `call_pure`, and `call_monitored`, while delegating controller state to
    `SchemeSupportAdaptive`.

    The inherited routing here remains only to keep unconverted schemes working
    during migration.
    """

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
            self.redirect_call = self.call_bind
            return

        self.redirect_call = (
            self.call_monitored
            if self.adaptive.monitor is not None
            else self.call_pure
        )

    def bind_advance_body(self, advance_body) -> None:
        """Legacy advance-body hook for unconverted adaptive schemes.

        Converted adaptive schemes should select their own pure call path in the
        concrete scheme instead of relying on this inherited hook.
        """

        self.redirect_advance_body = advance_body

    def call_bind(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        self.assign_executor(executor)
        return self.redirect_call(interval, state, executor)

    def bind_and_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        """Legacy alias for the old adaptive base-routing name."""

        return self.call_bind(interval, state, executor)

    def call_pure(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        self.redirect_advance_body(interval, state)
        return self.advance_report[_ADVANCE_ACCEPTED_DT]

    def pure_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        """Legacy alias for the old adaptive pure-call name."""

        return self.call_pure(interval, state, executor)

    def call_monitored(
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

    def monitored_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        """Legacy alias for the old adaptive monitored-call name."""

        return self.call_monitored(interval, state, executor)


class SchemeBaseExplicit(SchemeBase):
    def initialise_explicit(self, derivative: Derivative, workbench: Workbench) -> None:
        self.explicit = SchemeSupportExplicit.from_inputs(derivative, workbench)

        # Preserve the attributes existing concrete schemes already use.
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

        # Kept here for now. This will likely move into a support object when
        # IMEX schemes are converted to scheme-owned call routing.
        from stark.auditor import Auditor
        from stark.machinery.stage_solve.workspace import SchemeWorkspace

        Auditor.require_imex_scheme_inputs(derivative, workbench, translation_probe)
        self.workspace = SchemeWorkspace(workbench, translation_probe)

    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_imex_resolvent_problem(
            cls.tableau,
            cls.descriptor.short_name,
            cls.descriptor.full_name,
        )

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)


class SchemeBaseImplicit(SchemeBase):
    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_implicit_resolvent_problem(
            cls.tableau,
            cls.descriptor.short_name,
            cls.descriptor.full_name,
        )

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.stepper.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.stepper.snapshot_state(state)


class SchemeBaseExplicitFixed(SchemeBaseExplicit, SchemeBaseFixed):
    """Legacy fixed-explicit routing base for unconverted schemes.

    Converted fixed explicit schemes should define `__call__` in the concrete
    scheme and route through scheme-owned `call_pure` / `redirect_call`
    attributes. This inherited `__call__` remains only until the fixed explicit
    family has been migrated.
    """

    __slots__ = ("derivative", "explicit", "k1", "workspace", "redirect_call")

    def __init__(self, derivative: Derivative, workbench: Workbench) -> None:
        self.initialise_explicit(derivative, workbench)
        self.initialise_buffers()
        self.redirect_call = self.call_generic

    @abstractmethod
    def initialise_buffers(self) -> None:
        """Allocate stage-specific scratch storage beyond the shared `k1`."""

    @abstractmethod
    def call_generic(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        """Advance one fixed step using the generic translation operations."""

    def generic_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        """Legacy alias for the old fixed-explicit generic-call name."""

        return self.call_generic(interval, state, executor)

    def bind_fixed_call(self, call) -> None:
        """Legacy call-routing hook for unconverted fixed explicit schemes.

        New or converted schemes should select their own pure call path inside
        the concrete scheme instead of using this base hook.
        """

        self.redirect_call = call

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        """Legacy inherited call route for unconverted fixed explicit schemes."""

        return self.redirect_call(interval, state, executor)


class SchemeBaseExplicitAdaptive(SchemeBaseExplicit, SchemeBaseAdaptive):
    """Shared setup base for explicit adaptive schemes during migration.

    Converted adaptive schemes should own their public `__call__` and make their
    rejection/acceptance flow visible in the concrete scheme file. This base
    still provides setup and transitional adaptive runtime binding.
    """

    __slots__ = ("derivative", "explicit", "k1", "workspace")

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        regulator: Regulator | None = None,
    ) -> None:
        self.initialise_explicit(derivative, workbench)
        self.initialise_runtime(regulator)
        self.initialise_buffers()

    @abstractmethod
    def initialise_buffers(self) -> None:
        """Allocate stage-specific scratch storage beyond the shared `k1`."""


class SchemeBaseImplicitFixed(SchemeBaseImplicit, SchemeBaseFixed):
    """Shared implicit fixed-step support.

    Converted implicit fixed schemes should own their public `__call__` directly
    in the concrete scheme file.
    """


class SchemeBaseImplicitAdaptive(SchemeBaseImplicit, SchemeBaseAdaptive):
    """Shared implicit adaptive support during scheme-call migration.

    Converted implicit adaptive schemes should own their public `__call__`
    directly and keep adaptive accept/reject flow visible in the concrete scheme
    file.
    """


class SchemeBaseImExFixed(SchemeBaseImEx, SchemeBaseFixed):
    """Shared IMEX fixed-step setup.

    Converted IMEX fixed schemes should own their public `__call__` directly in
    the concrete scheme file.
    """

    __slots__ = ("workspace",)

    def __init__(self, derivative: ImExDerivative, workbench: Workbench) -> None:
        self.initialise_imex(derivative, workbench)


class SchemeBaseImExAdaptive(SchemeBaseImEx, SchemeBaseAdaptive):
    """Shared IMEX adaptive setup during scheme-call migration.

    Converted IMEX adaptive schemes should own their public `__call__` directly
    and keep explicit/implicit adaptive flow visible in the concrete scheme
    file.
    """

    __slots__ = ("workspace",)

    def __init__(
        self,
        derivative: ImExDerivative,
        workbench: Workbench,
        regulator: Regulator | None = None,
    ) -> None:
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
    "_ADVANCE_ACCEPTED_DT",
    "_ADVANCE_ERROR_RATIO",
    "_ADVANCE_NEXT_DT",
    "_ADVANCE_PROPOSED_DT",
    "_ADVANCE_REJECTION_COUNT",
    "_ADVANCE_T_START",
]