from __future__ import annotations

"""Shared setup classes for built-in scheme implementations.

A STARK scheme is usable when it provides `__call__(interval, state, executor)`,
`snapshot_state(state)`, and `set_apply_delta_safety(enabled)`.

Concrete built-in schemes own their public call routing. These base classes
provide shared setup, display, runtime binding, snapshot, and safety helpers;
they do not own numerical advance algorithms.
"""

from abc import ABC, abstractmethod

from stark.contracts import Derivative, ImExDerivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.monitor import Monitor
from stark.schemes.display import (
    display_imex_resolvent_problem,
    display_implicit_resolvent_problem,
)
from stark.schemes.support.adaptive import SchemeSupportAdaptive
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
    """Shared fixed-step monitor surface.

    Fixed schemes currently do not emit monitor records through this base.
    Concrete fixed schemes own their call routing.
    """

    def assign_monitor(self, monitor: Monitor) -> None:
        del monitor

    def unassign_monitor(self) -> None:
        pass


class SchemeBaseAdaptive(SchemeBase):
    """Shared adaptive runtime binding support.

    Adaptive concrete schemes own `__call__`, `call_bind`, `call_pure`,
    `call_monitored`, and their numerical accept/reject loops.

    This base owns only the adaptive support object and the executor/monitor
    binding operations that refresh the concrete scheme's selected call path.
    """

    __slots__ = (
        "adaptive",
        "redirect_call",
    )

    def initialise_runtime(self, regulator: Regulator | None = None) -> None:
        self.adaptive = SchemeSupportAdaptive(
            regulator if regulator is not None else self.default_regulator()
        )
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
    """Shared fixed explicit setup.

    Concrete fixed explicit schemes own their public `__call__`. This base
    provides common explicit setup and the call-selection hook used by explicit
    generated-algebra paths.
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
        """Advance one fixed step using generic translation operations."""

    def bind_fixed_call(self, call) -> None:
        self.redirect_call = call

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        return self.redirect_call(interval, state, executor)


class SchemeBaseExplicitAdaptive(SchemeBaseExplicit, SchemeBaseAdaptive):
    """Shared setup for explicit adaptive schemes."""

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
    """Shared setup for implicit fixed-step schemes."""


class SchemeBaseImplicitAdaptive(SchemeBaseImplicit, SchemeBaseAdaptive):
    """Shared setup for implicit adaptive schemes."""


class SchemeBaseImExFixed(SchemeBaseImEx, SchemeBaseFixed):
    """Shared setup for IMEX fixed-step schemes."""

    __slots__ = ("workspace",)

    def __init__(self, derivative: ImExDerivative, workbench: Workbench) -> None:
        self.initialise_imex(derivative, workbench)


class SchemeBaseImExAdaptive(SchemeBaseImEx, SchemeBaseAdaptive):
    """Shared setup for IMEX adaptive schemes."""

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
]