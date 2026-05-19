from __future__ import annotations

from stark.algebraist import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.support import (
    with_explicit_workspace_methods,
    with_fixed_step_monitoring,
    initialise_explicit_support,
    refresh_fixed_step_call,
    unbound_scheme_call,
    with_scheme_display,
)
from stark.schemes.tableau import ButcherTableau


EULER_TABLEAU = ButcherTableau(
    c=(0.0,),
    a=((),),
    b=(1.0,),
    order=1,
)

EULER_B = EULER_TABLEAU.b


@with_scheme_display
@with_fixed_step_monitoring
@with_explicit_workspace_methods
class SchemeEuler:
    """Forward Euler, the basic first-order explicit Runge-Kutta method.

    This is the simplest one-step method in the library: evaluate the
    derivative once at the start of the step and advance with that slope. It is
    useful as a baseline and for very cheap exploratory integrations, but it is
    only first-order accurate and has a small stability region.

    Further reading: https://en.wikipedia.org/wiki/Euler_method
    """

    __slots__ = (
        "_monitor",
        "advance_state",
        "call_pure",
        "delta",
        "derivative",
        "explicit",
        "k1",
        "redirect_call",
        "workspace",
    )

    descriptor = SchemeDescriptor("Euler", "Forward Euler")
    tableau = EULER_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        algebraist: Algebraist | None = None,
    ) -> None:
        self.advance_state = unbound_scheme_call
        self._monitor = None
        self.call_pure = self.call_generic
        self.redirect_call = self.call_pure
        initialise_explicit_support(self, derivative, workbench)
        self.delta = self.workspace.allocate_translation()
        refresh_fixed_step_call(self)

        if algebraist is not None:
            self.use_algebraist(algebraist)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def use_algebraist(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
        self.advance_state = calls.solution_state_call
        self.call_pure = self.call_algebraist
        refresh_fixed_step_call(self)

    def call_generic(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        apply_delta = workspace.apply_delta

        k1 = self.k1
        delta_buffer = self.delta

        dt = interval.step if interval.step <= remaining else remaining

        derivative(interval, state, k1)

        delta = scale(dt * EULER_B[0], k1, delta_buffer)
        apply_delta(delta, state)

        return dt

    def call_algebraist(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining

        k1 = self.k1
        derivative = self.derivative
        advance_state = self.advance_state

        derivative(interval, state, k1)

        advance_state(state, dt, k1, state)
        return dt


__all__ = ["EULER_TABLEAU", "SchemeEuler"]
