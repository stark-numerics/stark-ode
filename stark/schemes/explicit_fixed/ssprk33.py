from __future__ import annotations

from stark.algebraist.classic import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    with_explicit_workspace_methods,
    with_fixed_step_monitoring,
    initialise_explicit_support,
    refresh_fixed_step_call,
    unbound_scheme_call,
    with_scheme_display,
)
from stark.schemes.support.tableau import ButcherTableau


SSPRK33_TABLEAU = ButcherTableau(
    c=(0.0, 1.0, 0.5),
    a=((), (1.0,), (0.25, 0.25)),
    b=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    order=3,
)

SSPRK33_A = SSPRK33_TABLEAU.a
SSPRK33_B = SSPRK33_TABLEAU.b


@with_scheme_display
@with_fixed_step_monitoring
@with_explicit_workspace_methods
class SchemeSSPRK33:
    """The three-stage third-order strong-stability-preserving RK method.

    SSPRK33 is designed for problems where preserving monotonicity or other
    forward-Euler stability properties under a step restriction matters, such as
    hyperbolic PDE discretisations with nonlinear limiters.

    Further reading: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "_monitor",
        "advance_state",
        "call_pure",
        "combine_stage2",
        "combine_stage3",
        "derivative",
        "explicit",
        "k1",
        "k2",
        "k3",
        "redirect_call",
        "stage",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor("SSPRK33", "SSP RK33")
    tableau = SSPRK33_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        algebraist: Algebraist | None = None,
    ) -> None:
        self.advance_state = unbound_scheme_call
        self.combine_stage2 = unbound_scheme_call
        self.combine_stage3 = unbound_scheme_call
        self._monitor = None
        self.call_pure = self.call_inline
        self.redirect_call = self.call_pure
        initialise_explicit_support(self, derivative, workbench)
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2, self.k3 = workspace.allocate_translation_buffers(3)
        refresh_fixed_step_call(self)

        if algebraist is not None:
            self.use_specialist(algebraist)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def use_specialist(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
        self.combine_stage2 = calls.require_stage_state_call(1, type(self).__name__)
        self.combine_stage3 = calls.require_stage_state_call(2, type(self).__name__)
        self.advance_state = calls.solution_state_call
        self.call_pure = self.call_specialized
        refresh_fixed_step_call(self)

    def call_inline(
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
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval

        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3

        dt = interval.step if interval.step <= remaining else remaining
        half_dt = 0.5 * dt

        derivative(interval, state, k1)

        trial = scale(dt, k1, trial_buffer)
        trial(state, stage)
        derivative(stage_interval(interval, dt, dt), stage, k2)

        trial = combine2(
            dt * SSPRK33_A[2][0],
            k1,
            dt * SSPRK33_A[2][1],
            k2,
            trial_buffer,
        )
        trial(state, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k3)

        delta = combine3(
            dt * SSPRK33_B[0],
            k1,
            dt * SSPRK33_B[1],
            k2,
            dt * SSPRK33_B[2],
            k3,
            trial_buffer,
        )

        apply_delta(delta, state)
        return dt

    def call_specialized(
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
        half_dt = 0.5 * dt

        stage = self.stage
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3

        derivative = self.derivative
        stage_interval = self.workspace.stage_interval
        combine_stage2 = self.combine_stage2
        combine_stage3 = self.combine_stage3
        advance_state = self.advance_state

        derivative(interval, state, k1)

        combine_stage2(state, dt, k1, stage)
        derivative(stage_interval(interval, dt, dt), stage, k2)

        combine_stage3(state, dt, k1, k2, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k3)

        advance_state(state, dt, k1, k2, k3, state)
        return dt


__all__ = ["SSPRK33_TABLEAU", "SchemeSSPRK33"]
