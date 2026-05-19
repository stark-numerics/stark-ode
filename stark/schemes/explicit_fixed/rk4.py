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


RK4_TABLEAU = ButcherTableau(
    c=(0.0, 0.5, 0.5, 1.0),
    a=(
        (),
        (0.5,),
        (0.0, 0.5),
        (0.0, 0.0, 1.0),
    ),
    b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    order=4,
)

RK4_B = RK4_TABLEAU.b


@with_scheme_display
@with_fixed_step_monitoring
@with_explicit_workspace_methods
class SchemeRK4:
    """The classical four-stage fourth-order Runge-Kutta method.

    RK4 is the best-known fixed-step explicit Runge-Kutta scheme. It is a very
    common general-purpose baseline because it offers fourth-order accuracy,
    straightforward staging, and good behavior on many non-stiff problems.

    Further reading: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "_monitor",
        "advance_state",
        "call_pure",
        "combine_stage2",
        "combine_stage3",
        "combine_stage4",
        "derivative",
        "explicit",
        "k1",
        "k2",
        "k3",
        "k4",
        "redirect_call",
        "stage",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor("RK4", "Classical Runge-Kutta")
    tableau = RK4_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        algebraist: Algebraist | None = None,
    ) -> None:
        self.advance_state = unbound_scheme_call
        self.combine_stage2 = unbound_scheme_call
        self.combine_stage3 = unbound_scheme_call
        self.combine_stage4 = unbound_scheme_call
        self._monitor = None
        self.call_pure = self.call_generic
        self.redirect_call = self.call_pure
        initialise_explicit_support(self, derivative, workbench)
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2, self.k3, self.k4 = workspace.allocate_translation_buffers(4)
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
        self.combine_stage2 = calls.require_stage_state_call(1, type(self).__name__)
        self.combine_stage3 = calls.require_stage_state_call(2, type(self).__name__)
        self.combine_stage4 = calls.require_stage_state_call(3, type(self).__name__)
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
        combine4 = workspace.combine4
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval

        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        dt = interval.step if interval.step <= remaining else remaining
        half_dt = 0.5 * dt

        derivative(interval, state, k1)

        trial = scale(half_dt, k1, trial_buffer)
        trial(state, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k2)

        trial = scale(half_dt, k2, trial_buffer)
        trial(state, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k3)

        trial = scale(dt, k3, trial_buffer)
        trial(state, stage)
        derivative(stage_interval(interval, dt, dt), stage, k4)

        delta = combine4(
            dt * RK4_B[0],
            k1,
            dt * RK4_B[1],
            k2,
            dt * RK4_B[2],
            k3,
            dt * RK4_B[3],
            k4,
            trial_buffer,
        )

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
        half_dt = 0.5 * dt

        stage = self.stage
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        derivative = self.derivative
        stage_interval = self.workspace.stage_interval
        combine_stage2 = self.combine_stage2
        combine_stage3 = self.combine_stage3
        combine_stage4 = self.combine_stage4
        advance_state = self.advance_state

        derivative(interval, state, k1)

        combine_stage2(state, dt, k1, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k2)

        combine_stage3(state, dt, k2, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k3)

        combine_stage4(state, dt, k3, stage)
        derivative(stage_interval(interval, dt, dt), stage, k4)

        advance_state(state, dt, k1, k2, k3, k4, state)
        return dt


__all__ = ["RK4_TABLEAU", "SchemeRK4"]
