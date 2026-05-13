from __future__ import annotations

from stark.algebraist import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.schemes.base import SchemeBaseExplicitFixed
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


RK38_TABLEAU = ButcherTableau(
    c=(0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    a=((), (1.0 / 3.0,), (-1.0 / 3.0, 1.0), (1.0, -1.0, 1.0)),
    b=(1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0),
    order=4,
)

RK38_A = RK38_TABLEAU.a
RK38_B = RK38_TABLEAU.b


class SchemeRK38(SchemeBaseExplicitFixed):
    """The four-stage 3/8-rule fourth-order Runge-Kutta method.

    This is an alternative fourth-order RK4 family member with different stage
    coefficients from the classical RK4 method. It reaches the same formal order
    with a distinct tableau and is useful for comparisons or tableau studies.

    Further reading: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "advance_state",
        "combine_stage2",
        "combine_stage3",
        "combine_stage4",
        "k2",
        "k3",
        "k4",
        "call_pure",
        "redirect_call",
        "stage",
        "trial",
    )

    descriptor = SchemeDescriptor("RK38", "3/8 Rule Runge-Kutta")
    tableau = RK38_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        algebraist: Algebraist | None = None,
    ) -> None:
        self.advance_state = None
        self.combine_stage2 = None
        self.combine_stage3 = None
        self.combine_stage4 = None

        super().__init__(derivative, workbench)

        self.call_pure = self.call_generic
        self.redirect_call = self.call_pure

        if algebraist is not None:
            self.bind_algebraist_path(algebraist)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def initialise_buffers(self) -> None:
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2, self.k3, self.k4 = workspace.allocate_translation_buffers(4)

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
        self.combine_stage2 = calls.stage_state_calls[1]
        self.combine_stage3 = calls.stage_state_calls[2]
        self.combine_stage4 = calls.stage_state_calls[3]
        self.advance_state = calls.solution_state_call
        self.call_pure = self.call_algebraist
        self.redirect_call = self.call_pure

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
        combine2 = workspace.combine2
        combine3 = workspace.combine3
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
        one_third_dt = dt / 3.0
        two_thirds_dt = 2.0 * dt / 3.0

        derivative(interval, state, k1)

        trial = scale(trial_buffer, one_third_dt, k1)
        trial(state, stage)
        derivative(stage_interval(interval, dt, one_third_dt), stage, k2)

        trial = combine2(
            trial_buffer,
            dt * RK38_A[2][0],
            k1,
            dt * RK38_A[2][1],
            k2,
        )
        trial(state, stage)
        derivative(stage_interval(interval, dt, two_thirds_dt), stage, k3)

        trial = combine3(
            trial_buffer,
            dt * RK38_A[3][0],
            k1,
            dt * RK38_A[3][1],
            k2,
            dt * RK38_A[3][2],
            k3,
        )
        trial(state, stage)
        derivative(stage_interval(interval, dt, dt), stage, k4)

        delta = combine4(
            trial_buffer,
            dt * RK38_B[0],
            k1,
            dt * RK38_B[1],
            k2,
            dt * RK38_B[2],
            k3,
            dt * RK38_B[3],
            k4,
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
        one_third_dt = dt / 3.0
        two_thirds_dt = 2.0 * dt / 3.0

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

        combine_stage2(stage, state, dt, k1)
        derivative(stage_interval(interval, dt, one_third_dt), stage, k2)

        combine_stage3(stage, state, dt, k1, k2)
        derivative(stage_interval(interval, dt, two_thirds_dt), stage, k3)

        combine_stage4(stage, state, dt, k1, k2, k3)
        derivative(stage_interval(interval, dt, dt), stage, k4)

        advance_state(state, state, dt, k1, k2, k3, k4)
        return dt


__all__ = ["RK38_TABLEAU", "SchemeRK38"]