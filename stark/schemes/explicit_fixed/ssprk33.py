from __future__ import annotations

from stark.algebraist import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.schemes.base import SchemeBaseExplicitFixed
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


SSPRK33_TABLEAU = ButcherTableau(
    c=(0.0, 1.0, 0.5),
    a=((), (1.0,), (0.25, 0.25)),
    b=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    order=3,
)

SSPRK33_A = SSPRK33_TABLEAU.a
SSPRK33_B = SSPRK33_TABLEAU.b


class SchemeSSPRK33(SchemeBaseExplicitFixed):
    """The three-stage third-order strong-stability-preserving RK method.

    SSPRK33 is designed for problems where preserving monotonicity or other
    forward-Euler stability properties under a step restriction matters, such as
    hyperbolic PDE discretisations with nonlinear limiters.

    Further reading: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "advance_state",
        "combine_stage2",
        "combine_stage3",
        "k2",
        "k3",
        "call_pure",
        "redirect_call",
        "stage",
        "trial",
    )

    descriptor = SchemeDescriptor("SSPRK33", "SSP RK33")
    tableau = SSPRK33_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        algebraist: Algebraist | None = None,
    ) -> None:
        self.advance_state = None
        self.combine_stage2 = None
        self.combine_stage3 = None

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
        self.trial, self.k2, self.k3 = workspace.allocate_translation_buffers(3)

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
        self.combine_stage2 = calls.stages[1]
        self.combine_stage3 = calls.stages[2]
        self.advance_state = calls.solution_state
        self.call_pure = self.algebraist_call
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

        trial = scale(trial_buffer, dt, k1)
        trial(state, stage)
        derivative(stage_interval(interval, dt, dt), stage, k2)

        trial = combine2(
            trial_buffer,
            dt * SSPRK33_A[2][0],
            k1,
            dt * SSPRK33_A[2][1],
            k2,
        )
        trial(state, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k3)

        delta = combine3(
            trial_buffer,
            dt * SSPRK33_B[0],
            k1,
            dt * SSPRK33_B[1],
            k2,
            dt * SSPRK33_B[2],
            k3,
        )

        apply_delta(delta, state)
        return dt

    def algebraist_call(
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

        combine_stage2(stage, state, dt, k1)
        derivative(stage_interval(interval, dt, dt), stage, k2)

        combine_stage3(stage, state, dt, k1, k2)
        derivative(stage_interval(interval, dt, half_dt), stage, k3)

        advance_state(state, state, dt, k1, k2, k3)
        return dt


__all__ = ["SSPRK33_TABLEAU", "SchemeSSPRK33"]