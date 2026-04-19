from __future__ import annotations

from stark.execution.executor import Executor
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.schemes.tableau import ButcherTableau
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.base import SchemeBaseExplicitFixed


SSPRK33_TABLEAU = ButcherTableau(
    c=(0.0, 1.0, 0.5),
    a=((), (1.0,), (0.25, 0.25)),
    b=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    order=3,
)
SSPRK33_A = SSPRK33_TABLEAU.a
SSPRK33_B = SSPRK33_TABLEAU.b


class SchemeSSPRK33(SchemeBaseExplicitFixed):
    """
    The three-stage third-order strong-stability-preserving Runge-Kutta method.

    SSPRK33 is designed for problems where preserving monotonicity or other
    stability properties of forward Euler under a step restriction matters,
    such as hyperbolic PDE discretizations with nonlinear limiters.

    Further reading: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    __slots__ = ("k2", "k3", "stage", "trial")

    descriptor = SchemeDescriptor("SSPRK33", "SSP RK33")
    tableau = SSPRK33_TABLEAU

    def __init__(self, derivative: Derivative, workbench: Workbench) -> None:
        super().__init__(derivative, workbench)

    def initialise_buffers(self) -> None:
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2, self.k3 = workspace.allocate_translation_buffers(3)
    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
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
        derivative(interval, state, k1)

        trial = scale(trial_buffer, dt * SSPRK33_A[1][0], k1)
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
        derivative(stage_interval(interval, dt, 0.5 * dt), stage, k3)

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


__all__ = ["SSPRK33_TABLEAU", "SchemeSSPRK33"]
















