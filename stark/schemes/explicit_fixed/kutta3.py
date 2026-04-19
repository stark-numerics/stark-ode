from __future__ import annotations

from stark.execution.executor import Executor
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.schemes.tableau import ButcherTableau
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.base import SchemeBaseExplicitFixed


KUTTA3_TABLEAU = ButcherTableau(
    c=(0.0, 0.5, 1.0),
    a=((), (0.5,), (-1.0, 2.0)),
    b=(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    order=3,
)
KUTTA3_A = KUTTA3_TABLEAU.a
KUTTA3_B = KUTTA3_TABLEAU.b


class SchemeKutta3(SchemeBaseExplicitFixed):
    """
    The classical three-stage third-order Runge-Kutta method.

    This is the traditional third-order Kutta scheme often used as a compact
    fixed-step method when fourth-order accuracy is not needed but better
    behavior than second-order methods is still wanted.

    Further reading: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    __slots__ = ("k2", "k3", "stage", "trial")

    descriptor = SchemeDescriptor("Kutta3", "Kutta Third-Order")
    tableau = KUTTA3_TABLEAU

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

        trial = scale(trial_buffer, dt * KUTTA3_A[1][0], k1)
        trial(state, stage)
        derivative(stage_interval(interval, dt, 0.5 * dt), stage, k2)

        trial = combine2(
            trial_buffer,
            dt * KUTTA3_A[2][0],
            k1,
            dt * KUTTA3_A[2][1],
            k2,
        )
        trial(state, stage)
        derivative(stage_interval(interval, dt, dt), stage, k3)

        delta = combine3(
            trial_buffer,
            dt * KUTTA3_B[0],
            k1,
            dt * KUTTA3_B[1],
            k2,
            dt * KUTTA3_B[2],
            k3,
        )
        apply_delta(delta, state)
        return dt


__all__ = ["KUTTA3_TABLEAU", "SchemeKutta3"]
















