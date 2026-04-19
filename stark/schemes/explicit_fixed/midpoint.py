from __future__ import annotations

from stark.execution.executor import Executor
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.schemes.tableau import ButcherTableau
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.base import SchemeBaseExplicitFixed


MIDPOINT_TABLEAU = ButcherTableau(
    c=(0.0, 0.5),
    a=((), (0.5,)),
    b=(0.0, 1.0),
    order=2,
)
MIDPOINT_B = MIDPOINT_TABLEAU.b


class SchemeMidpoint(SchemeBaseExplicitFixed):
    """
    The explicit midpoint two-stage second-order Runge-Kutta method.

    This method samples the derivative at the midpoint predicted by an Euler
    half-step and then advances using that midpoint slope. It is one of the
    standard second-order explicit schemes.

    Further reading: https://en.wikipedia.org/wiki/Midpoint_method
    """

    __slots__ = ("k2", "stage", "trial")

    descriptor = SchemeDescriptor("Midpoint", "Explicit Midpoint")
    tableau = MIDPOINT_TABLEAU

    def __init__(self, derivative: Derivative, workbench: Workbench) -> None:
        super().__init__(derivative, workbench)

    def initialise_buffers(self) -> None:
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2 = workspace.allocate_translation_buffers(2)
    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval
        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2

        dt = interval.step if interval.step <= remaining else remaining
        derivative(interval, state, k1)

        trial = scale(trial_buffer, 0.5 * dt, k1)
        trial(state, stage)
        derivative(stage_interval(interval, dt, 0.5 * dt), stage, k2)

        delta = combine2(
            trial_buffer,
            dt * MIDPOINT_B[0],
            k1,
            dt * MIDPOINT_B[1],
            k2,
        )
        apply_delta(delta, state)
        return dt


__all__ = ["MIDPOINT_TABLEAU", "SchemeMidpoint"]
















