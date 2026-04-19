from __future__ import annotations

from stark.execution.executor import Executor
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.schemes.tableau import ButcherTableau
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.base import SchemeBaseExplicitFixed


EULER_TABLEAU = ButcherTableau(
    c=(0.0,),
    a=((),),
    b=(1.0,),
    order=1,
)
EULER_B = EULER_TABLEAU.b


class SchemeEuler(SchemeBaseExplicitFixed):
    """
    Forward Euler, the basic first-order explicit Runge-Kutta method.

    This is the simplest one-step method in the library: evaluate the
    derivative once at the start of the step and advance with that slope.
    It is useful as a baseline and for very cheap exploratory integrations,
    but it is only first-order accurate and has a small stability region.

    Further reading: https://en.wikipedia.org/wiki/Euler_method
    """

    __slots__ = ("delta",)

    descriptor = SchemeDescriptor("Euler", "Forward Euler")
    tableau = EULER_TABLEAU

    def __init__(self, derivative: Derivative, workbench: Workbench) -> None:
        super().__init__(derivative, workbench)

    def initialise_buffers(self) -> None:
        self.delta = self.workspace.allocate_translation()
    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
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
        delta = scale(delta_buffer, dt * EULER_B[0], k1)
        apply_delta(delta, state)
        return dt


__all__ = ["EULER_TABLEAU", "SchemeEuler"]
















