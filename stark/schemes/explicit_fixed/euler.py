from __future__ import annotations

from stark.algebraist import Algebraist
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

    __slots__ = ("advance_state", "delta")

    descriptor = SchemeDescriptor("Euler", "Forward Euler")
    tableau = EULER_TABLEAU

    def __init__(self, derivative: Derivative, workbench: Workbench, algebraist: Algebraist | None = None) -> None:
        self.advance_state = None
        super().__init__(derivative, workbench)
        if algebraist is not None:
            self.bind_algebraist_path(algebraist)

    def initialise_buffers(self) -> None:
        self.delta = self.workspace.allocate_translation()

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
        self.advance_state = calls.solution_state
        self.bind_fixed_call(self.algebraist_call)

    def generic_call(self, interval: IntervalLike, state: State, executor: Executor) -> float:
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

    def algebraist_call(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        k1 = self.k1
        self.derivative(interval, state, k1)
        self.advance_state(state, state, dt, k1)
        return dt


__all__ = ["EULER_TABLEAU", "SchemeEuler"]
















