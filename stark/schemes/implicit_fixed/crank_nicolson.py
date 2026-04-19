from __future__ import annotations

from stark.accelerators.binding import BoundDerivative
from stark.schemes.tableau import ButcherTableau
from stark.contracts import Block, Derivative, IntervalLike, Resolvent, State, Workbench
from stark.schemes.descriptor import SchemeDescriptor
from stark.machinery.stage_solve.workers import ShiftedOneStageResolventStep
from stark.schemes.base import SchemeBaseImplicitFixed
from stark.execution.executor import Executor


CRANK_NICOLSON_TABLEAU = ButcherTableau(
    c=(0.0, 1.0),
    a=((), (0.5, 0.5)),
    b=(0.5, 0.5),
    order=2,
    short_name="CN",
    full_name="Crank-Nicolson",
)


class SchemeCrankNicolson(SchemeBaseImplicitFixed):
    """
    The fixed-step Crank-Nicolson / trapezoidal Runge-Kutta method.

    This method evaluates one explicit stage at the start of the step, then
    resolves a one-stage shifted implicit problem at the end of the step:

        delta = (dt / 2) f(t_n, x_n) + (dt / 2) f(t_n + dt, x_n + delta).

    It is a simple but important bridge case between purely explicit methods
    and fully implicit sequential DIRK methods.

    Further reading: https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)
    """

    __slots__ = ("derivative", "k1", "known_block", "stepper")

    descriptor = SchemeDescriptor("CN", "Crank-Nicolson")
    tableau = CRANK_NICOLSON_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
    ) -> None:
        self.derivative = BoundDerivative(derivative)
        self.stepper = ShiftedOneStageResolventStep("Crank-Nicolson", self.tableau, derivative, workbench, resolvent)
        workspace = self.stepper.workspace
        self.k1 = workbench.allocate_translation()
        self.known_block = Block([workspace.allocate_translation()])

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.stepper.workspace
        derivative = self.derivative
        dt = interval.step if interval.step <= remaining else remaining

        derivative(interval, state, self.k1)
        self.known_block.items[0] = workspace.scale(self.known_block[0], 0.5 * dt, self.k1)
        delta = self.stepper.solve(interval, state, dt, alpha=0.5 * dt, stage_shift=dt, rhs=self.known_block)
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt

__all__ = ["CRANK_NICOLSON_TABLEAU", "SchemeCrankNicolson"]















