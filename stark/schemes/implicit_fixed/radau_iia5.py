from __future__ import annotations

from math import sqrt

from stark.schemes.tableau import ButcherTableau
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.schemes.descriptor import SchemeDescriptor
from stark.machinery.stage_solve.workers import CoupledCollocationResolventStep
from stark.schemes.base import SchemeBaseImplicitFixed
from stark.execution.executor import Executor


RADAU_IIA5_SQRT6 = sqrt(6.0)

RADAU_IIA5_TABLEAU = ButcherTableau(
    c=((4.0 - RADAU_IIA5_SQRT6) / 10.0, (4.0 + RADAU_IIA5_SQRT6) / 10.0, 1.0),
    a=(
        (
            (88.0 - 7.0 * RADAU_IIA5_SQRT6) / 360.0,
            (296.0 - 169.0 * RADAU_IIA5_SQRT6) / 1800.0,
            (-2.0 + 3.0 * RADAU_IIA5_SQRT6) / 225.0,
        ),
        (
            (296.0 + 169.0 * RADAU_IIA5_SQRT6) / 1800.0,
            (88.0 + 7.0 * RADAU_IIA5_SQRT6) / 360.0,
            (-2.0 - 3.0 * RADAU_IIA5_SQRT6) / 225.0,
        ),
        (
            (16.0 - RADAU_IIA5_SQRT6) / 36.0,
            (16.0 + RADAU_IIA5_SQRT6) / 36.0,
            1.0 / 9.0,
        ),
    ),
    b=(
        (16.0 - RADAU_IIA5_SQRT6) / 36.0,
        (16.0 + RADAU_IIA5_SQRT6) / 36.0,
        1.0 / 9.0,
    ),
    order=5,
    short_name="Radau5",
    full_name="Radau IIA 5",
)


class SchemeRadauIIA5(SchemeBaseImplicitFixed):
    """
    The three-stage fifth-order Radau IIA collocation method.

    This is the classical stiffly accurate Radau method used as the foundation
    of SciPy's `Radau` solver family. Like Gauss-Legendre it requires a fully
    coupled stage solve, but its stiff accuracy means the final stage delta is
    the step update.
    """

    __slots__ = ("stepper",)

    descriptor = SchemeDescriptor("Radau5", "Radau IIA 5")
    tableau = RADAU_IIA5_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
    ) -> None:
        self.stepper = CoupledCollocationResolventStep("Radau IIA 5", self.tableau, derivative, workbench, 3, resolvent)

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.stepper.workspace
        dt = interval.step if interval.step <= remaining else remaining
        stage_block = self.stepper.solve(interval, state, dt)
        workspace.apply_delta(stage_block[2], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = ["RADAU_IIA5_SQRT6", "RADAU_IIA5_TABLEAU", "SchemeRadauIIA5"]













