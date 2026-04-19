from __future__ import annotations

from math import sqrt

from stark.schemes.tableau import ButcherTableau
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.schemes.descriptor import SchemeDescriptor
from stark.machinery.stage_solve.workers import CoupledCollocationResolventStep
from stark.schemes.base import SchemeBaseImplicitFixed
from stark.execution.executor import Executor


GAUSS_LEGENDRE4_SQRT3 = sqrt(3.0)

GAUSS_LEGENDRE4_TABLEAU = ButcherTableau(
    c=(0.5 - GAUSS_LEGENDRE4_SQRT3 / 6.0, 0.5 + GAUSS_LEGENDRE4_SQRT3 / 6.0),
    a=(
        (0.25, 0.25 - GAUSS_LEGENDRE4_SQRT3 / 6.0),
        (0.25 + GAUSS_LEGENDRE4_SQRT3 / 6.0, 0.25),
    ),
    b=(0.5, 0.5),
    order=4,
    short_name="GL4",
    full_name="Gauss-Legendre 4",
)

_FINAL_WEIGHT_1 = -GAUSS_LEGENDRE4_SQRT3
_FINAL_WEIGHT_2 = GAUSS_LEGENDRE4_SQRT3


class SchemeGaussLegendre4(SchemeBaseImplicitFixed):
    """
    The two-stage fourth-order Gauss-Legendre collocation method.

    This is a genuinely fully implicit Runge-Kutta method: both stages are
    coupled and must be solved together. It is therefore a useful pressure
    test for STARK's coupled-stage resolvent layer.
    """

    __slots__ = ("stepper", "trial")

    descriptor = SchemeDescriptor("GL4", "Gauss-Legendre 4")
    tableau = GAUSS_LEGENDRE4_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
    ) -> None:
        self.stepper = CoupledCollocationResolventStep("Gauss-Legendre 4", self.tableau, derivative, workbench, 2, resolvent)
        self.trial = self.stepper.workspace.allocate_translation()

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.stepper.workspace
        dt = interval.step if interval.step <= remaining else remaining
        stage_block = self.stepper.solve(interval, state, dt)
        delta = workspace.combine2(
            self.trial,
            _FINAL_WEIGHT_1,
            stage_block[0],
            _FINAL_WEIGHT_2,
            stage_block[1],
        )
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = [
    "GAUSS_LEGENDRE4_SQRT3",
    "GAUSS_LEGENDRE4_TABLEAU",
    "SchemeGaussLegendre4",
]













