from __future__ import annotations

from math import sqrt

from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.machinery.stage_solve.workers import CoupledCollocationResolventStep
from stark.schemes.base import SchemeBaseImplicitFixed
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


GAUSS_LEGENDRE4_SQRT3 = sqrt(3.0)

GAUSS_LEGENDRE4_TABLEAU = ButcherTableau(
    c=(
        0.5 - GAUSS_LEGENDRE4_SQRT3 / 6.0,
        0.5 + GAUSS_LEGENDRE4_SQRT3 / 6.0,
    ),
    a=(
        (0.25, 0.25 - GAUSS_LEGENDRE4_SQRT3 / 6.0),
        (0.25 + GAUSS_LEGENDRE4_SQRT3 / 6.0, 0.25),
    ),
    b=(0.5, 0.5),
    order=4,
    short_name="GL4",
    full_name="Gauss-Legendre 4",
)

# `CoupledCollocationResolventStep` returns stage increments, not derivative
# rates. For Gauss-Legendre 4 the final RK update is reconstructed from the two
# stage increments using b^T A^-1, which reduces to (-sqrt(3), sqrt(3)).
GAUSS_LEGENDRE4_STAGE_INCREMENT_WEIGHTS = (
    -GAUSS_LEGENDRE4_SQRT3,
    GAUSS_LEGENDRE4_SQRT3,
)


class SchemeGaussLegendre4(SchemeBaseImplicitFixed):
    """The two-stage fourth-order Gauss-Legendre collocation method.

    Gauss-Legendre collocation methods use interior quadrature nodes and have
    order `2s` for `s` stages, so this two-stage method is fourth order. It is
    fully implicit: the two stage equations are coupled and must be solved as a
    block.

    STARK's coupled collocation stepper returns stage increments rather than
    derivative rates. Unlike stiffly accurate endpoint methods, the final
    Gauss-Legendre update is not equal to the final stage increment. The call
    body therefore explicitly reconstructs the step update from the two stage
    increments using the weights `b^T A^-1`.
    """

    __slots__ = (
        "call_pure",
        "redirect_call",
        "stepper",
        "trial",
    )

    descriptor = SchemeDescriptor("GL4", "Gauss-Legendre 4")
    tableau = GAUSS_LEGENDRE4_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
    ) -> None:
        self.stepper = CoupledCollocationResolventStep(
            "Gauss-Legendre 4",
            self.tableau,
            derivative,
            workbench,
            2,
            resolvent,
        )
        self.trial = self.stepper.workspace.allocate_translation()

        self.call_pure = self.call_generic
        self.redirect_call = self.call_pure

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

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

        workspace = self.stepper.workspace
        dt = interval.step if interval.step <= remaining else remaining

        stage_block = self.stepper.solve(interval, state, dt)

        delta = workspace.combine2(
            self.trial,
            GAUSS_LEGENDRE4_STAGE_INCREMENT_WEIGHTS[0],
            stage_block[0],
            GAUSS_LEGENDRE4_STAGE_INCREMENT_WEIGHTS[1],
            stage_block[1],
        )
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)

        return dt


__all__ = [
    "GAUSS_LEGENDRE4_SQRT3",
    "GAUSS_LEGENDRE4_STAGE_INCREMENT_WEIGHTS",
    "GAUSS_LEGENDRE4_TABLEAU",
    "SchemeGaussLegendre4",
]