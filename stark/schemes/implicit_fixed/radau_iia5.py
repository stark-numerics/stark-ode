from __future__ import annotations

from math import sqrt

from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.machinery.stage_solve.workers import CoupledCollocationResolventStep
from stark.schemes.base import SchemeBaseImplicitFixed
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


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
    """The three-stage fifth-order Radau IIA collocation method.

    Radau IIA methods are implicit Runge-Kutta collocation methods with the
    right endpoint included as a stage node. The three-stage member is fifth
    order and is a standard stiff ODE workhorse; SciPy's `Radau` solver is based
    on a fifth-order Radau IIA formula.

    STARK's coupled collocation stepper returns stage increments. For Radau IIA
    5 the final tableau row equals `b`, so the final stage increment is already
    the full step update. The call body applies `stage_block[2]` directly for
    that reason.
    """

    __slots__ = (
        "call_pure",
        "redirect_call",
        "stepper",
    )

    descriptor = SchemeDescriptor("Radau5", "Radau IIA 5")
    tableau = RADAU_IIA5_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
    ) -> None:
        self.stepper = CoupledCollocationResolventStep(
            "Radau IIA 5",
            self.tableau,
            derivative,
            workbench,
            3,
            resolvent,
        )

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
        workspace.apply_delta(stage_block[2], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)

        return dt


__all__ = [
    "RADAU_IIA5_SQRT6",
    "RADAU_IIA5_TABLEAU",
    "SchemeRadauIIA5",
]