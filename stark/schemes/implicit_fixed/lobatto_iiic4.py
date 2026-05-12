from __future__ import annotations

from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.machinery.stage_solve.workers import CoupledCollocationResolventStep
from stark.schemes.base import SchemeBaseImplicitFixed
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


LOBATTO_IIIC4_TABLEAU = ButcherTableau(
    c=(0.0, 0.5, 1.0),
    a=(
        (1.0 / 6.0, -1.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 5.0 / 12.0, -1.0 / 12.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    ),
    b=(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    order=4,
    short_name="Lobatto4",
    full_name="Lobatto IIIC 4",
)


class SchemeLobattoIIIC4(SchemeBaseImplicitFixed):
    """The three-stage fourth-order Lobatto IIIC collocation method.

    Lobatto collocation methods include both endpoints as stage nodes. With
    three stages this Lobatto IIIC method is fourth order.

    STARK's coupled collocation stepper returns stage increments. For this
    scheme the final tableau row equals `b`, so the final stage increment is
    already the full step update. The call body applies `stage_block[2]`
    directly for that reason, not as an optimisation shortcut.
    """

    __slots__ = (
        "call_pure",
        "redirect_call",
        "stepper",
    )

    descriptor = SchemeDescriptor("Lobatto4", "Lobatto IIIC 4")
    tableau = LOBATTO_IIIC4_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
    ) -> None:
        self.stepper = CoupledCollocationResolventStep(
            "Lobatto IIIC 4",
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
    "LOBATTO_IIIC4_TABLEAU",
    "SchemeLobattoIIIC4",
]