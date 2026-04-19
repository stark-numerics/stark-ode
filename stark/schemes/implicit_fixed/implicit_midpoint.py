from __future__ import annotations

from stark.schemes.tableau import ButcherTableau
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.schemes.descriptor import SchemeDescriptor
from stark.machinery.stage_solve.workers import ShiftedOneStageResolventStep
from stark.schemes.base import SchemeBaseImplicitFixed
from stark.execution.executor import Executor


IMPLICIT_MIDPOINT_TABLEAU = ButcherTableau(
    c=(0.5,),
    a=((0.5,),),
    b=(1.0,),
    order=2,
    short_name="IM",
    full_name="Implicit Midpoint",
)


class SchemeImplicitMidpoint(SchemeBaseImplicitFixed):
    """
    The one-stage implicit midpoint Runge-Kutta method.

    Implicit midpoint is the simplest nontrivial collocation method. In STARK's
    resolvent language it solves for a midpoint correction `z` satisfying

        z = (dt / 2) f(t_n + dt/2, x_n + z),

    then doubles that midpoint correction to advance the full step.

    Further reading: https://en.wikipedia.org/wiki/Midpoint_method
    """

    __slots__ = ("stepper", "trial")

    descriptor = SchemeDescriptor("IM", "Implicit Midpoint")
    tableau = IMPLICIT_MIDPOINT_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
    ) -> None:
        self.stepper = ShiftedOneStageResolventStep("Implicit Midpoint", self.tableau, derivative, workbench, resolvent)
        self.trial = self.stepper.workspace.allocate_translation()

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.stepper.workspace
        dt = interval.step if interval.step <= remaining else remaining
        midpoint = self.stepper.solve(interval, state, dt, alpha=0.5 * dt, stage_shift=0.5 * dt)
        delta = workspace.scale(self.trial, 2.0, midpoint)
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = ["IMPLICIT_MIDPOINT_TABLEAU", "SchemeImplicitMidpoint"]













