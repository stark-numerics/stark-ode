from __future__ import annotations

from stark.algebraist.classic import Algebraist, AlgebraistImplicitCombination
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.machinery.stage_solve.workers import ShiftedOneStageResolventStep
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    refresh_fixed_step_call,
    unbound_scheme_call,
    with_fixed_step_monitoring,
    with_implicit_stepper_methods,
    with_scheme_display,
)
from stark.schemes.support.tableau import ButcherTableau


IMPLICIT_MIDPOINT_TABLEAU = ButcherTableau(
    c=(0.5,),
    a=((0.5,),),
    b=(1.0,),
    order=2,
    short_name="IM",
    full_name="Implicit Midpoint",
)


@with_scheme_display
@with_fixed_step_monitoring
@with_implicit_stepper_methods
class SchemeImplicitMidpoint:
    """The one-stage implicit midpoint Runge-Kutta method.

    Implicit midpoint is the simplest nontrivial collocation method. In STARK's
    resolvent language it solves for a midpoint correction `z` satisfying

        z = (dt / 2) f(t_n + dt/2, x_n + z),

    then doubles that midpoint correction to advance the full step.

    Further reading: https://en.wikipedia.org/wiki/Midpoint_method
    """

    __slots__ = (
        "_monitor",
        "call_pure",
        "final_delta_call",
        "redirect_call",
        "stepper",
        "trial",
    )

    descriptor = SchemeDescriptor("IM", "Implicit Midpoint")
    tableau = IMPLICIT_MIDPOINT_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        *,
        algebraist: Algebraist | None = None,
    ) -> None:
        self.final_delta_call = unbound_scheme_call
        self._monitor = None
        self.stepper = ShiftedOneStageResolventStep(
            "Implicit Midpoint",
            self.tableau,
            derivative,
            workbench,
            resolvent,
        )
        self.trial = self.stepper.workspace.allocate_translation()

        self.call_pure = self.call_generic
        refresh_fixed_step_call(self)

        if algebraist is not None:
            self.bind_algebraist_path(algebraist)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_implicit_fixed_scheme(
            final_delta=AlgebraistImplicitCombination("final_delta", (2.0,)),
        )
        self.final_delta_call = calls.require_final_delta_call(type(self).__name__)
        self.call_pure = self.call_algebraist
        refresh_fixed_step_call(self)

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

        midpoint = self.stepper.solve(
            interval,
            state,
            dt,
            alpha=0.5 * dt,
            stage_shift=0.5 * dt,
        )
        delta = workspace.scale(2.0, midpoint, self.trial)
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)

        return dt

    def call_algebraist(
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
        final_delta_call = self.final_delta_call
        dt = interval.step if interval.step <= remaining else remaining

        midpoint = self.stepper.solve(
            interval,
            state,
            dt,
            alpha=0.5 * dt,
            stage_shift=0.5 * dt,
        )
        delta = final_delta_call(midpoint, self.trial)
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)

        return dt


__all__ = ["IMPLICIT_MIDPOINT_TABLEAU", "SchemeImplicitMidpoint"]
