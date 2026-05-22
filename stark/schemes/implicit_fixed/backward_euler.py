from __future__ import annotations

from stark.algebraist.classic import Algebraist
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.machinery.stage_solve.workers import ShiftedOneStageResolventStep
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    refresh_fixed_step_call,
    with_fixed_step_monitoring,
    with_implicit_stepper_methods,
    with_scheme_display,
)
from stark.schemes.support.tableau import ButcherTableau


BE_TABLEAU = ButcherTableau(
    c=(1.0,),
    a=((1.0,),),
    b=(1.0,),
    order=1,
)


@with_scheme_display
@with_fixed_step_monitoring
@with_implicit_stepper_methods
class SchemeBackwardEuler:
    """The implicit backward Euler method resolved by a stage resolvent.

    Backward Euler advances by solving

        x_{n+1} = x_n + dt f(x_{n+1}),

    so each step requires a shifted implicit solve rather than a direct explicit
    stage evaluation. In STARK that solve is owned by a `Resolvent`, which may be
    a generic automatic worker or a problem-specific custom one.

    Further reading: https://en.wikipedia.org/wiki/Backward_Euler_method
    """

    __slots__ = (
        "_monitor",
        "call_pure",
        "redirect_call",
        "stepper",
    )

    descriptor = SchemeDescriptor("BE", "Backward Euler")
    tableau = BE_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        *,
        algebraist: Algebraist | None = None,
    ) -> None:
        del algebraist
        self._monitor = None
        self.stepper = ShiftedOneStageResolventStep(
            "Backward Euler",
            self.tableau,
            derivative,
            workbench,
            resolvent,
        )
        self.call_pure = self.call_generic
        refresh_fixed_step_call(self)

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

        dt = interval.step if interval.step <= remaining else remaining

        stepper = self.stepper
        delta = stepper.solve(
            interval,
            state,
            dt,
            alpha=dt,
            stage_shift=dt,
        )
        stepper.workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)

        return dt


__all__ = ["BE_TABLEAU", "SchemeBackwardEuler"]
