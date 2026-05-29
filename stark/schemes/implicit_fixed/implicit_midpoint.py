from __future__ import annotations

from typing import Any, cast

from stark.contracts import Derivative, IntervalLike, Resolvent, State, Allocator
from stark.schemes.support.executor import SchemeExecutor
from stark.schemes.support import (
    SchemeDescriptor,
    refresh_fixed_step_call,
    with_fixed_step_monitoring,
    with_scheme_display,
)
from stark.schemes.support.implicit import (
    initialise_implicit_support,
    with_implicit_workspace_methods,
)
from stark.schemes.support.specialist import SchemeSpecialist
from stark.schemes.support.stage_problem import SchemeStageProblem
from stark.schemes.support.stencil import SchemeStencil
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
@with_implicit_workspace_methods
class SchemeImplicitMidpoint:
    """The one-stage implicit midpoint Runge-Kutta method.

    Implicit midpoint solves a midpoint increment and then doubles it to
    advance the full step.

    Algorithm sketch for one accepted step of size h:

        1. Solve the midpoint increment:
               z = h/2 * f(t + h/2, y + z)

        2. Advance with the full-step increment:
               y <- y + 2z
    """

    __slots__ = (
        "_monitor",
        "advance_update",
        "block_allocator",
        "call_monitorable",
        "derivative",
        "midpoint",
        "redirect_call",
        "resolvent",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor("IM", "Implicit Midpoint")
    tableau = IMPLICIT_MIDPOINT_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        allocator: Allocator,
        resolvent: Resolvent,
        *,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        self._monitor = None
        self.call_monitorable = self.call_inline
        self.redirect_call = self.call_monitorable
        self.resolvent = resolvent
        self.advance_update = None

        initialise_implicit_support(self, derivative, allocator)
        self.midpoint = self.block_allocator.allocate(1)
        self.trial = self.workspace.allocate_translation()

        refresh_fixed_step_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_monitorable = self.call_specialized
            refresh_fixed_step_call(self)

    def __call__(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        # Step 2 applies the doubled midpoint increment.
        self.advance_update = specialist.provide(SchemeStencil((2.0,), apply=True))

    def call_inline(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        workspace = self.workspace
        midpoint = self.midpoint

        # 1. Solve the midpoint increment:
        #        z = h/2 * f(t + h/2, y + z)
        problem = SchemeStageProblem(
            derivative=self.derivative,
            interval=workspace.stage_at(interval, dt, 0.5 * dt),
            origin=state,
            rhs=None,
            alpha=0.5 * dt,
        )
        self.resolvent(problem, midpoint)

        # 2. Advance y <- y + 2z.
        delta = workspace.scale(2.0, midpoint[0], self.trial)
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt

    def call_specialized(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        workspace = self.workspace
        midpoint = self.midpoint
        advance_update = cast(Any, self.advance_update)

        # 1. Solve the midpoint increment:
        #        z = h/2 * f(t + h/2, y + z)
        problem = SchemeStageProblem(
            derivative=self.derivative,
            interval=workspace.stage_at(interval, dt, 0.5 * dt),
            origin=state,
            rhs=None,
            alpha=0.5 * dt,
        )
        self.resolvent(problem, midpoint)

        # 2. Advance y <- y + 2z.
        advance_update(1.0, state, midpoint[0], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = ["IMPLICIT_MIDPOINT_TABLEAU", "SchemeImplicitMidpoint"]
