from __future__ import annotations

from stark.contracts import Derivative, IntervalLike, Resolvent, State, Allocator
from stark.schemes.support.executor import SchemeExecutor
from stark.schemes.support import (
    MonitorSchemeLike,
    SchemeDescriptor,
    with_fixed_step_monitoring,
    with_scheme_display,
)
from stark.schemes.support.implicit import (
    initialise_implicit_support,
    implicit_display_resolvent_problem,
    implicit_set_apply_delta_safety,
    implicit_snapshot_state,
)
from stark.schemes.support.specialist import SchemeSpecialist
from stark.schemes.support.stage_problem import SchemeStageProblem
from stark.schemes.support.tableau import ButcherTableau


BE_TABLEAU = ButcherTableau(
    c=(1.0,),
    a=((1.0,),),
    b=(1.0,),
    order=1,
)


@with_scheme_display
@with_fixed_step_monitoring
class SchemeBackwardEuler:
    """The implicit backward Euler method.

    Backward Euler advances by solving the implicit increment equation:

        delta = h * f(t + h, y + delta)

    and then applying the solved increment:

        y <- y + delta

    Algorithm sketch for one accepted step of size h:

        1. Build the stage problem at t + h:
               F(delta) = delta - h * f(t + h, y + delta)

        2. Use the configured resolvent to solve:
               F(delta) = 0

        3. Apply the solved increment:
               y <- y + delta

    The scheme owns the time-stepping formula. The resolvent owns the
    nonlinear solve for the implicit increment.
    """

    __slots__ = (
        "monitor",
        "block_allocator",
        "call_body",
        "call_step",
        "delta",
        "derivative",
        "redirect_call",
        "resolvent",
        "workspace",
    )

    descriptor = SchemeDescriptor("BE", "Backward Euler")
    display_resolvent_problem = classmethod(implicit_display_resolvent_problem)
    set_apply_delta_safety = implicit_set_apply_delta_safety
    snapshot_state = implicit_snapshot_state

    tableau = BE_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        allocator: Allocator,
        resolvent: Resolvent,
        *,
        specialist: SchemeSpecialist | None = None,
        monitor: MonitorSchemeLike | None = None,
    ) -> None:
        self.monitor = monitor
        self.call_body = self.call_inline
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step
        self.resolvent = resolvent

        initialise_implicit_support(self, derivative, allocator)
        self.delta = self.block_allocator.allocate(1)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_body = self.call_specialized
            if monitor is None:
                self.call_step = self.call_body
                self.redirect_call = self.call_step

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: SchemeExecutor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        """Accept specialist hooks for constructor consistency."""

        del specialist

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
        executor: SchemeExecutor,
    ) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining

        workspace = self.workspace
        resolvent = self.resolvent
        delta = self.delta

        # 1. Build the implicit stage problem:
        #        F(delta) = delta - h * f(t + h, y + delta)
        problem = SchemeStageProblem(
            derivative=self.derivative,
            interval=workspace.stage_at(interval, dt, dt),
            origin=state,
            rhs=None,
            alpha=dt,
        )

        # 2. Solve F(delta) = 0.
        resolvent(problem, delta)

        # 3. Apply the solved increment: y <- y + delta.
        workspace.apply_delta(delta[0], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt

    def call_specialized(
        self,
        interval: IntervalLike,
        state: State,
        executor: SchemeExecutor,
    ) -> float:
        return self.call_inline(interval, state, executor)


__all__ = ["BE_TABLEAU", "SchemeBackwardEuler"]
