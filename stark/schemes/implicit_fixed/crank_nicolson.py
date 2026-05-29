from __future__ import annotations

from typing import Any, cast

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
from stark.schemes.support.stencil import SchemeStencil
from stark.schemes.support.tableau import ButcherTableau


CRANK_NICOLSON_TABLEAU = ButcherTableau(
    c=(0.0, 1.0),
    a=((), (0.5, 0.5)),
    b=(0.5, 0.5),
    order=2,
    short_name="CN",
    full_name="Crank-Nicolson",
)


@with_scheme_display
@with_fixed_step_monitoring
class SchemeCrankNicolson:
    """The fixed-step Crank-Nicolson / trapezoidal Runge-Kutta method.

    Crank-Nicolson combines one explicit derivative at the start of the step
    with one implicit derivative at the end of the step.

    Algorithm sketch for one accepted step of size h:

        1. Compute k1 = f(t, y).
        2. Build the known explicit contribution h/2 * k1.
        3. Solve the shifted implicit increment:
               delta = h/2 * k1 + h/2 * f(t + h, y + delta)
        4. Apply the solved increment:
               y <- y + delta
    """

    __slots__ = (
        "monitor",
        "block_allocator",
        "call_body",
        "call_step",
        "derivative",
        "known_rhs",
        "known_rhs_kernel",
        "k1",
        "redirect_call",
        "resolvent",
        "stage_delta",
        "workspace",
    )

    descriptor = SchemeDescriptor("CN", "Crank-Nicolson")
    display_resolvent_problem = classmethod(implicit_display_resolvent_problem)
    set_apply_delta_safety = implicit_set_apply_delta_safety
    snapshot_state = implicit_snapshot_state

    tableau = CRANK_NICOLSON_TABLEAU

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
        self.known_rhs_kernel = None

        initialise_implicit_support(self, derivative, allocator)
        self.k1 = self.workspace.allocate_translation()
        self.known_rhs = self.block_allocator.allocate(1)
        self.stage_delta = self.block_allocator.allocate(1)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_body = self.call_specialized
            if monitor is None:
                self.call_step = self.call_body
                self.redirect_call = self.call_step

    def __call__(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        # Step 2 builds the known explicit contribution h/2 * k1.
        self.known_rhs_kernel = specialist.provide(SchemeStencil((0.5,)))

    def call_inline(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        workspace = self.workspace

        # 1. Compute k1 = f(t, y).
        self.derivative(interval, state, self.k1)

        # 2. Build the known explicit contribution h/2 * k1.
        self.known_rhs[0] = workspace.scale(0.5 * dt, self.k1, self.known_rhs[0])

        # 3. Solve delta = h/2 * k1 + h/2 * f(t + h, y + delta).
        problem = SchemeStageProblem(
            derivative=self.derivative,
            interval=workspace.stage_at(interval, dt, dt),
            origin=state,
            rhs=self.known_rhs,
            alpha=0.5 * dt,
        )
        self.resolvent(problem, self.stage_delta)

        # 4. Apply the solved increment: y <- y + delta.
        workspace.apply_delta(self.stage_delta[0], state)

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
        known_rhs_kernel = cast(Any, self.known_rhs_kernel)

        # 1. Compute k1 = f(t, y).
        self.derivative(interval, state, self.k1)

        # 2. Build the known explicit contribution h/2 * k1.
        self.known_rhs[0] = known_rhs_kernel(dt, self.k1, self.known_rhs[0])

        # 3. Solve delta = h/2 * k1 + h/2 * f(t + h, y + delta).
        problem = SchemeStageProblem(
            derivative=self.derivative,
            interval=workspace.stage_at(interval, dt, dt),
            origin=state,
            rhs=self.known_rhs,
            alpha=0.5 * dt,
        )
        self.resolvent(problem, self.stage_delta)

        # 4. Apply the solved increment: y <- y + delta.
        workspace.apply_delta(self.stage_delta[0], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = ["CRANK_NICOLSON_TABLEAU", "SchemeCrankNicolson"]
