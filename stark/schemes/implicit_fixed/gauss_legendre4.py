from __future__ import annotations

from math import sqrt
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
from stark.schemes.support.stage_problem import SchemeStageProblemCoupled
from stark.schemes.support.stencil import SchemeStencil
from stark.schemes.support.tableau import ButcherTableau


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
GAUSS_LEGENDRE4_STAGE_INCREMENT_WEIGHTS = (
    -GAUSS_LEGENDRE4_SQRT3,
    GAUSS_LEGENDRE4_SQRT3,
)


@with_scheme_display
@with_fixed_step_monitoring
@with_implicit_workspace_methods
class SchemeGaussLegendre4:
    """The two-stage fourth-order Gauss-Legendre collocation method.

    Algorithm sketch for one accepted step of size h:

        1. Solve the coupled stage increment problem:
               delta_i = h * sum_j a_ij f(t + c_j h, y + delta_j)

        2. Reconstruct the accepted step update from stage increments:
               y <- y + (-sqrt(3) * delta_1 + sqrt(3) * delta_2)

    The final update is reconstructed with b^T A^-1 because this collocation
    method is not stiffly accurate.
    """

    __slots__ = (
        "_monitor",
        "advance_update",
        "block_allocator",
        "call_pure",
        "derivative",
        "redirect_call",
        "resolvent",
        "stage_delta",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor("GL4", "Gauss-Legendre 4")
    tableau = GAUSS_LEGENDRE4_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        allocator: Allocator,
        resolvent: Resolvent,
        *,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        self._monitor = None
        self.call_pure = self.call_inline
        self.redirect_call = self.call_pure
        self.resolvent = resolvent
        self.advance_update = None

        initialise_implicit_support(self, derivative, allocator)
        self.stage_delta = self.block_allocator.allocate(2)
        self.trial = self.workspace.allocate_translation()

        refresh_fixed_step_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
            refresh_fixed_step_call(self)

    def __call__(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        # Step 2 reconstructs the accepted update from stage increments.
        self.advance_update = specialist.provide(
            SchemeStencil(GAUSS_LEGENDRE4_STAGE_INCREMENT_WEIGHTS, apply=True)
        )

    def _problem(self, interval: IntervalLike, state: State, dt: float) -> SchemeStageProblemCoupled:
        tableau = self.tableau
        return SchemeStageProblemCoupled(
            derivative=self.derivative,
            interval=interval,
            origin=state,
            rhs=None,
            step=dt,
            stage_shifts=tableau.c,
            matrix=tableau.a,
        )

    def call_inline(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        workspace = self.workspace
        stage_delta = self.stage_delta

        # 1. Solve the coupled stage increment problem.
        self.resolvent(self._problem(interval, state, dt), stage_delta)

        # 2. Reconstruct and apply the accepted step update.
        delta = workspace.combine2(
            GAUSS_LEGENDRE4_STAGE_INCREMENT_WEIGHTS[0],
            stage_delta[0],
            GAUSS_LEGENDRE4_STAGE_INCREMENT_WEIGHTS[1],
            stage_delta[1],
            self.trial,
        )
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
        advance_update = cast(Any, self.advance_update)

        # 1. Solve the coupled stage increment problem.
        self.resolvent(self._problem(interval, state, dt), self.stage_delta)

        # 2. Reconstruct and apply the accepted step update.
        advance_update(1.0, state, self.stage_delta[0], self.stage_delta[1], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = [
    "GAUSS_LEGENDRE4_SQRT3",
    "GAUSS_LEGENDRE4_STAGE_INCREMENT_WEIGHTS",
    "GAUSS_LEGENDRE4_TABLEAU",
    "SchemeGaussLegendre4",
]
