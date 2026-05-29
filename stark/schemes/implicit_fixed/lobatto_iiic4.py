from __future__ import annotations

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
from stark.schemes.support.tableau import ButcherTableau


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


@with_scheme_display
@with_fixed_step_monitoring
@with_implicit_workspace_methods
class SchemeLobattoIIIC4:
    """The three-stage fourth-order Lobatto IIIC collocation method.

    Algorithm sketch for one accepted step of size h:

        1. Solve the coupled Lobatto stage increment problem.
        2. Apply the final stage increment directly:
               y <- y + delta_3

    The method is stiffly accurate: the final tableau row equals b, so the
    final stage increment is already the full accepted step update.
    """

    __slots__ = (
        "_monitor",
        "block_allocator",
        "call_monitorable",
        "derivative",
        "redirect_call",
        "resolvent",
        "stage_delta",
        "workspace",
    )

    descriptor = SchemeDescriptor("Lobatto4", "Lobatto IIIC 4")
    tableau = LOBATTO_IIIC4_TABLEAU

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

        initialise_implicit_support(self, derivative, allocator)
        self.stage_delta = self.block_allocator.allocate(3)

        refresh_fixed_step_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_monitorable = self.call_specialized
            refresh_fixed_step_call(self)

    def __call__(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        """Accept specialist hooks for constructor consistency."""

        del specialist

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

        # 1. Solve the coupled Lobatto stage increment problem.
        self.resolvent(self._problem(interval, state, dt), self.stage_delta)

        # 2. Apply the final stage increment directly.
        self.workspace.apply_delta(self.stage_delta[2], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt

    def call_specialized(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.call_inline(interval, state, executor)


__all__ = ["LOBATTO_IIIC4_TABLEAU", "SchemeLobattoIIIC4"]
