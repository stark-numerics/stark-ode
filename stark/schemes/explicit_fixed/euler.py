from __future__ import annotations

from stark.contracts import Derivative, IntervalLike, State, Allocator
from stark.schemes.support.executor import SchemeExecutor
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    with_explicit_workspace_methods,
    with_fixed_step_monitoring,
    initialise_explicit_support,
    refresh_fixed_step_call,
    unbound_scheme_call,
    with_scheme_display,
)
from stark.schemes.support.specialist import SchemeSpecialist
from stark.schemes.support.stencil import SchemeStencilTableau
from stark.schemes.support.tableau import ButcherTableau


EULER_TABLEAU = ButcherTableau(
    c=(0.0,),
    a=((),),
    b=(1.0,),
    order=1,
)

EULER_B = EULER_TABLEAU.b


@with_scheme_display
@with_fixed_step_monitoring
@with_explicit_workspace_methods
class SchemeEuler:
    """Forward Euler, the basic first-order explicit Runge-Kutta method.

    This is the simplest one-step method in the library: evaluate the
    derivative once at the start of the step and advance with that slope. It is
    useful as a baseline and for very cheap exploratory integrations, but it is
    only first-order accurate and has a small stability region.

    Algorithm sketch for one accepted step of size h:

        1. k1 = f(t, y)
        2. y  <- y + h*k1

    The inline path expresses the method directly with workspace arithmetic.
    The specialized path uses a fixed-coefficient apply kernel prepared from
    the same tableau weights.

    Further reading: https://en.wikipedia.org/wiki/Euler_method
    """

    __slots__ = (
        "_monitor",
        "advance_delta_buffer",
        "advance_update",
        "call_monitorable",
        "derivative",
        "explicit",
        "k1",
        "redirect_call",
        "workspace",
    )

    descriptor = SchemeDescriptor("Euler", "Forward Euler")
    tableau = EULER_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        allocator: Allocator,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        self.advance_update = unbound_scheme_call

        self._monitor = None
        self.call_monitorable = self.call_inline
        self.redirect_call = self.call_monitorable

        initialise_explicit_support(self, derivative, allocator)
        self.advance_delta_buffer = self.workspace.allocate_translation()

        refresh_fixed_step_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_monitorable = self.call_specialized
            refresh_fixed_step_call(self)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: SchemeExecutor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(
        self,
        specialist: SchemeSpecialist,
    ) -> None:
        """Prepare fixed-coefficient kernels for the specialized path."""

        stencils = SchemeStencilTableau(self.tableau)

        # Step 2 advances the accepted state from the tableau's b weights.
        self.advance_update = specialist.provide(stencils.advance_update())

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

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        apply_delta = workspace.apply_delta

        k1 = self.k1
        advance_delta_buffer = self.advance_delta_buffer

        dt = interval.step if interval.step <= remaining else remaining

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. y <- y + h*k1
        advance_delta = scale(dt * EULER_B[0], k1, advance_delta_buffer)
        apply_delta(advance_delta, state)

        return dt

    def call_specialized(
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

        k1 = self.k1
        derivative = self.derivative
        advance_update = self.advance_update

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. y <- y + h*k1
        advance_update(dt, state, k1, state)

        return dt


__all__ = ["EULER_TABLEAU", "SchemeEuler"]
