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


SSPRK33_TABLEAU = ButcherTableau(
    c=(0.0, 1.0, 0.5),
    a=((), (1.0,), (0.25, 0.25)),
    b=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    order=3,
)

SSPRK33_A = SSPRK33_TABLEAU.a
SSPRK33_B = SSPRK33_TABLEAU.b


@with_scheme_display
@with_fixed_step_monitoring
@with_explicit_workspace_methods
class SchemeSSPRK33:
    """The three-stage third-order strong-stability-preserving RK method.

    SSPRK33 is designed for problems where preserving monotonicity or other
    forward-Euler stability properties under a step restriction matters, such
    as hyperbolic PDE discretisations with nonlinear limiters.

    Algorithm sketch for one accepted step of size h:

        1. k1 = f(t, y)
        2. k2 = f(t + h,   y + h*k1)
        3. k3 = f(t + h/2, y + h*(k1 + k2)/4)
        4. y  <- y + h*(k1 + k2 + 4k3)/6

    The inline path expresses the method directly with workspace arithmetic.
    The specialized path uses fixed-coefficient kernels prepared from the same
    tableau rows and weights.

    Further reading: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "_monitor",
        "advance_update",
        "call_pure",
        "derivative",
        "explicit",
        "k1",
        "k2",
        "k3",
        "redirect_call",
        "stage",
        "stage2_update",
        "stage3_update",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor("SSPRK33", "SSP RK33")
    tableau = SSPRK33_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        allocator: Allocator,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        self.advance_update = unbound_scheme_call
        self.stage2_update = unbound_scheme_call
        self.stage3_update = unbound_scheme_call

        self._monitor = None
        self.call_pure = self.call_inline
        self.redirect_call = self.call_pure

        initialise_explicit_support(self, derivative, allocator)

        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2, self.k3 = workspace.allocate_translation_buffers(3)

        refresh_fixed_step_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
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

        # Steps 2-3 build staged states from rows of the tableau's A matrix.
        self.stage2_update = specialist.provide(stencils.stage(1))
        self.stage3_update = specialist.provide(stencils.stage(2))

        # Step 4 advances the accepted state from the tableau's b weights.
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
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval

        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3

        dt = interval.step if interval.step <= remaining else remaining
        half_dt = 0.5 * dt

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + h, y + h*k1)
        stage_delta = scale(dt, k1, trial_buffer)
        stage_delta(state, stage)
        derivative(stage_interval(interval, dt, dt), stage, k2)

        # 3. k3 = f(t + h/2, y + h*(k1 + k2)/4)
        stage_delta = combine2(
            dt * SSPRK33_A[2][0],
            k1,
            dt * SSPRK33_A[2][1],
            k2,
            trial_buffer,
        )
        stage_delta(state, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k3)

        # 4. y <- y + h*(k1 + k2 + 4k3)/6
        advance_delta = combine3(
            dt * SSPRK33_B[0],
            k1,
            dt * SSPRK33_B[1],
            k2,
            dt * SSPRK33_B[2],
            k3,
            trial_buffer,
        )
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
        half_dt = 0.5 * dt

        stage = self.stage
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3

        derivative = self.derivative
        stage_interval = self.workspace.stage_interval
        stage2_update = self.stage2_update
        stage3_update = self.stage3_update
        advance_update = self.advance_update

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + h, y + h*k1)
        stage2_update(dt, state, k1, stage)
        derivative(stage_interval(interval, dt, dt), stage, k2)

        # 3. k3 = f(t + h/2, y + h*(k1 + k2)/4)
        stage3_update(dt, state, k1, k2, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k3)

        # 4. y <- y + h*(k1 + k2 + 4k3)/6
        advance_update(dt, state, k1, k2, k3, state)

        return dt


__all__ = ["SSPRK33_TABLEAU", "SchemeSSPRK33"]
