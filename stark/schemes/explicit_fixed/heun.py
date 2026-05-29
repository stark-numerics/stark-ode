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


HEUN_TABLEAU = ButcherTableau(
    c=(0.0, 1.0),
    a=((), (1.0,)),
    b=(0.5, 0.5),
    order=2,
)

HEUN_B = HEUN_TABLEAU.b


@with_scheme_display
@with_fixed_step_monitoring
@with_explicit_workspace_methods
class SchemeHeun:
    """Heun's explicit two-stage second-order Runge-Kutta method.

    This method averages a forward-Euler predictor slope with a slope evaluated
    at the end of the step, giving a simple second-order scheme sometimes
    called the explicit trapezoidal rule or improved Euler method.

    Algorithm sketch for one accepted step of size h:

        1. k1 = f(t, y)
        2. k2 = f(t + h, y + h*k1)
        3. y  <- y + h*(k1 + k2)/2

    The inline path expresses the method directly with workspace arithmetic.
    The specialized path uses fixed-coefficient kernels prepared from the same
    tableau rows and weights.

    Further reading: https://en.wikipedia.org/wiki/Heun%27s_method
    """

    __slots__ = (
        "_monitor",
        "advance_update",
        "call_pure",
        "derivative",
        "explicit",
        "k1",
        "k2",
        "redirect_call",
        "stage",
        "stage2_update",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor("Heun", "Heun")
    tableau = HEUN_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        allocator: Allocator,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        self.advance_update = unbound_scheme_call
        self.stage2_update = unbound_scheme_call

        self._monitor = None
        self.call_pure = self.call_inline
        self.redirect_call = self.call_pure

        initialise_explicit_support(self, derivative, allocator)

        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2 = workspace.allocate_translation_buffers(2)

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

        # Step 2 builds the second-stage state from row 1 of the A matrix.
        self.stage2_update = specialist.provide(stencils.stage(1))

        # Step 3 advances the accepted state from the tableau's b weights.
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
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval

        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2

        dt = interval.step if interval.step <= remaining else remaining

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + h, y + h*k1)
        stage_delta = scale(dt, k1, trial_buffer)
        stage_delta(state, stage)
        derivative(stage_interval(interval, dt, dt), stage, k2)

        # 3. y <- y + h*(k1 + k2)/2
        advance_delta = combine2(
            dt * HEUN_B[0],
            k1,
            dt * HEUN_B[1],
            k2,
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

        stage = self.stage
        k1 = self.k1
        k2 = self.k2

        derivative = self.derivative
        stage_interval = self.workspace.stage_interval
        stage2_update = self.stage2_update
        advance_update = self.advance_update

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + h, y + h*k1)
        stage2_update(dt, state, k1, stage)
        derivative(stage_interval(interval, dt, dt), stage, k2)

        # 3. y <- y + h*(k1 + k2)/2
        advance_update(dt, state, k1, k2, state)

        return dt


__all__ = ["HEUN_TABLEAU", "SchemeHeun"]
