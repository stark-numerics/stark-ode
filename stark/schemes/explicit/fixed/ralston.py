from __future__ import annotations

from stark.contracts import Derivative, IntervalLike, State, Allocator
from stark.schemes.execution.executor import SchemeExecutor
from stark.schemes.method.descriptor import SchemeDescriptor
from stark.schemes.monitoring.monitor import MonitorSchemeLike
from stark.schemes.monitoring.decorators import with_fixed_step_monitoring
from stark.schemes.explicit._support import (
    explicit_snapshot_state,
    initialise_explicit_support,
)
from stark.schemes.execution.unbound import unbound_scheme_call
from stark.schemes.display.decorators import with_scheme_display
from stark.schemes.specialization.specialist import SchemeSpecialist
from stark.schemes.specialization.stencil import SchemeStencilTableau
from stark.schemes.method.tableau import ButcherTableau


RALSTON_TABLEAU = ButcherTableau(
    c=(0.0, 2.0 / 3.0),
    a=((), (2.0 / 3.0,)),
    b=(0.25, 0.75),
    order=2,
)

RALSTON_B = RALSTON_TABLEAU.b
RALSTON_STAGE_FACTOR = 2.0 / 3.0


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, short_name, full_name, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records fixed-step monitor events.
# Provides: call_monitored.
@with_fixed_step_monitoring
class SchemeRalston:
    """Ralston's optimized two-stage second-order Runge-Kutta method.

    Among explicit RK2 methods, Ralston's choice of coefficients reduces the
    leading local truncation error constant, which often makes it a slightly
    sharper fixed-step second-order baseline than midpoint or Heun.

    Algorithm sketch for one accepted step of size h:

        1. k1 = f(t, y)
        2. k2 = f(t + 2h/3, y + 2h/3*k1)
        3. y  <- y + h*(k1 + 3k2)/4

    The inline path expresses the method directly with workspace arithmetic.
    The specialized path uses fixed-coefficient kernels prepared from the same
    tableau rows and weights.

    Further reading: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "monitor",
        "advance_update",
        "call_body",
        "call_step",
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

    descriptor = SchemeDescriptor("Ralston", "Ralston")
    snapshot_state = explicit_snapshot_state

    tableau = RALSTON_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        allocator: Allocator,
        specialist: SchemeSpecialist | None = None,
        monitor: MonitorSchemeLike | None = None,
    ) -> None:
        self.advance_update = unbound_scheme_call
        self.stage2_update = unbound_scheme_call

        self.monitor = monitor
        self.call_body = self.call_inline
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step

        initialise_explicit_support(self, derivative, allocator)

        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2 = workspace.allocate_translation_buffers(2)

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
        interval_at = workspace.interval_at

        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2

        dt = interval.step if interval.step <= remaining else remaining
        stage_dt = RALSTON_STAGE_FACTOR * dt

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + 2h/3, y + 2h/3*k1)
        stage_delta = scale(stage_dt, k1, trial_buffer)
        stage_delta(state, stage)
        derivative(interval_at(interval, dt, stage_dt), stage, k2)

        # 3. y <- y + h*(k1 + 3k2)/4
        advance_delta = combine2(
            dt * RALSTON_B[0],
            k1,
            dt * RALSTON_B[1],
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
        stage_dt = RALSTON_STAGE_FACTOR * dt

        stage = self.stage
        k1 = self.k1
        k2 = self.k2

        derivative = self.derivative
        interval_at = self.workspace.interval_at
        stage2_update = self.stage2_update
        advance_update = self.advance_update

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + 2h/3, y + 2h/3*k1)
        stage2_update(dt, state, k1, stage)
        derivative(interval_at(interval, dt, stage_dt), stage, k2)

        # 3. y <- y + h*(k1 + 3k2)/4
        advance_update(dt, state, k1, k2, state)

        return dt


__all__ = ["RALSTON_TABLEAU", "SchemeRalston"]
