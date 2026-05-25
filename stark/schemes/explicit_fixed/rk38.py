from __future__ import annotations

from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
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


RK38_TABLEAU = ButcherTableau(
    c=(0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    a=((), (1.0 / 3.0,), (-1.0 / 3.0, 1.0), (1.0, -1.0, 1.0)),
    b=(1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0),
    order=4,
)

RK38_A = RK38_TABLEAU.a
RK38_B = RK38_TABLEAU.b


@with_scheme_display
@with_fixed_step_monitoring
@with_explicit_workspace_methods
class SchemeRK38:
    """The four-stage 3/8-rule fourth-order Runge-Kutta method.

    The 3/8-rule is a fourth-order explicit Runge-Kutta method with four
    derivative evaluations per accepted step. It has the same formal order and
    stage count as the classical RK4 method, but uses a different tableau.

    Algorithm sketch for one accepted step of size h:

        1. k1 = f(t, y)
        2. k2 = f(t + h/3,  y + h/3*k1)
        3. k3 = f(t + 2h/3, y + h*(-1/3*k1 + k2))
        4. k4 = f(t + h,    y + h*(k1 - k2 + k3))
        5. y  <- y + h*(k1 + 3k2 + 3k3 + k4)/8

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
        "k4",
        "redirect_call",
        "stage",
        "stage2_update",
        "stage3_update",
        "stage4_update",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor("RK38", "3/8 Rule Runge-Kutta")
    tableau = RK38_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        self.advance_update = unbound_scheme_call
        self.stage2_update = unbound_scheme_call
        self.stage3_update = unbound_scheme_call
        self.stage4_update = unbound_scheme_call

        self._monitor = None
        self.call_pure = self.call_inline
        self.redirect_call = self.call_pure

        initialise_explicit_support(self, derivative, workbench)

        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2, self.k3, self.k4 = (
            workspace.allocate_translation_buffers(4)
        )

        refresh_fixed_step_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
            refresh_fixed_step_call(self)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(
        self,
        specialist: SchemeSpecialist,
    ) -> None:
        """Prepare fixed-coefficient kernels for the specialized path."""

        stencils = SchemeStencilTableau(self.tableau)

        # Steps 2-4 build staged states from rows of the tableau's A matrix.
        self.stage2_update = specialist.provide(stencils.stage(1))
        self.stage3_update = specialist.provide(stencils.stage(2))
        self.stage4_update = specialist.provide(stencils.stage(3))

        # Step 5 advances the accepted state from the tableau's b weights.
        self.advance_update = specialist.provide(stencils.advance_update())

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
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
        combine4 = workspace.combine4
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval

        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        dt = interval.step if interval.step <= remaining else remaining
        one_third_dt = dt / 3.0
        two_thirds_dt = 2.0 * dt / 3.0

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + h/3, y + h/3*k1)
        stage_delta = scale(one_third_dt, k1, trial_buffer)
        stage_delta(state, stage)
        derivative(stage_interval(interval, dt, one_third_dt), stage, k2)

        # 3. k3 = f(t + 2h/3, y + h*(-1/3*k1 + k2))
        stage_delta = combine2(
            dt * RK38_A[2][0],
            k1,
            dt * RK38_A[2][1],
            k2,
            trial_buffer,
        )
        stage_delta(state, stage)
        derivative(stage_interval(interval, dt, two_thirds_dt), stage, k3)

        # 4. k4 = f(t + h, y + h*(k1 - k2 + k3))
        stage_delta = combine3(
            dt * RK38_A[3][0],
            k1,
            dt * RK38_A[3][1],
            k2,
            dt * RK38_A[3][2],
            k3,
            trial_buffer,
        )
        stage_delta(state, stage)
        derivative(stage_interval(interval, dt, dt), stage, k4)

        # 5. y <- y + h*(k1 + 3k2 + 3k3 + k4)/8
        advance_delta = combine4(
            dt * RK38_B[0],
            k1,
            dt * RK38_B[1],
            k2,
            dt * RK38_B[2],
            k3,
            dt * RK38_B[3],
            k4,
            trial_buffer,
        )
        apply_delta(advance_delta, state)

        return dt

    def call_specialized(
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
        one_third_dt = dt / 3.0
        two_thirds_dt = 2.0 * dt / 3.0

        stage = self.stage
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        derivative = self.derivative
        stage_interval = self.workspace.stage_interval
        stage2_update = self.stage2_update
        stage3_update = self.stage3_update
        stage4_update = self.stage4_update
        advance_update = self.advance_update

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + h/3, y + h/3*k1)
        stage2_update(dt, state, k1, stage)
        derivative(stage_interval(interval, dt, one_third_dt), stage, k2)

        # 3. k3 = f(t + 2h/3, y + h*(-1/3*k1 + k2))
        stage3_update(dt, state, k1, k2, stage)
        derivative(stage_interval(interval, dt, two_thirds_dt), stage, k3)

        # 4. k4 = f(t + h, y + h*(k1 - k2 + k3))
        stage4_update(dt, state, k1, k2, k3, stage)
        derivative(stage_interval(interval, dt, dt), stage, k4)

        # 5. y <- y + h*(k1 + 3k2 + 3k3 + k4)/8
        advance_update(dt, state, k1, k2, k3, k4, state)

        return dt


__all__ = ["RK38_TABLEAU", "SchemeRK38"]
