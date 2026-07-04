from __future__ import annotations

from stark.methods.schemes.configuration import SchemeConfiguration
from stark.core.contracts import DerivativeLike, IntervalLike, State, Allocator
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_fixed_step_monitoring
from stark.methods.schemes.execution.call import SchemeCall
from stark.methods.schemes.explicit.runtime import SchemeRuntimeExplicit
from stark.methods.schemes.execution.unbound import unbound_scheme_call
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.specialization.specialist import SchemeSpecialist
from stark.methods.schemes.specialization.stencil import SchemeStencilTableau
from stark.methods.schemes.method.tableau import Tableau


RK4_TABLEAU = Tableau(
    c=(0.0, 0.5, 0.5, 1.0),
    a=(
        (),
        (0.5,),
        (0.0, 0.5),
        (0.0, 0.0, 1.0),
    ),
    b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    order=4,
)

RK4_B = RK4_TABLEAU.b


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records fixed-step monitor events.
# Provides: call_monitored.
@with_fixed_step_monitoring
class SchemeRK4:
    """The classical four-stage fourth-order Runge-Kutta method.

    RK4 is the best-known fixed-step explicit Runge-Kutta scheme. It is a very
    common general-purpose baseline because it offers fourth-order accuracy,
    straightforward staging, and good behavior on many non-stiff problems.

    Algorithm sketch for one accepted step of size h:

        1. k1 = f(t, y)
        2. k2 = f(t + h/2, y + h/2*k1)
        3. k3 = f(t + h/2, y + h/2*k2)
        4. k4 = f(t + h,   y + h*k3)
        5. y  <- y + h*(k1 + 2k2 + 2k3 + k4)/6

    The inline path expresses the method directly with workspace arithmetic.
    The specialized path uses fixed-coefficient kernels prepared from the same
    tableau rows and weights.

    Further reading: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    # Installed by the scheme monitoring decorator above this class.
    call_monitored: SchemeCall

    __slots__ = (
        "monitor",
        "advance_update",
        "call_body",
        "call_step",
        "derivative",
        "runtime",
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

    descriptor = SchemeDescriptor("RK4", "Classical Runge-Kutta")

    @classmethod
    def display_tableau(cls) -> str:
        """Installed by `with_scheme_display` from `stark.methods.schemes.display`."""

        raise NotImplementedError("with_scheme_display installs display_tableau.")

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = RK4_TABLEAU

    def __init__(
        self,
        derivative: DerivativeLike,
        allocator: Allocator,
        configuration: SchemeConfiguration | None = None,
        specialist: SchemeSpecialist | None = None,
        monitor: SchemeMonitor | None = None,
    ) -> None:
        self.advance_update = unbound_scheme_call
        self.stage2_update = unbound_scheme_call
        self.stage3_update = unbound_scheme_call
        self.stage4_update = unbound_scheme_call

        self.monitor = monitor
        self.call_body = self.call_inline
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step

        self.runtime = SchemeRuntimeExplicit(derivative, allocator)
        self.derivative = self.runtime.derivative
        self.workspace = self.runtime.workspace
        self.k1 = self.runtime.k1

        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2, self.k3, self.k4 = (
            workspace.allocate_translation_buffers(4)
        )

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
    ) -> float:
        return self.redirect_call(interval, state)

    def prepare_specialized_kernels(
        self,
        specialist: SchemeSpecialist,
    ) -> None:
        """Prepare fixed-coefficient kernels for the specialized path."""

        stencils = SchemeStencilTableau(self.tableau)

        # Steps 2-4 build staged states from rows of the tableau's A matrix.
        self.stage2_update = specialist.provide_apply(stencils.stage(1))
        self.stage3_update = specialist.provide_apply(stencils.stage(2))
        self.stage4_update = specialist.provide_apply(stencils.stage(3))

        # Step 5 advances the accepted state from the tableau's b weights.
        self.advance_update = specialist.provide_apply(stencils.advance_update())

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine4 = workspace.combine4
        apply_delta = workspace.apply_delta
        interval_at = workspace.interval_at

        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        dt = interval.step if interval.step <= remaining else remaining
        half_dt = 0.5 * dt

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + h/2, y + h/2*k1)
        stage_delta = scale(half_dt, k1, trial_buffer)
        stage_delta(state, stage)
        derivative(interval_at(interval, dt, half_dt), stage, k2)

        # 3. k3 = f(t + h/2, y + h/2*k2)
        stage_delta = scale(half_dt, k2, trial_buffer)
        stage_delta(state, stage)
        derivative(interval_at(interval, dt, half_dt), stage, k3)

        # 4. k4 = f(t + h, y + h*k3)
        stage_delta = scale(dt, k3, trial_buffer)
        stage_delta(state, stage)
        derivative(interval_at(interval, dt, dt), stage, k4)

        # 5. y <- y + h*(k1 + 2k2 + 2k3 + k4)/6
        advance_delta = combine4(
            dt * RK4_B[0],
            k1,
            dt * RK4_B[1],
            k2,
            dt * RK4_B[2],
            k3,
            dt * RK4_B[3],
            k4,
            trial_buffer,
        )
        apply_delta(advance_delta, state)

        return dt

    def call_specialized(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        half_dt = 0.5 * dt

        stage = self.stage
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        derivative = self.derivative
        interval_at = self.workspace.interval_at
        stage2_update = self.stage2_update
        stage3_update = self.stage3_update
        stage4_update = self.stage4_update
        advance_update = self.advance_update

        # 1. k1 = f(t, y)
        derivative(interval, state, k1)

        # 2. k2 = f(t + h/2, y + h/2*k1)
        stage2_update(dt, state, k1, stage)
        derivative(interval_at(interval, dt, half_dt), stage, k2)

        # 3. k3 = f(t + h/2, y + h/2*k2)
        stage3_update(dt, state, k1, k2, stage)
        derivative(interval_at(interval, dt, half_dt), stage, k3)

        # 4. k4 = f(t + h, y + h*k3)
        stage4_update(dt, state, k1, k2, k3, stage)
        derivative(interval_at(interval, dt, dt), stage, k4)

        # 5. y <- y + h*(k1 + 2k2 + 2k3 + k4)/6
        advance_update(dt, state, k1, k2, k3, k4, state)

        return dt


__all__ = ["RK4_TABLEAU", "SchemeRK4"]
