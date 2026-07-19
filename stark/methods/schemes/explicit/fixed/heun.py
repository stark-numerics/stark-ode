from __future__ import annotations

from stark.methods.schemes.configuration import SchemeConfiguration
from stark.core.contracts import DynamicsLike, IntervalLike, State, AllocatorLike
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_fixed_step_monitoring
from stark.methods.schemes.execution.call import SchemeCall
from stark.methods.schemes.explicit.runtime import SchemeRuntimeExplicit
from stark.methods.schemes.execution.unbound import unbound_scheme_call
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.linear_fixed_generation.linear_fixed import SchemeLinearFixedLike
from stark.methods.schemes.linear_fixed_generation.stencil import SchemeStencilTableau
from stark.methods.schemes.method.tableau import Tableau


HEUN_TABLEAU = Tableau(
    c=(0.0, 1.0),
    a=((), (1.0,)),
    b=(0.5, 0.5),
    order=2,
)

HEUN_B = HEUN_TABLEAU.b


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records fixed-step monitor events.
# Provides: call_monitored.
@with_fixed_step_monitoring
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

    # Installed by the scheme monitoring decorator above this class.
    call_monitored: SchemeCall

    __slots__ = (
        "monitor",
        "advance_update",
        "call_body",
        "call_step",
        "dynamics",
        "runtime",
        "k1",
        "k2",
        "redirect_call",
        "stage",
        "stage2_update",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor("Heun", "Heun")

    @classmethod
    def display_tableau(cls) -> str:
        """Installed by `with_scheme_display` from `stark.methods.schemes.display`."""

        raise NotImplementedError("with_scheme_display installs display_tableau.")

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = HEUN_TABLEAU

    def __init__(
        self,
        dynamics: DynamicsLike,
        allocator: AllocatorLike,
        configuration: SchemeConfiguration | None = None,
        linear_fixed: SchemeLinearFixedLike | None = None,
        monitor: SchemeMonitor | None = None,
    ) -> None:
        self.advance_update = unbound_scheme_call
        self.stage2_update = unbound_scheme_call

        self.monitor = monitor
        self.call_body = self.call_inline
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step

        self.runtime = SchemeRuntimeExplicit(dynamics, allocator)
        self.dynamics = self.runtime.dynamics
        self.workspace = self.runtime.workspace
        self.k1 = self.runtime.k1

        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2 = workspace.allocate_translation_buffers(2)

        if linear_fixed is not None:
            self.prepare_specialized_kernels(linear_fixed)
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
        linear_fixed: SchemeLinearFixedLike,
    ) -> None:
        """Prepare fixed-coefficient kernels for the specialized path."""

        stencils = SchemeStencilTableau(self.tableau)

        # Step 2 builds the second-stage state from row 1 of the A matrix.
        self.stage2_update = linear_fixed(stencils.stage(1))

        # Step 3 advances the accepted state from the tableau's b weights.
        self.advance_update = linear_fixed(stencils.advance_update())

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.workspace
        dynamics = self.dynamics
        scale = workspace.scale
        combine2 = workspace.combine2
        apply_delta = workspace.apply_delta
        interval_at = workspace.interval_at

        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2

        dt = interval.step if interval.step <= remaining else remaining

        # 1. k1 = f(t, y)
        dynamics(interval, state, k1)

        # 2. k2 = f(t + h, y + h*k1)
        stage_delta = scale(dt, k1, trial_buffer)
        stage_delta(state, stage)
        dynamics(interval_at(interval, dt, dt), stage, k2)

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
    ) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining

        stage = self.stage
        k1 = self.k1
        k2 = self.k2

        dynamics = self.dynamics
        interval_at = self.workspace.interval_at
        stage2_update = self.stage2_update
        advance_update = self.advance_update

        # 1. k1 = f(t, y)
        dynamics(interval, state, k1)

        # 2. k2 = f(t + h, y + h*k1)
        stage2_update(dt, state, k1, stage)
        dynamics(interval_at(interval, dt, dt), stage, k2)

        # 3. y <- y + h*(k1 + k2)/2
        advance_update(dt, state, k1, k2, state)

        return dt


__all__ = ["HEUN_TABLEAU", "SchemeHeun"]
