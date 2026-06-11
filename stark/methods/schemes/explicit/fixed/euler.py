from __future__ import annotations

from stark.methods.schemes.configuration import SchemeConfiguration
from stark.contracts import DerivativeLike, IntervalLike, State, Allocator
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_fixed_step_monitoring
from stark.methods.schemes.explicit._support import (
    explicit_snapshot_state,
    initialise_explicit_support,
)
from stark.methods.schemes.execution.unbound import unbound_scheme_call
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.specialization.specialist import SchemeSpecialist
from stark.methods.schemes.specialization.stencil import SchemeStencilTableau
from stark.methods.schemes.method.tableau import ButcherTableau


EULER_TABLEAU = ButcherTableau(
    c=(0.0,),
    a=((),),
    b=(1.0,),
    order=1,
)

EULER_B = EULER_TABLEAU.b


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, short_name, full_name, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records fixed-step monitor events.
# Provides: call_monitored.
@with_fixed_step_monitoring
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
        "monitor",
        "advance_delta_buffer",
        "advance_update",
        "call_body",
        "call_step",
        "derivative",
        "explicit",
        "k1",
        "redirect_call",
        "workspace",
    )

    descriptor = SchemeDescriptor("Euler", "Forward Euler")
    snapshot_state = explicit_snapshot_state

    tableau = EULER_TABLEAU

    def __init__(
        self,
        derivative: DerivativeLike,
        allocator: Allocator,
        configuration: SchemeConfiguration | None = None,
        specialist: SchemeSpecialist | None = None,
        monitor: SchemeMonitor | None = None,
    ) -> None:
        self.advance_update = unbound_scheme_call

        self.monitor = monitor
        self.call_body = self.call_inline
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step

        initialise_explicit_support(self, derivative, allocator)
        self.advance_delta_buffer = self.workspace.allocate_translation()

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

        # Step 2 advances the accepted state from the tableau's b weights.
        self.advance_update = specialist.provide(stencils.advance_update())

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
    ) -> float:
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
