from __future__ import annotations

from stark.methods.schemes.configuration import SchemeConfiguration
from typing import Any, cast

from stark.core.contracts import DerivativeLike, IntervalLike, Resolvent, State, Allocator
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_fixed_step_monitoring
from stark.methods.schemes.execution.call import SchemeCall
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.display.display import display_implicit_resolvent_problem
from stark.methods.schemes.implicit.runtime import SchemeRuntimeImplicit
from stark.methods.schemes.specialization.specialist import SchemeSpecialist
from stark.methods.schemes.request import SchemeResolventRequest
from stark.methods.schemes.specialization.stencil import SchemeStencil
from stark.methods.schemes.method.tableau import Tableau


IMPLICIT_MIDPOINT_TABLEAU = Tableau(
    c=(0.5,),
    a=((0.5,),),
    b=(1.0,),
    order=2,
    short_name="IM",
    full_name="Implicit Midpoint",
)


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records fixed-step monitor events.
# Provides: call_monitored.
@with_fixed_step_monitoring
class SchemeImplicitMidpoint:
    """The one-stage implicit midpoint Runge-Kutta method.

    Implicit midpoint solves a midpoint increment and then doubles it to
    advance the full step.

    Algorithm sketch for one accepted step of size h:

        1. Solve the midpoint increment:
               z = h/2 * f(t + h/2, y + z)

        2. Advance with the full-step increment:
               y <- y + 2z
    """

    # Installed by the scheme monitoring decorator above this class.
    call_monitored: SchemeCall

    __slots__ = (
        "monitor",
        "advance_update",
        "block_allocator",
        "call_body",
        "call_step",
        "derivative",
        "midpoint",
        "redirect_call",
        "resolvent",
        "trial",
        "runtime",
        "workspace",
    )

    descriptor = SchemeDescriptor("IM", "Implicit Midpoint")
    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_implicit_resolvent_problem(
            cls.tableau,
            cls.descriptor.short_name,
            cls.descriptor.full_name,
        )

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = IMPLICIT_MIDPOINT_TABLEAU

    def __init__(
        self,
        derivative: DerivativeLike,
        allocator: Allocator,
        resolvent: Resolvent,
        *,
        configuration: SchemeConfiguration | None = None,
        specialist: SchemeSpecialist | None = None,
        monitor: SchemeMonitor | None = None,
    ) -> None:
        self.monitor = monitor
        self.call_body = self.call_inline
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step
        self.resolvent = resolvent
        self.advance_update = None

        self.runtime = SchemeRuntimeImplicit(self, derivative, allocator)
        self.derivative = self.runtime.derivative
        self.workspace = self.runtime.workspace
        self.block_allocator = self.runtime.block_allocator
        self.midpoint = self.block_allocator.allocate(1)
        self.trial = self.workspace.allocate_translation()

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_body = self.call_specialized
            if monitor is None:
                self.call_step = self.call_body
                self.redirect_call = self.call_step

    def __call__(self, interval: IntervalLike, state: State) -> float:
        return self.redirect_call(interval, state)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        # Step 2 applies the doubled midpoint increment.
        self.advance_update = specialist.provide_apply(SchemeStencil((2.0,), apply=True))

    def call_inline(self, interval: IntervalLike, state: State) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        workspace = self.workspace
        midpoint = self.midpoint

        # 1. Solve the midpoint increment:
        #        z = h/2 * f(t + h/2, y + z)
        problem = SchemeResolventRequest(
            derivative=self.derivative,
            interval=workspace.interval_at(interval, dt, 0.5 * dt),
            origin=state,
            rhs=None,
            alpha=0.5 * dt,
        )
        self.resolvent(problem, midpoint)

        # 2. Advance y <- y + 2z.
        delta = workspace.scale(2.0, midpoint[0], self.trial)
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt

    def call_specialized(self, interval: IntervalLike, state: State) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        workspace = self.workspace
        midpoint = self.midpoint
        advance_update = cast(Any, self.advance_update)

        # 1. Solve the midpoint increment:
        #        z = h/2 * f(t + h/2, y + z)
        problem = SchemeResolventRequest(
            derivative=self.derivative,
            interval=workspace.interval_at(interval, dt, 0.5 * dt),
            origin=state,
            rhs=None,
            alpha=0.5 * dt,
        )
        self.resolvent(problem, midpoint)

        # 2. Advance y <- y + 2z.
        advance_update(1.0, state, midpoint[0], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = ["IMPLICIT_MIDPOINT_TABLEAU", "SchemeImplicitMidpoint"]
