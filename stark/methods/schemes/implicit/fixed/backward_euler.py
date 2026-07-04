from __future__ import annotations

from stark.methods.schemes.configuration import SchemeConfiguration
from stark.core.contracts import DynamicsLike, IntervalLike, Resolvent, State, Allocator
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_fixed_step_monitoring
from stark.methods.schemes.execution.call import SchemeCall
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.display.display import display_implicit_resolvent_problem
from stark.methods.schemes.implicit.runtime import SchemeRuntimeImplicit
from stark.methods.schemes.specialization.specialist import SchemeSpecialist
from stark.methods.schemes.request import SchemeResolventRequest
from stark.methods.schemes.method.tableau import Tableau


BE_TABLEAU = Tableau(
    c=(1.0,),
    a=((1.0,),),
    b=(1.0,),
    order=1,
)


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records fixed-step monitor events.
# Provides: call_monitored.
@with_fixed_step_monitoring
class SchemeBackwardEuler:
    """The implicit backward Euler method.

    Backward Euler advances by solving the implicit increment equation:

        delta = h * f(t + h, y + delta)

    and then applying the solved increment:

        y <- y + delta

    Algorithm sketch for one accepted step of size h:

        1. Build the stage problem at t + h:
               F(delta) = delta - h * f(t + h, y + delta)

        2. Use the configured resolvent to solve:
               F(delta) = 0

        3. Apply the solved increment:
               y <- y + delta

    The scheme owns the time-stepping formula. The resolvent owns the
    nonlinear solve for the implicit increment.
    """

    # Installed by the scheme monitoring decorator above this class.
    call_monitored: SchemeCall

    __slots__ = (
        "monitor",
        "block_allocator",
        "call_body",
        "call_step",
        "delta",
        "dynamics",
        "redirect_call",
        "resolvent",
        "runtime",
        "workspace",
    )

    descriptor = SchemeDescriptor("BE", "Backward Euler")

    @classmethod
    def display_tableau(cls) -> str:
        """Installed by `with_scheme_display` from `stark.methods.schemes.display`."""

        raise NotImplementedError("with_scheme_display installs display_tableau.")

    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_implicit_resolvent_problem(
            cls.tableau,
            cls.descriptor.short_name,
            cls.descriptor.full_name,
        )

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = BE_TABLEAU

    def __init__(
        self,
        dynamics: DynamicsLike,
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

        self.runtime = SchemeRuntimeImplicit(self, dynamics, allocator)
        self.dynamics = self.runtime.dynamics
        self.workspace = self.runtime.workspace
        self.block_allocator = self.runtime.block_allocator
        self.delta = self.block_allocator.allocate(1)

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

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        """Accept specialist hooks for constructor consistency."""

        del specialist

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining

        workspace = self.workspace
        resolvent = self.resolvent
        delta = self.delta

        # 1. Build the implicit stage problem:
        #        F(delta) = delta - h * f(t + h, y + delta)
        problem = SchemeResolventRequest(
            dynamics=self.dynamics,
            interval=workspace.interval_at(interval, dt, dt),
            origin=state,
            rhs=None,
            alpha=dt,
        )

        # 2. Solve F(delta) = 0.
        resolvent(problem, delta)

        # 3. Apply the solved increment: y <- y + delta.
        workspace.apply_delta(delta[0], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt

    def call_specialized(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
        return self.call_inline(interval, state)


__all__ = ["BE_TABLEAU", "SchemeBackwardEuler"]
