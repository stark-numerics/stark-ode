from __future__ import annotations

from stark.methods.schemes.configuration import SchemeConfiguration
from math import sqrt

from stark.core.contracts import DerivativeLike, IntervalLike, Resolvent, State, Allocator
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_fixed_step_monitoring
from stark.methods.schemes.execution.call import SchemeCall
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.display.display import display_implicit_resolvent_problem
from stark.methods.schemes.implicit.runtime import SchemeRuntimeImplicit
from stark.methods.schemes.specialization.specialist import SchemeSpecialist
from stark.methods.schemes.request import SchemeResolventRequestCoupled
from stark.methods.schemes.method.tableau import Tableau


RADAU_IIA5_SQRT6 = sqrt(6.0)
RADAU_IIA5_TABLEAU = Tableau(
    c=((4.0 - RADAU_IIA5_SQRT6) / 10.0, (4.0 + RADAU_IIA5_SQRT6) / 10.0, 1.0),
    a=(
        (
            (88.0 - 7.0 * RADAU_IIA5_SQRT6) / 360.0,
            (296.0 - 169.0 * RADAU_IIA5_SQRT6) / 1800.0,
            (-2.0 + 3.0 * RADAU_IIA5_SQRT6) / 225.0,
        ),
        (
            (296.0 + 169.0 * RADAU_IIA5_SQRT6) / 1800.0,
            (88.0 + 7.0 * RADAU_IIA5_SQRT6) / 360.0,
            (-2.0 - 3.0 * RADAU_IIA5_SQRT6) / 225.0,
        ),
        (
            (16.0 - RADAU_IIA5_SQRT6) / 36.0,
            (16.0 + RADAU_IIA5_SQRT6) / 36.0,
            1.0 / 9.0,
        ),
    ),
    b=(
        (16.0 - RADAU_IIA5_SQRT6) / 36.0,
        (16.0 + RADAU_IIA5_SQRT6) / 36.0,
        1.0 / 9.0,
    ),
    order=5,
    short_name="Radau5",
    full_name="Radau IIA 5",
)


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records fixed-step monitor events.
# Provides: call_monitored.
@with_fixed_step_monitoring
class SchemeRadauIIA5:
    """The three-stage fifth-order Radau IIA collocation method.

    Algorithm sketch for one accepted step of size h:

        1. Solve the coupled Radau stage increment problem.
        2. Apply the final stage increment directly:
               y <- y + delta_3

    Radau IIA 5 is stiffly accurate: the final tableau row equals b, so the
    final stage increment is already the full accepted step update.
    """

    # Installed by the scheme monitoring decorator above this class.
    call_monitored: SchemeCall

    __slots__ = (
        "monitor",
        "block_allocator",
        "call_body",
        "call_step",
        "derivative",
        "redirect_call",
        "resolvent",
        "stage_delta",
        "runtime",
        "workspace",
    )

    descriptor = SchemeDescriptor("Radau5", "Radau IIA 5")
    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_implicit_resolvent_problem(
            cls.tableau,
            cls.descriptor.short_name,
            cls.descriptor.full_name,
        )

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = RADAU_IIA5_TABLEAU

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

        self.runtime = SchemeRuntimeImplicit(self, derivative, allocator)
        self.derivative = self.runtime.derivative
        self.workspace = self.runtime.workspace
        self.block_allocator = self.runtime.block_allocator
        self.stage_delta = self.block_allocator.allocate(3)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_body = self.call_specialized
            if monitor is None:
                self.call_step = self.call_body
                self.redirect_call = self.call_step

    def __call__(self, interval: IntervalLike, state: State) -> float:
        return self.redirect_call(interval, state)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        """Accept specialist hooks for constructor consistency."""

        del specialist

    def _problem(self, interval: IntervalLike, state: State, dt: float) -> SchemeResolventRequestCoupled:
        tableau = self.tableau
        return SchemeResolventRequestCoupled(
            derivative=self.derivative,
            interval=interval,
            origin=state,
            rhs=None,
            step=dt,
            stage_shifts=tableau.c,
            matrix=tableau.a,
        )

    def call_inline(self, interval: IntervalLike, state: State) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining

        # 1. Solve the coupled Radau stage increment problem.
        self.resolvent(self._problem(interval, state, dt), self.stage_delta)

        # 2. Apply the final stage increment directly.
        self.workspace.apply_delta(self.stage_delta[2], state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt

    def call_specialized(self, interval: IntervalLike, state: State) -> float:
        return self.call_inline(interval, state)


__all__ = ["RADAU_IIA5_SQRT6", "RADAU_IIA5_TABLEAU", "SchemeRadauIIA5"]
