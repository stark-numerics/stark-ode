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
from stark.methods.schemes.specialization.stencil import (
    SchemeStencil,
    esdirk_stage_increment_stencils,
)
from stark.methods.schemes.method.tableau import Tableau


CROUZEIX_DIRK3_GAMMA = 0.5
CROUZEIX_DIRK3_TABLEAU = Tableau(
    c=(0.5, 2.0 / 3.0, 0.5, 1.0),
    a=(
        (0.5,),
        (1.0 / 6.0, 0.5),
        (-0.5, 0.5, 0.5),
        (1.5, -1.5, 0.5, 0.5),
    ),
    b=(1.5, -1.5, 0.5, 0.5),
    order=3,
    short_name="Crouzeix3",
    full_name="Crouzeix DIRK3",
)

_STAGE_STENCILS = esdirk_stage_increment_stencils(
    CROUZEIX_DIRK3_TABLEAU,
    CROUZEIX_DIRK3_GAMMA,
)
_KNOWN2_WEIGHTS = _STAGE_STENCILS.known_shifts[1]
_KNOWN3_WEIGHTS = _STAGE_STENCILS.known_shifts[2]
_KNOWN4_WEIGHTS = _STAGE_STENCILS.known_shifts[3]
_STAGE_INCREMENT_WEIGHTS = _STAGE_STENCILS.high_delta


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records fixed-step monitor events.
# Provides: call_monitored.
@with_fixed_step_monitoring
class SchemeCrouzeixDIRK3:
    """Crouzeix's fixed-step third-order sequential DIRK method.

    Algorithm sketch for one accepted step of size h:

        1. Solve stage 1:
               delta_1 = gamma h f(t + h/2, y + delta_1)

        2. Build the known shift for stage 2 from delta_1, then solve stage 2.

        3. Build the known shift for stage 3 from delta_1 and delta_2, then
           solve stage 3.

        4. Build the known shift for stage 4 from delta_1, delta_2, and
           delta_3, then solve stage 4.

        5. Advance using the stage-increment representation:
               y <- y + w_1 delta_1 + w_2 delta_2 + w_3 delta_3 + delta_4

    The known-shift and advance constants are the tableau coefficients
    expressed in the diagonal stage-increment basis used by the sequential
    implicit solves.
    """

    # Installed by the scheme monitoring decorator above this class.
    call_monitored: SchemeCall

    __slots__ = (
        "monitor",
        "block_allocator",
        "call_body",
        "call_step",
        "delta1",
        "delta2",
        "delta3",
        "delta4",
        "derivative",
        "final_update",
        "known2",
        "known3",
        "known4",
        "known2_kernel",
        "known3_kernel",
        "known4_kernel",
        "redirect_call",
        "resolvent",
        "trial",
        "runtime",
        "workspace",
    )

    descriptor = SchemeDescriptor("Crouzeix3", "Crouzeix DIRK3")
    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_implicit_resolvent_problem(
            cls.tableau,
            cls.descriptor.short_name,
            cls.descriptor.full_name,
        )

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = CROUZEIX_DIRK3_TABLEAU

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
        self.known2_kernel = None
        self.known3_kernel = None
        self.known4_kernel = None
        self.final_update = None

        self.runtime = SchemeRuntimeImplicit(self, derivative, allocator)
        self.derivative = self.runtime.derivative
        self.workspace = self.runtime.workspace
        self.block_allocator = self.runtime.block_allocator
        self.delta1 = self.block_allocator.allocate(1)
        self.delta2 = self.block_allocator.allocate(1)
        self.delta3 = self.block_allocator.allocate(1)
        self.delta4 = self.block_allocator.allocate(1)
        self.known2 = self.block_allocator.allocate(1)
        self.known3 = self.block_allocator.allocate(1)
        self.known4 = self.block_allocator.allocate(1)
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
        # Steps 2-4 build known shifts from previously solved stage increments.
        self.known2_kernel = specialist.provide_delta(SchemeStencil(_KNOWN2_WEIGHTS))
        self.known3_kernel = specialist.provide_delta(SchemeStencil(_KNOWN3_WEIGHTS))
        self.known4_kernel = specialist.provide_delta(
            SchemeStencil(_KNOWN4_WEIGHTS)
        )

        # Step 5 applies the final stage-increment combination to the state.
        self.final_update = specialist.provide_apply(
            SchemeStencil(_STAGE_INCREMENT_WEIGHTS, apply=True)
        )

    def _stage_problem(
        self,
        interval: IntervalLike,
        state: State,
        dt: float,
        *,
        stage_shift: float,
        rhs,
    ) -> SchemeResolventRequest:
        return SchemeResolventRequest(
            derivative=self.derivative,
            interval=self.workspace.interval_at(interval, dt, stage_shift),
            origin=state,
            rhs=rhs,
            alpha=CROUZEIX_DIRK3_GAMMA * dt,
        )

    def call_inline(self, interval: IntervalLike, state: State) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.workspace
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        combine4 = workspace.combine4
        dt = interval.step if interval.step <= remaining else remaining

        # 1. Solve stage 1.
        self.resolvent(
            self._stage_problem(interval, state, dt, stage_shift=0.5 * dt, rhs=None),
            self.delta1,
        )

        # 2. Build known shift for stage 2 and solve stage 2.
        self.known2[0] = scale(_KNOWN2_WEIGHTS[0], self.delta1[0], self.known2[0])
        self.resolvent(
            self._stage_problem(
                interval,
                state,
                dt,
                stage_shift=(2.0 / 3.0) * dt,
                rhs=self.known2,
            ),
            self.delta2,
        )

        # 3. Build known shift for stage 3 and solve stage 3.
        self.known3[0] = combine2(
            _KNOWN3_WEIGHTS[0],
            self.delta1[0],
            _KNOWN3_WEIGHTS[1],
            self.delta2[0],
            self.known3[0],
        )
        self.resolvent(
            self._stage_problem(interval, state, dt, stage_shift=0.5 * dt, rhs=self.known3),
            self.delta3,
        )

        # 4. Build known shift for stage 4 and solve stage 4.
        self.known4[0] = combine3(
            _KNOWN4_WEIGHTS[0],
            self.delta1[0],
            _KNOWN4_WEIGHTS[1],
            self.delta2[0],
            _KNOWN4_WEIGHTS[2],
            self.delta3[0],
            self.known4[0],
        )
        self.resolvent(
            self._stage_problem(interval, state, dt, stage_shift=dt, rhs=self.known4),
            self.delta4,
        )

        # 5. Advance with the stage-increment representation.
        delta = combine4(
            _STAGE_INCREMENT_WEIGHTS[0],
            self.delta1[0],
            _STAGE_INCREMENT_WEIGHTS[1],
            self.delta2[0],
            _STAGE_INCREMENT_WEIGHTS[2],
            self.delta3[0],
            _STAGE_INCREMENT_WEIGHTS[3],
            self.delta4[0],
            self.trial,
        )
        workspace.apply_delta(delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt

    def call_specialized(self, interval: IntervalLike, state: State) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        known2_kernel = cast(Any, self.known2_kernel)
        known3_kernel = cast(Any, self.known3_kernel)
        known4_kernel = cast(Any, self.known4_kernel)
        final_update = cast(Any, self.final_update)

        dt = interval.step if interval.step <= remaining else remaining

        # 1. Solve stage 1.
        self.resolvent(
            self._stage_problem(interval, state, dt, stage_shift=0.5 * dt, rhs=None),
            self.delta1,
        )

        # 2. Build known shift for stage 2 and solve stage 2.
        self.known2[0] = known2_kernel(1.0, self.delta1[0], self.known2[0])
        self.resolvent(
            self._stage_problem(
                interval,
                state,
                dt,
                stage_shift=(2.0 / 3.0) * dt,
                rhs=self.known2,
            ),
            self.delta2,
        )

        # 3. Build known shift for stage 3 and solve stage 3.
        self.known3[0] = known3_kernel(
            1.0,
            self.delta1[0],
            self.delta2[0],
            self.known3[0],
        )
        self.resolvent(
            self._stage_problem(interval, state, dt, stage_shift=0.5 * dt, rhs=self.known3),
            self.delta3,
        )

        # 4. Build known shift for stage 4 and solve stage 4.
        self.known4[0] = known4_kernel(
            1.0,
            self.delta1[0],
            self.delta2[0],
            self.delta3[0],
            self.known4[0],
        )
        self.resolvent(
            self._stage_problem(interval, state, dt, stage_shift=dt, rhs=self.known4),
            self.delta4,
        )

        # 5. Advance with the stage-increment representation.
        final_update(
            1.0,
            state,
            self.delta1[0],
            self.delta2[0],
            self.delta3[0],
            self.delta4[0],
            state,
        )

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = [
    "CROUZEIX_DIRK3_GAMMA",
    "CROUZEIX_DIRK3_TABLEAU",
    "SchemeCrouzeixDIRK3",
]
