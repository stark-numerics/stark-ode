from __future__ import annotations

from typing import Any, cast

from stark.contracts import Derivative, IntervalLike, Resolvent, State, Allocator
from stark.schemes.support.executor import SchemeExecutor
from stark.schemes.support import (
    SchemeDescriptor,
    refresh_fixed_step_call,
    with_fixed_step_monitoring,
    with_scheme_display,
)
from stark.schemes.support.implicit import (
    initialise_implicit_support,
    with_implicit_workspace_methods,
)
from stark.schemes.support.specialist import SchemeSpecialist
from stark.schemes.support.stage_problem import SchemeStageProblem
from stark.schemes.support.stencil import (
    SchemeStencil,
    esdirk_stage_increment_stencils,
)
from stark.schemes.support.tableau import ButcherTableau


CROUZEIX_DIRK3_GAMMA = 0.5
CROUZEIX_DIRK3_TABLEAU = ButcherTableau(
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


@with_scheme_display
@with_fixed_step_monitoring
@with_implicit_workspace_methods
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

    __slots__ = (
        "_monitor",
        "block_allocator",
        "call_monitorable",
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
        "workspace",
    )

    descriptor = SchemeDescriptor("Crouzeix3", "Crouzeix DIRK3")
    tableau = CROUZEIX_DIRK3_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        allocator: Allocator,
        resolvent: Resolvent,
        *,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        self._monitor = None
        self.call_monitorable = self.call_inline
        self.redirect_call = self.call_monitorable
        self.resolvent = resolvent
        self.known2_kernel = None
        self.known3_kernel = None
        self.known4_kernel = None
        self.final_update = None

        initialise_implicit_support(self, derivative, allocator)
        self.delta1 = self.block_allocator.allocate(1)
        self.delta2 = self.block_allocator.allocate(1)
        self.delta3 = self.block_allocator.allocate(1)
        self.delta4 = self.block_allocator.allocate(1)
        self.known2 = self.block_allocator.allocate(1)
        self.known3 = self.block_allocator.allocate(1)
        self.known4 = self.block_allocator.allocate(1)
        self.trial = self.workspace.allocate_translation()

        refresh_fixed_step_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_monitorable = self.call_specialized
            refresh_fixed_step_call(self)

    def __call__(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        # Steps 2-4 build known shifts from previously solved stage increments.
        self.known2_kernel = specialist.provide(SchemeStencil(_KNOWN2_WEIGHTS))
        self.known3_kernel = specialist.provide(SchemeStencil(_KNOWN3_WEIGHTS))
        self.known4_kernel = specialist.provide(
            SchemeStencil(_KNOWN4_WEIGHTS)
        )

        # Step 5 applies the final stage-increment combination to the state.
        self.final_update = specialist.provide(
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
    ) -> SchemeStageProblem:
        return SchemeStageProblem(
            derivative=self.derivative,
            interval=self.workspace.stage_at(interval, dt, stage_shift),
            origin=state,
            rhs=rhs,
            alpha=CROUZEIX_DIRK3_GAMMA * dt,
        )

    def call_inline(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        del executor

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

    def call_specialized(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        del executor

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
