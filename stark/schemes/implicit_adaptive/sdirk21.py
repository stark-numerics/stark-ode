from __future__ import annotations

from stark.accelerators.binding import DerivativeAccelerated
from stark.block import Block
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.resolvents.support.failure import ResolventError
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    SchemeStepControl,
    initialise_adaptive_runtime,
    refresh_adaptive_call,
    unbound_scheme_call,
    with_adaptive_runtime_methods,
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

from math import sqrt


SDIRK21_GAMMA = (2.0 - sqrt(2.0)) / 2.0
SDIRK21_B2 = (1.0 - 2.0 * SDIRK21_GAMMA) / (4.0 * SDIRK21_GAMMA)
SDIRK21_B1 = 1.0 - SDIRK21_B2 - SDIRK21_GAMMA
SDIRK21_BHAT2 = (
    SDIRK21_GAMMA
    * (
        -2.0
        + 7.0 * SDIRK21_GAMMA
        - 5.0 * SDIRK21_GAMMA * SDIRK21_GAMMA
        + 4.0 * SDIRK21_GAMMA * SDIRK21_GAMMA * SDIRK21_GAMMA
    )
    / (2.0 * (2.0 * SDIRK21_GAMMA - 1.0))
)
SDIRK21_BHAT3 = (
    -2.0
    * SDIRK21_GAMMA
    * SDIRK21_GAMMA
    * (1.0 - SDIRK21_GAMMA + SDIRK21_GAMMA * SDIRK21_GAMMA)
    / (2.0 * SDIRK21_GAMMA - 1.0)
)
SDIRK21_BHAT1 = 1.0 - SDIRK21_BHAT2 - SDIRK21_BHAT3

SDIRK21_TABLEAU = ButcherTableau(
    c=(0.0, 2.0 * SDIRK21_GAMMA, 1.0),
    a=(
        (),
        (SDIRK21_GAMMA, SDIRK21_GAMMA),
        (SDIRK21_B1, SDIRK21_B2, SDIRK21_GAMMA),
    ),
    b=(SDIRK21_B1, SDIRK21_B2, SDIRK21_GAMMA),
    order=2,
    b_embedded=(SDIRK21_BHAT1, SDIRK21_BHAT2, SDIRK21_BHAT3),
    embedded_order=1,
    short_name="SDIRK21",
    full_name="ESDIRK 2(1)",
)

_STAGE_STENCILS = esdirk_stage_increment_stencils(SDIRK21_TABLEAU, SDIRK21_GAMMA)
_KNOWN3_WEIGHTS = _STAGE_STENCILS.known_shifts[2]
_STAGE_INCREMENT_WEIGHTS_HIGH = _STAGE_STENCILS.high_delta
_STAGE_INCREMENT_WEIGHTS_LOW = _STAGE_STENCILS.low_delta
_STAGE_INCREMENT_WEIGHTS_ERROR = _STAGE_STENCILS.error_delta


@with_scheme_display
@with_adaptive_runtime_methods
@with_implicit_workspace_methods
class SchemeSDIRK21:
    """Adaptive ESDIRK 2(1) with sequential implicit stage solves.

    Algorithm sketch for one trial step of size h:

        1. Compute the explicit first-stage rate k1 = f(t, y).
        2. Form delta1 = gamma h k1.
        3. Solve the second-stage increment:
               delta2 = delta1 + gamma h f(t + 2 gamma h, y + delta2)
        4. Form the known shift for the final stage from delta1 and delta2.
        5. Solve the final-stage increment:
               delta3 = known3 + gamma h f(t + h, y + delta3)
        6. Build the high-order accepted increment and embedded error estimate.
        7. Accept/reject through the adaptive step controller.

    The scheme owns the diagonal stage recipe. The resolvent owns each
    nonlinear one-block solve.
    """

    step_control: SchemeStepControl

    __slots__ = (
        "step_control", "block_allocator", "call_pure", "delta1", "delta2",
        "delta2_block", "delta3", "delta3_block", "derivative", "error",
        "error_delta_call", "high_delta_call", "implicit", "known2_call",
        "known2_block", "known3_call", "known3", "known3_block",
        "redirect_call", "resolvent", "stage1_rate", "trial", "workspace",
    )

    descriptor = SchemeDescriptor("SDIRK21", "ESDIRK 2(1)")
    tableau = SDIRK21_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
        *,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        self.error_delta_call = unbound_scheme_call
        self.high_delta_call = unbound_scheme_call
        self.known2_call = unbound_scheme_call
        self.known3_call = unbound_scheme_call
        self.resolvent = resolvent

        initialise_implicit_support(self, derivative, workbench)
        self.derivative = DerivativeAccelerated(derivative)

        workspace = self.workspace
        self.stage1_rate = workspace.allocate_translation()
        self.delta1, self.delta2, self.delta3, self.trial, self.error, self.known3 = (
            workspace.allocate_translation_buffers(6)
        )
        self.delta2_block = Block([self.delta2])
        self.delta3_block = Block([self.delta3])
        self.known2_block = Block([self.delta1])
        self.known3_block = Block([self.known3])

        initialise_adaptive_runtime(self, regulator)
        self.call_pure = self.call_inline
        refresh_adaptive_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
            refresh_adaptive_call(self)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.5)

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        # Step 2 forms delta1 = gamma h k1.
        self.known2_call = specialist.provide(SchemeStencil((1.0,), scale=SDIRK21_GAMMA))
        # Step 4 forms the known final-stage shift from solved increments.
        self.known3_call = specialist.provide(SchemeStencil(_KNOWN3_WEIGHTS))
        # Step 6 builds high-order and error increments in the stage-increment basis.
        self.high_delta_call = specialist.provide(SchemeStencil(_STAGE_INCREMENT_WEIGHTS_HIGH))
        self.error_delta_call = specialist.provide(SchemeStencil(_STAGE_INCREMENT_WEIGHTS_ERROR))

    def _solve_stage(
        self,
        interval: IntervalLike,
        state: State,
        dt: float,
        stage_shift: float,
        alpha: float,
        known_shift,
        known_block: Block,
        delta_block: Block,
    ):
        known_block[0] = known_shift
        problem = SchemeStageProblem(
            derivative=self.derivative,
            interval=self.workspace.stage_at(interval, dt, stage_shift),
            origin=state,
            rhs=known_block,
            alpha=alpha,
        )
        self.resolvent(problem, delta_block)
        return delta_block[0]

    def call_inline(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        proposal = self.step_control.propose_step(interval)
        if proposal.remaining <= 0.0:
            self.step_control.record_stopped(interval)
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        apply_delta = workspace.apply_delta
        ratio = self.step_control.ratio

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__

        while True:
            # 1. k1 = f(t, y).
            derivative(interval, state, self.stage1_rate)

            # 2. delta1 = gamma h k1.
            delta1 = scale(dt * SDIRK21_GAMMA, self.stage1_rate, self.delta1)

            try:
                # 3. Solve the second diagonal implicit stage.
                delta2 = self._solve_stage(
                    interval, state, dt, 2.0 * SDIRK21_GAMMA * dt,
                    dt * SDIRK21_GAMMA, delta1, self.known2_block, self.delta2_block,
                )
                # 4. Known shift for the final stage.
                known3 = combine2(
                    _KNOWN3_WEIGHTS[0],
                    delta1,
                    _KNOWN3_WEIGHTS[1],
                    delta2,
                    self.known3,
                )
                # 5. Solve the final diagonal implicit stage.
                delta3 = self._solve_stage(
                    interval, state, dt, dt, dt * SDIRK21_GAMMA,
                    known3, self.known3_block, self.delta3_block,
                )
            except ResolventError:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, scheme_name)
                continue

            # 6. Build high-order increment and embedded error estimate.
            delta_high = combine3(
                _STAGE_INCREMENT_WEIGHTS_HIGH[0], delta1,
                _STAGE_INCREMENT_WEIGHTS_HIGH[1], delta2,
                _STAGE_INCREMENT_WEIGHTS_HIGH[2], delta3,
                self.trial,
            )
            error = combine3(
                _STAGE_INCREMENT_WEIGHTS_ERROR[0], delta1,
                _STAGE_INCREMENT_WEIGHTS_ERROR[1], delta2,
                _STAGE_INCREMENT_WEIGHTS_ERROR[2], delta3,
                self.error,
            )
            error_ratio = ratio(error.norm(), delta_high.norm())
            if error_ratio <= 1.0:
                break
            rejection_count += 1
            dt = self.step_control.rejected_step(dt, error_ratio, remaining, scheme_name)

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = self.step_control.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        interval.step = next_dt
        apply_delta(delta_high, state)
        report = self.step_control.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return report.accepted_dt

    def call_specialized(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        del executor
        proposal = self.step_control.propose_step(interval)
        if proposal.remaining <= 0.0:
            self.step_control.record_stopped(interval)
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        apply_delta = workspace.apply_delta
        ratio = self.step_control.ratio
        known2_call = self.known2_call
        known3_call = self.known3_call
        high_delta_call = self.high_delta_call
        error_delta_call = self.error_delta_call

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__

        while True:
            # 1. k1 = f(t, y).
            derivative(interval, state, self.stage1_rate)
            # 2. delta1 = gamma h k1.
            delta1 = known2_call(dt, self.stage1_rate, self.delta1)
            try:
                # 3. Solve the second diagonal implicit stage.
                delta2 = self._solve_stage(
                    interval, state, dt, 2.0 * SDIRK21_GAMMA * dt,
                    dt * SDIRK21_GAMMA, delta1, self.known2_block, self.delta2_block,
                )
                # 4. Known shift for the final stage.
                known3 = known3_call(1.0, delta1, delta2, self.known3)
                # 5. Solve the final diagonal implicit stage.
                delta3 = self._solve_stage(
                    interval, state, dt, dt, dt * SDIRK21_GAMMA,
                    known3, self.known3_block, self.delta3_block,
                )
            except ResolventError:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, scheme_name)
                continue

            # 6. Build high-order increment and embedded error estimate.
            delta_high = high_delta_call(1.0, delta1, delta2, delta3, self.trial)
            error = error_delta_call(1.0, delta1, delta2, delta3, self.error)
            error_ratio = ratio(error.norm(), delta_high.norm())
            if error_ratio <= 1.0:
                break
            rejection_count += 1
            dt = self.step_control.rejected_step(dt, error_ratio, remaining, scheme_name)

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = self.step_control.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        interval.step = next_dt
        apply_delta(delta_high, state)
        report = self.step_control.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return report.accepted_dt


__all__ = ["SDIRK21_GAMMA", "SDIRK21_TABLEAU", "SchemeSDIRK21"]
