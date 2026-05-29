from __future__ import annotations

from stark.schemes.support.derivative import SchemeDerivative
from stark.block import Block
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Allocator
from stark.schemes.support.executor import SchemeExecutor
from stark.executor.adaptivity import ExecutorAdaptivity
from stark.contracts.errors import StarkErrorRecoverable
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


KVAERNO3_GAMMA = 0.43586652150
KVAERNO3_A21 = KVAERNO3_GAMMA
KVAERNO3_A31 = (-4.0 * KVAERNO3_GAMMA * KVAERNO3_GAMMA + 6.0 * KVAERNO3_GAMMA - 1.0) / (4.0 * KVAERNO3_GAMMA)
KVAERNO3_A32 = (-2.0 * KVAERNO3_GAMMA + 1.0) / (4.0 * KVAERNO3_GAMMA)
KVAERNO3_A41 = (6.0 * KVAERNO3_GAMMA - 1.0) / (12.0 * KVAERNO3_GAMMA)
KVAERNO3_A42 = -1.0 / ((24.0 * KVAERNO3_GAMMA - 12.0) * KVAERNO3_GAMMA)
KVAERNO3_A43 = (-6.0 * KVAERNO3_GAMMA * KVAERNO3_GAMMA + 6.0 * KVAERNO3_GAMMA - 1.0) / (6.0 * KVAERNO3_GAMMA - 3.0)

KVAERNO3_TABLEAU = ButcherTableau(
    c=(0.0, 2.0 * KVAERNO3_GAMMA, 1.0, 1.0),
    a=(
        (),
        (KVAERNO3_A21, KVAERNO3_GAMMA),
        (KVAERNO3_A31, KVAERNO3_A32, KVAERNO3_GAMMA),
        (KVAERNO3_A41, KVAERNO3_A42, KVAERNO3_A43, KVAERNO3_GAMMA),
    ),
    b=(KVAERNO3_A41, KVAERNO3_A42, KVAERNO3_A43, KVAERNO3_GAMMA),
    order=3,
    b_embedded=(KVAERNO3_A31, KVAERNO3_A32, KVAERNO3_GAMMA, 0.0),
    embedded_order=2,
    short_name="Kvaerno3",
    full_name="Kvaerno 3(2)",
)

_STAGE_STENCILS = esdirk_stage_increment_stencils(KVAERNO3_TABLEAU, KVAERNO3_GAMMA)
_KNOWN3_WEIGHTS = _STAGE_STENCILS.known_shifts[2]
_KNOWN4_WEIGHTS = _STAGE_STENCILS.known_shifts[3]
_STAGE_INCREMENT_WEIGHTS_HIGH = _STAGE_STENCILS.high_delta
_STAGE_INCREMENT_WEIGHTS_LOW = _STAGE_STENCILS.low_delta
_STAGE_INCREMENT_WEIGHTS_ERROR = _STAGE_STENCILS.error_delta


@with_scheme_display
@with_adaptive_runtime_methods
@with_implicit_workspace_methods
class SchemeKvaerno3:
    """Kvaerno's adaptive ESDIRK 3(2) method.

    Algorithm sketch for one trial step of size h:

        1. Compute the explicit first-stage rate k1 = f(t, y).
        2. Form delta1 = gamma h k1.
        3. Solve the second diagonal implicit stage.
        4. Form the third-stage known shift and solve the third stage.
        5. Form the final-stage known shift and solve the final stage.
        6. Build the high-order increment and embedded error estimate.
        7. Accept/reject through the adaptive controller.
    """

    step_control: SchemeStepControl

    __slots__ = (
        "step_control", "block_allocator", "call_pure", "delta1", "delta2", "delta2_block",
        "delta3", "delta3_block", "delta4", "delta4_block", "derivative", "error",
        "error_delta_call", "high_delta_call", "implicit", "known2_call", "known2_block",
        "known3_call", "known3", "known3_block", "known4_call", "known4", "known4_block",
        "redirect_call", "resolvent", "stage1_rate", "trial", "workspace",
    )

    descriptor = SchemeDescriptor("Kvaerno3", "Kvaerno 3(2)")
    tableau = KVAERNO3_TABLEAU

    def __init__(self, derivative: Derivative, allocator: Allocator, resolvent: Resolvent, adaptivity: ExecutorAdaptivity | None = None, *, specialist: SchemeSpecialist | None = None) -> None:
        self.error_delta_call = unbound_scheme_call
        self.high_delta_call = unbound_scheme_call
        self.known2_call = unbound_scheme_call
        self.known3_call = unbound_scheme_call
        self.known4_call = unbound_scheme_call
        self.resolvent = resolvent
        initialise_implicit_support(self, derivative, allocator)
        self.derivative = SchemeDerivative(derivative)
        workspace = self.workspace
        self.stage1_rate = workspace.allocate_translation()
        self.delta1, self.delta2, self.delta3, self.delta4, self.known3, self.known4, self.trial, self.error = workspace.allocate_translation_buffers(8)
        self.delta2_block = Block([self.delta2])
        self.delta3_block = Block([self.delta3])
        self.delta4_block = Block([self.delta4])
        self.known2_block = Block([self.delta1])
        self.known3_block = Block([self.known3])
        self.known4_block = Block([self.known4])
        initialise_adaptive_runtime(self, adaptivity)
        self.call_pure = self.call_inline
        refresh_adaptive_call(self)
        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
            refresh_adaptive_call(self)

    @staticmethod
    def default_adaptivity() -> ExecutorAdaptivity:
        return ExecutorAdaptivity(error_exponent=1.0 / 3.0)

    def __call__(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        self.known2_call = specialist.provide(SchemeStencil((1.0,), scale=KVAERNO3_GAMMA))
        self.known3_call = specialist.provide(SchemeStencil(_KNOWN3_WEIGHTS))
        self.known4_call = specialist.provide(SchemeStencil(_KNOWN4_WEIGHTS))
        self.high_delta_call = specialist.provide(SchemeStencil(_STAGE_INCREMENT_WEIGHTS_HIGH))
        self.error_delta_call = specialist.provide(SchemeStencil(_STAGE_INCREMENT_WEIGHTS_ERROR))

    def _solve_stage(self, interval: IntervalLike, state: State, dt: float, stage_shift: float, alpha: float, known_shift, known_block: Block, delta_block: Block):
        known_block[0] = known_shift
        problem = SchemeStageProblem(derivative=self.derivative, interval=self.workspace.stage_at(interval, dt, stage_shift), origin=state, rhs=known_block, alpha=alpha)
        self.resolvent(problem, delta_block)
        return delta_block[0]

    def call_inline(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
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
        combine4 = workspace.combine4
        apply_delta = workspace.apply_delta
        ratio = self.step_control.ratio
        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__
        derivative(interval, state, self.stage1_rate)
        while True:
            # 2. delta1 = gamma h k1.
            delta1 = scale(dt * KVAERNO3_GAMMA, self.stage1_rate, self.delta1)
            try:
                # 3. Solve the second diagonal implicit stage.
                delta2 = self._solve_stage(interval, state, dt, 2.0 * KVAERNO3_GAMMA * dt, dt * KVAERNO3_GAMMA, delta1, self.known2_block, self.delta2_block)
                # 4. Known shift and solve for the third stage.
                known3 = combine2(
                    _KNOWN3_WEIGHTS[0],
                    delta1,
                    _KNOWN3_WEIGHTS[1],
                    delta2,
                    self.known3,
                )
                delta3 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO3_GAMMA, known3, self.known3_block, self.delta3_block)
                # 5. Known shift and solve for the final stage.
                known4 = combine3(
                    _KNOWN4_WEIGHTS[0],
                    delta1,
                    _KNOWN4_WEIGHTS[1],
                    delta2,
                    _KNOWN4_WEIGHTS[2],
                    delta3,
                    self.known4,
                )
                delta4 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO3_GAMMA, known4, self.known4_block, self.delta4_block)
            except StarkErrorRecoverable:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, scheme_name)
                continue
            # 6. High-order increment and embedded error.
            delta_high = combine4(_STAGE_INCREMENT_WEIGHTS_HIGH[0], delta1, _STAGE_INCREMENT_WEIGHTS_HIGH[1], delta2, _STAGE_INCREMENT_WEIGHTS_HIGH[2], delta3, _STAGE_INCREMENT_WEIGHTS_HIGH[3], delta4, self.trial)
            error = combine4(_STAGE_INCREMENT_WEIGHTS_ERROR[0], delta1, _STAGE_INCREMENT_WEIGHTS_ERROR[1], delta2, _STAGE_INCREMENT_WEIGHTS_ERROR[2], delta3, _STAGE_INCREMENT_WEIGHTS_ERROR[3], delta4, self.error)
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
        report = self.step_control.record_accepted(accepted_dt=accepted_dt, t_start=t_start, proposed_dt=proposed_dt, next_dt=next_dt, error_ratio=error_ratio, rejection_count=rejection_count)
        return report.accepted_dt

    def call_specialized(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        del executor
        proposal = self.step_control.propose_step(interval)
        if proposal.remaining <= 0.0:
            self.step_control.record_stopped(interval)
            return 0.0
        workspace = self.workspace
        derivative = self.derivative
        apply_delta = workspace.apply_delta
        ratio = self.step_control.ratio
        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__
        derivative(interval, state, self.stage1_rate)
        while True:
            delta1 = self.known2_call(dt, self.stage1_rate, self.delta1)
            try:
                delta2 = self._solve_stage(interval, state, dt, 2.0 * KVAERNO3_GAMMA * dt, dt * KVAERNO3_GAMMA, delta1, self.known2_block, self.delta2_block)
                known3 = self.known3_call(1.0, delta1, delta2, self.known3)
                delta3 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO3_GAMMA, known3, self.known3_block, self.delta3_block)
                known4 = self.known4_call(1.0, delta1, delta2, delta3, self.known4)
                delta4 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO3_GAMMA, known4, self.known4_block, self.delta4_block)
            except StarkErrorRecoverable:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, scheme_name)
                continue
            delta_high = self.high_delta_call(1.0, delta1, delta2, delta3, delta4, self.trial)
            error = self.error_delta_call(1.0, delta1, delta2, delta3, delta4, self.error)
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
        report = self.step_control.record_accepted(accepted_dt=accepted_dt, t_start=t_start, proposed_dt=proposed_dt, next_dt=next_dt, error_ratio=error_ratio, rejection_count=rejection_count)
        return report.accepted_dt


__all__ = ["KVAERNO3_GAMMA", "KVAERNO3_TABLEAU", "SchemeKvaerno3"]
