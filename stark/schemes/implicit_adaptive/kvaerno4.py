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


KVAERNO4_GAMMA = 0.5728160625

def _poly(*coefficients: float) -> float:
    value = 0.0
    for coefficient in coefficients:
        value = value * KVAERNO4_GAMMA + coefficient
    return value

KVAERNO4_A21 = KVAERNO4_GAMMA
KVAERNO4_A31 = (_poly(144.0, -180.0, 81.0, -15.0, 1.0) * KVAERNO4_GAMMA / (_poly(12.0, -6.0, 1.0) ** 2))
KVAERNO4_A32 = (_poly(-36.0, 39.0, -15.0, 2.0) * KVAERNO4_GAMMA / (_poly(12.0, -6.0, 1.0) ** 2))
KVAERNO4_A41 = _poly(-144.0, 396.0, -330.0, 117.0, -18.0, 1.0) / (12.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(12.0, -9.0, 2.0))
KVAERNO4_A42 = _poly(72.0, -126.0, 69.0, -15.0, 1.0) / (12.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(3.0, -1.0))
KVAERNO4_A43 = (_poly(-6.0, 6.0, -1.0) * (_poly(12.0, -6.0, 1.0) ** 2)) / (12.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(12.0, -9.0, 2.0) * _poly(3.0, -1.0))
KVAERNO4_A51 = _poly(288.0, -312.0, 120.0, -18.0, 1.0) / (48.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(12.0, -9.0, 2.0))
KVAERNO4_A52 = _poly(24.0, -12.0, 1.0) / (48.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(3.0, -1.0))
KVAERNO4_A53 = -(_poly(12.0, -6.0, 1.0) ** 3) / (48.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(3.0, -1.0) * _poly(12.0, -9.0, 2.0) * _poly(6.0, -6.0, 1.0))
KVAERNO4_A54 = _poly(-24.0, 36.0, -12.0, 1.0) / _poly(24.0, -24.0, 4.0)
KVAERNO4_C2 = KVAERNO4_GAMMA + KVAERNO4_A21
KVAERNO4_C3 = KVAERNO4_GAMMA + KVAERNO4_A31 + KVAERNO4_A32

KVAERNO4_TABLEAU = ButcherTableau(
    c=(0.0, KVAERNO4_C2, KVAERNO4_C3, 1.0, 1.0),
    a=(
        (),
        (KVAERNO4_A21, KVAERNO4_GAMMA),
        (KVAERNO4_A31, KVAERNO4_A32, KVAERNO4_GAMMA),
        (KVAERNO4_A41, KVAERNO4_A42, KVAERNO4_A43, KVAERNO4_GAMMA),
        (KVAERNO4_A51, KVAERNO4_A52, KVAERNO4_A53, KVAERNO4_A54, KVAERNO4_GAMMA),
    ),
    b=(KVAERNO4_A51, KVAERNO4_A52, KVAERNO4_A53, KVAERNO4_A54, KVAERNO4_GAMMA),
    order=4,
    b_embedded=(KVAERNO4_A41, KVAERNO4_A42, KVAERNO4_A43, KVAERNO4_GAMMA, 0.0),
    embedded_order=3,
    short_name="Kvaerno4",
    full_name="Kvaerno 4(3)",
)

_STAGE_STENCILS = esdirk_stage_increment_stencils(KVAERNO4_TABLEAU, KVAERNO4_GAMMA)
_KNOWN3_WEIGHTS = _STAGE_STENCILS.known_shifts[2]
_KNOWN4_WEIGHTS = _STAGE_STENCILS.known_shifts[3]
_KNOWN5_WEIGHTS = _STAGE_STENCILS.known_shifts[4]
_STAGE_INCREMENT_WEIGHTS_HIGH = _STAGE_STENCILS.high_delta
_STAGE_INCREMENT_WEIGHTS_LOW = _STAGE_STENCILS.low_delta
_STAGE_INCREMENT_WEIGHTS_ERROR = _STAGE_STENCILS.error_delta


@with_scheme_display
@with_adaptive_runtime_methods
@with_implicit_workspace_methods
class SchemeKvaerno4:
    """Kvaerno's adaptive ESDIRK 4(3) method.

    Algorithm sketch for one trial step of size h:

        1. Compute the explicit first-stage rate k1 = f(t, y).
        2. Form delta1 = gamma h k1.
        3. Sequentially solve the four diagonal implicit stage increments.
        4. Build the high-order increment and embedded error estimate.
        5. Accept/reject through the adaptive controller.
    """

    step_control: SchemeStepControl

    __slots__ = (
        "step_control", "block_allocator", "call_monitorable", "delta1", "delta2", "delta2_block",
        "delta3", "delta3_block", "delta4", "delta4_block", "delta5", "delta5_block",
        "derivative", "error", "error_delta_call", "high_delta_call", "implicit", "known2_call",
        "known2_block", "known3_call", "known3", "known3_block", "known4_call", "known4",
        "known4_block", "known5_call", "known5", "known5_block", "redirect_call", "resolvent",
        "stage1_rate", "trial", "workspace",
    )

    descriptor = SchemeDescriptor("Kvaerno4", "Kvaerno 4(3)")
    tableau = KVAERNO4_TABLEAU

    def __init__(self, derivative: Derivative, allocator: Allocator, resolvent: Resolvent, adaptivity: ExecutorAdaptivity | None = None, *, specialist: SchemeSpecialist | None = None) -> None:
        self.error_delta_call = unbound_scheme_call
        self.high_delta_call = unbound_scheme_call
        self.known2_call = unbound_scheme_call
        self.known3_call = unbound_scheme_call
        self.known4_call = unbound_scheme_call
        self.known5_call = unbound_scheme_call
        self.resolvent = resolvent
        initialise_implicit_support(self, derivative, allocator)
        self.derivative = SchemeDerivative(derivative)
        workspace = self.workspace
        self.stage1_rate = workspace.allocate_translation()
        (self.delta1, self.delta2, self.delta3, self.delta4, self.delta5, self.known3, self.known4, self.known5, self.trial, self.error) = workspace.allocate_translation_buffers(10)
        self.delta2_block = Block([self.delta2])
        self.delta3_block = Block([self.delta3])
        self.delta4_block = Block([self.delta4])
        self.delta5_block = Block([self.delta5])
        self.known2_block = Block([self.delta1])
        self.known3_block = Block([self.known3])
        self.known4_block = Block([self.known4])
        self.known5_block = Block([self.known5])
        initialise_adaptive_runtime(self, adaptivity)
        self.call_monitorable = self.call_inline
        refresh_adaptive_call(self)
        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_monitorable = self.call_specialized
            refresh_adaptive_call(self)

    @staticmethod
    def default_adaptivity() -> ExecutorAdaptivity:
        return ExecutorAdaptivity(error_exponent=1.0 / 4.0)

    def __call__(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        self.known2_call = specialist.provide(SchemeStencil((1.0,), scale=KVAERNO4_GAMMA))
        self.known3_call = specialist.provide(SchemeStencil(_KNOWN3_WEIGHTS))
        self.known4_call = specialist.provide(SchemeStencil(_KNOWN4_WEIGHTS))
        self.known5_call = specialist.provide(SchemeStencil(_KNOWN5_WEIGHTS))
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
        combine5 = workspace.combine5
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
            delta1 = scale(dt * KVAERNO4_GAMMA, self.stage1_rate, self.delta1)
            try:
                delta2 = self._solve_stage(interval, state, dt, KVAERNO4_C2 * dt, dt * KVAERNO4_GAMMA, delta1, self.known2_block, self.delta2_block)
                known3 = combine2(
                    _KNOWN3_WEIGHTS[0],
                    delta1,
                    _KNOWN3_WEIGHTS[1],
                    delta2,
                    self.known3,
                )
                delta3 = self._solve_stage(interval, state, dt, KVAERNO4_C3 * dt, dt * KVAERNO4_GAMMA, known3, self.known3_block, self.delta3_block)
                known4 = combine3(
                    _KNOWN4_WEIGHTS[0],
                    delta1,
                    _KNOWN4_WEIGHTS[1],
                    delta2,
                    _KNOWN4_WEIGHTS[2],
                    delta3,
                    self.known4,
                )
                delta4 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO4_GAMMA, known4, self.known4_block, self.delta4_block)
                known5 = combine4(
                    _KNOWN5_WEIGHTS[0],
                    delta1,
                    _KNOWN5_WEIGHTS[1],
                    delta2,
                    _KNOWN5_WEIGHTS[2],
                    delta3,
                    _KNOWN5_WEIGHTS[3],
                    delta4,
                    self.known5,
                )
                delta5 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO4_GAMMA, known5, self.known5_block, self.delta5_block)
            except StarkErrorRecoverable:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, scheme_name)
                continue
            delta_high = combine5(_STAGE_INCREMENT_WEIGHTS_HIGH[0], delta1, _STAGE_INCREMENT_WEIGHTS_HIGH[1], delta2, _STAGE_INCREMENT_WEIGHTS_HIGH[2], delta3, _STAGE_INCREMENT_WEIGHTS_HIGH[3], delta4, _STAGE_INCREMENT_WEIGHTS_HIGH[4], delta5, self.trial)
            error = combine5(_STAGE_INCREMENT_WEIGHTS_ERROR[0], delta1, _STAGE_INCREMENT_WEIGHTS_ERROR[1], delta2, _STAGE_INCREMENT_WEIGHTS_ERROR[2], delta3, _STAGE_INCREMENT_WEIGHTS_ERROR[3], delta4, _STAGE_INCREMENT_WEIGHTS_ERROR[4], delta5, self.error)
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
                delta2 = self._solve_stage(interval, state, dt, KVAERNO4_C2 * dt, dt * KVAERNO4_GAMMA, delta1, self.known2_block, self.delta2_block)
                known3 = self.known3_call(1.0, delta1, delta2, self.known3)
                delta3 = self._solve_stage(interval, state, dt, KVAERNO4_C3 * dt, dt * KVAERNO4_GAMMA, known3, self.known3_block, self.delta3_block)
                known4 = self.known4_call(1.0, delta1, delta2, delta3, self.known4)
                delta4 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO4_GAMMA, known4, self.known4_block, self.delta4_block)
                known5 = self.known5_call(1.0, delta1, delta2, delta3, delta4, self.known5)
                delta5 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO4_GAMMA, known5, self.known5_block, self.delta5_block)
            except StarkErrorRecoverable:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, scheme_name)
                continue
            delta_high = self.high_delta_call(1.0, delta1, delta2, delta3, delta4, delta5, self.trial)
            error = self.error_delta_call(1.0, delta1, delta2, delta3, delta4, delta5, self.error)
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


__all__ = ["KVAERNO4_GAMMA", "KVAERNO4_TABLEAU", "SchemeKvaerno4"]
