from __future__ import annotations

from stark.methods.schemes.configuration import SchemeConfiguration, SchemeConfigurationDefault
from stark.core.block import Block
from stark.core.contracts import DerivativeLike, IntervalLike, Resolvent, State, Allocator
from stark.core.contracts.errors import StarkErrorRecoverable
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_adaptive_step_monitoring
from stark.methods.schemes.execution.step_control import SchemeStepControl
from stark.methods.schemes.execution.unbound import unbound_scheme_call
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.implicit._support import (
    initialise_implicit_support,
    implicit_display_resolvent_problem,
    implicit_snapshot_state,
)
from stark.methods.schemes.specialization.specialist import SchemeSpecialist
from stark.methods.schemes.requests.resolvent import SchemeResolventRequest
from stark.methods.schemes.specialization.stencil import (
    SchemeStencil,
    esdirk_stage_increment_stencils,
)
from stark.methods.schemes.method.tableau import ButcherTableau


KVAERNO5_GAMMA = 0.26
KVAERNO5_A21 = KVAERNO5_GAMMA
KVAERNO5_A31 = 0.13
KVAERNO5_A32 = 0.84033320996790809
KVAERNO5_A41 = 0.22371961478320505
KVAERNO5_A42 = 0.47675532319799699
KVAERNO5_A43 = -0.06470895363112615
KVAERNO5_A51 = 0.16648564323248321
KVAERNO5_A52 = 0.10450018841591720
KVAERNO5_A53 = 0.03631482272098715
KVAERNO5_A54 = -0.13090704451073998
KVAERNO5_A61 = 0.13855640231268224
KVAERNO5_A62 = 0.0
KVAERNO5_A63 = -0.04245337201752043
KVAERNO5_A64 = 0.02446657898003141
KVAERNO5_A65 = 0.61943039072480676
KVAERNO5_A71 = 0.13659751177640291
KVAERNO5_A72 = 0.0
KVAERNO5_A73 = -0.05496908796538376
KVAERNO5_A74 = -0.04118626728321046
KVAERNO5_A75 = 0.62993304899016403
KVAERNO5_A76 = 0.06962479448202728

KVAERNO5_C2 = 0.52
KVAERNO5_C3 = 1.230333209967908
KVAERNO5_C4 = 0.8957659843500759
KVAERNO5_C5 = 0.43639360985864756

KVAERNO5_TABLEAU = ButcherTableau(
    c=(0.0, KVAERNO5_C2, KVAERNO5_C3, KVAERNO5_C4, KVAERNO5_C5, 1.0, 1.0),
    a=(
        (),
        (KVAERNO5_A21, KVAERNO5_GAMMA),
        (KVAERNO5_A31, KVAERNO5_A32, KVAERNO5_GAMMA),
        (KVAERNO5_A41, KVAERNO5_A42, KVAERNO5_A43, KVAERNO5_GAMMA),
        (KVAERNO5_A51, KVAERNO5_A52, KVAERNO5_A53, KVAERNO5_A54, KVAERNO5_GAMMA),
        (KVAERNO5_A61, KVAERNO5_A62, KVAERNO5_A63, KVAERNO5_A64, KVAERNO5_A65, KVAERNO5_GAMMA),
        (KVAERNO5_A71, KVAERNO5_A72, KVAERNO5_A73, KVAERNO5_A74, KVAERNO5_A75, KVAERNO5_A76, KVAERNO5_GAMMA),
    ),
    b=(KVAERNO5_A71, KVAERNO5_A72, KVAERNO5_A73, KVAERNO5_A74, KVAERNO5_A75, KVAERNO5_A76, KVAERNO5_GAMMA),
    order=5,
    b_embedded=(KVAERNO5_A61, KVAERNO5_A62, KVAERNO5_A63, KVAERNO5_A64, KVAERNO5_A65, KVAERNO5_GAMMA, 0.0),
    embedded_order=4,
    short_name="Kvaerno5",
    full_name="Kvaerno 5(4)",
)

_STAGE_STENCILS = esdirk_stage_increment_stencils(KVAERNO5_TABLEAU, KVAERNO5_GAMMA)
_KNOWN3_WEIGHTS = _STAGE_STENCILS.known_shifts[2]
_KNOWN4_WEIGHTS = _STAGE_STENCILS.known_shifts[3]
_KNOWN5_WEIGHTS = _STAGE_STENCILS.known_shifts[4]
_KNOWN6_WEIGHTS = _STAGE_STENCILS.known_shifts[5]
_KNOWN7_WEIGHTS = _STAGE_STENCILS.known_shifts[6]
_STAGE_INCREMENT_WEIGHTS_HIGH = _STAGE_STENCILS.high_delta
_STAGE_INCREMENT_WEIGHTS_LOW = _STAGE_STENCILS.low_delta
_STAGE_INCREMENT_WEIGHTS_ERROR = _STAGE_STENCILS.error_delta


@with_scheme_display
@with_adaptive_step_monitoring
class SchemeKvaerno5:
    """Kvaerno's adaptive ESDIRK 5(4) method.

    This is the seven-stage, stiffly accurate Kvaerno method commonly used as
    the higher-order sibling of the built-in Kvaerno3 and Kvaerno4 schemes.
    Each trial step computes one explicit first-stage rate, solves six
    diagonal implicit stage increments, then compares the fifth-order advance
    with the embedded fourth-order estimate for adaptive step control.
    """

    step_control: SchemeStepControl

    __slots__ = (
        "monitor",
        "call_body",
        "step_control", "block_allocator", "call_step", "delta1", "delta2", "delta2_block",
        "delta3", "delta3_block", "delta4", "delta4_block", "delta5", "delta5_block",
        "delta6", "delta6_block", "delta7", "delta7_block", "derivative", "error",
        "error_delta_call", "high_delta_call", "implicit", "known2_call", "known2_block",
        "known3_call", "known3", "known3_block", "known4_call", "known4", "known4_block",
        "known5_call", "known5", "known5_block", "known6_call", "known6", "known6_block",
        "known7_call", "known7", "known7_block", "redirect_call", "resolvent",
        "stage1_rate", "trial", "workspace",
    )

    descriptor = SchemeDescriptor("Kvaerno5", "Kvaerno 5(4)")
    display_resolvent_problem = classmethod(implicit_display_resolvent_problem)
    snapshot_state = implicit_snapshot_state
    tableau = KVAERNO5_TABLEAU

    def __init__(self, derivative: DerivativeLike, allocator: Allocator, resolvent: Resolvent, *, configuration: SchemeConfiguration | None = None, specialist: SchemeSpecialist | None = None, monitor: SchemeMonitor | None = None) -> None:
        self.error_delta_call = unbound_scheme_call
        self.high_delta_call = unbound_scheme_call
        self.known2_call = unbound_scheme_call
        self.known3_call = unbound_scheme_call
        self.known4_call = unbound_scheme_call
        self.known5_call = unbound_scheme_call
        self.known6_call = unbound_scheme_call
        self.known7_call = unbound_scheme_call
        self.resolvent = resolvent
        initialise_implicit_support(self, derivative, allocator)
        self.derivative = derivative
        workspace = self.workspace
        (
            self.delta1, self.delta2, self.delta3, self.delta4, self.delta5, self.delta6,
            self.delta7, self.known3, self.known4, self.known5, self.known6, self.known7,
            self.trial, self.error,
        ) = workspace.allocate_translation_buffers(14)
        self.stage1_rate = workspace.allocate_translation()
        self.delta2_block = Block([self.delta2])
        self.delta3_block = Block([self.delta3])
        self.delta4_block = Block([self.delta4])
        self.delta5_block = Block([self.delta5])
        self.delta6_block = Block([self.delta6])
        self.delta7_block = Block([self.delta7])
        self.known2_block = Block([self.delta1])
        self.known3_block = Block([self.known3])
        self.known4_block = Block([self.known4])
        self.known5_block = Block([self.known5])
        self.known6_block = Block([self.known6])
        self.known7_block = Block([self.known7])

        self.step_control = SchemeStepControl(configuration if configuration is not None else SchemeConfigurationDefault())
        self.call_body = self.call_inline
        self.monitor = monitor
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step
        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_body = self.call_specialized
            if monitor is None:
                self.call_step = self.call_body
                self.redirect_call = self.call_step

    @staticmethod
    def default_adaptivity() -> float:
        return 1.0 / 5.0

    def __call__(self, interval: IntervalLike, state: State) -> float:
        return self.redirect_call(interval, state)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        self.known2_call = specialist.provide(SchemeStencil((1.0,), scale=KVAERNO5_GAMMA))
        self.known3_call = specialist.provide(SchemeStencil(_KNOWN3_WEIGHTS))
        self.known4_call = specialist.provide(SchemeStencil(_KNOWN4_WEIGHTS))
        self.known5_call = specialist.provide(SchemeStencil(_KNOWN5_WEIGHTS))
        self.known6_call = specialist.provide(SchemeStencil(_KNOWN6_WEIGHTS))
        self.known7_call = specialist.provide(SchemeStencil(_KNOWN7_WEIGHTS))
        self.high_delta_call = specialist.provide(SchemeStencil(_STAGE_INCREMENT_WEIGHTS_HIGH))
        self.error_delta_call = specialist.provide(SchemeStencil(_STAGE_INCREMENT_WEIGHTS_ERROR))

    def _solve_stage(self, interval: IntervalLike, state: State, dt: float, stage_shift: float, alpha: float, known_shift, known_block: Block, delta_block: Block):
        known_block[0] = known_shift
        problem = SchemeResolventRequest(derivative=self.derivative, interval=self.workspace.interval_at(interval, dt, stage_shift), origin=state, rhs=known_block, alpha=alpha)
        self.resolvent(problem, delta_block)
        return delta_block[0]

    def call_inline(self, interval: IntervalLike, state: State) -> float:
        step_control = self.step_control
        proposal = step_control.propose_step(interval)
        if proposal.remaining <= 0.0:
            step_control.record_stopped(interval)
            return 0.0
        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        combine4 = workspace.combine4
        combine5 = workspace.combine5
        combine6 = workspace.combine6
        combine7 = workspace.combine7
        apply_delta = workspace.apply_delta
        ratio = step_control.ratio
        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__
        derivative(interval, state, self.stage1_rate)
        while True:
            delta1 = scale(dt * KVAERNO5_GAMMA, self.stage1_rate, self.delta1)
            try:
                delta2 = self._solve_stage(interval, state, dt, KVAERNO5_C2 * dt, dt * KVAERNO5_GAMMA, delta1, self.known2_block, self.delta2_block)
                known3 = combine2(_KNOWN3_WEIGHTS[0], delta1, _KNOWN3_WEIGHTS[1], delta2, self.known3)
                delta3 = self._solve_stage(interval, state, dt, KVAERNO5_C3 * dt, dt * KVAERNO5_GAMMA, known3, self.known3_block, self.delta3_block)
                known4 = combine3(_KNOWN4_WEIGHTS[0], delta1, _KNOWN4_WEIGHTS[1], delta2, _KNOWN4_WEIGHTS[2], delta3, self.known4)
                delta4 = self._solve_stage(interval, state, dt, KVAERNO5_C4 * dt, dt * KVAERNO5_GAMMA, known4, self.known4_block, self.delta4_block)
                known5 = combine4(_KNOWN5_WEIGHTS[0], delta1, _KNOWN5_WEIGHTS[1], delta2, _KNOWN5_WEIGHTS[2], delta3, _KNOWN5_WEIGHTS[3], delta4, self.known5)
                delta5 = self._solve_stage(interval, state, dt, KVAERNO5_C5 * dt, dt * KVAERNO5_GAMMA, known5, self.known5_block, self.delta5_block)
                known6 = combine5(_KNOWN6_WEIGHTS[0], delta1, _KNOWN6_WEIGHTS[1], delta2, _KNOWN6_WEIGHTS[2], delta3, _KNOWN6_WEIGHTS[3], delta4, _KNOWN6_WEIGHTS[4], delta5, self.known6)
                delta6 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO5_GAMMA, known6, self.known6_block, self.delta6_block)
                known7 = combine6(_KNOWN7_WEIGHTS[0], delta1, _KNOWN7_WEIGHTS[1], delta2, _KNOWN7_WEIGHTS[2], delta3, _KNOWN7_WEIGHTS[3], delta4, _KNOWN7_WEIGHTS[4], delta5, _KNOWN7_WEIGHTS[5], delta6, self.known7)
                delta7 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO5_GAMMA, known7, self.known7_block, self.delta7_block)
            except StarkErrorRecoverable:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, scheme_name)
                continue
            delta_high = combine7(_STAGE_INCREMENT_WEIGHTS_HIGH[0], delta1, _STAGE_INCREMENT_WEIGHTS_HIGH[1], delta2, _STAGE_INCREMENT_WEIGHTS_HIGH[2], delta3, _STAGE_INCREMENT_WEIGHTS_HIGH[3], delta4, _STAGE_INCREMENT_WEIGHTS_HIGH[4], delta5, _STAGE_INCREMENT_WEIGHTS_HIGH[5], delta6, _STAGE_INCREMENT_WEIGHTS_HIGH[6], delta7, self.trial)
            error = combine7(_STAGE_INCREMENT_WEIGHTS_ERROR[0], delta1, _STAGE_INCREMENT_WEIGHTS_ERROR[1], delta2, _STAGE_INCREMENT_WEIGHTS_ERROR[2], delta3, _STAGE_INCREMENT_WEIGHTS_ERROR[3], delta4, _STAGE_INCREMENT_WEIGHTS_ERROR[4], delta5, _STAGE_INCREMENT_WEIGHTS_ERROR[5], delta6, _STAGE_INCREMENT_WEIGHTS_ERROR[6], delta7, self.error)
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

    def call_specialized(self, interval: IntervalLike, state: State) -> float:
        step_control = self.step_control
        proposal = step_control.propose_step(interval)
        if proposal.remaining <= 0.0:
            step_control.record_stopped(interval)
            return 0.0
        workspace = self.workspace
        derivative = self.derivative
        apply_delta = workspace.apply_delta
        ratio = step_control.ratio
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
                delta2 = self._solve_stage(interval, state, dt, KVAERNO5_C2 * dt, dt * KVAERNO5_GAMMA, delta1, self.known2_block, self.delta2_block)
                known3 = self.known3_call(1.0, delta1, delta2, self.known3)
                delta3 = self._solve_stage(interval, state, dt, KVAERNO5_C3 * dt, dt * KVAERNO5_GAMMA, known3, self.known3_block, self.delta3_block)
                known4 = self.known4_call(1.0, delta1, delta2, delta3, self.known4)
                delta4 = self._solve_stage(interval, state, dt, KVAERNO5_C4 * dt, dt * KVAERNO5_GAMMA, known4, self.known4_block, self.delta4_block)
                known5 = self.known5_call(1.0, delta1, delta2, delta3, delta4, self.known5)
                delta5 = self._solve_stage(interval, state, dt, KVAERNO5_C5 * dt, dt * KVAERNO5_GAMMA, known5, self.known5_block, self.delta5_block)
                known6 = self.known6_call(1.0, delta1, delta2, delta3, delta4, delta5, self.known6)
                delta6 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO5_GAMMA, known6, self.known6_block, self.delta6_block)
                known7 = self.known7_call(1.0, delta1, delta2, delta3, delta4, delta5, delta6, self.known7)
                delta7 = self._solve_stage(interval, state, dt, dt, dt * KVAERNO5_GAMMA, known7, self.known7_block, self.delta7_block)
            except StarkErrorRecoverable:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, scheme_name)
                continue
            delta_high = self.high_delta_call(1.0, delta1, delta2, delta3, delta4, delta5, delta6, delta7, self.trial)
            error = self.error_delta_call(1.0, delta1, delta2, delta3, delta4, delta5, delta6, delta7, self.error)
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


__all__ = ["KVAERNO5_GAMMA", "KVAERNO5_TABLEAU", "SchemeKvaerno5"]
