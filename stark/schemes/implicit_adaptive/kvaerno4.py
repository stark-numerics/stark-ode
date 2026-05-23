from __future__ import annotations

from stark.accelerators.binding import DerivativeAccelerated
from stark.algebraist.classic import Algebraist, AlgebraistImplicitCombination
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.machinery.stage_solve.workers import SequentialDIRKResolventStep
from stark.resolvents.support.failure import ResolventError
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    SchemeStepControl,
    initialise_adaptive_runtime,
    refresh_adaptive_call,
    unbound_scheme_call,
    with_adaptive_runtime_methods,
    with_implicit_stepper_methods,
    with_scheme_display,
)
from stark.schemes.support.tableau import ButcherTableau


KVAERNO4_GAMMA = 0.5728160625


def _poly(*coefficients: float) -> float:
    value = 0.0
    for coefficient in coefficients:
        value = value * KVAERNO4_GAMMA + coefficient
    return value


KVAERNO4_A21 = KVAERNO4_GAMMA
KVAERNO4_A31 = (
    _poly(144.0, -180.0, 81.0, -15.0, 1.0)
    * KVAERNO4_GAMMA
    / (_poly(12.0, -6.0, 1.0) ** 2)
)
KVAERNO4_A32 = (
    _poly(-36.0, 39.0, -15.0, 2.0)
    * KVAERNO4_GAMMA
    / (_poly(12.0, -6.0, 1.0) ** 2)
)
KVAERNO4_A41 = _poly(-144.0, 396.0, -330.0, 117.0, -18.0, 1.0) / (
    12.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(12.0, -9.0, 2.0)
)
KVAERNO4_A42 = _poly(72.0, -126.0, 69.0, -15.0, 1.0) / (
    12.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(3.0, -1.0)
)
KVAERNO4_A43 = (
    _poly(-6.0, 6.0, -1.0) * (_poly(12.0, -6.0, 1.0) ** 2)
) / (
    12.0
    * KVAERNO4_GAMMA
    * KVAERNO4_GAMMA
    * _poly(12.0, -9.0, 2.0)
    * _poly(3.0, -1.0)
)
KVAERNO4_A51 = _poly(288.0, -312.0, 120.0, -18.0, 1.0) / (
    48.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(12.0, -9.0, 2.0)
)
KVAERNO4_A52 = _poly(24.0, -12.0, 1.0) / (
    48.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(3.0, -1.0)
)
KVAERNO4_A53 = -(_poly(12.0, -6.0, 1.0) ** 3) / (
    48.0
    * KVAERNO4_GAMMA
    * KVAERNO4_GAMMA
    * _poly(3.0, -1.0)
    * _poly(12.0, -9.0, 2.0)
    * _poly(6.0, -6.0, 1.0)
)
KVAERNO4_A54 = _poly(-24.0, 36.0, -12.0, 1.0) / _poly(
    24.0,
    -24.0,
    4.0,
)

KVAERNO4_C2 = KVAERNO4_GAMMA + KVAERNO4_A21
KVAERNO4_C3 = KVAERNO4_GAMMA + KVAERNO4_A31 + KVAERNO4_A32

KVAERNO4_TABLEAU = ButcherTableau(
    c=(0.0, KVAERNO4_C2, KVAERNO4_C3, 1.0, 1.0),
    a=(
        (),
        (KVAERNO4_A21, KVAERNO4_GAMMA),
        (KVAERNO4_A31, KVAERNO4_A32, KVAERNO4_GAMMA),
        (KVAERNO4_A41, KVAERNO4_A42, KVAERNO4_A43, KVAERNO4_GAMMA),
        (
            KVAERNO4_A51,
            KVAERNO4_A52,
            KVAERNO4_A53,
            KVAERNO4_A54,
            KVAERNO4_GAMMA,
        ),
    ),
    b=(KVAERNO4_A51, KVAERNO4_A52, KVAERNO4_A53, KVAERNO4_A54, KVAERNO4_GAMMA),
    order=4,
    b_embedded=(KVAERNO4_A41, KVAERNO4_A42, KVAERNO4_A43, KVAERNO4_GAMMA, 0.0),
    embedded_order=3,
    short_name="Kvaerno4",
    full_name="Kvaerno 4(3)",
)

# `SequentialDIRKResolventStep` solves each implicit stage in a diagonal
# stage-increment basis:
#
#     delta_i = known_shift + gamma * dt * f(...)
#
# Off-diagonal tableau coefficients, final weights, and embedded weights are
# therefore represented as stage-increment weights by dividing by `gamma`.
_DELTA21 = KVAERNO4_A21 / KVAERNO4_GAMMA
_DELTA31 = KVAERNO4_A31 / KVAERNO4_GAMMA
_DELTA32 = KVAERNO4_A32 / KVAERNO4_GAMMA
_DELTA41 = KVAERNO4_A41 / KVAERNO4_GAMMA
_DELTA42 = KVAERNO4_A42 / KVAERNO4_GAMMA
_DELTA43 = KVAERNO4_A43 / KVAERNO4_GAMMA
_DELTA51 = KVAERNO4_A51 / KVAERNO4_GAMMA
_DELTA52 = KVAERNO4_A52 / KVAERNO4_GAMMA
_DELTA53 = KVAERNO4_A53 / KVAERNO4_GAMMA
_DELTA54 = KVAERNO4_A54 / KVAERNO4_GAMMA

_STAGE_INCREMENT_WEIGHTS_HIGH = (_DELTA51, _DELTA52, _DELTA53, _DELTA54, 1.0)
_STAGE_INCREMENT_WEIGHTS_LOW = (_DELTA41, _DELTA42, _DELTA43, 1.0, 0.0)
_STAGE_INCREMENT_WEIGHTS_ERROR = tuple(
    high - low
    for high, low in zip(
        _STAGE_INCREMENT_WEIGHTS_HIGH,
        _STAGE_INCREMENT_WEIGHTS_LOW,
        strict=True,
    )
)


@with_scheme_display
@with_adaptive_runtime_methods
@with_implicit_stepper_methods
class SchemeKvaerno4:
    """Kvaerno's adaptive ESDIRK 4(3) method.

    This is a fourth-order, stiffly accurate, explicit-first-stage singly
    diagonally implicit Runge-Kutta method with a third-order embedded estimate
    for adaptive step control.

    The first stage is explicit. The remaining stages share the same diagonal
    coefficient `gamma`, so STARK solves them sequentially through
    `SequentialDIRKResolventStep` rather than through one coupled nonlinear
    system.

    The coefficient definitions are deliberately left in formula form. They are
    generated from Kvaerno's construction and are easier to audit this way than
    as unexplained decimal constants.

    The `_DELTA...` constants are written in the stage-increment basis used by
    the stepper. They are derived from the tableau coefficients, not separate
    method parameters.

    Further reading: Anne Kvaerno, "Singly diagonally implicit Runge-Kutta
    methods with an explicit first stage" (BIT Numerical Mathematics, 2004).
    """

    # Assigned by initialise_adaptive_runtime from stark.schemes.support.
    step_control: SchemeStepControl

    __slots__ = (
        "step_control",
        "call_pure",
        "delta1",
        "delta2",
        "delta3",
        "delta4",
        "delta5",
        "derivative",
        "error",
        "error_delta_call",
        "high_delta_call",
        "known2_call",
        "known3_call",
        "known4_call",
        "known5_call",
        "known4",
        "known5",
        "redirect_call",
        "stage1_rate",
        "stepper",
        "trial",
    )

    descriptor = SchemeDescriptor("Kvaerno4", "Kvaerno 4(3)")
    tableau = KVAERNO4_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
        *,
        algebraist: Algebraist | None = None,
    ) -> None:
        self.error_delta_call = unbound_scheme_call
        self.high_delta_call = unbound_scheme_call
        self.known2_call = unbound_scheme_call
        self.known3_call = unbound_scheme_call
        self.known4_call = unbound_scheme_call
        self.known5_call = unbound_scheme_call

        self.derivative = DerivativeAccelerated(derivative)
        self.stepper = SequentialDIRKResolventStep(
            "Kvaerno4",
            self.tableau,
            derivative,
            workbench,
            4,
            resolvent,
        )
        self.stage1_rate = workbench.allocate_translation()

        workspace = self.stepper.workspace
        (
            self.delta1,
            self.delta2,
            self.delta3,
            self.delta4,
            self.delta5,
            self.known4,
            self.known5,
            self.trial,
            self.error,
        ) = workspace.allocate_translation_buffers(9)

        initialise_adaptive_runtime(self, regulator)
        self.call_pure = self.call_inline
        refresh_adaptive_call(self)

        if algebraist is not None:
            self.use_specialists(algebraist)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.25)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def use_specialists(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_implicit_adaptive_scheme(
            known_shifts=(
                AlgebraistImplicitCombination(
                    "known2",
                    (KVAERNO4_GAMMA,),
                    step_scale=True,
                ),
                None,
                AlgebraistImplicitCombination(
                    "known3",
                    (_DELTA31, _DELTA32),
                ),
                AlgebraistImplicitCombination(
                    "known4",
                    (_DELTA41, _DELTA42, _DELTA43),
                ),
                AlgebraistImplicitCombination(
                    "known5",
                    (_DELTA51, _DELTA52, _DELTA53, _DELTA54),
                ),
            ),
            high_delta=AlgebraistImplicitCombination(
                "high_delta",
                _STAGE_INCREMENT_WEIGHTS_HIGH,
            ),
            error_delta=AlgebraistImplicitCombination(
                "error_delta",
                _STAGE_INCREMENT_WEIGHTS_ERROR,
            ),
        )
        scheme_name = type(self).__name__
        self.known2_call = calls.require_known_shift_call(0, scheme_name)
        self.known3_call = calls.require_known_shift_call(2, scheme_name)
        self.known4_call = calls.require_known_shift_call(3, scheme_name)
        self.known5_call = calls.require_known_shift_call(4, scheme_name)
        self.high_delta_call = calls.require_high_delta_call(scheme_name)
        self.error_delta_call = calls.require_error_delta_call(scheme_name)
        self.call_pure = self.call_specialized
        refresh_adaptive_call(self)

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        proposal = self.step_control.propose_step(interval)

        if proposal.remaining <= 0.0:
            self.step_control.record_stopped(interval)
            return 0.0

        stepper = self.stepper
        workspace = stepper.workspace
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
            delta1 = scale(
                dt * KVAERNO4_GAMMA,
                self.stage1_rate,
                self.delta1,
            )

            try:
                delta2 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=0,
                    stage_shift=KVAERNO4_C2 * dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=delta1,
                    out=self.delta2,
                )

                known3 = combine2(
                    _DELTA31,
                    delta1,
                    _DELTA32,
                    delta2,
                    self.trial,
                )

                delta3 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=1,
                    stage_shift=KVAERNO4_C3 * dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known3,
                    out=self.delta3,
                )

                known4 = combine3(
                    _DELTA41,
                    delta1,
                    _DELTA42,
                    delta2,
                    _DELTA43,
                    delta3,
                    self.known4,
                )

                delta4 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=2,
                    stage_shift=dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known4,
                    out=self.delta4,
                )

                known5 = combine4(
                    _DELTA51,
                    delta1,
                    _DELTA52,
                    delta2,
                    _DELTA53,
                    delta3,
                    _DELTA54,
                    delta4,
                    self.known5,
                )

                delta5 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=3,
                    stage_shift=dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known5,
                    out=self.delta5,
                )

            except ResolventError:
                rejection_count += 1
                dt = self.step_control.rejected_step(
                    dt,
                    1.0,
                    remaining,
                    scheme_name,
                )
                continue

            delta_high = combine5(
                _STAGE_INCREMENT_WEIGHTS_HIGH[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_HIGH[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_HIGH[2],
                delta3,
                _STAGE_INCREMENT_WEIGHTS_HIGH[3],
                delta4,
                _STAGE_INCREMENT_WEIGHTS_HIGH[4],
                delta5,
                self.trial,
            )

            error = combine5(
                _STAGE_INCREMENT_WEIGHTS_ERROR[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_ERROR[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_ERROR[2],
                delta3,
                _STAGE_INCREMENT_WEIGHTS_ERROR[3],
                delta4,
                _STAGE_INCREMENT_WEIGHTS_ERROR[4],
                delta5,
                self.error,
            )

            error_ratio = ratio(error.norm(), delta_high.norm())

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = self.step_control.rejected_step(
                dt,
                error_ratio,
                remaining,
                scheme_name,
            )

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = self.step_control.accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )

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

    def call_specialized(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        proposal = self.step_control.propose_step(interval)

        if proposal.remaining <= 0.0:
            self.step_control.record_stopped(interval)
            return 0.0

        stepper = self.stepper
        workspace = stepper.workspace
        derivative = self.derivative
        apply_delta = workspace.apply_delta
        ratio = self.step_control.ratio
        known2_call = self.known2_call
        known3_call = self.known3_call
        known4_call = self.known4_call
        known5_call = self.known5_call
        high_delta_call = self.high_delta_call
        error_delta_call = self.error_delta_call

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__

        derivative(interval, state, self.stage1_rate)

        while True:
            delta1 = known2_call(
                dt,
                self.stage1_rate,
                self.delta1,
            )

            try:
                delta2 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=0,
                    stage_shift=KVAERNO4_C2 * dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=delta1,
                    out=self.delta2,
                )

                known3 = known3_call(
                    delta1,
                    delta2,
                    self.trial,
                )

                delta3 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=1,
                    stage_shift=KVAERNO4_C3 * dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known3,
                    out=self.delta3,
                )

                known4 = known4_call(
                    delta1,
                    delta2,
                    delta3,
                    self.known4,
                )

                delta4 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=2,
                    stage_shift=dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known4,
                    out=self.delta4,
                )

                known5 = known5_call(
                    delta1,
                    delta2,
                    delta3,
                    delta4,
                    self.known5,
                )

                delta5 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=3,
                    stage_shift=dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known5,
                    out=self.delta5,
                )

            except ResolventError:
                rejection_count += 1
                dt = self.step_control.rejected_step(
                    dt,
                    1.0,
                    remaining,
                    scheme_name,
                )
                continue

            delta_high = high_delta_call(
                delta1,
                delta2,
                delta3,
                delta4,
                delta5,
                self.trial,
            )

            error = error_delta_call(
                delta1,
                delta2,
                delta3,
                delta4,
                delta5,
                self.error,
            )

            error_ratio = ratio(error.norm(), delta_high.norm())

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = self.step_control.rejected_step(
                dt,
                error_ratio,
                remaining,
                scheme_name,
            )

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = self.step_control.accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )

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


__all__ = [
    "KVAERNO4_GAMMA",
    "KVAERNO4_TABLEAU",
    "SchemeKvaerno4",
]
