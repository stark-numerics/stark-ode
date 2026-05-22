from __future__ import annotations

from stark.accelerators.binding import DerivativeAccelerated
from stark.algebraist.classic import Algebraist, AlgebraistImplicitCombination
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.resolvents.support.failure import ResolventError
from stark.machinery.stage_solve.workers import SequentialDIRKResolventStep
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


KVAERNO3_GAMMA = 0.43586652150

KVAERNO3_A21 = KVAERNO3_GAMMA
KVAERNO3_A31 = (
    -4.0 * KVAERNO3_GAMMA * KVAERNO3_GAMMA
    + 6.0 * KVAERNO3_GAMMA
    - 1.0
) / (4.0 * KVAERNO3_GAMMA)
KVAERNO3_A32 = (-2.0 * KVAERNO3_GAMMA + 1.0) / (4.0 * KVAERNO3_GAMMA)
KVAERNO3_A41 = (6.0 * KVAERNO3_GAMMA - 1.0) / (12.0 * KVAERNO3_GAMMA)
KVAERNO3_A42 = -1.0 / ((24.0 * KVAERNO3_GAMMA - 12.0) * KVAERNO3_GAMMA)
KVAERNO3_A43 = (
    -6.0 * KVAERNO3_GAMMA * KVAERNO3_GAMMA
    + 6.0 * KVAERNO3_GAMMA
    - 1.0
) / (6.0 * KVAERNO3_GAMMA - 3.0)

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

# `SequentialDIRKResolventStep` solves each implicit stage in a diagonal
# stage-increment basis:
#
#     delta_i = known_shift + gamma * dt * f(...)
#
# Off-diagonal tableau coefficients, final weights, and embedded weights are
# therefore converted into stage-increment weights by dividing by `gamma`.
_DELTA21 = KVAERNO3_A21 / KVAERNO3_GAMMA
_DELTA31 = KVAERNO3_A31 / KVAERNO3_GAMMA
_DELTA32 = KVAERNO3_A32 / KVAERNO3_GAMMA
_DELTA41 = KVAERNO3_A41 / KVAERNO3_GAMMA
_DELTA42 = KVAERNO3_A42 / KVAERNO3_GAMMA
_DELTA43 = KVAERNO3_A43 / KVAERNO3_GAMMA

_STAGE_INCREMENT_WEIGHTS_HIGH = (_DELTA41, _DELTA42, _DELTA43, 1.0)
_STAGE_INCREMENT_WEIGHTS_LOW = (_DELTA31, _DELTA32, 1.0, 0.0)
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
class SchemeKvaerno3:
    """Kvaerno's adaptive ESDIRK 3(2) method.

    This is a third-order, stiffly accurate, explicit-first-stage singly
    diagonally implicit Runge-Kutta method with a second-order embedded estimate
    for step control.

    The first stage is explicit. The remaining stages share the same diagonal
    coefficient `gamma`, so STARK can solve them sequentially with
    `SequentialDIRKResolventStep`. Compared with `SchemeSDIRK21`, this method is
    a more capable stiff adaptive option while still keeping the stage recipe
    small enough to inspect directly.

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
        "derivative",
        "error",
        "error_delta_call",
        "high_delta_call",
        "known2_call",
        "known3_call",
        "known4_call",
        "known4",
        "redirect_call",
        "stage1_rate",
        "stepper",
        "trial",
    )

    descriptor = SchemeDescriptor("Kvaerno3", "Kvaerno 3(2)")
    tableau = KVAERNO3_TABLEAU

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

        self.derivative = DerivativeAccelerated(derivative)
        self.stepper = SequentialDIRKResolventStep(
            "Kvaerno3",
            self.tableau,
            derivative,
            workbench,
            3,
            resolvent,
        )
        self.stage1_rate = workbench.allocate_translation()

        workspace = self.stepper.workspace
        (
            self.delta1,
            self.delta2,
            self.delta3,
            self.delta4,
            self.known4,
            self.trial,
            self.error,
        ) = workspace.allocate_translation_buffers(7)

        initialise_adaptive_runtime(self, regulator)
        self.call_pure = self.call_generic
        refresh_adaptive_call(self)

        if algebraist is not None:
            self.bind_algebraist_path(algebraist)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=1.0 / 3.0)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_implicit_adaptive_scheme(
            known_shifts=(
                AlgebraistImplicitCombination(
                    "known2",
                    (KVAERNO3_GAMMA,),
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
        self.high_delta_call = calls.require_high_delta_call(scheme_name)
        self.error_delta_call = calls.require_error_delta_call(scheme_name)
        self.call_pure = self.call_algebraist
        refresh_adaptive_call(self)

    def call_generic(
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
                dt * KVAERNO3_GAMMA,
                self.stage1_rate,
                self.delta1,
            )

            try:
                delta2 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=0,
                    stage_shift=2.0 * KVAERNO3_GAMMA * dt,
                    alpha=dt * KVAERNO3_GAMMA,
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
                    stage_shift=dt,
                    alpha=dt * KVAERNO3_GAMMA,
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
                    alpha=dt * KVAERNO3_GAMMA,
                    known_shift=known4,
                    out=self.delta4,
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

            delta_high = combine4(
                _STAGE_INCREMENT_WEIGHTS_HIGH[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_HIGH[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_HIGH[2],
                delta3,
                _STAGE_INCREMENT_WEIGHTS_HIGH[3],
                delta4,
                self.trial,
            )

            error = combine4(
                _STAGE_INCREMENT_WEIGHTS_ERROR[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_ERROR[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_ERROR[2],
                delta3,
                _STAGE_INCREMENT_WEIGHTS_ERROR[3],
                delta4,
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

    def call_algebraist(
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
                    stage_shift=2.0 * KVAERNO3_GAMMA * dt,
                    alpha=dt * KVAERNO3_GAMMA,
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
                    stage_shift=dt,
                    alpha=dt * KVAERNO3_GAMMA,
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
                    alpha=dt * KVAERNO3_GAMMA,
                    known_shift=known4,
                    out=self.delta4,
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
                self.trial,
            )

            error = error_delta_call(
                delta1,
                delta2,
                delta3,
                delta4,
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
    "KVAERNO3_GAMMA",
    "KVAERNO3_TABLEAU",
    "SchemeKvaerno3",
]
