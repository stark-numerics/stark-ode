from __future__ import annotations

from math import sqrt

from stark.accelerators.binding import BoundDerivative
from stark.algebraist import Algebraist, AlgebraistImplicitCombination
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

# `SequentialDIRKResolventStep` solves each implicit stage in a diagonal
# stage-increment basis:
#
#     delta_i = known_shift + gamma * dt * f(...)
#
# High, embedded, and error weights are therefore represented as tableau
# weights divided by `gamma`, except for the final diagonal stage contribution,
# which already has unit weight in this increment basis.
_DELTA1_HIGH = SDIRK21_B1 / SDIRK21_GAMMA
_DELTA2_HIGH = SDIRK21_B2 / SDIRK21_GAMMA
_DELTA3_HIGH = 1.0

_DELTA1_LOW = SDIRK21_BHAT1 / SDIRK21_GAMMA
_DELTA2_LOW = SDIRK21_BHAT2 / SDIRK21_GAMMA
_DELTA3_LOW = SDIRK21_BHAT3 / SDIRK21_GAMMA

_STAGE_INCREMENT_WEIGHTS_HIGH = (_DELTA1_HIGH, _DELTA2_HIGH, _DELTA3_HIGH)
_STAGE_INCREMENT_WEIGHTS_LOW = (_DELTA1_LOW, _DELTA2_LOW, _DELTA3_LOW)
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
class SchemeSDIRK21:
    """Adaptive ESDIRK 2(1) with sequential implicit stage solves.

    This is a singly diagonally implicit Runge-Kutta pair. The first stage is
    explicit, and each later implicit stage uses the same diagonal coefficient
    `gamma`. That lets STARK solve the stages one at a time through the
    configured resolvent instead of forming one fully coupled nonlinear block.

    The method advances with a second-order formula and estimates local error
    with an embedded first-order formula. It is a compact stiff adaptive
    exemplar because it exercises the implicit solver stack, resolvent failure
    rejection, embedded error control, and adaptive step-size update without the
    coefficient volume of the larger Kvaerno schemes.

    The `_DELTA...` constants are written in the stage-increment basis used by
    `SequentialDIRKResolventStep`, not as raw tableau entries.
    """

    # Assigned by initialise_adaptive_runtime from stark.schemes.support.
    step_control: SchemeStepControl

    __slots__ = (
        "step_control",
        "call_pure",
        "delta1",
        "delta2",
        "delta3",
        "derivative",
        "error",
        "error_delta_call",
        "high_delta_call",
        "known2_call",
        "known3_call",
        "known3",
        "redirect_call",
        "stage1_rate",
        "stepper",
        "trial",
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
        algebraist: Algebraist | None = None,
    ) -> None:
        self.error_delta_call = unbound_scheme_call
        self.high_delta_call = unbound_scheme_call
        self.known2_call = unbound_scheme_call
        self.known3_call = unbound_scheme_call

        self.derivative = BoundDerivative(derivative)
        self.stepper = SequentialDIRKResolventStep(
            "SDIRK21",
            self.tableau,
            derivative,
            workbench,
            2,
            resolvent,
        )
        self.stage1_rate = workbench.allocate_translation()

        workspace = self.stepper.workspace
        (
            self.delta1,
            self.delta2,
            self.delta3,
            self.trial,
            self.error,
            self.known3,
        ) = workspace.allocate_translation_buffers(6)

        initialise_adaptive_runtime(self, regulator)
        self.call_pure = self.call_generic
        refresh_adaptive_call(self)

        if algebraist is not None:
            self.bind_algebraist_path(algebraist)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.5)

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
                    (SDIRK21_GAMMA,),
                    step_scale=True,
                ),
                None,
                AlgebraistImplicitCombination(
                    "known3",
                    (_DELTA1_HIGH, _DELTA2_HIGH),
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
        apply_delta = workspace.apply_delta
        ratio = self.step_control.ratio

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__

        while True:
            derivative(interval, state, self.stage1_rate)
            delta1 = scale(
                dt * SDIRK21_GAMMA,
                self.stage1_rate,
                self.delta1,
            )

            try:
                delta2 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=0,
                    stage_shift=2.0 * SDIRK21_GAMMA * dt,
                    alpha=dt * SDIRK21_GAMMA,
                    known_shift=delta1,
                    out=self.delta2,
                )

                known3 = combine2(
                    _DELTA1_HIGH,
                    delta1,
                    _DELTA2_HIGH,
                    delta2,
                    self.known3,
                )

                delta3 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=1,
                    stage_shift=dt,
                    alpha=dt * SDIRK21_GAMMA,
                    known_shift=known3,
                    out=self.delta3,
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

            delta_high = combine3(
                _STAGE_INCREMENT_WEIGHTS_HIGH[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_HIGH[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_HIGH[2],
                delta3,
                self.trial,
            )

            error = combine3(
                _STAGE_INCREMENT_WEIGHTS_ERROR[0],
                delta1,
                _STAGE_INCREMENT_WEIGHTS_ERROR[1],
                delta2,
                _STAGE_INCREMENT_WEIGHTS_ERROR[2],
                delta3,
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
        high_delta_call = self.high_delta_call
        error_delta_call = self.error_delta_call

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__

        while True:
            derivative(interval, state, self.stage1_rate)
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
                    stage_shift=2.0 * SDIRK21_GAMMA * dt,
                    alpha=dt * SDIRK21_GAMMA,
                    known_shift=delta1,
                    out=self.delta2,
                )

                known3 = known3_call(
                    delta1,
                    delta2,
                    self.known3,
                )

                delta3 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=1,
                    stage_shift=dt,
                    alpha=dt * SDIRK21_GAMMA,
                    known_shift=known3,
                    out=self.delta3,
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
                self.trial,
            )

            error = error_delta_call(
                delta1,
                delta2,
                delta3,
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
    "SDIRK21_GAMMA",
    "SDIRK21_TABLEAU",
    "SchemeSDIRK21",
]
