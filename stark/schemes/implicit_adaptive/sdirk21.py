from __future__ import annotations

from stark.accelerators.binding import BoundDerivative
from math import sqrt

from stark.schemes.tableau import ButcherTableau
from stark.execution.regulator import Regulator
from stark.execution.executor import Executor
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Workbench
from stark.resolvents.failure import ResolventError
from stark.schemes.descriptor import SchemeDescriptor
from stark.machinery.stage_solve.workers import SequentialDIRKResolventStep
from stark.schemes.base import (
    SchemeBaseImplicitAdaptive,
    _ADVANCE_ACCEPTED_DT,
    _ADVANCE_ERROR_RATIO,
    _ADVANCE_NEXT_DT,
    _ADVANCE_PROPOSED_DT,
    _ADVANCE_REJECTION_COUNT,
    _ADVANCE_T_START,
)


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

_DELTA1_HIGH = SDIRK21_B1 / SDIRK21_GAMMA
_DELTA2_HIGH = SDIRK21_B2 / SDIRK21_GAMMA
_DELTA3_HIGH = 1.0
_DELTA1_LOW = SDIRK21_BHAT1 / SDIRK21_GAMMA
_DELTA2_LOW = SDIRK21_BHAT2 / SDIRK21_GAMMA
_DELTA3_LOW = SDIRK21_BHAT3 / SDIRK21_GAMMA
_DELTA1_ERR = _DELTA1_HIGH - _DELTA1_LOW
_DELTA2_ERR = _DELTA2_HIGH - _DELTA2_LOW
_DELTA3_ERR = _DELTA3_HIGH - _DELTA3_LOW


class SchemeSDIRK21(SchemeBaseImplicitAdaptive):
    """
    An adaptive ESDIRK 2(1) method with sequential implicit stage solves.

    This is a singly diagonally implicit Runge-Kutta pair: each implicit stage
    uses the same diagonal coefficient, which lets STARK resolve the stages one
    at a time with the configured resolvent. The method advances with
    a second-order formula and estimates local error with an embedded
    first-order formula.

    It is a useful first stiff adaptive method because it exercises the
    implicit solver stack without requiring a fully coupled block solve.

    Further reading: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "derivative",
        "stepper",
        "stage1_rate",
        "delta1",
        "delta2",
        "delta3",
        "trial",
        "error",
        "known3",
    )

    descriptor = SchemeDescriptor("SDIRK21", "ESDIRK 2(1)")
    tableau = SDIRK21_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        self.derivative = BoundDerivative(derivative)
        self.stepper = SequentialDIRKResolventStep("SDIRK21", self.tableau, derivative, workbench, 2, resolvent)
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
        self.initialise_runtime(regulator)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.5)

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        return self.redirect_call(interval, state, executor)

    def advance_body(self, interval: IntervalLike, state: State) -> None:
        remaining = interval.stop - interval.present
        advance_report = self.advance_report
        if remaining <= 0.0:
            advance_report[_ADVANCE_ACCEPTED_DT] = 0.0
            advance_report[_ADVANCE_T_START] = interval.present
            advance_report[_ADVANCE_PROPOSED_DT] = 0.0
            advance_report[_ADVANCE_NEXT_DT] = 0.0
            advance_report[_ADVANCE_ERROR_RATIO] = 0.0
            advance_report[_ADVANCE_REJECTION_COUNT] = 0
            return

        workspace = self.stepper.workspace
        stepper = self.stepper
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        apply_delta = workspace.apply_delta
        controller = self._controller
        ratio = self._ratio
        assert controller is not None
        assert ratio is not None
        dt = interval.step if interval.step <= remaining else remaining
        proposed_dt = dt
        rejection_count = 0

        while True:
            derivative(interval, state, self.stage1_rate)
            delta1 = scale(self.delta1, dt * SDIRK21_GAMMA, self.stage1_rate)

            try:
                stage2 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=0,
                    stage_shift=2.0 * SDIRK21_GAMMA * dt,
                    alpha=dt * SDIRK21_GAMMA,
                    known_shift=delta1,
                    out=self.delta2,
                )
                delta2 = stage2

                known3 = combine2(
                    self.known3,
                    (1.0 - SDIRK21_B2 - SDIRK21_GAMMA) / SDIRK21_GAMMA,
                    delta1,
                    SDIRK21_B2 / SDIRK21_GAMMA,
                    delta2,
                )
                stage3 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=1,
                    stage_shift=dt,
                    alpha=dt * SDIRK21_GAMMA,
                    known_shift=known3,
                    out=self.delta3,
                )
                delta3 = stage3
            except ResolventError:
                rejection_count += 1
                dt = controller.rejected_step(dt, 1.0, remaining, "SDIRK21")
                continue

            delta_high = combine3(
                self.trial,
                _DELTA1_HIGH,
                delta1,
                _DELTA2_HIGH,
                delta2,
                _DELTA3_HIGH,
                delta3,
            )
            error = combine3(
                self.error,
                _DELTA1_ERR,
                delta1,
                _DELTA2_ERR,
                delta2,
                _DELTA3_ERR,
                delta3,
            )
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = ratio(error_norm, delta_high_norm)

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "SDIRK21")

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        interval.step = next_dt
        apply_delta(delta_high, state)
        advance_report[_ADVANCE_ACCEPTED_DT] = accepted_dt
        advance_report[_ADVANCE_T_START] = interval.present
        advance_report[_ADVANCE_PROPOSED_DT] = proposed_dt
        advance_report[_ADVANCE_NEXT_DT] = next_dt
        advance_report[_ADVANCE_ERROR_RATIO] = error_ratio
        advance_report[_ADVANCE_REJECTION_COUNT] = rejection_count


__all__ = ["SDIRK21_GAMMA", "SDIRK21_TABLEAU", "SchemeSDIRK21"]

















