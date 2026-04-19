from __future__ import annotations

from stark.accelerators.binding import BoundDerivative
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

KVAERNO4_GAMMA = 0.5728160625


def _poly(*coefficients: float) -> float:
    value = 0.0
    for coefficient in coefficients:
        value = value * KVAERNO4_GAMMA + coefficient
    return value
KVAERNO4_A21 = KVAERNO4_GAMMA
KVAERNO4_A31 = _poly(144.0, -180.0, 81.0, -15.0, 1.0) * KVAERNO4_GAMMA / (_poly(12.0, -6.0, 1.0) ** 2)
KVAERNO4_A32 = _poly(-36.0, 39.0, -15.0, 2.0) * KVAERNO4_GAMMA / (_poly(12.0, -6.0, 1.0) ** 2)
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
_DELTA_HIGH = (_DELTA51, _DELTA52, _DELTA53, _DELTA54, 1.0)
_DELTA_LOW = (_DELTA41, _DELTA42, _DELTA43, 1.0, 0.0)
_DELTA_ERR = tuple(high - low for high, low in zip(_DELTA_HIGH, _DELTA_LOW, strict=True))


class SchemeKvaerno4(SchemeBaseImplicitAdaptive):
    """
    Kvaerno's adaptive ESDIRK 4(3) method with sequential stage solves.

    This is a fourth-order stiffly accurate ESDIRK pair with a third-order
    embedded estimate. It aims at the same stiff adaptive territory as Diffrax
    Kvaerno-family methods while staying inside STARK's sequential stage
    resolvent structure.

    Further reading: Anne Kvaerno, "Singly diagonally implicit Runge-Kutta
    methods with an explicit first stage" (BIT Numerical Mathematics, 2004).
    """

    __slots__ = (
        "derivative",
        "stepper",
        "stage1_rate",
        "delta1",
        "delta2",
        "delta3",
        "delta4",
        "delta5",
        "known4",
        "known5",
        "trial",
        "error",
    )

    descriptor = SchemeDescriptor("Kvaerno4", "Kvaerno 4(3)")
    tableau = KVAERNO4_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        self.derivative = BoundDerivative(derivative)
        self.stepper = SequentialDIRKResolventStep("Kvaerno4", self.tableau, derivative, workbench, 4, resolvent)
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
        self.initialise_runtime(regulator)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.25)

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

        stepper = self.stepper
        workspace = stepper.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        combine4 = workspace.combine4
        combine5 = workspace.combine5
        apply_delta = workspace.apply_delta
        controller = self._controller
        ratio = self._ratio
        assert controller is not None
        assert ratio is not None
        dt = interval.step if interval.step <= remaining else remaining
        proposed_dt = dt
        rejection_count = 0
        derivative(interval, state, self.stage1_rate)

        while True:
            delta1 = scale(self.delta1, dt * KVAERNO4_GAMMA, self.stage1_rate)

            try:
                stage2 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=0,
                    stage_shift=KVAERNO4_C2 * dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=delta1,
                    out=self.delta2,
                )
                delta2 = stage2

                known3 = combine2(self.trial, _DELTA31, delta1, _DELTA32, delta2)
                stage3 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=1,
                    stage_shift=KVAERNO4_C3 * dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known3,
                    out=self.delta3,
                )
                delta3 = stage3

                known4 = combine3(self.known4, _DELTA41, delta1, _DELTA42, delta2, _DELTA43, delta3)
                stage4 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=2,
                    stage_shift=dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known4,
                    out=self.delta4,
                )
                delta4 = stage4

                known5 = combine4(
                    self.known5,
                    _DELTA51,
                    delta1,
                    _DELTA52,
                    delta2,
                    _DELTA53,
                    delta3,
                    _DELTA54,
                    delta4,
                )
                stage5 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=3,
                    stage_shift=dt,
                    alpha=dt * KVAERNO4_GAMMA,
                    known_shift=known5,
                    out=self.delta5,
                )
                delta5 = stage5
            except ResolventError:
                rejection_count += 1
                dt = controller.rejected_step(dt, 1.0, remaining, "Kvaerno4")
                continue

            delta_high = combine5(
                self.trial,
                _DELTA_HIGH[0],
                delta1,
                _DELTA_HIGH[1],
                delta2,
                _DELTA_HIGH[2],
                delta3,
                _DELTA_HIGH[3],
                delta4,
                _DELTA_HIGH[4],
                delta5,
            )
            error = combine5(
                self.error,
                _DELTA_ERR[0],
                delta1,
                _DELTA_ERR[1],
                delta2,
                _DELTA_ERR[2],
                delta3,
                _DELTA_ERR[3],
                delta4,
                _DELTA_ERR[4],
                delta5,
            )
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = ratio(error_norm, delta_high_norm)

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "Kvaerno4")

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


__all__ = ["KVAERNO4_GAMMA", "KVAERNO4_TABLEAU", "SchemeKvaerno4"]

















