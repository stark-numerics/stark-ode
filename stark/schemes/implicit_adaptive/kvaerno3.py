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


KVAERNO3_GAMMA = 0.43586652150
KVAERNO3_A21 = KVAERNO3_GAMMA
KVAERNO3_A31 = (-4.0 * KVAERNO3_GAMMA * KVAERNO3_GAMMA + 6.0 * KVAERNO3_GAMMA - 1.0) / (4.0 * KVAERNO3_GAMMA)
KVAERNO3_A32 = (-2.0 * KVAERNO3_GAMMA + 1.0) / (4.0 * KVAERNO3_GAMMA)
KVAERNO3_A41 = (6.0 * KVAERNO3_GAMMA - 1.0) / (12.0 * KVAERNO3_GAMMA)
KVAERNO3_A42 = -1.0 / ((24.0 * KVAERNO3_GAMMA - 12.0) * KVAERNO3_GAMMA)
KVAERNO3_A43 = (
    -6.0 * KVAERNO3_GAMMA * KVAERNO3_GAMMA + 6.0 * KVAERNO3_GAMMA - 1.0
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

_DELTA21 = KVAERNO3_A21 / KVAERNO3_GAMMA
_DELTA31 = KVAERNO3_A31 / KVAERNO3_GAMMA
_DELTA32 = KVAERNO3_A32 / KVAERNO3_GAMMA
_DELTA41 = KVAERNO3_A41 / KVAERNO3_GAMMA
_DELTA42 = KVAERNO3_A42 / KVAERNO3_GAMMA
_DELTA43 = KVAERNO3_A43 / KVAERNO3_GAMMA
_DELTA_HIGH = (_DELTA41, _DELTA42, _DELTA43, 1.0)
_DELTA_LOW = (_DELTA31, _DELTA32, 1.0, 0.0)
_DELTA_ERR = tuple(high - low for high, low in zip(_DELTA_HIGH, _DELTA_LOW, strict=True))


class SchemeKvaerno3(SchemeBaseImplicitAdaptive):
    """
    Kvaerno's adaptive ESDIRK 3(2) method with sequential stage solves.

    This is a third-order stiffly accurate ESDIRK scheme with a second-order
    embedded estimate for step control. It keeps the same diagonal coefficient
    across implicit stages, so it fits naturally into STARK's stage-resolvent
    architecture while taking materially larger steps than SDIRK21 on
    smooth stiff problems.

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
        "known4",
        "trial",
        "error",
    )

    descriptor = SchemeDescriptor("Kvaerno3", "Kvaerno 3(2)")
    tableau = KVAERNO3_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        self.derivative = BoundDerivative(derivative)
        self.stepper = SequentialDIRKResolventStep("Kvaerno3", self.tableau, derivative, workbench, 3, resolvent)
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
        self.initialise_runtime(regulator)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=1.0 / 3.0)

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
            delta1 = scale(self.delta1, dt * KVAERNO3_GAMMA, self.stage1_rate)

            try:
                stage2 = stepper.solve(
                    interval,
                    state,
                    dt,
                    block_index=0,
                    stage_shift=2.0 * KVAERNO3_GAMMA * dt,
                    alpha=dt * KVAERNO3_GAMMA,
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
                    stage_shift=dt,
                    alpha=dt * KVAERNO3_GAMMA,
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
                    alpha=dt * KVAERNO3_GAMMA,
                    known_shift=known4,
                    out=self.delta4,
                )
                delta4 = stage4
            except ResolventError:
                rejection_count += 1
                dt = controller.rejected_step(dt, 1.0, remaining, "Kvaerno3")
                continue

            delta_high = combine4(
                self.trial,
                _DELTA_HIGH[0],
                delta1,
                _DELTA_HIGH[1],
                delta2,
                _DELTA_HIGH[2],
                delta3,
                _DELTA_HIGH[3],
                delta4,
            )
            error = combine4(
                self.error,
                _DELTA_ERR[0],
                delta1,
                _DELTA_ERR[1],
                delta2,
                _DELTA_ERR[2],
                delta3,
                _DELTA_ERR[3],
                delta4,
            )
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = ratio(error_norm, delta_high_norm)

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "Kvaerno3")

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


__all__ = ["KVAERNO3_GAMMA", "KVAERNO3_TABLEAU", "SchemeKvaerno3"]

















