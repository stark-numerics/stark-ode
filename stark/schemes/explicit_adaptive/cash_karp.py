from __future__ import annotations

from stark.execution.regulator import Regulator
from stark.execution.executor import Executor
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.schemes.tableau import ButcherTableau
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.base import (
    SchemeBaseExplicitAdaptive,
    _ADVANCE_ACCEPTED_DT,
    _ADVANCE_ERROR_RATIO,
    _ADVANCE_NEXT_DT,
    _ADVANCE_PROPOSED_DT,
    _ADVANCE_REJECTION_COUNT,
    _ADVANCE_T_START,
)


RKCK_TABLEAU = ButcherTableau(
    c=(0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0),
    a=(
        (),
        (1.0 / 5.0,),
        (3.0 / 40.0, 9.0 / 40.0),
        (3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0),
        (-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0),
        (
            1631.0 / 55296.0,
            175.0 / 512.0,
            575.0 / 13824.0,
            44275.0 / 110592.0,
            253.0 / 4096.0,
        ),
    ),
    b=(37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0),
    order=5,
    b_embedded=(
        2825.0 / 27648.0,
        0.0,
        18575.0 / 48384.0,
        13525.0 / 55296.0,
        277.0 / 14336.0,
        1.0 / 4.0,
    ),
    embedded_order=4,
)
RKCK_A = RKCK_TABLEAU.a
RKCK_B_HIGH = RKCK_TABLEAU.b
RKCK_B_LOW = RKCK_TABLEAU.b_embedded
assert RKCK_B_LOW is not None
RKCK_B_HIGH_NZ = (RKCK_B_HIGH[0], RKCK_B_HIGH[2], RKCK_B_HIGH[3], RKCK_B_HIGH[5])
RKCK_B_ERR_NZ = (
    RKCK_B_HIGH[0] - RKCK_B_LOW[0],
    RKCK_B_HIGH[2] - RKCK_B_LOW[2],
    RKCK_B_HIGH[3] - RKCK_B_LOW[3],
    RKCK_B_HIGH[4] - RKCK_B_LOW[4],
    RKCK_B_HIGH[5] - RKCK_B_LOW[5],
)


class SchemeCashKarp(SchemeBaseExplicitAdaptive):
    """
    The adaptive Cash-Karp embedded 5(4) Runge-Kutta pair.

    Cash-Karp advances with a fifth-order explicit method and estimates the
    local error with an embedded fourth-order formula. It is a classic adaptive
    explicit solver for smooth non-stiff problems.

    Further reading: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "error",
        "k2",
        "k3",
        "k4",
        "k5",
        "k6",
        "stage",
        "trial",
    )

    descriptor = SchemeDescriptor("RKCK", "Cash Karp")
    tableau = RKCK_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        regulator: Regulator | None = None,
    ) -> None:
        super().__init__(derivative, workbench, regulator)

    def initialise_buffers(self) -> None:
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.error, self.k2, self.k3, self.k4, self.k5, self.k6 = workspace.allocate_translation_buffers(7)

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

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        combine4 = workspace.combine4
        combine5 = workspace.combine5
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval
        controller = self._controller
        ratio = self._ratio
        assert controller is not None
        assert ratio is not None
        stage = self.stage
        trial_buffer = self.trial
        error_buffer = self.error
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4
        k5 = self.k5
        k6 = self.k6
        dt = interval.step if interval.step <= remaining else remaining
        proposed_dt = dt
        rejection_count = 0
        derivative(interval, state, k1)

        while True:
            trial = scale(trial_buffer, dt * RKCK_A[1][0], k1)
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt / 5.0), stage, k2)

            trial = combine2(
                trial_buffer,
                dt * RKCK_A[2][0],
                k1,
                dt * RKCK_A[2][1],
                k2,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, 3.0 * dt / 10.0), stage, k3)

            trial = combine3(
                trial_buffer,
                dt * RKCK_A[3][0],
                k1,
                dt * RKCK_A[3][1],
                k2,
                dt * RKCK_A[3][2],
                k3,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, 3.0 * dt / 5.0), stage, k4)

            trial = combine4(
                trial_buffer,
                dt * RKCK_A[4][0],
                k1,
                dt * RKCK_A[4][1],
                k2,
                dt * RKCK_A[4][2],
                k3,
                dt * RKCK_A[4][3],
                k4,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt), stage, k5)

            trial = combine5(
                trial_buffer,
                dt * RKCK_A[5][0],
                k1,
                dt * RKCK_A[5][1],
                k2,
                dt * RKCK_A[5][2],
                k3,
                dt * RKCK_A[5][3],
                k4,
                dt * RKCK_A[5][4],
                k5,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, 7.0 * dt / 8.0), stage, k6)

            delta_high = combine4(
                trial_buffer,
                dt * RKCK_B_HIGH_NZ[0],
                k1,
                dt * RKCK_B_HIGH_NZ[1],
                k3,
                dt * RKCK_B_HIGH_NZ[2],
                k4,
                dt * RKCK_B_HIGH_NZ[3],
                k6,
            )
            error = combine5(
                error_buffer,
                dt * RKCK_B_ERR_NZ[0],
                k1,
                dt * RKCK_B_ERR_NZ[1],
                k3,
                dt * RKCK_B_ERR_NZ[2],
                k4,
                dt * RKCK_B_ERR_NZ[3],
                k5,
                dt * RKCK_B_ERR_NZ[4],
                k6,
            )
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = ratio(error_norm, delta_high_norm)

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "RKCK")

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

__all__ = ["RKCK_TABLEAU", "SchemeCashKarp"]

















