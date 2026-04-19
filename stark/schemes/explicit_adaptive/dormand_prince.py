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


RKDP_TABLEAU = ButcherTableau(
    c=(0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0),
    a=(
        (),
        (1.0 / 5.0,),
        (3.0 / 40.0, 9.0 / 40.0),
        (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
        (
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ),
        (
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ),
        (),
    ),
    b=(35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0),
    order=5,
    b_embedded=(
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ),
    embedded_order=4,
)
RKDP_A = RKDP_TABLEAU.a
RKDP_B_HIGH = RKDP_TABLEAU.b
RKDP_B_LOW = RKDP_TABLEAU.b_embedded
assert RKDP_B_LOW is not None
RKDP_B_HIGH_NZ = (
    RKDP_B_HIGH[0],
    RKDP_B_HIGH[2],
    RKDP_B_HIGH[3],
    RKDP_B_HIGH[4],
    RKDP_B_HIGH[5],
)
RKDP_B_ERR_NZ = (
    RKDP_B_HIGH[0] - RKDP_B_LOW[0],
    RKDP_B_HIGH[2] - RKDP_B_LOW[2],
    RKDP_B_HIGH[3] - RKDP_B_LOW[3],
    RKDP_B_HIGH[4] - RKDP_B_LOW[4],
    RKDP_B_HIGH[5] - RKDP_B_LOW[5],
    RKDP_B_HIGH[6] - RKDP_B_LOW[6],
)


class SchemeDormandPrince(SchemeBaseExplicitAdaptive):
    """
    The adaptive Dormand-Prince embedded 5(4) Runge-Kutta pair.

    This is the RK45 family most users meet first: a fifth-order explicit
    method with a fourth-order embedded error estimate. It is a strong default
    choice for smooth non-stiff problems and is the basis of many classic
    adaptive ODE drivers.

    Further reading: https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
    """

    __slots__ = (
        "error",
        "k2",
        "k3",
        "k4",
        "k5",
        "k6",
        "k7",
        "stage",
        "trial",
    )

    descriptor = SchemeDescriptor("RKDP", "Dormand-Prince")
    tableau = RKDP_TABLEAU

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
        self.trial, self.error, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7 = workspace.allocate_translation_buffers(8)

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
        combine6 = workspace.combine6
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval
        controller = self._controller
        bound = self._bound
        assert controller is not None
        assert bound is not None
        stage = self.stage
        trial_buffer = self.trial
        error_buffer = self.error
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4
        k5 = self.k5
        k6 = self.k6
        k7 = self.k7
        dt = interval.step if interval.step <= remaining else remaining
        proposed_dt = dt
        rejection_count = 0
        derivative(interval, state, k1)

        while True:
            trial = scale(trial_buffer, dt * RKDP_A[1][0], k1)
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt / 5.0), stage, k2)

            trial = combine2(
                trial_buffer,
                dt * RKDP_A[2][0],
                k1,
                dt * RKDP_A[2][1],
                k2,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, 3.0 * dt / 10.0), stage, k3)

            trial = combine3(
                trial_buffer,
                dt * RKDP_A[3][0],
                k1,
                dt * RKDP_A[3][1],
                k2,
                dt * RKDP_A[3][2],
                k3,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, 4.0 * dt / 5.0), stage, k4)

            trial = combine4(
                trial_buffer,
                dt * RKDP_A[4][0],
                k1,
                dt * RKDP_A[4][1],
                k2,
                dt * RKDP_A[4][2],
                k3,
                dt * RKDP_A[4][3],
                k4,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, 8.0 * dt / 9.0), stage, k5)

            trial = combine5(
                trial_buffer,
                dt * RKDP_A[5][0],
                k1,
                dt * RKDP_A[5][1],
                k2,
                dt * RKDP_A[5][2],
                k3,
                dt * RKDP_A[5][3],
                k4,
                dt * RKDP_A[5][4],
                k5,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt), stage, k6)

            delta_high = combine5(
                trial_buffer,
                dt * RKDP_B_HIGH_NZ[0],
                k1,
                dt * RKDP_B_HIGH_NZ[1],
                k3,
                dt * RKDP_B_HIGH_NZ[2],
                k4,
                dt * RKDP_B_HIGH_NZ[3],
                k5,
                dt * RKDP_B_HIGH_NZ[4],
                k6,
            )
            delta_high(state, stage)
            derivative(stage_interval(interval, dt, dt), stage, k7)

            error = combine6(
                error_buffer,
                dt * RKDP_B_ERR_NZ[0],
                k1,
                dt * RKDP_B_ERR_NZ[1],
                k3,
                dt * RKDP_B_ERR_NZ[2],
                k4,
                dt * RKDP_B_ERR_NZ[3],
                k5,
                dt * RKDP_B_ERR_NZ[4],
                k6,
                dt * RKDP_B_ERR_NZ[5],
                k7,
            )
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = error_norm / bound(delta_high_norm)

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "RKDP")

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


__all__ = ["RKDP_TABLEAU", "SchemeDormandPrince"]

















