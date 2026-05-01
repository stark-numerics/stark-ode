from __future__ import annotations

from stark.algebraist import Algebraist
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


TSIT5_TABLEAU = ButcherTableau(
    c=(0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0),
    a=(
        (),
        (0.161,),
        (-0.008480655492357, 0.335480655492357),
        (2.897153057105494, -6.359448489975075, 4.362295432869581),
        (
            5.325864828439259,
            -11.74888356406283,
            7.495539342889836,
            -0.09249506636175525,
        ),
        (
            5.86145544294642,
            -12.92096931784711,
            8.159367898576159,
            -0.071584973281401,
            -0.02826905039406838,
        ),
        (
            0.09646076681806523,
            0.01,
            0.4798896504144996,
            1.379008574103742,
            -3.290069515436081,
            2.324710524099774,
        ),
    ),
    b=(
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0.0,
    ),
    order=5,
    b_embedded=(
        0.09468075576583923,
        0.009183565540343,
        0.4877705284247616,
        1.234297566930479,
        -2.707712349983526,
        1.866628418170587,
        1.0 / 66.0,
    ),
    embedded_order=4,
)
TSIT5_A = TSIT5_TABLEAU.a
TSIT5_B_HIGH = TSIT5_TABLEAU.b
TSIT5_B_LOW = TSIT5_TABLEAU.b_embedded
assert TSIT5_B_LOW is not None
TSIT5_B_ERR = tuple(high - low for high, low in zip(TSIT5_B_HIGH, TSIT5_B_LOW, strict=True))


class SchemeTsitouras5(SchemeBaseExplicitAdaptive):
    """
    The adaptive Tsitouras embedded 5(4) Runge-Kutta pair.

    Tsitouras 5 is a modern fifth-order explicit adaptive method designed to
    offer strong practical performance with a carefully tuned tableau and
    embedded fourth-order error estimate.

    Further reading: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
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
        "combine_stage2",
        "combine_stage3",
        "combine_stage4",
        "combine_stage5",
        "combine_stage6",
        "combine_stage7",
        "combine_solution",
        "combine_error",
        "bound_apply_delta",
        "bound_stage_interval",
    )

    descriptor = SchemeDescriptor("TSIT5", "Tsitouras 5")
    tableau = TSIT5_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        regulator: Regulator | None = None,
        algebraist: Algebraist | None = None,
    ) -> None:
        super().__init__(derivative, workbench, regulator)
        if algebraist is not None:
            self.bind_algebraist_path(algebraist)

    def initialise_buffers(self) -> None:
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.error, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7 = workspace.allocate_translation_buffers(8)
        self.combine_stage2 = None
        self.combine_stage3 = None
        self.combine_stage4 = None
        self.combine_stage5 = None
        self.combine_stage6 = None
        self.combine_stage7 = None
        self.combine_solution = None
        self.combine_error = None
        self.bound_apply_delta = workspace.apply_delta
        self.bound_stage_interval = workspace.stage_interval

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
        error = calls.error
        if error is None:
            raise ValueError("Tsitouras5 requires an embedded error combination.")
        if len(calls.stages) < 7:
            raise ValueError("Tsitouras5 requires seven tableau stage combinations.")
        self.combine_stage2 = calls.stages[1]
        self.combine_stage3 = calls.stages[2]
        self.combine_stage4 = calls.stages[3]
        self.combine_stage5 = calls.stages[4]
        self.combine_stage6 = calls.stages[5]
        self.combine_stage7 = calls.stages[6]
        self.combine_solution = calls.solution
        self.combine_error = error
        self.bind_advance_body(self.advance_body_algebraist)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        super().set_apply_delta_safety(enabled)
        self.bound_apply_delta = self.workspace.apply_delta

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
        combine7 = workspace.combine7
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
            trial = scale(trial_buffer, dt * TSIT5_A[1][0], k1)
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[1]), stage, k2)

            trial = combine2(
                trial_buffer,
                dt * TSIT5_A[2][0],
                k1,
                dt * TSIT5_A[2][1],
                k2,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[2]), stage, k3)

            trial = combine3(
                trial_buffer,
                dt * TSIT5_A[3][0],
                k1,
                dt * TSIT5_A[3][1],
                k2,
                dt * TSIT5_A[3][2],
                k3,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[3]), stage, k4)

            trial = combine4(
                trial_buffer,
                dt * TSIT5_A[4][0],
                k1,
                dt * TSIT5_A[4][1],
                k2,
                dt * TSIT5_A[4][2],
                k3,
                dt * TSIT5_A[4][3],
                k4,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[4]), stage, k5)

            trial = combine5(
                trial_buffer,
                dt * TSIT5_A[5][0],
                k1,
                dt * TSIT5_A[5][1],
                k2,
                dt * TSIT5_A[5][2],
                k3,
                dt * TSIT5_A[5][3],
                k4,
                dt * TSIT5_A[5][4],
                k5,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[5]), stage, k6)

            trial = combine6(
                trial_buffer,
                dt * TSIT5_A[6][0],
                k1,
                dt * TSIT5_A[6][1],
                k2,
                dt * TSIT5_A[6][2],
                k3,
                dt * TSIT5_A[6][3],
                k4,
                dt * TSIT5_A[6][4],
                k5,
                dt * TSIT5_A[6][5],
                k6,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[6]), stage, k7)

            delta_high = combine6(
                trial_buffer,
                dt * TSIT5_B_HIGH[0],
                k1,
                dt * TSIT5_B_HIGH[1],
                k2,
                dt * TSIT5_B_HIGH[2],
                k3,
                dt * TSIT5_B_HIGH[3],
                k4,
                dt * TSIT5_B_HIGH[4],
                k5,
                dt * TSIT5_B_HIGH[5],
                k6,
            )
            error = combine7(
                error_buffer,
                dt * TSIT5_B_ERR[0],
                k1,
                dt * TSIT5_B_ERR[1],
                k2,
                dt * TSIT5_B_ERR[2],
                k3,
                dt * TSIT5_B_ERR[3],
                k4,
                dt * TSIT5_B_ERR[4],
                k5,
                dt * TSIT5_B_ERR[5],
                k6,
                dt * TSIT5_B_ERR[6],
                k7,
            )
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = error_norm / bound(delta_high_norm)

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "TSIT5")

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

    def advance_body_algebraist(self, interval: IntervalLike, state: State) -> None:
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

        derivative = self.derivative
        apply_delta = self.bound_apply_delta
        stage_interval = self.bound_stage_interval
        controller = self._controller
        bound = self._bound
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
        combine_stage2 = self.combine_stage2
        combine_stage3 = self.combine_stage3
        combine_stage4 = self.combine_stage4
        combine_stage5 = self.combine_stage5
        combine_stage6 = self.combine_stage6
        combine_stage7 = self.combine_stage7
        combine_solution = self.combine_solution
        combine_error = self.combine_error
        dt = interval.step if interval.step <= remaining else remaining
        proposed_dt = dt
        rejection_count = 0
        derivative(interval, state, k1)

        while True:
            combine_stage2(stage, state, dt, k1)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[1]), stage, k2)

            combine_stage3(stage, state, dt, k1, k2)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[2]), stage, k3)

            combine_stage4(stage, state, dt, k1, k2, k3)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[3]), stage, k4)

            combine_stage5(stage, state, dt, k1, k2, k3, k4)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[4]), stage, k5)

            combine_stage6(stage, state, dt, k1, k2, k3, k4, k5)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[5]), stage, k6)

            combine_stage7(stage, state, dt, k1, k2, k3, k4, k5, k6)
            derivative(stage_interval(interval, dt, dt * TSIT5_TABLEAU.c[6]), stage, k7)

            delta_high = combine_solution(trial_buffer, dt, k1, k2, k3, k4, k5, k6)
            error = combine_error(error_buffer, dt, k1, k2, k3, k4, k5, k6, k7)
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = error_norm / bound(delta_high_norm)

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "TSIT5")

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


__all__ = ["TSIT5_TABLEAU", "SchemeTsitouras5"]

















