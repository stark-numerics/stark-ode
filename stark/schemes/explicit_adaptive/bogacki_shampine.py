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


BS23_TABLEAU = ButcherTableau(
    c=(0.0, 0.5, 0.75, 1.0),
    a=(
        (),
        (0.5,),
        (0.0, 0.75),
        (2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0),
    ),
    b=(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
    order=3,
    b_embedded=(7.0 / 24.0, 0.25, 1.0 / 3.0, 0.125),
    embedded_order=2,
)
BS23_A = BS23_TABLEAU.a
BS23_B_HIGH = BS23_TABLEAU.b
BS23_B_LOW = BS23_TABLEAU.b_embedded
assert BS23_B_LOW is not None


class SchemeBogackiShampine(SchemeBaseExplicitAdaptive):
    """
    The adaptive Bogacki-Shampine embedded 3(2) Runge-Kutta pair.

    This method advances with a third-order explicit Runge-Kutta formula while
    estimating error with an embedded second-order formula. That makes it a
    light adaptive method for non-stiff problems where a cheap error estimate
    is more valuable than very high order.

    Further reading: https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method
    """

    __slots__ = (
        "error",
        "k2",
        "k3",
        "k4",
        "stage",
        "trial",
        "combine_stage2",
        "combine_stage3",
        "combine_stage4",
        "combine_solution",
        "combine_error",
        "bound_apply_delta",
        "bound_stage_interval",
    )

    descriptor = SchemeDescriptor("BS23", "Bogacki-Shampine")
    tableau = BS23_TABLEAU

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

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=1.0 / 3.0)

    def initialise_buffers(self) -> None:
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.error, self.k2, self.k3, self.k4 = workspace.allocate_translation_buffers(5)
        self.combine_stage2 = None
        self.combine_stage3 = None
        self.combine_stage4 = None
        self.combine_solution = None
        self.combine_error = None
        self.bound_apply_delta = workspace.apply_delta
        self.bound_stage_interval = workspace.stage_interval

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau, self.workspace)
        error = calls.error
        if error is None:
            raise ValueError("Bogacki-Shampine requires an embedded error combination.")
        if len(calls.stages) < 4:
            raise ValueError("Bogacki-Shampine requires four tableau stage combinations.")
        self.combine_stage2 = calls.stages[1]
        self.combine_stage3 = calls.stages[2]
        self.combine_stage4 = calls.stages[3]
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
        combine3 = workspace.combine3
        combine4 = workspace.combine4
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
        dt = interval.step if interval.step <= remaining else remaining
        proposed_dt = dt
        rejection_count = 0
        derivative(interval, state, k1)
        while True:
            trial = scale(trial_buffer, 0.5 * dt, k1)
            trial(state, stage)
            derivative(stage_interval(interval, dt, 0.5 * dt), stage, k2)

            trial = scale(trial_buffer, 0.75 * dt, k2)
            trial(state, stage)
            derivative(stage_interval(interval, dt, 0.75 * dt), stage, k3)

            trial = combine3(
                trial_buffer,
                dt * BS23_A[3][0],
                k1,
                dt * BS23_A[3][1],
                k2,
                dt * BS23_A[3][2],
                k3,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt), stage, k4)

            delta_high = combine3(
                trial_buffer,
                dt * BS23_B_HIGH[0],
                k1,
                dt * BS23_B_HIGH[1],
                k2,
                dt * BS23_B_HIGH[2],
                k3,
            )
            error = combine4(
                error_buffer,
                dt * (BS23_B_HIGH[0] - BS23_B_LOW[0]),
                k1,
                dt * (BS23_B_HIGH[1] - BS23_B_LOW[1]),
                k2,
                dt * (BS23_B_HIGH[2] - BS23_B_LOW[2]),
                k3,
                dt * (BS23_B_HIGH[3] - BS23_B_LOW[3]),
                k4,
            )
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = ratio(error_norm, delta_high_norm)

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "BS23")

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
        ratio = self._ratio
        stage = self.stage
        trial_buffer = self.trial
        error_buffer = self.error
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4
        combine_stage2 = self.combine_stage2
        combine_stage3 = self.combine_stage3
        combine_stage4 = self.combine_stage4
        combine_solution = self.combine_solution
        combine_error = self.combine_error
        dt = interval.step if interval.step <= remaining else remaining
        proposed_dt = dt
        rejection_count = 0
        derivative(interval, state, k1)
        while True:
            combine_stage2(stage, state, dt, k1)
            derivative(stage_interval(interval, dt, 0.5 * dt), stage, k2)

            combine_stage3(stage, state, dt, k2)
            derivative(stage_interval(interval, dt, 0.75 * dt), stage, k3)

            combine_stage4(stage, state, dt, k1, k2, k3)
            derivative(stage_interval(interval, dt, dt), stage, k4)

            delta_high = combine_solution(trial_buffer, dt, k1, k2, k3)
            error = combine_error(error_buffer, dt, k1, k2, k3, k4)
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = ratio(error_norm, delta_high_norm)

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "BS23")

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


__all__ = ["BS23_TABLEAU", "SchemeBogackiShampine"]

















