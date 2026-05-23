from __future__ import annotations

from stark.algebraist.classic import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    SchemeStepControl,
    initialise_adaptive_runtime,
    initialise_explicit_support,
    refresh_adaptive_call,
    unbound_scheme_call,
    with_adaptive_runtime_methods,
    with_explicit_workspace_methods,
    with_scheme_display,
)
from stark.schemes.support.tableau import ButcherTableau


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

BS23_A_STAGE4 = BS23_A[3]
BS23_ERROR_WEIGHTS = tuple(
    high - low for high, low in zip(BS23_B_HIGH, BS23_B_LOW, strict=True)
)


@with_scheme_display
@with_adaptive_runtime_methods
@with_explicit_workspace_methods
class SchemeBogackiShampine:
    """The adaptive Bogacki-Shampine embedded 3(2) Runge-Kutta pair.

    This method advances with a third-order explicit Runge-Kutta formula while
    estimating error with an embedded second-order formula. That makes it a
    light adaptive method for non-stiff problems where a cheap error estimate
    is more valuable than very high order.

    Further reading: https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method
    """

    # Assigned by initialise_adaptive_runtime from stark.schemes.support.
    step_control: SchemeStepControl

    __slots__ = (
        "step_control",
        "bound_apply_delta",
        "bound_stage_interval",
        "call_pure",
        "combine_error",
        "combine_solution",
        "combine_stage2",
        "combine_stage3",
        "combine_stage4",
        "derivative",
        "error",
        "explicit",
        "k1",
        "k2",
        "k3",
        "k4",
        "redirect_call",
        "stage",
        "trial",
        "workspace",
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
        initialise_explicit_support(self, derivative, workbench)
        initialise_adaptive_runtime(self, regulator)
        self.initialise_buffers()

        self.call_pure = self.call_inline
        refresh_adaptive_call(self)

        if algebraist is not None:
            self.use_specialists(algebraist)

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

    def initialise_buffers(self) -> None:
        workspace = self.workspace

        self.stage = workspace.allocate_state_buffer()
        self.trial, self.error, self.k2, self.k3, self.k4 = (
            workspace.allocate_translation_buffers(5)
        )

        self.combine_stage2 = unbound_scheme_call
        self.combine_stage3 = unbound_scheme_call
        self.combine_stage4 = unbound_scheme_call
        self.combine_solution = unbound_scheme_call
        self.combine_error = unbound_scheme_call

        self.bound_apply_delta = workspace.apply_delta
        self.bound_stage_interval = workspace.stage_interval

    def use_specialists(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
        error = calls.require_error_delta_call(type(self).__name__)

        self.combine_stage2 = calls.require_stage_state_call(1, type(self).__name__)
        self.combine_stage3 = calls.require_stage_state_call(2, type(self).__name__)
        self.combine_stage4 = calls.require_stage_state_call(3, type(self).__name__)
        self.combine_solution = calls.solution_delta_call
        self.combine_error = error

        self.call_pure = self.call_specialized
        refresh_adaptive_call(self)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.explicit.set_apply_delta_safety(enabled)
        self.bound_apply_delta = self.workspace.apply_delta

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

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine3 = workspace.combine3
        combine4 = workspace.combine4
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval
        ratio = self.step_control.ratio
        stage = self.stage
        trial_buffer = self.trial
        error_buffer = self.error

        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        derivative(interval, state, k1)

        while True:
            half_dt = 0.5 * dt
            three_quarter_dt = 0.75 * dt

            trial = scale(half_dt, k1, trial_buffer)
            trial(state, stage)
            derivative(stage_interval(interval, dt, half_dt), stage, k2)

            trial = scale(three_quarter_dt, k2, trial_buffer)
            trial(state, stage)
            derivative(stage_interval(interval, dt, three_quarter_dt), stage, k3)

            trial = combine3(
                dt * BS23_A_STAGE4[0],
                k1,
                dt * BS23_A_STAGE4[1],
                k2,
                dt * BS23_A_STAGE4[2],
                k3,
                trial_buffer,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt), stage, k4)

            delta_high = combine3(
                dt * BS23_B_HIGH[0],
                k1,
                dt * BS23_B_HIGH[1],
                k2,
                dt * BS23_B_HIGH[2],
                k3,
                trial_buffer,
            )

            error = combine4(
                dt * BS23_ERROR_WEIGHTS[0],
                k1,
                dt * BS23_ERROR_WEIGHTS[1],
                k2,
                dt * BS23_ERROR_WEIGHTS[2],
                k3,
                dt * BS23_ERROR_WEIGHTS[3],
                k4,
                error_buffer,
            )

            error_ratio = ratio(error.norm(), delta_high.norm())
            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = self.step_control.rejected_step(
                dt,
                error_ratio,
                remaining,
                self.short_name,
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

        derivative = self.derivative
        apply_delta = self.bound_apply_delta
        stage_interval = self.bound_stage_interval
        ratio = self.step_control.ratio
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

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        derivative(interval, state, k1)

        while True:
            half_dt = 0.5 * dt
            three_quarter_dt = 0.75 * dt

            combine_stage2(state, dt, k1, stage)
            derivative(stage_interval(interval, dt, half_dt), stage, k2)

            combine_stage3(state, dt, k2, stage)
            derivative(stage_interval(interval, dt, three_quarter_dt), stage, k3)

            combine_stage4(state, dt, k1, k2, k3, stage)
            derivative(stage_interval(interval, dt, dt), stage, k4)

            delta_high = combine_solution(dt, k1, k2, k3, trial_buffer)
            error = combine_error(dt, k1, k2, k3, k4, error_buffer)

            error_ratio = ratio(error.norm(), delta_high.norm())
            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = self.step_control.rejected_step(
                dt,
                error_ratio,
                remaining,
                self.short_name,
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


__all__ = ["BS23_TABLEAU", "SchemeBogackiShampine"]
