from __future__ import annotations

from stark.algebraist import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.monitor import MonitorStep
from stark.schemes.base import SchemeBaseExplicitAdaptive
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


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

RKCK_B_HIGH_NZ = (
    RKCK_B_HIGH[0],
    RKCK_B_HIGH[2],
    RKCK_B_HIGH[3],
    RKCK_B_HIGH[5],
)
RKCK_B_ERR_NZ = (
    RKCK_B_HIGH[0] - RKCK_B_LOW[0],
    RKCK_B_HIGH[2] - RKCK_B_LOW[2],
    RKCK_B_HIGH[3] - RKCK_B_LOW[3],
    RKCK_B_HIGH[4] - RKCK_B_LOW[4],
    RKCK_B_HIGH[5] - RKCK_B_LOW[5],
)


class SchemeCashKarp(SchemeBaseExplicitAdaptive):
    """The adaptive Cash-Karp embedded 5(4) Runge-Kutta pair.

    Cash-Karp advances with a fifth-order explicit method and estimates the
    local error with an embedded fourth-order formula. It is a classic adaptive
    explicit solver for smooth non-stiff problems.

    Further reading:
    https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "bound_apply_delta",
        "bound_stage_interval",
        "call_pure",
        "combine_error",
        "combine_solution",
        "combine_stage2",
        "combine_stage3",
        "combine_stage4",
        "combine_stage5",
        "combine_stage6",
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
        algebraist: Algebraist | None = None,
    ) -> None:
        super().__init__(derivative, workbench, regulator)

        self.call_pure = self.call_generic
        self.refresh_call()

        if algebraist is not None:
            self.bind_algebraist_path(algebraist)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def call_bind(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        self.assign_executor(executor)
        return self.redirect_call(interval, state, executor)

    def call_monitored(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        accepted_dt = self.call_pure(interval, state, executor)
        report = self.adaptive.report()
        monitor = self.adaptive.monitor

        if monitor is not None:
            monitor(
                MonitorStep(
                    scheme=self.short_name,
                    t_start=report.t_start,
                    t_end=report.t_end,
                    proposed_dt=report.proposed_dt,
                    accepted_dt=report.accepted_dt,
                    next_dt=report.next_dt,
                    error_ratio=report.error_ratio,
                    rejection_count=report.rejection_count,
                )
            )

        return accepted_dt

    def initialise_buffers(self) -> None:
        workspace = self.workspace

        self.stage = workspace.allocate_state_buffer()
        self.trial, self.error, self.k2, self.k3, self.k4, self.k5, self.k6 = (
            workspace.allocate_translation_buffers(7)
        )

        self.combine_stage2 = None
        self.combine_stage3 = None
        self.combine_stage4 = None
        self.combine_stage5 = None
        self.combine_stage6 = None
        self.combine_solution = None
        self.combine_error = None

        self.bound_apply_delta = workspace.apply_delta
        self.bound_stage_interval = workspace.stage_interval

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
        error = calls.require_error_delta_call(type(self).__name__)



        self.combine_stage2 = calls.require_stage_state_call(1, type(self).__name__)
        self.combine_stage3 = calls.require_stage_state_call(2, type(self).__name__)
        self.combine_stage4 = calls.require_stage_state_call(3, type(self).__name__)
        self.combine_stage5 = calls.require_stage_state_call(4, type(self).__name__)
        self.combine_stage6 = calls.require_stage_state_call(5, type(self).__name__)
        self.combine_solution = calls.solution_delta_call
        self.combine_error = error

        self.call_pure = self.call_algebraist
        self.refresh_call()

    def set_apply_delta_safety(self, enabled: bool) -> None:
        super().set_apply_delta_safety(enabled)
        self.bound_apply_delta = self.workspace.apply_delta

    def call_generic(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        proposal = self.adaptive.propose_step(interval)

        if proposal.remaining <= 0.0:
            self.adaptive.record_stopped(interval)
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        combine4 = workspace.combine4
        combine5 = workspace.combine5
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval
        ratio = self.adaptive.ratio

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

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
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

            error_ratio = ratio(error.norm(), delta_high.norm())

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = self.adaptive.rejected_step(
                dt,
                error_ratio,
                remaining,
                self.short_name,
            )

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = self.adaptive.accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )

        interval.step = next_dt
        apply_delta(delta_high, state)

        report = self.adaptive.record_accepted(
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

        proposal = self.adaptive.propose_step(interval)

        if proposal.remaining <= 0.0:
            self.adaptive.record_stopped(interval)
            return 0.0

        derivative = self.derivative
        apply_delta = self.bound_apply_delta
        stage_interval = self.bound_stage_interval
        ratio = self.adaptive.ratio

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

        combine_stage2 = self.combine_stage2
        combine_stage3 = self.combine_stage3
        combine_stage4 = self.combine_stage4
        combine_stage5 = self.combine_stage5
        combine_stage6 = self.combine_stage6
        combine_solution = self.combine_solution
        combine_error = self.combine_error

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        derivative(interval, state, k1)

        while True:
            combine_stage2(stage, state, dt, k1)
            derivative(stage_interval(interval, dt, dt / 5.0), stage, k2)

            combine_stage3(stage, state, dt, k1, k2)
            derivative(stage_interval(interval, dt, 3.0 * dt / 10.0), stage, k3)

            combine_stage4(stage, state, dt, k1, k2, k3)
            derivative(stage_interval(interval, dt, 3.0 * dt / 5.0), stage, k4)

            combine_stage5(stage, state, dt, k1, k2, k3, k4)
            derivative(stage_interval(interval, dt, dt), stage, k5)

            combine_stage6(stage, state, dt, k1, k2, k3, k4, k5)
            derivative(stage_interval(interval, dt, 7.0 * dt / 8.0), stage, k6)

            delta_high = combine_solution(trial_buffer, dt, k1, k3, k4, k6)
            error = combine_error(error_buffer, dt, k1, k3, k4, k5, k6)
            error_ratio = ratio(error.norm(), delta_high.norm())

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = self.adaptive.rejected_step(
                dt,
                error_ratio,
                remaining,
                self.short_name,
            )

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = self.adaptive.accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )

        interval.step = next_dt
        apply_delta(delta_high, state)

        report = self.adaptive.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return report.accepted_dt


__all__ = ["RKCK_TABLEAU", "SchemeCashKarp"]
