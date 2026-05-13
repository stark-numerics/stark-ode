from __future__ import annotations

from stark.algebraist import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.monitor import MonitorStep
from stark.schemes.base import SchemeBaseExplicitAdaptive
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


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
    """The adaptive Dormand-Prince embedded 5(4) Runge-Kutta pair.

    This is the RK45 family most users meet first: a fifth-order explicit method
    with a fourth-order embedded error estimate. It is a strong default choice
    for smooth non-stiff problems and is the basis of many classic adaptive ODE
    drivers.

    Further reading: https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
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
        self.trial, self.error, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7 = (
            workspace.allocate_translation_buffers(8)
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
        error = calls.error_delta_call

        if error is None:
            raise ValueError("Dormand-Prince requires an embedded error combination.")

        if len(calls.stage_state_calls) < 6:
            raise ValueError("Dormand-Prince requires six tableau stage combinations.")

        self.combine_stage2 = calls.stage_state_calls[1]
        self.combine_stage3 = calls.stage_state_calls[2]
        self.combine_stage4 = calls.stage_state_calls[3]
        self.combine_stage5 = calls.stage_state_calls[4]
        self.combine_stage6 = calls.stage_state_calls[5]
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
        combine6 = workspace.combine6
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval
        bound = self.adaptive.bound

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

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
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

            error_ratio = error.norm() / bound(delta_high.norm())

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
        bound = self.adaptive.bound

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
            derivative(stage_interval(interval, dt, 4.0 * dt / 5.0), stage, k4)

            combine_stage5(stage, state, dt, k1, k2, k3, k4)
            derivative(stage_interval(interval, dt, 8.0 * dt / 9.0), stage, k5)

            combine_stage6(stage, state, dt, k1, k2, k3, k4, k5)
            derivative(stage_interval(interval, dt, dt), stage, k6)

            delta_high = combine_solution(trial_buffer, dt, k1, k3, k4, k5, k6)
            delta_high(state, stage)
            derivative(stage_interval(interval, dt, dt), stage, k7)

            error = combine_error(error_buffer, dt, k1, k3, k4, k5, k6, k7)
            error_ratio = error.norm() / bound(delta_high.norm())

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


__all__ = ["RKDP_TABLEAU", "SchemeDormandPrince"]