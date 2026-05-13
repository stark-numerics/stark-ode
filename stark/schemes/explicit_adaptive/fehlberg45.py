from __future__ import annotations

from stark.algebraist import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.monitor import MonitorStep
from stark.schemes.base import SchemeBaseExplicitAdaptive
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


RKF45_TABLEAU = ButcherTableau(
    c=(0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0),
    a=(
        (),
        (1.0 / 4.0,),
        (3.0 / 32.0, 9.0 / 32.0),
        (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0),
        (439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0),
        (-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0),
    ),
    b=(16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0),
    order=5,
    b_embedded=(25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0),
    embedded_order=4,
)

RKF45_A = RKF45_TABLEAU.a
RKF45_B_HIGH = RKF45_TABLEAU.b
RKF45_B_LOW = RKF45_TABLEAU.b_embedded
assert RKF45_B_LOW is not None

RKF45_B_HIGH_NZ = (
    RKF45_B_HIGH[0],
    RKF45_B_HIGH[2],
    RKF45_B_HIGH[3],
    RKF45_B_HIGH[4],
    RKF45_B_HIGH[5],
)
RKF45_B_ERR_NZ = (
    RKF45_B_HIGH[0] - RKF45_B_LOW[0],
    RKF45_B_HIGH[2] - RKF45_B_LOW[2],
    RKF45_B_HIGH[3] - RKF45_B_LOW[3],
    RKF45_B_HIGH[4] - RKF45_B_LOW[4],
    RKF45_B_HIGH[5] - RKF45_B_LOW[5],
)


class SchemeFehlberg45(SchemeBaseExplicitAdaptive):
    """The adaptive Runge-Kutta-Fehlberg embedded 5(4) pair.

    Fehlberg's method is one of the classic adaptive explicit Runge-Kutta
    constructions: a fifth-order update paired with a fourth-order estimate for
    step control.

    Further reading:
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
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

    descriptor = SchemeDescriptor("RKF45", "Fehlberg 4(5)")
    tableau = RKF45_TABLEAU

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
            trial = scale(trial_buffer, dt * RKF45_A[1][0], k1)
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt / 4.0), stage, k2)

            trial = combine2(
                trial_buffer,
                dt * RKF45_A[2][0],
                k1,
                dt * RKF45_A[2][1],
                k2,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, 3.0 * dt / 8.0), stage, k3)

            trial = combine3(
                trial_buffer,
                dt * RKF45_A[3][0],
                k1,
                dt * RKF45_A[3][1],
                k2,
                dt * RKF45_A[3][2],
                k3,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, 12.0 * dt / 13.0), stage, k4)

            trial = combine4(
                trial_buffer,
                dt * RKF45_A[4][0],
                k1,
                dt * RKF45_A[4][1],
                k2,
                dt * RKF45_A[4][2],
                k3,
                dt * RKF45_A[4][3],
                k4,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, dt), stage, k5)

            trial = combine5(
                trial_buffer,
                dt * RKF45_A[5][0],
                k1,
                dt * RKF45_A[5][1],
                k2,
                dt * RKF45_A[5][2],
                k3,
                dt * RKF45_A[5][3],
                k4,
                dt * RKF45_A[5][4],
                k5,
            )
            trial(state, stage)
            derivative(stage_interval(interval, dt, 0.5 * dt), stage, k6)

            delta_high = combine5(
                trial_buffer,
                dt * RKF45_B_HIGH_NZ[0],
                k1,
                dt * RKF45_B_HIGH_NZ[1],
                k3,
                dt * RKF45_B_HIGH_NZ[2],
                k4,
                dt * RKF45_B_HIGH_NZ[3],
                k5,
                dt * RKF45_B_HIGH_NZ[4],
                k6,
            )

            error = combine5(
                error_buffer,
                dt * RKF45_B_ERR_NZ[0],
                k1,
                dt * RKF45_B_ERR_NZ[1],
                k3,
                dt * RKF45_B_ERR_NZ[2],
                k4,
                dt * RKF45_B_ERR_NZ[3],
                k5,
                dt * RKF45_B_ERR_NZ[4],
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
            derivative(stage_interval(interval, dt, dt / 4.0), stage, k2)

            combine_stage3(stage, state, dt, k1, k2)
            derivative(stage_interval(interval, dt, 3.0 * dt / 8.0), stage, k3)

            combine_stage4(stage, state, dt, k1, k2, k3)
            derivative(stage_interval(interval, dt, 12.0 * dt / 13.0), stage, k4)

            combine_stage5(stage, state, dt, k1, k2, k3, k4)
            derivative(stage_interval(interval, dt, dt), stage, k5)

            combine_stage6(stage, state, dt, k1, k2, k3, k4, k5)
            derivative(stage_interval(interval, dt, 0.5 * dt), stage, k6)

            delta_high = combine_solution(trial_buffer, dt, k1, k3, k4, k5, k6)
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


__all__ = ["RKF45_TABLEAU", "SchemeFehlberg45"]
