from __future__ import annotations

from stark.algebraist import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.monitor import MonitorStep
from stark.schemes.base import SchemeBaseExplicitAdaptive
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.support.adaptive import ReportAdaptiveAdvance
from stark.schemes.tableau import ButcherTableau


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


class SchemeBogackiShampine(SchemeBaseExplicitAdaptive):
    """The adaptive Bogacki-Shampine embedded 3(2) Runge-Kutta pair.

    This method advances with a third-order explicit Runge-Kutta formula while
    estimating error with an embedded second-order formula. That makes it a
    light adaptive method for non-stiff problems where a cheap error estimate is
    more valuable than very high order.

    Further reading:
    https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method
    """

    __slots__ = (
        "bound_apply_delta",
        "bound_stage_interval",
        "combine_error",
        "combine_solution",
        "combine_stage2",
        "combine_stage3",
        "combine_stage4",
        "error",
        "k2",
        "k3",
        "k4",
        "pure_advance",
        "stage",
        "trial",
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
        self.pure_advance = self.advance_generic
        super().__init__(derivative, workbench, regulator)

        # BS23 owns the advance algorithm selection. The adaptive base still
        # owns executor/monitor assignment during the transition.
        self.pure_advance = self.advance_generic

        if algebraist is not None:
            self.bind_algebraist_path(algebraist)

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

    def bind_and_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        self.assign_executor(executor)
        return self.redirect_call(interval, state, executor)

    def pure_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor
        return self.pure_advance(interval, state).accepted_dt

    def monitored_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        report = self.pure_advance(interval, state)
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

        return report.accepted_dt

    def advance_body(self, interval: IntervalLike, state: State) -> None:
        """Compatibility bridge for the transitional adaptive base."""

        self.pure_advance(interval, state)

    def initialise_buffers(self) -> None:
        workspace = self.workspace

        self.stage = workspace.allocate_state_buffer()
        self.trial, self.error, self.k2, self.k3, self.k4 = (
            workspace.allocate_translation_buffers(5)
        )

        self.combine_stage2 = None
        self.combine_stage3 = None
        self.combine_stage4 = None
        self.combine_solution = None
        self.combine_error = None

        self.bound_apply_delta = workspace.apply_delta
        self.bound_stage_interval = workspace.stage_interval

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
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
        self.pure_advance = self.advance_algebraist

    def set_apply_delta_safety(self, enabled: bool) -> None:
        super().set_apply_delta_safety(enabled)
        self.bound_apply_delta = self.workspace.apply_delta

    def advance_generic(
        self,
        interval: IntervalLike,
        state: State,
    ) -> ReportAdaptiveAdvance:
        proposal = self.adaptive.propose_step(interval)

        if proposal.remaining <= 0.0:
            self.adaptive.record_stopped(interval)
            return self.adaptive.report()

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine3 = workspace.combine3
        combine4 = workspace.combine4
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

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        derivative(interval, state, k1)

        while True:
            half_dt = 0.5 * dt
            three_quarter_dt = 0.75 * dt

            trial = scale(trial_buffer, half_dt, k1)
            trial(state, stage)
            derivative(stage_interval(interval, dt, half_dt), stage, k2)

            trial = scale(trial_buffer, three_quarter_dt, k2)
            trial(state, stage)
            derivative(stage_interval(interval, dt, three_quarter_dt), stage, k3)

            trial = combine3(
                trial_buffer,
                dt * BS23_A_STAGE4[0],
                k1,
                dt * BS23_A_STAGE4[1],
                k2,
                dt * BS23_A_STAGE4[2],
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
                dt * BS23_ERROR_WEIGHTS[0],
                k1,
                dt * BS23_ERROR_WEIGHTS[1],
                k2,
                dt * BS23_ERROR_WEIGHTS[2],
                k3,
                dt * BS23_ERROR_WEIGHTS[3],
                k4,
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

        return self.adaptive.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )

    def advance_algebraist(
        self,
        interval: IntervalLike,
        state: State,
    ) -> ReportAdaptiveAdvance:
        proposal = self.adaptive.propose_step(interval)

        if proposal.remaining <= 0.0:
            self.adaptive.record_stopped(interval)
            return self.adaptive.report()

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

            combine_stage2(stage, state, dt, k1)
            derivative(stage_interval(interval, dt, half_dt), stage, k2)

            combine_stage3(stage, state, dt, k2)
            derivative(stage_interval(interval, dt, three_quarter_dt), stage, k3)

            combine_stage4(stage, state, dt, k1, k2, k3)
            derivative(stage_interval(interval, dt, dt), stage, k4)

            delta_high = combine_solution(trial_buffer, dt, k1, k2, k3)
            error = combine_error(error_buffer, dt, k1, k2, k3, k4)
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

        return self.adaptive.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )


__all__ = ["BS23_TABLEAU", "SchemeBogackiShampine"]