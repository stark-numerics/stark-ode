from __future__ import annotations

from stark.algebraist import Algebraist
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.execution.executor import Executor
from stark.schemes.base import SchemeBaseExplicitFixed
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau


MIDPOINT_TABLEAU = ButcherTableau(
    c=(0.0, 0.5),
    a=((), (0.5,)),
    b=(0.0, 1.0),
    order=2,
)

MIDPOINT_B = MIDPOINT_TABLEAU.b


class SchemeMidpoint(SchemeBaseExplicitFixed):
    """The explicit midpoint two-stage second-order Runge-Kutta method.

    This method samples the derivative at the midpoint predicted by an Euler
    half-step and then advances using that midpoint slope. It is one of the
    standard second-order explicit schemes.

    Further reading: https://en.wikipedia.org/wiki/Midpoint_method
    """

    __slots__ = (
        "advance_state",
        "combine_stage2",
        "k2",
        "pure_call",
        "redirect_call",
        "stage",
        "trial",
    )

    descriptor = SchemeDescriptor("Midpoint", "Explicit Midpoint")
    tableau = MIDPOINT_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        algebraist: Algebraist | None = None,
    ) -> None:
        self.advance_state = None
        self.combine_stage2 = None

        super().__init__(derivative, workbench)

        self.pure_call = self.generic_call
        self.redirect_call = self.pure_call

        if algebraist is not None:
            self.bind_algebraist_path(algebraist)

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
        self.trial, self.k2 = workspace.allocate_translation_buffers(2)

    def bind_algebraist_path(self, algebraist: Algebraist) -> None:
        calls = algebraist.bind_explicit_scheme(self.tableau)
        self.combine_stage2 = calls.stages[1]
        self.advance_state = calls.solution_state
        self.pure_call = self.algebraist_call
        self.redirect_call = self.pure_call

    def generic_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval

        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2

        dt = interval.step if interval.step <= remaining else remaining
        half_dt = 0.5 * dt

        derivative(interval, state, k1)

        trial = scale(trial_buffer, half_dt, k1)
        trial(state, stage)
        derivative(stage_interval(interval, dt, half_dt), stage, k2)

        delta = combine2(
            trial_buffer,
            dt * MIDPOINT_B[0],
            k1,
            dt * MIDPOINT_B[1],
            k2,
        )
        apply_delta(delta, state)

        return dt

    def algebraist_call(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        half_dt = 0.5 * dt

        stage = self.stage
        k1 = self.k1
        k2 = self.k2
        derivative = self.derivative
        stage_interval = self.workspace.stage_interval
        combine_stage2 = self.combine_stage2
        advance_state = self.advance_state

        derivative(interval, state, k1)

        combine_stage2(stage, state, dt, k1)
        derivative(stage_interval(interval, dt, half_dt), stage, k2)

        advance_state(state, state, dt, k2)
        return dt


__all__ = ["MIDPOINT_TABLEAU", "SchemeMidpoint"]