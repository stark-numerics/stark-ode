from __future__ import annotations

from stark.schemes.support.derivative import SchemeDerivative
from stark.block import Block
from stark.contracts import Derivative, IntervalLike, Resolvent, State, Allocator
from stark.schemes.support.executor import SchemeExecutor
from stark.executor.adaptivity import ExecutorAdaptivity
from stark.contracts.errors import StarkErrorRecoverable
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    SchemeStepControl,
    initialise_adaptive_runtime,
    refresh_adaptive_call,
    with_adaptive_runtime_methods,
)
from stark.schemes.support.implicit import (
    initialise_implicit_support,
    with_implicit_workspace_methods,
)
from stark.schemes.support.specialist import SchemeSpecialist
from stark.schemes.support.stage_problem import SchemeStageProblem


@with_adaptive_runtime_methods
@with_implicit_workspace_methods
class SchemeBDF2:
    """Adaptive second-order backward differentiation formula.

    Algorithm sketch for one trial step of size h:

        1. Use a backward-Euler startup solve when no stable BDF2 history exists.
        2. Otherwise build the BDF2 shifted residual from the previous accepted
           displacement and current step ratio.
        3. Solve the one-block implicit displacement problem with the configured
           resolvent.
        4. Compare the BDF2 displacement with a backward-Euler companion estimate.
        5. Accept/reject through the adaptive controller.
    """

    step_control: SchemeStepControl

    __slots__ = (
        "step_control", "block_allocator", "call_pure", "derivative", "error",
        "has_history", "implicit", "known_shift", "known_shift_block", "low",
        "previous_delta", "previous_step", "redirect_call", "resolvent",
        "startup_rate", "trial_block", "workspace",
    )

    descriptor = SchemeDescriptor("BDF2", "Backward Differentiation Formula 2")

    def __init__(
        self,
        derivative: Derivative,
        allocator: Allocator,
        resolvent: Resolvent,
        adaptivity: ExecutorAdaptivity | None = None,
        *,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        del specialist
        self.resolvent = resolvent
        initialise_implicit_support(self, derivative, allocator)
        self.derivative = SchemeDerivative(derivative)

        workspace = self.workspace
        self.startup_rate = workspace.allocate_translation()
        self.trial_block = Block([workspace.allocate_translation()])
        self.previous_delta, self.low, self.error, self.known_shift = (
            workspace.allocate_translation_buffers(4)
        )
        self.known_shift_block = Block([self.known_shift])
        self.previous_step = 0.0
        self.has_history = False

        initialise_adaptive_runtime(self, adaptivity)
        self.call_pure = self.call_inline
        refresh_adaptive_call(self)

    @staticmethod
    def default_adaptivity() -> ExecutorAdaptivity:
        return ExecutorAdaptivity(error_exponent=0.5)

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    @classmethod
    def display_method(cls) -> str:
        return "\n".join(
            [
                f"{cls.descriptor.short_name} method ({cls.descriptor.full_name})",
                "startup: backward Euler",
                "main step: variable-step BDF2",
                "error estimate: backward-Euler companion",
            ]
        )

    @classmethod
    def display_resolvent_problem(cls) -> str:
        return "\n".join(
            [
                f"{cls.descriptor.short_name} resolvent problem ({cls.descriptor.full_name})",
                "",
                "Unknown step displacement:",
                "  Delta = x_{n+1} - x_n",
                "",
                "Startup solve:",
                "  Delta - h f(t_{n+1}, x_n + Delta) = 0",
                "",
                "BDF2 solve:",
                "  Delta - shift - alpha f(t_{n+1}, x_n + Delta) = 0",
            ]
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"{type(self).__name__}(short_name={self.short_name!r}, full_name={self.full_name!r})",
                self.display_method(),
            ]
        )

    def __str__(self) -> str:
        return self.display_method()

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def __call__(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.redirect_call(interval, state, executor)

    def call_specialized(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        return self.call_inline(interval, state, executor)

    def call_inline(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        del executor
        proposal = self.step_control.propose_step(interval)
        if proposal.remaining <= 0.0:
            self.step_control.record_stopped(interval)
            return 0.0

        workspace = self.workspace
        apply_delta = workspace.apply_delta
        scale = workspace.scale
        ratio_fn = self.step_control.ratio
        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        while True:
            step_ratio_is_safe = (
                self.has_history
                and self.previous_step > 0.0
                and 0.5 <= dt / self.previous_step <= 2.0
            )

            try:
                if step_ratio_is_safe:
                    # 2. Solve the variable-step BDF2 shifted residual.
                    delta_high, error_ratio = self.solve_bdf2_step(interval, state, dt, ratio_fn)
                else:
                    # 1. Use a backward-Euler startup solve.
                    delta_high, error_ratio = self.solve_startup_step(interval, state, dt, ratio_fn)
            except StarkErrorRecoverable:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, self.short_name)
                continue

            if error_ratio <= 1.0:
                break
            rejection_count += 1
            dt = self.step_control.rejected_step(dt, error_ratio, remaining, self.short_name)

        accepted_dt = dt
        apply_delta(delta_high, state)
        self.previous_delta = scale(1.0, delta_high, self.previous_delta)
        self.previous_step = accepted_dt
        self.has_history = True

        remaining_after = remaining - accepted_dt
        controller_next_dt = self.step_control.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        interval.step = 0.0 if remaining_after <= 0.0 else max(accepted_dt, controller_next_dt)
        report = self.step_control.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=interval.step,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return report.accepted_dt

    def solve_startup_step(self, interval: IntervalLike, state: State, dt: float, ratio_fn):
        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        trial_block = self.trial_block

        derivative(interval, state, self.startup_rate)

        # Startup problem: Delta - h f(t+h, y + Delta) = 0.
        self.known_shift = scale(0.0, self.known_shift, self.known_shift)
        self.known_shift_block[0] = self.known_shift
        trial_block[0] = scale(0.0, trial_block[0], trial_block[0])
        problem = SchemeStageProblem(
            derivative=derivative,
            interval=workspace.stage_at(interval, dt, dt),
            origin=state,
            rhs=self.known_shift_block,
            alpha=dt,
        )
        self.resolvent(problem, trial_block)

        delta_high = trial_block[0]
        delta_low = scale(dt, self.startup_rate, self.low)
        error = combine2(1.0, delta_high, -1.0, delta_low, self.error)
        error_ratio = ratio_fn(error.norm(), delta_high.norm())
        return delta_high, error_ratio

    def solve_bdf2_step(self, interval: IntervalLike, state: State, dt: float, ratio_fn):
        workspace = self.workspace
        scale = workspace.scale
        combine2 = workspace.combine2
        trial_block = self.trial_block

        step_ratio = dt / self.previous_step
        alpha0 = (2.0 * step_ratio + 1.0) / (step_ratio + 1.0)
        alpha2 = (step_ratio * step_ratio) / (step_ratio + 1.0)
        beta = (step_ratio * step_ratio) / (2.0 * step_ratio + 1.0)
        alpha = dt * (step_ratio + 1.0) / (2.0 * step_ratio + 1.0)

        self.known_shift = scale(beta, self.previous_delta, self.known_shift)
        self.known_shift_block[0] = self.known_shift
        trial_block[0] = scale(0.0, trial_block[0], trial_block[0])
        problem = SchemeStageProblem(
            derivative=self.derivative,
            interval=workspace.stage_at(interval, dt, dt),
            origin=state,
            rhs=self.known_shift_block,
            alpha=alpha,
        )
        self.resolvent(problem, trial_block)

        delta_high = trial_block[0]
        delta_low = combine2(alpha0, delta_high, -alpha2, self.previous_delta, self.low)
        error = combine2(1.0, delta_high, -1.0, delta_low, self.error)
        error_ratio = ratio_fn(error.norm(), delta_high.norm())
        return delta_high, error_ratio


__all__ = ["SchemeBDF2"]
