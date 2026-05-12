from __future__ import annotations

from stark.accelerators.binding import BoundDerivative
from stark.auditor import Auditor
from stark.contracts import Block, Derivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.monitor import MonitorStep
from stark.resolvents.failure import ResolventError
from stark.schemes.base import SchemeBaseImplicitAdaptive
from stark.schemes.descriptor import SchemeDescriptor


class SchemeBDF2(SchemeBaseImplicitAdaptive):
    """Adaptive second-order backward differentiation formula.

    BDF2 is a two-step implicit method. After a first backward-Euler startup
    step, it advances with

        alpha0 x_{n+1} + alpha1 x_n + alpha2 x_{n-1}
        = dt f(t_{n+1}, x_{n+1}),

    using a variable-step ratio and a backward-Euler-style embedded estimate for
    step control.

    This implementation stores the previous accepted displacement rather than a
    full previous state. If the method has no usable history, or if the proposed
    step changes too sharply relative to the previous accepted step, it falls
    back to a backward-Euler startup solve so the history can settle before
    resuming the multistep update.

    The nonlinear solve still has the STARK-friendly shifted form

        delta - shift - alpha f(t_{n+1}, state + delta) = 0,

    so it slots into the one-stage resolvent layer without needing a coupled
    block stage solve.

    Further reading: https://en.wikipedia.org/wiki/Backward_differentiation_formula
    """

    __slots__ = (
        "call_pure",
        "derivative",
        "error",
        "has_history",
        "known_shift",
        "known_shift_block",
        "low",
        "previous_delta",
        "previous_step",
        "resolvent",
        "startup_rate",
        "trial_block",
        "workspace",
    )

    descriptor = SchemeDescriptor("BDF2", "Backward Differentiation Formula 2")

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)

        if resolvent is None:
            raise TypeError("BDF2 requires an explicit resolvent.")

        self.derivative = BoundDerivative(derivative)
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.startup_rate = translation_probe
        self.resolvent = resolvent

        workspace = self.workspace
        self.trial_block = Block([workspace.allocate_translation()])
        self.previous_delta, self.low, self.error, self.known_shift = (
            workspace.allocate_translation_buffers(4)
        )
        self.known_shift_block = Block([self.known_shift])

        self.previous_step = 0.0
        self.has_history = False

        self.initialise_runtime(regulator)
        self.call_pure = self.call_generic
        self.refresh_call()

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.5)

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
                "Backward-Euler startup solve:",
                "",
                "  Delta - h f(t_{n+1}, x_n + Delta) = 0",
                "",
                "Variable-step BDF2 solve after startup:",
                "",
                "  alpha_0 Delta - shift - h f(t_{n+1}, x_n + Delta) = 0",
                "",
                "where the shift term is built from the previous accepted step displacement",
                "and the current step ratio.",
                "",
                "A custom resolvent for this method must accept arguments",
                "`(out, alpha, rhs=None)` and overwrite `out` with the solved",
                "one-block displacement, stored as `Block(Translation)`.",
            ]
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                (
                    f"{type(self).__name__}("
                    f"short_name={self.short_name!r}, "
                    f"full_name={self.full_name!r})"
                ),
                self.display_method(),
            ]
        )

    def __str__(self) -> str:
        return self.display_method()

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)

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

    def advance_body(self, interval: IntervalLike, state: State) -> None:
        """Compatibility bridge for the transitional adaptive base."""

        self.call_pure(interval, state, Executor())

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
        apply_delta = workspace.apply_delta
        scale = workspace.scale
        ratio_fn = self.adaptive.ratio

        assert ratio_fn is not None

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

            if step_ratio_is_safe:
                try:
                    delta_high, error_ratio = self.solve_bdf2_step(
                        interval,
                        state,
                        dt,
                        ratio_fn,
                    )
                except ResolventError:
                    rejection_count += 1
                    dt = self.adaptive.rejected_step(
                        dt,
                        1.0,
                        remaining,
                        self.short_name,
                    )
                    continue
            else:
                try:
                    delta_high, error_ratio = self.solve_startup_step(
                        interval,
                        state,
                        dt,
                        ratio_fn,
                    )
                except ResolventError:
                    rejection_count += 1
                    dt = self.adaptive.rejected_step(
                        dt,
                        1.0,
                        remaining,
                        self.short_name,
                    )
                    continue

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
        apply_delta(delta_high, state)

        # `scale` may return a new translation rather than mutating its first
        # argument, so keep the returned object as the next history value.
        self.previous_delta = scale(self.previous_delta, 1.0, delta_high)
        self.previous_step = accepted_dt
        self.has_history = True

        remaining_after = remaining - accepted_dt
        controller_next_dt = self.adaptive.accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )

        if remaining_after <= 0.0:
            next_dt = 0.0
        else:
            # Preserve the existing BDF2 policy: avoid immediately shrinking the
            # next proposed step below the just-accepted step. Large changes are
            # handled by the startup fallback on the following call.
            next_dt = max(accepted_dt, controller_next_dt)

        interval.step = next_dt

        report = self.adaptive.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return report.accepted_dt

    def solve_startup_step(
        self,
        interval: IntervalLike,
        state: State,
        dt: float,
        ratio_fn,
    ):
        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        stage_interval = workspace.stage_interval
        trial_block = self.trial_block

        derivative(interval, state, self.startup_rate)

        self.resolvent.bind(stage_interval(interval, dt, dt), state)

        # `scale` may return a fresh zero translation; keep it in the block so
        # fallback algebra and in-place algebra both behave correctly.
        trial_block.items[0] = scale(trial_block[0], 0.0, trial_block[0])
        self.resolvent(trial_block, dt)

        delta_high = trial_block[0]
        delta_low = scale(self.low, dt, self.startup_rate)
        error = combine2(self.error, 1.0, delta_high, -1.0, delta_low)
        error_ratio = ratio_fn(error.norm(), delta_high.norm())

        return delta_high, error_ratio

    def solve_bdf2_step(
        self,
        interval: IntervalLike,
        state: State,
        dt: float,
        ratio_fn,
    ):
        workspace = self.workspace
        scale = workspace.scale
        combine2 = workspace.combine2
        stage_interval = workspace.stage_interval
        trial_block = self.trial_block

        step_ratio = dt / self.previous_step

        alpha0 = (2.0 * step_ratio + 1.0) / (step_ratio + 1.0)
        alpha2 = (step_ratio * step_ratio) / (step_ratio + 1.0)
        beta = (step_ratio * step_ratio) / (2.0 * step_ratio + 1.0)
        alpha = dt * (step_ratio + 1.0) / (2.0 * step_ratio + 1.0)

        # `scale` may return a new object, so update both the attribute and the
        # block that the resolvent receives as its right-hand-side shift.
        self.known_shift = scale(self.known_shift, beta, self.previous_delta)
        self.known_shift_block.items[0] = self.known_shift

        self.resolvent.bind(stage_interval(interval, dt, dt), state)
        trial_block.items[0] = scale(trial_block[0], 0.0, trial_block[0])
        self.resolvent(trial_block, alpha, rhs=self.known_shift_block)

        delta_high = trial_block[0]
        delta_low = combine2(self.low, alpha0, delta_high, -alpha2, self.previous_delta)
        error = combine2(self.error, 1.0, delta_high, -1.0, delta_low)
        error_ratio = ratio_fn(error.norm(), delta_high.norm())

        return delta_high, error_ratio


__all__ = ["SchemeBDF2"]