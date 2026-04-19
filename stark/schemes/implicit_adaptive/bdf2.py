from __future__ import annotations

from stark.accelerators.binding import BoundDerivative
from stark.auditor import Auditor
from stark.execution.regulator import Regulator
from stark.execution.executor import Executor
from stark.contracts import Block, Derivative, IntervalLike, Resolvent, State, Workbench
from stark.resolvents.failure import ResolventError
from stark.schemes.descriptor import SchemeDescriptor
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.schemes.base import (
    SchemeBaseImplicitAdaptive,
    _ADVANCE_ACCEPTED_DT,
    _ADVANCE_ERROR_RATIO,
    _ADVANCE_NEXT_DT,
    _ADVANCE_PROPOSED_DT,
    _ADVANCE_REJECTION_COUNT,
    _ADVANCE_T_START,
)


class SchemeBDF2(SchemeBaseImplicitAdaptive):
    """
    Adaptive second-order backward differentiation formula with BE startup.

    BDF2 is a two-step implicit method. After a first backward-Euler startup
    step, it advances with

        alpha0 x_{n+1} + alpha1 x_n + alpha2 x_{n-1} = dt f(x_{n+1}),

    using a variable-step ratio and a backward-Euler-style embedded estimate for
    step control. This first pass falls back to backward Euler after large step
    changes so the history can settle before resuming the multistep update. The
    nonlinear solve still has the STARK-friendly shifted form
    `delta - shift - alpha f(state + delta) = 0`, so it slots into the current
    resolvent layer without needing a coupled block stage solve.

    Further reading: https://en.wikipedia.org/wiki/Backward_differentiation_formula
    """

    __slots__ = (
        "derivative",
        "resolvent",
        "workspace",
        "startup_rate",
        "trial_block",
        "known_shift_block",
        "previous_delta",
        "low",
        "error",
        "known_shift",
        "previous_step",
        "has_history",
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
        self.derivative = BoundDerivative(derivative)
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.startup_rate = translation_probe
        if resolvent is None:
            raise TypeError("BDF2 requires an explicit resolvent.")
        self.resolvent = resolvent
        workspace = self.workspace
        self.trial_block = Block([workspace.allocate_translation()])
        self.previous_delta, self.low, self.error, self.known_shift = workspace.allocate_translation_buffers(4)
        self.known_shift_block = Block([self.known_shift])
        self.previous_step = 0.0
        self.has_history = False
        self.initialise_runtime(regulator)

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
                "    Delta = x_{n+1} - x_n",
                "",
                "Backward-Euler startup solve:",
                "",
                "    Delta - h f(t_{n+1}, x_n + Delta) = 0",
                "",
                "Variable-step BDF2 solve after startup:",
                "",
                "    alpha_0 Delta - shift - h f(t_{n+1}, x_n + Delta) = 0",
                "",
                "where the shift term is built from the previous accepted step displacement",
                "and the current step ratio.",
                "",
                "A custom resolvent for this method must accept arguments `(out, alpha, rhs=None)` and overwrite `out` with the solved one-block displacement, stored as `Block(Translation)`.",
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

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)

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
        apply_delta = workspace.apply_delta
        stage_interval = workspace.stage_interval
        controller = self._controller
        ratio_fn = self._ratio
        assert controller is not None
        assert ratio_fn is not None
        trial_block = self.trial_block
        dt = interval.step if interval.step <= remaining else remaining
        proposed_dt = dt
        rejection_count = 0

        use_bdf2 = self.has_history and 0.5 <= dt / self.previous_step <= 2.0

        if not use_bdf2:
            derivative(interval, state, self.startup_rate)
            while True:
                try:
                    self.resolvent.bind(stage_interval(interval, dt, dt), state)
                    scale(trial_block[0], 0.0, trial_block[0])
                    self.resolvent(trial_block, dt)
                    delta_high = trial_block[0]
                except ResolventError:
                    rejection_count += 1
                    dt = controller.rejected_step(dt, 1.0, remaining, "BDF2")
                    continue
                break

            delta_low = scale(self.low, dt, self.startup_rate)
            error = combine2(self.error, 1.0, delta_high, -1.0, delta_low)
            error_norm = error.norm()
            delta_high_norm = delta_high.norm()
            error_ratio = ratio_fn(error_norm, delta_high_norm)
        else:
            while True:
                try:
                    ratio = dt / self.previous_step
                    alpha0 = (2.0 * ratio + 1.0) / (ratio + 1.0)
                    alpha2 = (ratio * ratio) / (ratio + 1.0)
                    beta = (ratio * ratio) / (2.0 * ratio + 1.0)
                    alpha = dt * (ratio + 1.0) / (2.0 * ratio + 1.0)
                    known_shift = scale(self.known_shift, beta, self.previous_delta)
                    self.resolvent.bind(stage_interval(interval, dt, dt), state)
                    scale(trial_block[0], 0.0, trial_block[0])
                    self.resolvent(trial_block, alpha, rhs=self.known_shift_block)
                    delta_high = trial_block[0]
                except ResolventError:
                    rejection_count += 1
                    dt = controller.rejected_step(dt, 1.0, remaining, "BDF2")
                    continue

                delta_low = combine2(self.low, alpha0, delta_high, -alpha2, self.previous_delta)
                error = combine2(self.error, 1.0, delta_high, -1.0, delta_low)
                error_norm = error.norm()
                delta_high_norm = delta_high.norm()
                error_ratio = ratio_fn(error_norm, delta_high_norm)
                break

        accepted_dt = dt
        apply_delta(delta_high, state)
        scale(self.previous_delta, 1.0, delta_high)
        self.previous_step = accepted_dt
        self.has_history = True
        remaining_after = remaining - accepted_dt
        next_step = controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        if remaining_after <= 0.0:
            interval.step = 0.0
        else:
            interval.step = max(accepted_dt, next_step)
        advance_report[_ADVANCE_ACCEPTED_DT] = accepted_dt
        advance_report[_ADVANCE_T_START] = interval.present
        advance_report[_ADVANCE_PROPOSED_DT] = proposed_dt
        advance_report[_ADVANCE_NEXT_DT] = interval.step
        advance_report[_ADVANCE_ERROR_RATIO] = error_ratio
        advance_report[_ADVANCE_REJECTION_COUNT] = rejection_count


__all__ = ["SchemeBDF2"]
















