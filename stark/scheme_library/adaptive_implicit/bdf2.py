from __future__ import annotations

from stark.audit import Auditor
from stark.regulator import Regulator
from stark.tolerance import Tolerance
from stark.contracts import Block, Derivative, IntervalLike, Linearizer, ResolverLike, State, Workbench
from stark.resolver_library.picard import ResolverPicard
from stark.resolver_support.failure import ResolutionError
from stark.scheme_support.adaptive_controller import AdaptiveController
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.implicit_residual import ShiftedJacobianOperator, ShiftedResidualOperator
from stark.scheme_support.workspace import SchemeWorkspace


class _BDF2Residual:
    __slots__ = (
        "method_name",
        "scale",
        "combine2",
        "combine3",
        "copy_state",
        "base_state",
        "trial_state",
        "shift",
        "derivative",
        "derivative_buffer",
        "alpha",
        "linearizer",
        "jacobian_operator",
        "residual_operator",
        "_linearize",
    )

    def __init__(
        self,
        derivative: Derivative,
        workspace: SchemeWorkspace,
        linearizer: Linearizer | None,
    ) -> None:
        self.method_name = "BDF2"
        self.scale = workspace.scale
        self.combine2 = workspace.combine2
        self.combine3 = workspace.combine3
        self.copy_state = workspace.copy_state
        self.base_state = workspace.allocate_state_buffer()
        self.trial_state = workspace.allocate_state_buffer()
        self.shift = workspace.allocate_translation()
        self.derivative = derivative
        self.derivative_buffer = workspace.allocate_translation()
        self.alpha = 0.0
        self.linearizer = linearizer
        self.jacobian_operator = ShiftedJacobianOperator(self.method_name)
        self.residual_operator = ShiftedResidualOperator(workspace, self.jacobian_operator)
        self._linearize = self._linearize_configured if linearizer is not None else self._linearize_missing

    def configure(self, state: State, alpha: float, shift=None) -> None:
        self.copy_state(self.base_state, state)
        self.alpha = alpha
        if shift is None:
            self.scale(self.shift, 0.0, self.shift)
            return
        self.combine2(self.shift, 0.0, self.shift, 1.0, shift)

    def __call__(self, out: Block, block: Block) -> None:
        delta = block[0]
        delta(self.base_state, self.trial_state)
        self.derivative(self.trial_state, self.derivative_buffer)
        self.combine3(out[0], 1.0, delta, -1.0, self.shift, -self.alpha, self.derivative_buffer)

    def linearize(self, out, block: Block) -> None:
        self._linearize(out, block)

    def _linearize_missing(self, out, block: Block) -> None:
        del out, block
        raise RuntimeError(f"{self.method_name} Newton resolution requires a linearizer.")

    def _linearize_configured(self, out, block: Block) -> None:
        linearizer = self.linearizer
        assert linearizer is not None
        block[0](self.base_state, self.trial_state)
        linearizer(self.jacobian_operator, self.trial_state)
        self.residual_operator.alpha = self.alpha
        out.operators[0] = self.residual_operator


class SchemeBDF2:
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
    resolver layer without needing a coupled block stage solve.

    Further reading: https://en.wikipedia.org/wiki/Backward_differentiation_formula
    """

    __slots__ = (
        "derivative",
        "resolver",
        "workspace",
        "regulator",
        "controller",
        "startup_rate",
        "trial_block",
        "residual",
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
        linearizer: Linearizer | None,
        resolver: ResolverLike | None = None,
        regulator: Regulator | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        if linearizer is not None:
            Auditor.require_linearizer_inputs(linearizer, workbench, translation_probe)
        self.derivative = derivative
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.regulator = regulator if regulator is not None else Regulator(error_exponent=0.5)
        self.controller = AdaptiveController(self.regulator)
        self.startup_rate = translation_probe
        workspace = self.workspace
        self.trial_block = Block([workspace.allocate_translation()])
        self.residual = _BDF2Residual(derivative, workspace, linearizer)
        self.previous_delta, self.low, self.error, self.known_shift = workspace.allocate_translation_buffers(4)
        self.previous_step = 0.0
        self.has_history = False
        self.resolver = resolver if resolver is not None else ResolverPicard(workbench)

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

    def __call__(self, interval: IntervalLike, state: State, tolerance: Tolerance) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        apply_delta = workspace.apply_delta
        controller = self.controller
        trial_block = self.trial_block
        dt = interval.step if interval.step <= remaining else remaining

        use_bdf2 = self.has_history and 0.5 <= dt / self.previous_step <= 2.0

        if not use_bdf2:
            while True:
                try:
                    self.residual.configure(state, dt)
                    scale(trial_block[0], 0.0, trial_block[0])
                    self.resolver(trial_block, self.residual)
                    delta_high = trial_block[0]
                except ResolutionError:
                    dt = controller.rejected_step(dt, 1.0, remaining, "BDF2")
                    continue
                break

            derivative(state, self.startup_rate)
            delta_low = scale(self.low, dt, self.startup_rate)
            error = combine2(self.error, 1.0, delta_high, -1.0, delta_low)
            error_ratio = tolerance.ratio(error.norm(), delta_high.norm())
        else:
            while True:
                try:
                    ratio = dt / self.previous_step
                    alpha0 = (2.0 * ratio + 1.0) / (ratio + 1.0)
                    alpha2 = (ratio * ratio) / (ratio + 1.0)
                    beta = (ratio * ratio) / (2.0 * ratio + 1.0)
                    alpha = dt * (ratio + 1.0) / (2.0 * ratio + 1.0)
                    known_shift = scale(self.known_shift, beta, self.previous_delta)
                    self.residual.configure(state, alpha, known_shift)
                    scale(trial_block[0], 0.0, trial_block[0])
                    self.resolver(trial_block, self.residual)
                    delta_high = trial_block[0]
                except ResolutionError:
                    dt = controller.rejected_step(dt, 1.0, remaining, "BDF2")
                    continue

                delta_low = combine2(self.low, alpha0, delta_high, -alpha2, self.previous_delta)
                error = combine2(self.error, 1.0, delta_high, -1.0, delta_low)
                error_ratio = tolerance.ratio(error.norm(), delta_high.norm())
                break

        accepted_dt = dt
        apply_delta(delta_high, state)
        combine2(self.previous_delta, 0.0, self.previous_delta, 1.0, delta_high)
        self.previous_step = accepted_dt
        self.has_history = True
        remaining_after = interval.stop - (interval.present + accepted_dt)
        next_step = controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        if remaining_after <= 0.0:
            interval.step = 0.0
        else:
            interval.step = max(accepted_dt, next_step)
        return accepted_dt


__all__ = ["SchemeBDF2"]

