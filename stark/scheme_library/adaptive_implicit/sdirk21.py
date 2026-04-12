from __future__ import annotations

from math import sqrt

from stark.audit import Auditor
from stark.butcher_tableau import ButcherTableau
from stark.control import Regulator, Tolerance
from stark.contracts import Block, Derivative, IntervalLike, Linearizer, ResolverLike, State, Workbench
from stark.resolver_library.picard import ResolverPicard
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.workspace import SchemeWorkspace


SDIRK21_GAMMA = (2.0 - sqrt(2.0)) / 2.0
SDIRK21_B2 = (1.0 - 2.0 * SDIRK21_GAMMA) / (4.0 * SDIRK21_GAMMA)
SDIRK21_B1 = 1.0 - SDIRK21_B2 - SDIRK21_GAMMA
SDIRK21_BHAT2 = (
    SDIRK21_GAMMA
    * (
        -2.0
        + 7.0 * SDIRK21_GAMMA
        - 5.0 * SDIRK21_GAMMA * SDIRK21_GAMMA
        + 4.0 * SDIRK21_GAMMA * SDIRK21_GAMMA * SDIRK21_GAMMA
    )
    / (2.0 * (2.0 * SDIRK21_GAMMA - 1.0))
)
SDIRK21_BHAT3 = (
    -2.0
    * SDIRK21_GAMMA
    * SDIRK21_GAMMA
    * (1.0 - SDIRK21_GAMMA + SDIRK21_GAMMA * SDIRK21_GAMMA)
    / (2.0 * SDIRK21_GAMMA - 1.0)
)
SDIRK21_BHAT1 = 1.0 - SDIRK21_BHAT2 - SDIRK21_BHAT3

SDIRK21_TABLEAU = ButcherTableau(
    c=(0.0, 2.0 * SDIRK21_GAMMA, 1.0),
    a=(
        (),
        (SDIRK21_GAMMA, SDIRK21_GAMMA),
        (SDIRK21_B1, SDIRK21_B2, SDIRK21_GAMMA),
    ),
    b=(SDIRK21_B1, SDIRK21_B2, SDIRK21_GAMMA),
    order=2,
    b_embedded=(SDIRK21_BHAT1, SDIRK21_BHAT2, SDIRK21_BHAT3),
    embedded_order=1,
    short_name="SDIRK21",
    full_name="ESDIRK 2(1)",
)

_DELTA1_HIGH = SDIRK21_B1 / SDIRK21_GAMMA
_DELTA2_HIGH = SDIRK21_B2 / SDIRK21_GAMMA
_DELTA3_HIGH = 1.0
_DELTA1_LOW = SDIRK21_BHAT1 / SDIRK21_GAMMA
_DELTA2_LOW = SDIRK21_BHAT2 / SDIRK21_GAMMA
_DELTA3_LOW = SDIRK21_BHAT3 / SDIRK21_GAMMA
_DELTA1_ERR = _DELTA1_HIGH - _DELTA1_LOW
_DELTA2_ERR = _DELTA2_HIGH - _DELTA2_LOW
_DELTA3_ERR = _DELTA3_HIGH - _DELTA3_LOW


class _SingleStageDIRKResidual:
    __slots__ = (
        "combine2",
        "copy_state",
        "base_state",
        "trial_state",
        "known_shift",
        "known_state",
        "derivative",
        "derivative_buffer",
        "alpha",
        "linearizer",
        "jacobian_operator",
        "residual_operator",
    )

    def __init__(self, derivative: Derivative, workspace: SchemeWorkspace, linearizer: Linearizer | None) -> None:
        self.combine2 = workspace.combine2
        self.copy_state = workspace.copy_state
        self.base_state = workspace.allocate_state_buffer()
        self.trial_state = workspace.allocate_state_buffer()
        self.known_state = workspace.allocate_state_buffer()
        self.known_shift = workspace.allocate_translation()
        self.derivative = derivative
        self.derivative_buffer = workspace.allocate_translation()
        self.alpha = 0.0
        self.linearizer = linearizer
        self.jacobian_operator = _DIRKJacobianOperator()
        self.residual_operator = _DIRKResidualOperator(workspace, self.jacobian_operator)

    def configure(self, state: State, known_shift, alpha: float) -> None:
        self.copy_state(self.base_state, state)
        self.known_shift = self.combine2(self.known_shift, 0.0, self.known_shift, 1.0, known_shift)
        self.alpha = alpha

    def __call__(self, out: Block, block: Block) -> None:
        if len(out) != 1 or len(block) != 1:
            raise ValueError("Single-stage DIRK residual expects one-translation blocks.")

        delta = block[0]
        self.known_shift(self.base_state, self.known_state)
        delta(self.known_state, self.trial_state)
        self.derivative(self.trial_state, self.derivative_buffer)
        out.items[0] = self.combine2(out[0], 1.0, delta, -self.alpha, self.derivative_buffer)

    def linearize(self, out, block: Block) -> None:
        if len(block) != 1:
            raise ValueError("Single-stage DIRK residual expects one-translation blocks.")
        if self.linearizer is None:
            raise RuntimeError("Newton resolution requires a linearizer.")
        if len(out.operators) != 1:
            raise ValueError("Single-stage DIRK linearization expects a one-operator block.")

        self.known_shift(self.base_state, self.known_state)
        block[0](self.known_state, self.trial_state)
        self.linearizer(self.jacobian_operator, self.trial_state)
        self.residual_operator.alpha = self.alpha
        out.operators[0] = self.residual_operator


class _DIRKJacobianOperator:
    __slots__ = ("apply",)

    def __init__(self) -> None:
        self.apply = _unconfigured_operator

    def __call__(self, out, translation) -> None:
        self.apply(out, translation)


class _DIRKResidualOperator:
    __slots__ = ("combine2", "jacobian_buffer", "jacobian", "alpha")

    def __init__(self, workspace: SchemeWorkspace, jacobian) -> None:
        self.combine2 = workspace.combine2
        self.jacobian_buffer = workspace.allocate_translation()
        self.jacobian = jacobian
        self.alpha = 0.0

    def __call__(self, out, translation):
        self.jacobian(self.jacobian_buffer, translation)
        return self.combine2(out, 1.0, translation, -self.alpha, self.jacobian_buffer)


def _unconfigured_operator(out, translation) -> None:
    del out, translation
    raise RuntimeError("DIRK Jacobian operator was used before the linearizer configured it.")


class SchemeSDIRK21:
    """Adaptive ESDIRK 2(1) scheme with sequential stage solves."""

    __slots__ = (
        "derivative",
        "linearizer",
        "resolver",
        "workspace",
        "regulator",
        "stage1_rate",
        "delta1",
        "delta2",
        "delta3",
        "trial",
        "error",
        "known3",
        "stage_block2",
        "stage_block3",
        "residual2",
        "residual3",
    )

    descriptor = SchemeDescriptor("SDIRK21", "ESDIRK 2(1)")
    tableau = SDIRK21_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        linearizer: Linearizer,
        resolver: ResolverLike | None = None,
        regulator: Regulator | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.derivative = derivative
        self.linearizer = linearizer
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.regulator = regulator if regulator is not None else Regulator(error_exponent=0.5)
        self.stage1_rate = translation_probe
        workspace = self.workspace
        (
            self.delta1,
            self.delta2,
            self.delta3,
            self.trial,
            self.error,
            self.known3,
        ) = workspace.allocate_translation_buffers(6)
        self.stage_block2 = Block([workspace.allocate_translation()])
        self.stage_block3 = Block([workspace.allocate_translation()])
        self.residual2 = _SingleStageDIRKResidual(derivative, workspace, linearizer)
        self.residual3 = _SingleStageDIRKResidual(derivative, workspace, linearizer)
        self.resolver = resolver if resolver is not None else ResolverPicard(workbench)

    @classmethod
    def display_tableau(cls) -> str:
        return cls.descriptor.display_tableau(cls.tableau)

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return self.descriptor.repr_for(type(self).__name__, self.tableau)

    def __str__(self) -> str:
        return self.display_tableau()

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
        combine3 = workspace.combine3
        apply_delta = workspace.apply_delta
        regulator = self.regulator
        safety = regulator.safety
        min_factor = regulator.min_factor
        max_factor = regulator.max_factor
        exponent = regulator.error_exponent
        dt = interval.step if interval.step <= remaining else remaining

        while True:
            derivative(state, self.stage1_rate)
            delta1 = scale(self.delta1, dt * SDIRK21_GAMMA, self.stage1_rate)

            try:
                self.residual2.configure(state, delta1, dt * SDIRK21_GAMMA)
                self.stage_block2.items[0] = scale(self.stage_block2[0], 0.0, self.stage_block2[0])
                self.resolver(self.stage_block2, self.residual2)
                delta2 = combine2(self.delta2, 0.0, self.delta2, 1.0, self.stage_block2[0])

                known3 = combine2(
                    self.known3,
                    (1.0 - SDIRK21_B2 - SDIRK21_GAMMA) / SDIRK21_GAMMA,
                    delta1,
                    SDIRK21_B2 / SDIRK21_GAMMA,
                    delta2,
                )
                self.residual3.configure(state, known3, dt * SDIRK21_GAMMA)
                self.stage_block3.items[0] = scale(self.stage_block3[0], 0.0, self.stage_block3[0])
                self.resolver(self.stage_block3, self.residual3)
                delta3 = combine2(self.delta3, 0.0, self.delta3, 1.0, self.stage_block3[0])
            except RuntimeError:
                dt *= min_factor
                if dt <= 0.0:
                    raise RuntimeError("SDIRK21 step size underflowed to zero.")
                if dt > remaining:
                    dt = remaining
                continue

            delta_high = combine3(
                self.trial,
                _DELTA1_HIGH,
                delta1,
                _DELTA2_HIGH,
                delta2,
                _DELTA3_HIGH,
                delta3,
            )
            error = combine3(
                self.error,
                _DELTA1_ERR,
                delta1,
                _DELTA2_ERR,
                delta2,
                _DELTA3_ERR,
                delta3,
            )
            error_ratio = tolerance.ratio(error.norm(), delta_high.norm())

            if error_ratio <= 1.0:
                break

            if error_ratio == 0.0:
                factor = max_factor
            else:
                factor = safety * (1.0 / error_ratio) ** exponent
                factor = min(max_factor, max(min_factor, factor))
            dt *= factor
            if dt <= 0.0:
                raise RuntimeError("SDIRK21 step size underflowed to zero.")
            if dt > remaining:
                dt = remaining

        accepted_dt = dt
        remaining_after = interval.stop - (interval.present + accepted_dt)
        if remaining_after <= 0.0:
            interval.step = 0.0
        elif error_ratio == 0.0:
            interval.step = min(accepted_dt * max_factor, remaining_after)
        else:
            factor = safety * (1.0 / error_ratio) ** exponent
            factor = min(max_factor, max(min_factor, factor))
            interval.step = min(accepted_dt * factor, remaining_after)
        apply_delta(delta_high, state)
        return accepted_dt


__all__ = ["SDIRK21_GAMMA", "SDIRK21_TABLEAU", "SchemeSDIRK21"]
