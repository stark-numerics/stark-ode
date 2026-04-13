from __future__ import annotations

from math import sqrt

from stark.audit import Auditor
from stark.butcher_tableau import ButcherTableau
from stark.regulator import Regulator
from stark.tolerance import Tolerance
from stark.contracts import Block, Derivative, IntervalLike, Linearizer, ResolverLike, State, Workbench
from stark.resolver_library.picard import ResolverPicard
from stark.resolver_support.failure import ResolutionError
from stark.scheme_support.adaptive_controller import AdaptiveController
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.implicit_residual import ShiftedImplicitResidual
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


class SchemeSDIRK21:
    """
    An adaptive ESDIRK 2(1) method with sequential implicit stage solves.

    This is a singly diagonally implicit Runge-Kutta pair: each implicit stage
    uses the same diagonal coefficient, which lets STARK resolve the stages one
    at a time with the configured nonlinear resolver. The method advances with
    a second-order formula and estimates local error with an embedded
    first-order formula.

    It is a useful first stiff adaptive method because it exercises the
    implicit solver stack without requiring a fully coupled block solve.

    Further reading: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

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
        "controller",
    )

    descriptor = SchemeDescriptor("SDIRK21", "ESDIRK 2(1)")
    tableau = SDIRK21_TABLEAU

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
        self.linearizer = linearizer
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.regulator = regulator if regulator is not None else Regulator(error_exponent=0.5)
        self.controller = AdaptiveController(self.regulator)
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
        self.residual2 = ShiftedImplicitResidual("SDIRK21", derivative, workspace, linearizer)
        self.residual3 = ShiftedImplicitResidual("SDIRK21", derivative, workspace, linearizer)
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
        controller = self.controller
        dt = interval.step if interval.step <= remaining else remaining

        while True:
            derivative(state, self.stage1_rate)
            delta1 = scale(self.delta1, dt * SDIRK21_GAMMA, self.stage1_rate)

            try:
                self.residual2.configure(state, dt * SDIRK21_GAMMA, delta1)
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
                self.residual3.configure(state, dt * SDIRK21_GAMMA, known3)
                self.stage_block3.items[0] = scale(self.stage_block3[0], 0.0, self.stage_block3[0])
                self.resolver(self.stage_block3, self.residual3)
                delta3 = combine2(self.delta3, 0.0, self.delta3, 1.0, self.stage_block3[0])
            except ResolutionError:
                dt = controller.rejected_step(dt, 1.0, remaining, "SDIRK21")
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

            dt = controller.rejected_step(dt, error_ratio, remaining, "SDIRK21")

        accepted_dt = dt
        remaining_after = interval.stop - (interval.present + accepted_dt)
        interval.step = controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        apply_delta(delta_high, state)
        return accepted_dt


__all__ = ["SDIRK21_GAMMA", "SDIRK21_TABLEAU", "SchemeSDIRK21"]

