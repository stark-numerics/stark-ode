from __future__ import annotations

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

KVAERNO4_GAMMA = 0.5728160625


def _poly(*coefficients: float) -> float:
    value = 0.0
    for coefficient in coefficients:
        value = value * KVAERNO4_GAMMA + coefficient
    return value
KVAERNO4_A21 = KVAERNO4_GAMMA
KVAERNO4_A31 = _poly(144.0, -180.0, 81.0, -15.0, 1.0) * KVAERNO4_GAMMA / (_poly(12.0, -6.0, 1.0) ** 2)
KVAERNO4_A32 = _poly(-36.0, 39.0, -15.0, 2.0) * KVAERNO4_GAMMA / (_poly(12.0, -6.0, 1.0) ** 2)
KVAERNO4_A41 = _poly(-144.0, 396.0, -330.0, 117.0, -18.0, 1.0) / (
    12.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(12.0, -9.0, 2.0)
)
KVAERNO4_A42 = _poly(72.0, -126.0, 69.0, -15.0, 1.0) / (
    12.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(3.0, -1.0)
)
KVAERNO4_A43 = (
    _poly(-6.0, 6.0, -1.0) * (_poly(12.0, -6.0, 1.0) ** 2)
) / (
    12.0
    * KVAERNO4_GAMMA
    * KVAERNO4_GAMMA
    * _poly(12.0, -9.0, 2.0)
    * _poly(3.0, -1.0)
)
KVAERNO4_A51 = _poly(288.0, -312.0, 120.0, -18.0, 1.0) / (
    48.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(12.0, -9.0, 2.0)
)
KVAERNO4_A52 = _poly(24.0, -12.0, 1.0) / (
    48.0 * KVAERNO4_GAMMA * KVAERNO4_GAMMA * _poly(3.0, -1.0)
)
KVAERNO4_A53 = -(_poly(12.0, -6.0, 1.0) ** 3) / (
    48.0
    * KVAERNO4_GAMMA
    * KVAERNO4_GAMMA
    * _poly(3.0, -1.0)
    * _poly(12.0, -9.0, 2.0)
    * _poly(6.0, -6.0, 1.0)
)
KVAERNO4_A54 = _poly(-24.0, 36.0, -12.0, 1.0) / _poly(24.0, -24.0, 4.0)
KVAERNO4_C2 = KVAERNO4_GAMMA + KVAERNO4_A21
KVAERNO4_C3 = KVAERNO4_GAMMA + KVAERNO4_A31 + KVAERNO4_A32

KVAERNO4_TABLEAU = ButcherTableau(
    c=(0.0, KVAERNO4_C2, KVAERNO4_C3, 1.0, 1.0),
    a=(
        (),
        (KVAERNO4_A21, KVAERNO4_GAMMA),
        (KVAERNO4_A31, KVAERNO4_A32, KVAERNO4_GAMMA),
        (KVAERNO4_A41, KVAERNO4_A42, KVAERNO4_A43, KVAERNO4_GAMMA),
        (KVAERNO4_A51, KVAERNO4_A52, KVAERNO4_A53, KVAERNO4_A54, KVAERNO4_GAMMA),
    ),
    b=(KVAERNO4_A51, KVAERNO4_A52, KVAERNO4_A53, KVAERNO4_A54, KVAERNO4_GAMMA),
    order=4,
    b_embedded=(KVAERNO4_A41, KVAERNO4_A42, KVAERNO4_A43, KVAERNO4_GAMMA, 0.0),
    embedded_order=3,
    short_name="Kvaerno4",
    full_name="Kvaerno 4(3)",
)

_DELTA21 = KVAERNO4_A21 / KVAERNO4_GAMMA
_DELTA31 = KVAERNO4_A31 / KVAERNO4_GAMMA
_DELTA32 = KVAERNO4_A32 / KVAERNO4_GAMMA
_DELTA41 = KVAERNO4_A41 / KVAERNO4_GAMMA
_DELTA42 = KVAERNO4_A42 / KVAERNO4_GAMMA
_DELTA43 = KVAERNO4_A43 / KVAERNO4_GAMMA
_DELTA51 = KVAERNO4_A51 / KVAERNO4_GAMMA
_DELTA52 = KVAERNO4_A52 / KVAERNO4_GAMMA
_DELTA53 = KVAERNO4_A53 / KVAERNO4_GAMMA
_DELTA54 = KVAERNO4_A54 / KVAERNO4_GAMMA
_DELTA_HIGH = (_DELTA51, _DELTA52, _DELTA53, _DELTA54, 1.0)
_DELTA_LOW = (_DELTA41, _DELTA42, _DELTA43, 1.0, 0.0)
_DELTA_ERR = tuple(high - low for high, low in zip(_DELTA_HIGH, _DELTA_LOW, strict=True))


class SchemeKvaerno4:
    """
    Kvaerno's adaptive ESDIRK 4(3) method with sequential stage solves.

    This is a fourth-order stiffly accurate ESDIRK pair with a third-order
    embedded estimate. It aims at the same stiff adaptive territory as Diffrax
    Kvaerno-family methods while staying inside STARK's sequential stage
    resolver structure.

    Further reading: Anne Kvaerno, "Singly diagonally implicit Runge-Kutta
    methods with an explicit first stage" (BIT Numerical Mathematics, 2004).
    """

    __slots__ = (
        "derivative",
        "linearizer",
        "resolver",
        "workspace",
        "regulator",
        "controller",
        "stage1_rate",
        "delta1",
        "delta2",
        "delta3",
        "delta4",
        "delta5",
        "known4",
        "known5",
        "trial",
        "error",
        "stage_block2",
        "stage_block3",
        "stage_block4",
        "stage_block5",
        "residual2",
        "residual3",
        "residual4",
        "residual5",
    )

    descriptor = SchemeDescriptor("Kvaerno4", "Kvaerno 4(3)")
    tableau = KVAERNO4_TABLEAU

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
        self.regulator = regulator if regulator is not None else Regulator(error_exponent=0.25)
        self.controller = AdaptiveController(self.regulator)
        self.stage1_rate = translation_probe
        workspace = self.workspace
        (
            self.delta1,
            self.delta2,
            self.delta3,
            self.delta4,
            self.delta5,
            self.known4,
            self.known5,
            self.trial,
            self.error,
        ) = workspace.allocate_translation_buffers(9)
        self.stage_block2 = Block([workspace.allocate_translation()])
        self.stage_block3 = Block([workspace.allocate_translation()])
        self.stage_block4 = Block([workspace.allocate_translation()])
        self.stage_block5 = Block([workspace.allocate_translation()])
        self.residual2 = ShiftedImplicitResidual("Kvaerno4", derivative, workspace, linearizer)
        self.residual3 = ShiftedImplicitResidual("Kvaerno4", derivative, workspace, linearizer)
        self.residual4 = ShiftedImplicitResidual("Kvaerno4", derivative, workspace, linearizer)
        self.residual5 = ShiftedImplicitResidual("Kvaerno4", derivative, workspace, linearizer)
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
        combine4 = workspace.combine4
        combine5 = workspace.combine5
        apply_delta = workspace.apply_delta
        controller = self.controller
        dt = interval.step if interval.step <= remaining else remaining
        derivative(state, self.stage1_rate)

        while True:
            delta1 = scale(self.delta1, dt * KVAERNO4_GAMMA, self.stage1_rate)

            try:
                self.residual2.configure(state, dt * KVAERNO4_GAMMA, delta1)
                workspace.scale(self.stage_block2[0], 0.0, self.stage_block2[0])
                self.resolver(self.stage_block2, self.residual2)
                delta2 = combine2(self.delta2, 0.0, self.delta2, 1.0, self.stage_block2[0])

                known3 = combine2(self.trial, _DELTA31, delta1, _DELTA32, delta2)
                self.residual3.configure(state, dt * KVAERNO4_GAMMA, known3)
                workspace.scale(self.stage_block3[0], 0.0, self.stage_block3[0])
                self.resolver(self.stage_block3, self.residual3)
                delta3 = combine2(self.delta3, 0.0, self.delta3, 1.0, self.stage_block3[0])

                known4 = combine3(self.known4, _DELTA41, delta1, _DELTA42, delta2, _DELTA43, delta3)
                self.residual4.configure(state, dt * KVAERNO4_GAMMA, known4)
                workspace.scale(self.stage_block4[0], 0.0, self.stage_block4[0])
                self.resolver(self.stage_block4, self.residual4)
                delta4 = combine2(self.delta4, 0.0, self.delta4, 1.0, self.stage_block4[0])

                known5 = combine4(
                    self.known5,
                    _DELTA51,
                    delta1,
                    _DELTA52,
                    delta2,
                    _DELTA53,
                    delta3,
                    _DELTA54,
                    delta4,
                )
                self.residual5.configure(state, dt * KVAERNO4_GAMMA, known5)
                workspace.scale(self.stage_block5[0], 0.0, self.stage_block5[0])
                self.resolver(self.stage_block5, self.residual5)
                delta5 = combine2(self.delta5, 0.0, self.delta5, 1.0, self.stage_block5[0])
            except ResolutionError:
                dt = controller.rejected_step(dt, 1.0, remaining, "Kvaerno4")
                continue

            delta_high = combine5(
                self.trial,
                _DELTA_HIGH[0],
                delta1,
                _DELTA_HIGH[1],
                delta2,
                _DELTA_HIGH[2],
                delta3,
                _DELTA_HIGH[3],
                delta4,
                _DELTA_HIGH[4],
                delta5,
            )
            error = combine5(
                self.error,
                _DELTA_ERR[0],
                delta1,
                _DELTA_ERR[1],
                delta2,
                _DELTA_ERR[2],
                delta3,
                _DELTA_ERR[3],
                delta4,
                _DELTA_ERR[4],
                delta5,
            )
            error_ratio = tolerance.ratio(error.norm(), delta_high.norm())

            if error_ratio <= 1.0:
                break

            dt = controller.rejected_step(dt, error_ratio, remaining, "Kvaerno4")

        accepted_dt = dt
        remaining_after = interval.stop - (interval.present + accepted_dt)
        interval.step = controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        apply_delta(delta_high, state)
        return accepted_dt


__all__ = ["KVAERNO4_GAMMA", "KVAERNO4_TABLEAU", "SchemeKvaerno4"]

