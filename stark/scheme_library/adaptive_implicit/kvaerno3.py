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


KVAERNO3_GAMMA = 0.43586652150
KVAERNO3_A21 = KVAERNO3_GAMMA
KVAERNO3_A31 = (-4.0 * KVAERNO3_GAMMA * KVAERNO3_GAMMA + 6.0 * KVAERNO3_GAMMA - 1.0) / (4.0 * KVAERNO3_GAMMA)
KVAERNO3_A32 = (-2.0 * KVAERNO3_GAMMA + 1.0) / (4.0 * KVAERNO3_GAMMA)
KVAERNO3_A41 = (6.0 * KVAERNO3_GAMMA - 1.0) / (12.0 * KVAERNO3_GAMMA)
KVAERNO3_A42 = -1.0 / ((24.0 * KVAERNO3_GAMMA - 12.0) * KVAERNO3_GAMMA)
KVAERNO3_A43 = (
    -6.0 * KVAERNO3_GAMMA * KVAERNO3_GAMMA + 6.0 * KVAERNO3_GAMMA - 1.0
) / (6.0 * KVAERNO3_GAMMA - 3.0)

KVAERNO3_TABLEAU = ButcherTableau(
    c=(0.0, 2.0 * KVAERNO3_GAMMA, 1.0, 1.0),
    a=(
        (),
        (KVAERNO3_A21, KVAERNO3_GAMMA),
        (KVAERNO3_A31, KVAERNO3_A32, KVAERNO3_GAMMA),
        (KVAERNO3_A41, KVAERNO3_A42, KVAERNO3_A43, KVAERNO3_GAMMA),
    ),
    b=(KVAERNO3_A41, KVAERNO3_A42, KVAERNO3_A43, KVAERNO3_GAMMA),
    order=3,
    b_embedded=(KVAERNO3_A31, KVAERNO3_A32, KVAERNO3_GAMMA, 0.0),
    embedded_order=2,
    short_name="Kvaerno3",
    full_name="Kvaerno 3(2)",
)

_DELTA21 = KVAERNO3_A21 / KVAERNO3_GAMMA
_DELTA31 = KVAERNO3_A31 / KVAERNO3_GAMMA
_DELTA32 = KVAERNO3_A32 / KVAERNO3_GAMMA
_DELTA41 = KVAERNO3_A41 / KVAERNO3_GAMMA
_DELTA42 = KVAERNO3_A42 / KVAERNO3_GAMMA
_DELTA43 = KVAERNO3_A43 / KVAERNO3_GAMMA
_DELTA_HIGH = (_DELTA41, _DELTA42, _DELTA43, 1.0)
_DELTA_LOW = (_DELTA31, _DELTA32, 1.0, 0.0)
_DELTA_ERR = tuple(high - low for high, low in zip(_DELTA_HIGH, _DELTA_LOW, strict=True))


class SchemeKvaerno3:
    """
    Kvaerno's adaptive ESDIRK 3(2) method with sequential stage solves.

    This is a third-order stiffly accurate ESDIRK scheme with a second-order
    embedded estimate for step control. It keeps the same diagonal coefficient
    across implicit stages, so it fits naturally into STARK's residual-plus-
    resolver architecture while taking materially larger steps than SDIRK21 on
    smooth stiff problems.

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
        "known4",
        "trial",
        "error",
        "stage_block2",
        "stage_block3",
        "stage_block4",
        "residual2",
        "residual3",
        "residual4",
    )

    descriptor = SchemeDescriptor("Kvaerno3", "Kvaerno 3(2)")
    tableau = KVAERNO3_TABLEAU

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
        self.regulator = regulator if regulator is not None else Regulator(error_exponent=1.0 / 3.0)
        self.controller = AdaptiveController(self.regulator)
        self.stage1_rate = translation_probe
        workspace = self.workspace
        (
            self.delta1,
            self.delta2,
            self.delta3,
            self.delta4,
            self.known4,
            self.trial,
            self.error,
        ) = workspace.allocate_translation_buffers(7)
        self.stage_block2 = Block([workspace.allocate_translation()])
        self.stage_block3 = Block([workspace.allocate_translation()])
        self.stage_block4 = Block([workspace.allocate_translation()])
        self.residual2 = ShiftedImplicitResidual("Kvaerno3", derivative, workspace, linearizer)
        self.residual3 = ShiftedImplicitResidual("Kvaerno3", derivative, workspace, linearizer)
        self.residual4 = ShiftedImplicitResidual("Kvaerno3", derivative, workspace, linearizer)
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
        apply_delta = workspace.apply_delta
        controller = self.controller
        dt = interval.step if interval.step <= remaining else remaining
        derivative(state, self.stage1_rate)

        while True:
            delta1 = scale(self.delta1, dt * KVAERNO3_GAMMA, self.stage1_rate)

            try:
                self.residual2.configure(state, dt * KVAERNO3_GAMMA, delta1)
                workspace.scale(self.stage_block2[0], 0.0, self.stage_block2[0])
                self.resolver(self.stage_block2, self.residual2)
                delta2 = combine2(self.delta2, 0.0, self.delta2, 1.0, self.stage_block2[0])

                known3 = combine2(self.trial, _DELTA31, delta1, _DELTA32, delta2)
                self.residual3.configure(state, dt * KVAERNO3_GAMMA, known3)
                workspace.scale(self.stage_block3[0], 0.0, self.stage_block3[0])
                self.resolver(self.stage_block3, self.residual3)
                delta3 = combine2(self.delta3, 0.0, self.delta3, 1.0, self.stage_block3[0])

                known4 = combine3(self.known4, _DELTA41, delta1, _DELTA42, delta2, _DELTA43, delta3)
                self.residual4.configure(state, dt * KVAERNO3_GAMMA, known4)
                workspace.scale(self.stage_block4[0], 0.0, self.stage_block4[0])
                self.resolver(self.stage_block4, self.residual4)
                delta4 = combine2(self.delta4, 0.0, self.delta4, 1.0, self.stage_block4[0])
            except ResolutionError:
                dt = controller.rejected_step(dt, 1.0, remaining, "Kvaerno3")
                continue

            delta_high = combine4(
                self.trial,
                _DELTA_HIGH[0],
                delta1,
                _DELTA_HIGH[1],
                delta2,
                _DELTA_HIGH[2],
                delta3,
                _DELTA_HIGH[3],
                delta4,
            )
            error = combine4(
                self.error,
                _DELTA_ERR[0],
                delta1,
                _DELTA_ERR[1],
                delta2,
                _DELTA_ERR[2],
                delta3,
                _DELTA_ERR[3],
                delta4,
            )
            error_ratio = tolerance.ratio(error.norm(), delta_high.norm())

            if error_ratio <= 1.0:
                break

            dt = controller.rejected_step(dt, error_ratio, remaining, "Kvaerno3")

        accepted_dt = dt
        remaining_after = interval.stop - (interval.present + accepted_dt)
        interval.step = controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        apply_delta(delta_high, state)
        return accepted_dt


__all__ = ["KVAERNO3_GAMMA", "KVAERNO3_TABLEAU", "SchemeKvaerno3"]

