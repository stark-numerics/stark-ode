from __future__ import annotations

from stark.audit import Auditor
from stark.tolerance import Tolerance
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.butcher_tableau import ButcherTableau
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.workspace import SchemeWorkspace


SSPRK33_TABLEAU = ButcherTableau(
    c=(0.0, 1.0, 0.5),
    a=((), (1.0,), (0.25, 0.25)),
    b=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    order=3,
)
SSPRK33_A = SSPRK33_TABLEAU.a
SSPRK33_B = SSPRK33_TABLEAU.b


class SchemeSSPRK33:
    """
    The three-stage third-order strong-stability-preserving Runge-Kutta method.

    SSPRK33 is designed for problems where preserving monotonicity or other
    stability properties of forward Euler under a step restriction matters,
    such as hyperbolic PDE discretizations with nonlinear limiters.

    Further reading: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    __slots__ = ("derivative", "k1", "k2", "k3", "workspace", "stage", "trial")

    descriptor = SchemeDescriptor("SSPRK33", "SSP RK33")
    tableau = SSPRK33_TABLEAU

    def __init__(self, derivative: Derivative, workbench: Workbench) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.derivative = derivative
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.k1 = translation_probe
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2, self.k3 = workspace.allocate_translation_buffers(3)

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
        del tolerance
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        apply_delta = workspace.apply_delta
        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3

        dt = interval.step if interval.step <= remaining else remaining
        derivative(state, k1)

        trial = scale(trial_buffer, dt * SSPRK33_A[1][0], k1)
        trial(state, stage)
        derivative(stage, k2)

        trial = combine2(
            trial_buffer,
            dt * SSPRK33_A[2][0],
            k1,
            dt * SSPRK33_A[2][1],
            k2,
        )
        trial(state, stage)
        derivative(stage, k3)

        delta = combine3(
            trial_buffer,
            dt * SSPRK33_B[0],
            k1,
            dt * SSPRK33_B[1],
            k2,
            dt * SSPRK33_B[2],
            k3,
        )
        apply_delta(delta, state)
        return dt


__all__ = ["SSPRK33_TABLEAU", "SchemeSSPRK33"]


