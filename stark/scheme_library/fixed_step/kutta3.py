from __future__ import annotations

from stark.audit import Auditor
from stark.control import Tolerance
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.butcher_tableau import ButcherTableau
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.workspace import SchemeWorkspace


KUTTA3_TABLEAU = ButcherTableau(
    c=(0.0, 0.5, 1.0),
    a=((), (0.5,), (-1.0, 2.0)),
    b=(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    order=3,
)
KUTTA3_A = KUTTA3_TABLEAU.a
KUTTA3_B = KUTTA3_TABLEAU.b


class SchemeKutta3:
    """Classic third-order Kutta method."""

    __slots__ = ("derivative", "k1", "k2", "k3", "workspace", "stage", "trial")

    descriptor = SchemeDescriptor("Kutta3", "Kutta Third-Order")
    tableau = KUTTA3_TABLEAU

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

        trial = scale(trial_buffer, dt * KUTTA3_A[1][0], k1)
        trial(state, stage)
        derivative(stage, k2)

        trial = combine2(
            trial_buffer,
            dt * KUTTA3_A[2][0],
            k1,
            dt * KUTTA3_A[2][1],
            k2,
        )
        trial(state, stage)
        derivative(stage, k3)

        delta = combine3(
            trial_buffer,
            dt * KUTTA3_B[0],
            k1,
            dt * KUTTA3_B[1],
            k2,
            dt * KUTTA3_B[2],
            k3,
        )
        apply_delta(delta, state)
        return dt


__all__ = ["KUTTA3_TABLEAU", "SchemeKutta3"]

