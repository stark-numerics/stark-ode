from __future__ import annotations

from stark.audit import Auditor
from stark.control import Tolerance
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.scheme_butcher_tableau import ButcherTableau
from stark.scheme_descriptor import SchemeDescriptor
from stark.scheme_workspace import SchemeWorkspace


RK4_TABLEAU = ButcherTableau(
    c=(0.0, 0.5, 0.5, 1.0),
    a=(
        (),
        (0.5,),
        (0.0, 0.5),
        (0.0, 0.0, 1.0),
    ),
    b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    order=4,
)
RK4_B = RK4_TABLEAU.b


class SchemeRK4:
    """Classic fixed-step fourth-order Runge-Kutta method."""

    __slots__ = ("derivative", "k1", "k2", "k3", "k4", "workspace", "stage", "trial")

    descriptor = SchemeDescriptor("RK4", "Classical Runge-Kutta")
    tableau = RK4_TABLEAU

    def __init__(self, derivative: Derivative, workbench: Workbench) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.derivative = derivative
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.k1 = translation_probe
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2, self.k3, self.k4 = workspace.allocate_translation_buffers(4)

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
        combine4 = workspace.combine4
        apply_delta = workspace.apply_delta
        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        dt = interval.step if interval.step <= remaining else remaining

        derivative(state, k1)

        trial = scale(trial_buffer, 0.5 * dt, k1)
        trial(state, stage)
        derivative(stage, k2)

        trial = scale(trial_buffer, 0.5 * dt, k2)
        trial(state, stage)
        derivative(stage, k3)

        trial = scale(trial_buffer, dt, k3)
        trial(state, stage)
        derivative(stage, k4)

        delta = combine4(
            trial_buffer,
            dt * RK4_B[0],
            k1,
            dt * RK4_B[1],
            k2,
            dt * RK4_B[2],
            k3,
            dt * RK4_B[3],
            k4,
        )
        apply_delta(delta, state)
        return dt


__all__ = ["RK4_TABLEAU", "SchemeRK4"]
