from __future__ import annotations

from stark.audit import Auditor
from stark.tolerance import Tolerance
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.butcher_tableau import ButcherTableau
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.workspace import SchemeWorkspace


RALSTON_TABLEAU = ButcherTableau(
    c=(0.0, 2.0 / 3.0),
    a=((), (2.0 / 3.0,)),
    b=(0.25, 0.75),
    order=2,
)
RALSTON_B = RALSTON_TABLEAU.b


class SchemeRalston:
    """
    Ralston's optimized two-stage second-order Runge-Kutta method.

    Among explicit RK2 methods, Ralston's choice of coefficients reduces the
    leading local truncation error constant, which often makes it a slightly
    sharper fixed-step second-order baseline than midpoint or Heun.

    Further reading: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    __slots__ = ("derivative", "k1", "k2", "workspace", "stage", "trial")

    descriptor = SchemeDescriptor("Ralston", "Ralston")
    tableau = RALSTON_TABLEAU

    def __init__(self, derivative: Derivative, workbench: Workbench) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.derivative = derivative
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.k1 = translation_probe
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.k2 = workspace.allocate_translation_buffers(2)

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
        apply_delta = workspace.apply_delta
        stage = self.stage
        trial_buffer = self.trial
        k1 = self.k1
        k2 = self.k2

        dt = interval.step if interval.step <= remaining else remaining
        derivative(state, k1)

        trial = scale(trial_buffer, dt * (2.0 / 3.0), k1)
        trial(state, stage)
        derivative(stage, k2)

        delta = combine2(
            trial_buffer,
            dt * RALSTON_B[0],
            k1,
            dt * RALSTON_B[1],
            k2,
        )
        apply_delta(delta, state)
        return dt


__all__ = ["RALSTON_TABLEAU", "SchemeRalston"]


