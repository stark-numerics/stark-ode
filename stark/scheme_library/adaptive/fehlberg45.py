from __future__ import annotations

from stark.audit import Auditor
from stark.control import Regulator, Tolerance
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.butcher_tableau import ButcherTableau
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.workspace import SchemeWorkspace


RKF45_TABLEAU = ButcherTableau(
    c=(0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0),
    a=(
        (),
        (1.0 / 4.0,),
        (3.0 / 32.0, 9.0 / 32.0),
        (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0),
        (439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0),
        (-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0),
    ),
    b=(16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0),
    order=5,
    b_embedded=(25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0),
    embedded_order=4,
)
RKF45_A = RKF45_TABLEAU.a
RKF45_B_HIGH = RKF45_TABLEAU.b
RKF45_B_LOW = RKF45_TABLEAU.b_embedded
assert RKF45_B_LOW is not None


class SchemeFehlberg45:
    """Adaptive Fehlberg 5(4) Runge-Kutta method."""

    __slots__ = ("regulator", "derivative", "error", "k1", "k2", "k3", "k4", "k5", "k6", "workspace", "stage", "trial")

    descriptor = SchemeDescriptor("RKF45", "Fehlberg 4(5)")
    tableau = RKF45_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        regulator: Regulator | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.derivative = derivative
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.regulator = regulator if regulator is not None else Regulator()
        self.k1 = translation_probe
        workspace = self.workspace
        self.stage = workspace.allocate_state_buffer()
        self.trial, self.error, self.k2, self.k3, self.k4, self.k5, self.k6 = workspace.allocate_translation_buffers(7)

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
        combine6 = workspace.combine6
        apply_delta = workspace.apply_delta
        regulator = self.regulator
        stage = self.stage
        trial_buffer = self.trial
        error_buffer = self.error
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4
        k5 = self.k5
        k6 = self.k6
        dt = interval.step if interval.step <= remaining else remaining
        safety = regulator.safety
        min_factor = regulator.min_factor
        max_factor = regulator.max_factor
        error_exponent = regulator.error_exponent

        while True:
            derivative(state, k1)

            trial = scale(trial_buffer, dt * RKF45_A[1][0], k1)
            trial(state, stage)
            derivative(stage, k2)

            trial = combine2(
                trial_buffer,
                dt * RKF45_A[2][0],
                k1,
                dt * RKF45_A[2][1],
                k2,
            )
            trial(state, stage)
            derivative(stage, k3)

            trial = combine3(
                trial_buffer,
                dt * RKF45_A[3][0],
                k1,
                dt * RKF45_A[3][1],
                k2,
                dt * RKF45_A[3][2],
                k3,
            )
            trial(state, stage)
            derivative(stage, k4)

            trial = combine4(
                trial_buffer,
                dt * RKF45_A[4][0],
                k1,
                dt * RKF45_A[4][1],
                k2,
                dt * RKF45_A[4][2],
                k3,
                dt * RKF45_A[4][3],
                k4,
            )
            trial(state, stage)
            derivative(stage, k5)

            trial = combine5(
                trial_buffer,
                dt * RKF45_A[5][0],
                k1,
                dt * RKF45_A[5][1],
                k2,
                dt * RKF45_A[5][2],
                k3,
                dt * RKF45_A[5][3],
                k4,
                dt * RKF45_A[5][4],
                k5,
            )
            trial(state, stage)
            derivative(stage, k6)

            delta_high = combine6(
                trial_buffer,
                dt * RKF45_B_HIGH[0],
                k1,
                dt * RKF45_B_HIGH[1],
                k2,
                dt * RKF45_B_HIGH[2],
                k3,
                dt * RKF45_B_HIGH[3],
                k4,
                dt * RKF45_B_HIGH[4],
                k5,
                dt * RKF45_B_HIGH[5],
                k6,
            )
            error = combine6(
                error_buffer,
                dt * (RKF45_B_HIGH[0] - RKF45_B_LOW[0]),
                k1,
                dt * (RKF45_B_HIGH[1] - RKF45_B_LOW[1]),
                k2,
                dt * (RKF45_B_HIGH[2] - RKF45_B_LOW[2]),
                k3,
                dt * (RKF45_B_HIGH[3] - RKF45_B_LOW[3]),
                k4,
                dt * (RKF45_B_HIGH[4] - RKF45_B_LOW[4]),
                k5,
                dt * (RKF45_B_HIGH[5] - RKF45_B_LOW[5]),
                k6,
            )
            err = error.norm()
            error_ratio = tolerance.ratio(err, delta_high.norm())

            if error_ratio <= 1.0:
                break

            if error_ratio == 0.0:
                factor = max_factor
            else:
                factor = safety * (1.0 / error_ratio) ** error_exponent
                factor = min(max_factor, max(min_factor, factor))
            dt = dt * factor
            if dt <= 0.0:
                raise RuntimeError("RKF45 step size underflowed to zero.")
            if dt > remaining:
                dt = remaining

        accepted_dt = dt
        remaining_after = interval.stop - (interval.present + accepted_dt)
        if remaining_after <= 0.0:
            interval.step = 0.0
        elif error_ratio == 0.0:
            interval.step = min(accepted_dt * max_factor, remaining_after)
        else:
            factor = safety * (1.0 / error_ratio) ** error_exponent
            factor = min(max_factor, max(min_factor, factor))
            interval.step = min(accepted_dt * factor, remaining_after)
        apply_delta(delta_high, state)
        return accepted_dt


__all__ = ["RKF45_TABLEAU", "SchemeFehlberg45"]

