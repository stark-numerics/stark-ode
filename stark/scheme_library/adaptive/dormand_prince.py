from __future__ import annotations

from stark.audit import Auditor
from stark.control import Regulator, Tolerance
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.butcher_tableau import ButcherTableau
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.workspace import SchemeWorkspace


RKDP_TABLEAU = ButcherTableau(
    c=(0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0),
    a=(
        (),
        (1.0 / 5.0,),
        (3.0 / 40.0, 9.0 / 40.0),
        (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
        (
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ),
        (
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ),
        (),
    ),
    b=(35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0),
    order=5,
    b_embedded=(
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ),
    embedded_order=4,
)
RKDP_A = RKDP_TABLEAU.a
RKDP_B_HIGH = RKDP_TABLEAU.b
RKDP_B_LOW = RKDP_TABLEAU.b_embedded
assert RKDP_B_LOW is not None
RKDP_B_HIGH_NZ = (
    RKDP_B_HIGH[0],
    RKDP_B_HIGH[2],
    RKDP_B_HIGH[3],
    RKDP_B_HIGH[4],
    RKDP_B_HIGH[5],
)
RKDP_B_ERR_NZ = (
    RKDP_B_HIGH[0] - RKDP_B_LOW[0],
    RKDP_B_HIGH[2] - RKDP_B_LOW[2],
    RKDP_B_HIGH[3] - RKDP_B_LOW[3],
    RKDP_B_HIGH[4] - RKDP_B_LOW[4],
    RKDP_B_HIGH[5] - RKDP_B_LOW[5],
    RKDP_B_HIGH[6] - RKDP_B_LOW[6],
)


class SchemeDormandPrince:
    """Adaptive Dormand-Prince 5(4) scheme."""

    __slots__ = ("regulator", "derivative", "error", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "workspace", "stage", "trial")

    descriptor = SchemeDescriptor("RKDP", "Dormand-Prince")
    tableau = RKDP_TABLEAU

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
        self.trial, self.error, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7 = workspace.allocate_translation_buffers(8)

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
        bound = tolerance.bound
        stage = self.stage
        trial_buffer = self.trial
        error_buffer = self.error
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4
        k5 = self.k5
        k6 = self.k6
        k7 = self.k7
        dt = interval.step if interval.step <= remaining else remaining
        safety = regulator.safety
        min_factor = regulator.min_factor
        max_factor = regulator.max_factor
        error_exponent = regulator.error_exponent

        derivative(state, k1)

        while True:
            trial = scale(trial_buffer, dt * RKDP_A[1][0], k1)
            trial(state, stage)
            derivative(stage, k2)

            trial = combine2(
                trial_buffer,
                dt * RKDP_A[2][0],
                k1,
                dt * RKDP_A[2][1],
                k2,
            )
            trial(state, stage)
            derivative(stage, k3)

            trial = combine3(
                trial_buffer,
                dt * RKDP_A[3][0],
                k1,
                dt * RKDP_A[3][1],
                k2,
                dt * RKDP_A[3][2],
                k3,
            )
            trial(state, stage)
            derivative(stage, k4)

            trial = combine4(
                trial_buffer,
                dt * RKDP_A[4][0],
                k1,
                dt * RKDP_A[4][1],
                k2,
                dt * RKDP_A[4][2],
                k3,
                dt * RKDP_A[4][3],
                k4,
            )
            trial(state, stage)
            derivative(stage, k5)

            trial = combine5(
                trial_buffer,
                dt * RKDP_A[5][0],
                k1,
                dt * RKDP_A[5][1],
                k2,
                dt * RKDP_A[5][2],
                k3,
                dt * RKDP_A[5][3],
                k4,
                dt * RKDP_A[5][4],
                k5,
            )
            trial(state, stage)
            derivative(stage, k6)

            delta_high = combine5(
                trial_buffer,
                dt * RKDP_B_HIGH_NZ[0],
                k1,
                dt * RKDP_B_HIGH_NZ[1],
                k3,
                dt * RKDP_B_HIGH_NZ[2],
                k4,
                dt * RKDP_B_HIGH_NZ[3],
                k5,
                dt * RKDP_B_HIGH_NZ[4],
                k6,
            )
            delta_high(state, stage)
            derivative(stage, k7)

            error = combine6(
                error_buffer,
                dt * RKDP_B_ERR_NZ[0],
                k1,
                dt * RKDP_B_ERR_NZ[1],
                k3,
                dt * RKDP_B_ERR_NZ[2],
                k4,
                dt * RKDP_B_ERR_NZ[3],
                k5,
                dt * RKDP_B_ERR_NZ[4],
                k6,
                dt * RKDP_B_ERR_NZ[5],
                k7,
            )
            err = error.norm()
            error_ratio = err / bound(delta_high.norm())

            if error_ratio <= 1.0:
                break

            if error_ratio == 0.0:
                factor = max_factor
            else:
                factor = safety * (1.0 / error_ratio) ** error_exponent
                factor = min(max_factor, max(min_factor, factor))

            dt = dt * factor
            if dt <= 0.0:
                raise RuntimeError("RKDP step size underflowed to zero.")
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


__all__ = ["RKDP_TABLEAU", "SchemeDormandPrince"]

