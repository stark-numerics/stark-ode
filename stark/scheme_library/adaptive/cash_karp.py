from __future__ import annotations

from stark.audit import Auditor
from stark.regulator import Regulator
from stark.tolerance import Tolerance
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.butcher_tableau import ButcherTableau
from stark.scheme_support.adaptive_controller import AdaptiveController
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.workspace import SchemeWorkspace


RKCK_TABLEAU = ButcherTableau(
    c=(0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0),
    a=(
        (),
        (1.0 / 5.0,),
        (3.0 / 40.0, 9.0 / 40.0),
        (3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0),
        (-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0),
        (
            1631.0 / 55296.0,
            175.0 / 512.0,
            575.0 / 13824.0,
            44275.0 / 110592.0,
            253.0 / 4096.0,
        ),
    ),
    b=(37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0),
    order=5,
    b_embedded=(
        2825.0 / 27648.0,
        0.0,
        18575.0 / 48384.0,
        13525.0 / 55296.0,
        277.0 / 14336.0,
        1.0 / 4.0,
    ),
    embedded_order=4,
)
RKCK_A = RKCK_TABLEAU.a
RKCK_B_HIGH = RKCK_TABLEAU.b
RKCK_B_LOW = RKCK_TABLEAU.b_embedded
assert RKCK_B_LOW is not None
RKCK_B_HIGH_NZ = (RKCK_B_HIGH[0], RKCK_B_HIGH[2], RKCK_B_HIGH[3], RKCK_B_HIGH[5])
RKCK_B_ERR_NZ = (
    RKCK_B_HIGH[0] - RKCK_B_LOW[0],
    RKCK_B_HIGH[2] - RKCK_B_LOW[2],
    RKCK_B_HIGH[3] - RKCK_B_LOW[3],
    RKCK_B_HIGH[4] - RKCK_B_LOW[4],
    RKCK_B_HIGH[5] - RKCK_B_LOW[5],
)


class SchemeCashKarp:
    """
    The adaptive Cash-Karp embedded 5(4) Runge-Kutta pair.

    Cash-Karp advances with a fifth-order explicit method and estimates the
    local error with an embedded fourth-order formula. It is a classic adaptive
    explicit solver for smooth non-stiff problems.

    Further reading: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    __slots__ = ("regulator", "controller", "derivative", "error", "k1", "k2", "k3", "k4", "k5", "k6", "workspace", "stage", "trial")

    descriptor = SchemeDescriptor("RKCK", "Cash Karp")
    tableau = RKCK_TABLEAU

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
        self.controller = AdaptiveController(self.regulator)
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
        apply_delta = workspace.apply_delta
        controller = self.controller
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

        while True:
            derivative(state, k1)

            trial = scale(trial_buffer, dt * RKCK_A[1][0], k1)
            trial(state, stage)
            derivative(stage, k2)

            trial = combine2(
                trial_buffer,
                dt * RKCK_A[2][0],
                k1,
                dt * RKCK_A[2][1],
                k2,
            )
            trial(state, stage)
            derivative(stage, k3)

            trial = combine3(
                trial_buffer,
                dt * RKCK_A[3][0],
                k1,
                dt * RKCK_A[3][1],
                k2,
                dt * RKCK_A[3][2],
                k3,
            )
            trial(state, stage)
            derivative(stage, k4)

            trial = combine4(
                trial_buffer,
                dt * RKCK_A[4][0],
                k1,
                dt * RKCK_A[4][1],
                k2,
                dt * RKCK_A[4][2],
                k3,
                dt * RKCK_A[4][3],
                k4,
            )
            trial(state, stage)
            derivative(stage, k5)

            trial = combine5(
                trial_buffer,
                dt * RKCK_A[5][0],
                k1,
                dt * RKCK_A[5][1],
                k2,
                dt * RKCK_A[5][2],
                k3,
                dt * RKCK_A[5][3],
                k4,
                dt * RKCK_A[5][4],
                k5,
            )
            trial(state, stage)
            derivative(stage, k6)

            delta_high = combine4(
                trial_buffer,
                dt * RKCK_B_HIGH_NZ[0],
                k1,
                dt * RKCK_B_HIGH_NZ[1],
                k3,
                dt * RKCK_B_HIGH_NZ[2],
                k4,
                dt * RKCK_B_HIGH_NZ[3],
                k6,
            )
            error = combine5(
                error_buffer,
                dt * RKCK_B_ERR_NZ[0],
                k1,
                dt * RKCK_B_ERR_NZ[1],
                k3,
                dt * RKCK_B_ERR_NZ[2],
                k4,
                dt * RKCK_B_ERR_NZ[3],
                k5,
                dt * RKCK_B_ERR_NZ[4],
                k6,
            )
            err = error.norm()
            error_ratio = tolerance.ratio(err, delta_high.norm())

            if error_ratio <= 1.0:
                break

            dt = controller.rejected_step(dt, error_ratio, remaining, "RKCK")

        accepted_dt = dt
        remaining_after = interval.stop - (interval.present + accepted_dt)
        interval.step = controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        apply_delta(delta_high, state)
        return accepted_dt


SchemeRKCK = SchemeCashKarp

__all__ = ["RKCK_TABLEAU", "SchemeCashKarp", "SchemeRKCK"]


