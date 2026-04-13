from __future__ import annotations

from stark.audit import Auditor
from stark.regulator import Regulator
from stark.tolerance import Tolerance
from stark.contracts import Derivative, IntervalLike, State, Workbench
from stark.butcher_tableau import ButcherTableau
from stark.scheme_support.adaptive_controller import AdaptiveController
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.workspace import SchemeWorkspace


TSIT5_TABLEAU = ButcherTableau(
    c=(0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0),
    a=(
        (),
        (0.161,),
        (-0.008480655492357, 0.335480655492357),
        (2.897153057105494, -6.359448489975075, 4.362295432869581),
        (
            5.325864828439259,
            -11.74888356406283,
            7.495539342889836,
            -0.09249506636175525,
        ),
        (
            5.86145544294642,
            -12.92096931784711,
            8.159367898576159,
            -0.071584973281401,
            -0.02826905039406838,
        ),
        (
            0.09646076681806523,
            0.01,
            0.4798896504144996,
            1.379008574103742,
            -3.290069515436081,
            2.324710524099774,
        ),
    ),
    b=(
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0.0,
    ),
    order=5,
    b_embedded=(
        0.09468075576583923,
        0.009183565540343,
        0.4877705284247616,
        1.234297566930479,
        -2.707712349983526,
        1.866628418170587,
        1.0 / 66.0,
    ),
    embedded_order=4,
)
TSIT5_A = TSIT5_TABLEAU.a
TSIT5_B_HIGH = TSIT5_TABLEAU.b
TSIT5_B_LOW = TSIT5_TABLEAU.b_embedded
assert TSIT5_B_LOW is not None
TSIT5_B_ERR = tuple(high - low for high, low in zip(TSIT5_B_HIGH, TSIT5_B_LOW, strict=True))


class SchemeTsitouras5:
    """
    The adaptive Tsitouras embedded 5(4) Runge-Kutta pair.

    Tsitouras 5 is a modern fifth-order explicit adaptive method designed to
    offer strong practical performance with a carefully tuned tableau and
    embedded fourth-order error estimate.

    Further reading: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    __slots__ = ("regulator", "controller", "derivative", "error", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "workspace", "stage", "trial")

    descriptor = SchemeDescriptor("TSIT5", "Tsitouras 5")
    tableau = TSIT5_TABLEAU

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
        combine7 = workspace.combine7
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
        k7 = self.k7
        dt = interval.step if interval.step <= remaining else remaining
        derivative(state, k1)

        while True:
            trial = scale(trial_buffer, dt * TSIT5_A[1][0], k1)
            trial(state, stage)
            derivative(stage, k2)

            trial = combine2(
                trial_buffer,
                dt * TSIT5_A[2][0],
                k1,
                dt * TSIT5_A[2][1],
                k2,
            )
            trial(state, stage)
            derivative(stage, k3)

            trial = combine3(
                trial_buffer,
                dt * TSIT5_A[3][0],
                k1,
                dt * TSIT5_A[3][1],
                k2,
                dt * TSIT5_A[3][2],
                k3,
            )
            trial(state, stage)
            derivative(stage, k4)

            trial = combine4(
                trial_buffer,
                dt * TSIT5_A[4][0],
                k1,
                dt * TSIT5_A[4][1],
                k2,
                dt * TSIT5_A[4][2],
                k3,
                dt * TSIT5_A[4][3],
                k4,
            )
            trial(state, stage)
            derivative(stage, k5)

            trial = combine5(
                trial_buffer,
                dt * TSIT5_A[5][0],
                k1,
                dt * TSIT5_A[5][1],
                k2,
                dt * TSIT5_A[5][2],
                k3,
                dt * TSIT5_A[5][3],
                k4,
                dt * TSIT5_A[5][4],
                k5,
            )
            trial(state, stage)
            derivative(stage, k6)

            trial = combine6(
                trial_buffer,
                dt * TSIT5_A[6][0],
                k1,
                dt * TSIT5_A[6][1],
                k2,
                dt * TSIT5_A[6][2],
                k3,
                dt * TSIT5_A[6][3],
                k4,
                dt * TSIT5_A[6][4],
                k5,
                dt * TSIT5_A[6][5],
                k6,
            )
            trial(state, stage)
            derivative(stage, k7)

            delta_high = combine6(
                trial_buffer,
                dt * TSIT5_B_HIGH[0],
                k1,
                dt * TSIT5_B_HIGH[1],
                k2,
                dt * TSIT5_B_HIGH[2],
                k3,
                dt * TSIT5_B_HIGH[3],
                k4,
                dt * TSIT5_B_HIGH[4],
                k5,
                dt * TSIT5_B_HIGH[5],
                k6,
            )
            error = combine7(
                error_buffer,
                dt * TSIT5_B_ERR[0],
                k1,
                dt * TSIT5_B_ERR[1],
                k2,
                dt * TSIT5_B_ERR[2],
                k3,
                dt * TSIT5_B_ERR[3],
                k4,
                dt * TSIT5_B_ERR[4],
                k5,
                dt * TSIT5_B_ERR[5],
                k6,
                dt * TSIT5_B_ERR[6],
                k7,
            )
            err = error.norm()
            error_ratio = err / tolerance.bound(delta_high.norm())

            if error_ratio <= 1.0:
                break

            dt = controller.rejected_step(dt, error_ratio, remaining, "TSIT5")

        accepted_dt = dt
        remaining_after = interval.stop - (interval.present + accepted_dt)
        interval.step = controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        apply_delta(delta_high, state)
        return accepted_dt


__all__ = ["TSIT5_TABLEAU", "SchemeTsitouras5"]


