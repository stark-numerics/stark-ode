from __future__ import annotations

from stark.schemes.tableau import ButcherTableau, ImExButcherTableau
from stark.contracts import ImExDerivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.regulator import Regulator
from stark.resolvents.support.guard import ResolventTableauGuard
from stark.resolvents.failure import ResolventError
from stark.schemes.descriptor import SchemeDescriptor
from stark.machinery.stage_solve.workers import ImExStepper
from stark.schemes.base import (
    SchemeBaseImExAdaptive,
    _ADVANCE_ACCEPTED_DT,
    _ADVANCE_ERROR_RATIO,
    _ADVANCE_NEXT_DT,
    _ADVANCE_PROPOSED_DT,
    _ADVANCE_REJECTION_COUNT,
    _ADVANCE_T_START,
)
from stark.execution.executor import Executor


ARK324L2SA_EXPLICIT = ButcherTableau(
    c=(0.0, 1767732205903.0 / 2027836641118.0, 3.0 / 5.0, 1.0),
    a=(
        (),
        (1767732205903.0 / 2027836641118.0,),
        (5535828885825.0 / 10492691773637.0, 788022342437.0 / 10882634858940.0),
        (
            6485989280629.0 / 16251701735622.0,
            -4246266847089.0 / 9704473918619.0,
            10755448449292.0 / 10357097424841.0,
        ),
    ),
    b=(
        1471266399579.0 / 7840856788654.0,
        -4482444167858.0 / 7529755066697.0,
        11266239266428.0 / 11593286722821.0,
        1767732205903.0 / 4055673282236.0,
    ),
    order=3,
    b_embedded=(
        2756255671327.0 / 12835298489170.0,
        -10771552573575.0 / 22201958757719.0,
        9247589265047.0 / 10645013368117.0,
        2193209047091.0 / 5459859503100.0,
    ),
    embedded_order=2,
)

ARK324L2SA_IMPLICIT = ButcherTableau(
    c=(0.0, 1767732205903.0 / 2027836641118.0, 3.0 / 5.0, 1.0),
    a=(
        (),
        (1767732205903.0 / 4055673282236.0, 1767732205903.0 / 4055673282236.0),
        (
            2746238789719.0 / 10658868560708.0,
            -640167445237.0 / 6845629431997.0,
            1767732205903.0 / 4055673282236.0,
        ),
        (
            1471266399579.0 / 7840856788654.0,
            -4482444167858.0 / 7529755066697.0,
            11266239266428.0 / 11593286722821.0,
            1767732205903.0 / 4055673282236.0,
        ),
    ),
    b=(
        1471266399579.0 / 7840856788654.0,
        -4482444167858.0 / 7529755066697.0,
        11266239266428.0 / 11593286722821.0,
        1767732205903.0 / 4055673282236.0,
    ),
    order=3,
    b_embedded=(
        2756255671327.0 / 12835298489170.0,
        -10771552573575.0 / 22201958757719.0,
        9247589265047.0 / 10645013368117.0,
        2193209047091.0 / 5459859503100.0,
    ),
    embedded_order=2,
)

ARK324L2SA_TABLEAU = ImExButcherTableau(
    explicit=ARK324L2SA_EXPLICIT,
    implicit=ARK324L2SA_IMPLICIT,
    short_name="ARK324L2SA",
    full_name="ARK3(2)4L[2]SA",
)


class SchemeKennedyCarpenter32(SchemeBaseImExAdaptive):
    """
    Adaptive Kennedy-Carpenter ARK3(2)4L[2]SA IMEX method.

    This is a low-order embedded additive Runge-Kutta pair with a stiffly
    accurate, L-stable implicit partner. It is a good first adaptive IMEX
    method because it keeps the stage structure small while still exercising
    the full explicit-plus-resolvent coupling.

    Further reading: Kennedy and Carpenter, Applied Numerical Mathematics 44,
    2003; SUNDIALS ARKODE Butcher tables.
    """

    __slots__ = (
        "resolvent",
        "stepper",
        "tableau_guard",
    )

    descriptor = SchemeDescriptor("KC32", "Kennedy-Carpenter 3(2)")
    tableau = ARK324L2SA_TABLEAU

    def __init__(
        self,
        derivative: ImExDerivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        super().__init__(derivative, workbench, regulator)
        self.tableau_guard = ResolventTableauGuard("KennedyCarpenter32", self.tableau)
        if resolvent is None:
            raise TypeError("KennedyCarpenter32 requires an explicit resolvent.")
        self.resolvent = resolvent
        self.tableau_guard(self.resolvent)
        self.stepper = ImExStepper(derivative, self.workspace, self.resolvent, self.tableau)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=1.0 / 3.0)


    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        return self.redirect_call(interval, state, executor)

    def advance_body(self, interval: IntervalLike, state: State) -> None:
        remaining = interval.stop - interval.present
        advance_report = self.advance_report
        if remaining <= 0.0:
            advance_report[_ADVANCE_ACCEPTED_DT] = 0.0
            advance_report[_ADVANCE_T_START] = interval.present
            advance_report[_ADVANCE_PROPOSED_DT] = 0.0
            advance_report[_ADVANCE_NEXT_DT] = 0.0
            advance_report[_ADVANCE_ERROR_RATIO] = 0.0
            advance_report[_ADVANCE_REJECTION_COUNT] = 0
            return

        controller = self._controller
        ratio = self._ratio
        stepper = self.stepper
        workspace = self.workspace
        assert controller is not None
        assert ratio is not None
        dt = interval.step if interval.step <= remaining else remaining
        proposed_dt = dt
        rejection_count = 0
        while True:
            try:
                delta_high, error, delta_high_norm, error_norm = stepper.step(interval, state, dt, include_norms=True)
            except ResolventError:
                rejection_count += 1
                dt = controller.rejected_step(dt, 1.0, remaining, "KC32")
                continue

            assert error is not None
            assert delta_high_norm is not None
            assert error_norm is not None
            error_ratio = ratio(error_norm, delta_high_norm)
            if error_ratio <= 1.0:
                break
            rejection_count += 1
            dt = controller.rejected_step(dt, error_ratio, remaining, "KC32")

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = controller.accepted_next_step(accepted_dt, error_ratio, remaining_after)
        interval.step = next_dt
        workspace.apply_delta(delta_high, state)
        advance_report[_ADVANCE_ACCEPTED_DT] = accepted_dt
        advance_report[_ADVANCE_T_START] = interval.present
        advance_report[_ADVANCE_PROPOSED_DT] = proposed_dt
        advance_report[_ADVANCE_NEXT_DT] = next_dt
        advance_report[_ADVANCE_ERROR_RATIO] = error_ratio
        advance_report[_ADVANCE_REJECTION_COUNT] = rejection_count

__all__ = ["ARK324L2SA_TABLEAU", "SchemeKennedyCarpenter32"]














