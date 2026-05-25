from __future__ import annotations

from stark.contracts import IntervalLike, State
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.schemes.support import (
    SchemeStepControl,
    with_adaptive_runtime_methods,
    with_imex_workspace_methods,
    with_scheme_display,
)
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support.tableau import ButcherTableau, ButcherTableauImex
from stark.schemes.imex_adaptive._kennedy_carpenter import SchemeKennedyCarpenterAdaptive


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
ARK324L2SA_TABLEAU = ButcherTableauImex(
    explicit=ARK324L2SA_EXPLICIT,
    implicit=ARK324L2SA_IMPLICIT,
    short_name="ARK324L2SA",
    full_name="ARK3(2)4L[2]SA",
)
KENNEDY_CARPENTER32_TABLEAU = ARK324L2SA_TABLEAU


@with_scheme_display
@with_adaptive_runtime_methods
@with_imex_workspace_methods
class SchemeKennedyCarpenter32(SchemeKennedyCarpenterAdaptive):
    """Adaptive Kennedy-Carpenter ARK3(2)4L[2]SA IMEX method.

    Algorithm sketch for one trial step of size h:

        1. Build each known IMEX stage right-hand side from previous split
           derivative evaluations.
        2. Solve each diagonal implicit stage problem with the configured
           resolvent.
        3. Recompute both split derivatives at the solved stage state.
        4. Build the high-order accepted increment and embedded error estimate.
        5. Accept or reject through the adaptive step controller.

    This is the smallest Kennedy-Carpenter adaptive IMEX scheme and is the
    reference implementation for the family.
    """

    step_control: SchemeStepControl
    descriptor = SchemeDescriptor("KC32", "Kennedy-Carpenter 3(2)")
    tableau = KENNEDY_CARPENTER32_TABLEAU

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        return self.redirect_call(interval, state, executor)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=1.0 / 3.0)


__all__ = [
    "ARK324L2SA_EXPLICIT",
    "ARK324L2SA_IMPLICIT",
    "ARK324L2SA_TABLEAU",
    "KENNEDY_CARPENTER32_TABLEAU",
    "SchemeKennedyCarpenter32",
]
