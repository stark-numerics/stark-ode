from __future__ import annotations

from stark.core.contracts import Allocator, DerivativeSplitLike, IntervalLike, Resolvent, State
from stark.methods.schemes.configuration import SchemeConfiguration, SchemeConfigurationDefault
from stark.methods.schemes.execution.call import SchemeCall
from stark.methods.schemes.execution.step_control import SchemeStepControl
from stark.methods.schemes.monitoring.decorators import with_adaptive_step_monitoring
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.display.display import display_imex_resolvent_problem
from stark.methods.schemes.imex.runtime import SchemeRuntimeImex
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.method.tableau import Tableau, TableauImex
from stark.methods.schemes.imex.adaptive.kennedy_carpenter import KennedyCarpenterAdaptiveStep
from stark.methods.schemes.specialization.specialist import SchemeSpecialist


ARK324L2SA_EXPLICIT = Tableau(
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
ARK324L2SA_IMPLICIT = Tableau(
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
ARK324L2SA_TABLEAU = TableauImex(
    explicit=ARK324L2SA_EXPLICIT,
    implicit=ARK324L2SA_IMPLICIT,
    short_name="ARK324L2SA",
    full_name="ARK3(2)4L[2]SA",
)
KENNEDY_CARPENTER32_TABLEAU = ARK324L2SA_TABLEAU


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records accepted/rejected adaptive-step monitor events.
# Provides: call_monitored.
@with_adaptive_step_monitoring
class SchemeKennedyCarpenter32:
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

    # Installed by the scheme monitoring decorator above this class.
    call_monitored: SchemeCall

    __slots__ = (
        "adaptive_step",
        "call_body",
        "call_step",
        "monitor",
        "redirect_call",
        "step_control",
        "runtime",
        "workspace",
    )

    descriptor = SchemeDescriptor("KC32", "Kennedy-Carpenter 3(2)")
    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_imex_resolvent_problem(
            cls.tableau,
            cls.descriptor.short_name,
            cls.descriptor.full_name,
        )

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = KENNEDY_CARPENTER32_TABLEAU

    def __init__(
        self,
        derivative: DerivativeSplitLike,
        allocator: Allocator,
        resolvent: Resolvent,
        *,
        configuration: SchemeConfiguration | None = None,
        specialist: SchemeSpecialist | None = None,
        monitor: SchemeMonitor | None = None,
    ) -> None:
        self.runtime = SchemeRuntimeImex(derivative, allocator)
        self.workspace = self.runtime.workspace
        self.adaptive_step = KennedyCarpenterAdaptiveStep(
            tableau=self.tableau,
            derivative=derivative,
            workspace=self.workspace,
            resolvent=resolvent,
            configuration=configuration if configuration is not None else SchemeConfigurationDefault(),
            specialist=specialist,
        )
        self.step_control = self.adaptive_step.step_control
        self.call_body = self.call_specialized if specialist is not None else self.call_inline
        self.monitor = monitor
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step

    def call_inline(self, interval: IntervalLike, state: State) -> float:
        return self.adaptive_step.call_inline(interval, state)

    def call_specialized(self, interval: IntervalLike, state: State) -> float:
        return self.adaptive_step.call_specialized(interval, state)

    def __call__(self, interval: IntervalLike, state: State) -> float:
        return self.redirect_call(interval, state)

__all__ = [
    "ARK324L2SA_EXPLICIT",
    "ARK324L2SA_IMPLICIT",
    "ARK324L2SA_TABLEAU",
    "KENNEDY_CARPENTER32_TABLEAU",
    "SchemeKennedyCarpenter32",
]
