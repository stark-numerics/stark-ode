from __future__ import annotations

from stark.core.contracts import AllocatorLike, DynamicsSplitLike, IntervalLike, Resolvent, State
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


ARK436L2SA_EXPLICIT = Tableau(
    c=(0.0, 0.5, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0),
    a=(
        (),
        (0.5,),
        (13861.0 / 62500.0, 6889.0 / 62500.0),
        (-116923316275.0 / 2393684061468.0, -2731218467317.0 / 15368042101831.0, 9408046702089.0 / 11113171139209.0),
        (-451086348788.0 / 2902428689909.0, -2682348792572.0 / 7519795681897.0, 12662868775082.0 / 11960479115383.0, 3355817975965.0 / 11060851509271.0),
        (647845179188.0 / 3216320057751.0, 73281519250.0 / 8382639484533.0, 552539513391.0 / 3454668386233.0, 3354512671639.0 / 8306763924573.0, 4040.0 / 17871.0),
    ),
    b=(82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0, -2260.0 / 8211.0, 0.25),
    order=4,
    b_embedded=(4586570599.0 / 29645900160.0, 0.0, 178811875.0 / 945068544.0, 814220225.0 / 1159782912.0, -3700637.0 / 11593932.0, 61727.0 / 225920.0),
    embedded_order=3,
)
ARK436L2SA_IMPLICIT = Tableau(
    c=(0.0, 0.5, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0),
    a=(
        (),
        (0.25, 0.25),
        (8611.0 / 62500.0, -1743.0 / 31250.0, 0.25),
        (5012029.0 / 34652500.0, -654441.0 / 2922500.0, 174375.0 / 388108.0, 0.25),
        (15267082809.0 / 155376265600.0, -71443401.0 / 120774400.0, 730878875.0 / 902184768.0, 2285395.0 / 8070912.0, 0.25),
        (82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0, -2260.0 / 8211.0, 0.25),
    ),
    b=(82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0, -2260.0 / 8211.0, 0.25),
    order=4,
    b_embedded=(4586570599.0 / 29645900160.0, 0.0, 178811875.0 / 945068544.0, 814220225.0 / 1159782912.0, -3700637.0 / 11593932.0, 61727.0 / 225920.0),
    embedded_order=3,
)
ARK436L2SA_TABLEAU = TableauImex(
    explicit=ARK436L2SA_EXPLICIT,
    implicit=ARK436L2SA_IMPLICIT,
    short_name="ARK436L2SA",
    full_name="ARK4(3)6L[2]SA",
)
KENNEDY_CARPENTER43_6_TABLEAU = ARK436L2SA_TABLEAU


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records accepted/rejected adaptive-step monitor events.
# Provides: call_monitored.
@with_adaptive_step_monitoring
class SchemeKennedyCarpenter43_6:
    """Adaptive Kennedy-Carpenter ARK4(3)6L[2]SA IMEX method.

    Six-stage fourth-order additive Runge-Kutta pair using the shared
    Kennedy-Carpenter IMEX trial-step algorithm.
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

    descriptor = SchemeDescriptor("KC43-6", "Kennedy-Carpenter 4(3) 6-stage")
    @classmethod
    def display_tableau(cls) -> str:
        """Installed by `with_scheme_display` from `stark.methods.schemes.display`."""

        raise NotImplementedError("with_scheme_display installs display_tableau.")

    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_imex_resolvent_problem(
            cls.tableau,
            cls.descriptor.short_name,
            cls.descriptor.full_name,
        )

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = KENNEDY_CARPENTER43_6_TABLEAU

    def __init__(
        self,
        dynamics: DynamicsSplitLike,
        allocator: AllocatorLike,
        resolvent: Resolvent,
        *,
        configuration: SchemeConfiguration | None = None,
        specialist: SchemeSpecialist | None = None,
        monitor: SchemeMonitor | None = None,
    ) -> None:
        self.runtime = SchemeRuntimeImex(dynamics, allocator)
        self.workspace = self.runtime.workspace
        self.adaptive_step = KennedyCarpenterAdaptiveStep(
            tableau=self.tableau,
            dynamics=dynamics,
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
    "ARK436L2SA_EXPLICIT",
    "ARK436L2SA_IMPLICIT",
    "ARK436L2SA_TABLEAU",
    "KENNEDY_CARPENTER43_6_TABLEAU",
    "SchemeKennedyCarpenter43_6",
]
