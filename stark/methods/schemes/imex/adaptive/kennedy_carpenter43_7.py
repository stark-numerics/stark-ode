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
from stark.methods.schemes.linear_fixed_generation.linear_fixed import SchemeLinearFixedLike


ARK437L2SA_EXPLICIT = Tableau(
    c=(0.0, 247.0 / 1000.0, 4276536705230.0 / 10142255878289.0, 67.0 / 200.0, 3.0 / 40.0, 7.0 / 10.0, 1.0),
    a=(
        (),
        (247.0 / 1000.0,),
        (247.0 / 4000.0, 2694949928731.0 / 7487940209513.0),
        (464650059369.0 / 8764239774964.0, 878889893998.0 / 2444806327765.0, -952945855348.0 / 12294611323341.0),
        (476636172619.0 / 8159180917465.0, -1271469283451.0 / 7793814740893.0, -859560642026.0 / 4356155882851.0, 1723805262919.0 / 4571918432560.0),
        (6338158500785.0 / 11769362343261.0, -4970555480458.0 / 10924838743837.0, 3326578051521.0 / 2647936831840.0, -880713585975.0 / 1841400956686.0, -1428733748635.0 / 8843423958496.0),
        (760814592956.0 / 3276306540349.0, 760814592956.0 / 3276306540349.0, -47223648122716.0 / 6934462133451.0, 71187472546993.0 / 9669769126921.0, -13330509492149.0 / 9695768672337.0, 11565764226357.0 / 8513123442827.0),
    ),
    b=(0.0, 0.0, 9164257142617.0 / 17756377923965.0, -10812980402763.0 / 74029279521829.0, 1335994250573.0 / 5691609445217.0, 2273837961795.0 / 8368240463276.0, 247.0 / 2000.0),
    order=4,
    b_embedded=(0.0, 0.0, 4469248916618.0 / 8635866897933.0, -621260224600.0 / 4094290005349.0, 696572312987.0 / 2942599194819.0, 1532940081127.0 / 5565293938103.0, 2441.0 / 20000.0),
    embedded_order=3,
)
ARK437L2SA_IMPLICIT = Tableau(
    c=ARK437L2SA_EXPLICIT.c,
    a=(
        (),
        (1235.0 / 10000.0, 1235.0 / 10000.0),
        (624185399699.0 / 4186980696204.0, 624185399699.0 / 4186980696204.0, 1235.0 / 10000.0),
        (1258591069120.0 / 10082082980243.0, 1258591069120.0 / 10082082980243.0, -322722984531.0 / 8455138723562.0, 1235.0 / 10000.0),
        (-436103496990.0 / 5971407786587.0, -436103496990.0 / 5971407786587.0, -2689175662187.0 / 11046760208243.0, 4431412449334.0 / 12995360898505.0, 1235.0 / 10000.0),
        (-2207373168298.0 / 14430576638973.0, -2207373168298.0 / 14430576638973.0, 242511121179.0 / 3358618340039.0, 3145666661981.0 / 7780404714551.0, 5882073923981.0 / 14490790706663.0, 1235.0 / 10000.0),
        (0.0, 0.0, 9164257142617.0 / 17756377923965.0, -10812980402763.0 / 74029279521829.0, 1335994250573.0 / 5691609445217.0, 2273837961795.0 / 8368240463276.0, 1235.0 / 10000.0),
    ),
    b=(0.0, 0.0, 9164257142617.0 / 17756377923965.0, -10812980402763.0 / 74029279521829.0, 1335994250573.0 / 5691609445217.0, 2273837961795.0 / 8368240463276.0, 1235.0 / 10000.0),
    order=4,
    b_embedded=(0.0, 0.0, 4469248916618.0 / 8635866897933.0, -621260224600.0 / 4094290005349.0, 696572312987.0 / 2942599194819.0, 1532940081127.0 / 5565293938103.0, 2441.0 / 20000.0),
    embedded_order=3,
)
ARK437L2SA_TABLEAU = TableauImex(
    explicit=ARK437L2SA_EXPLICIT,
    implicit=ARK437L2SA_IMPLICIT,
    short_name="ARK437L2SA",
    full_name="ARK4(3)7L[2]SA",
)
KENNEDY_CARPENTER43_7_TABLEAU = ARK437L2SA_TABLEAU


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records accepted/rejected adaptive-step monitor events.
# Provides: call_monitored.
@with_adaptive_step_monitoring
class SchemeKennedyCarpenter43_7:
    """Adaptive Kennedy-Carpenter ARK4(3)7L[2]SA IMEX method."""

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

    descriptor = SchemeDescriptor("KC43-7", "Kennedy-Carpenter 4(3) 7-stage")
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

    tableau = KENNEDY_CARPENTER43_7_TABLEAU

    def __init__(
        self,
        dynamics: DynamicsSplitLike,
        allocator: AllocatorLike,
        resolvent: Resolvent,
        *,
        configuration: SchemeConfiguration | None = None,
        linear_fixed: SchemeLinearFixedLike | None = None,
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
            linear_fixed=linear_fixed,
        )
        self.step_control = self.adaptive_step.step_control
        self.call_body = self.call_specialized if linear_fixed is not None else self.call_inline
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
    "ARK437L2SA_EXPLICIT",
    "ARK437L2SA_IMPLICIT",
    "ARK437L2SA_TABLEAU",
    "KENNEDY_CARPENTER43_7_TABLEAU",
    "SchemeKennedyCarpenter43_7",
]
