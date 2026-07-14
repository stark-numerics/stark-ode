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


ARK548L2SA_EXPLICIT = Tableau(
    c=(0.0, 41.0 / 100.0, 2935347310677.0 / 11292855782101.0, 1426016391358.0 / 7196633302097.0, 92.0 / 100.0, 24.0 / 100.0, 3.0 / 5.0, 1.0),
    a=(
        (),
        (41.0 / 100.0,),
        (367902744464.0 / 2072280473677.0, 677623207551.0 / 8224143866563.0),
        (1268023523408.0 / 10340822734521.0, 0.0, 1029933939417.0 / 13636558850479.0),
        (14463281900351.0 / 6315353703477.0, 0.0, 66114435211212.0 / 5879490589093.0, -54053170152839.0 / 4284798021562.0),
        (14090043504691.0 / 34967701212078.0, 0.0, 15191511035443.0 / 11219624916014.0, -18461159152457.0 / 12425892160975.0, -281667163811.0 / 9011619295870.0),
        (19230459214898.0 / 13134317526959.0, 0.0, 21275331358303.0 / 2942455364971.0, -38145345988419.0 / 4862620318723.0, -1.0 / 8.0, -1.0 / 8.0),
        (-19977161125411.0 / 11928030595625.0, 0.0, -40795976796054.0 / 6384907823539.0, 177454434618887.0 / 12078138498510.0, 782672205425.0 / 8267701900261.0, -69563011059811.0 / 9646580694205.0, 7356628210526.0 / 4942186776405.0),
    ),
    b=(-872700587467.0 / 9133579230613.0, 0.0, 0.0, 22348218063261.0 / 9555858737531.0, -1143369518992.0 / 8141816002931.0, -39379526789629.0 / 19018526304540.0, 32727382324388.0 / 42900044865799.0, 41.0 / 200.0),
    order=5,
    b_embedded=(-975461918565.0 / 9796059967033.0, 0.0, 0.0, 78070527104295.0 / 32432590147079.0, -548382580838.0 / 3424219808633.0, -33438840321285.0 / 15594753105479.0, 3629800801594.0 / 4656183773603.0, 4035322873751.0 / 18575991585200.0),
    embedded_order=4,
)
ARK548L2SA_IMPLICIT = Tableau(
    c=ARK548L2SA_EXPLICIT.c,
    a=(
        (),
        (41.0 / 200.0, 41.0 / 200.0),
        (41.0 / 400.0, -567603406766.0 / 11931857230679.0, 41.0 / 200.0),
        (683785636431.0 / 9252920307686.0, 0.0, -110385047103.0 / 1367015193373.0, 41.0 / 200.0),
        (3016520224154.0 / 10081342136671.0, 0.0, 30586259806659.0 / 12414158314087.0, -22760509404356.0 / 11113319521817.0, 41.0 / 200.0),
        (218866479029.0 / 1489978393911.0, 0.0, 638256894668.0 / 5436446318841.0, -1179710474555.0 / 5321154724896.0, -60928119172.0 / 8023461067671.0, 41.0 / 200.0),
        (1020004230633.0 / 5715676835656.0, 0.0, 25762820946817.0 / 25263940353407.0, -2161375909145.0 / 9755907335909.0, -211217309593.0 / 5846859502534.0, -4269925059573.0 / 7827059040749.0, 41.0 / 200.0),
        (-872700587467.0 / 9133579230613.0, 0.0, 0.0, 22348218063261.0 / 9555858737531.0, -1143369518992.0 / 8141816002931.0, -39379526789629.0 / 19018526304540.0, 32727382324388.0 / 42900044865799.0, 41.0 / 200.0),
    ),
    b=ARK548L2SA_EXPLICIT.b,
    order=5,
    b_embedded=ARK548L2SA_EXPLICIT.b_embedded,
    embedded_order=4,
)
ARK548L2SA_TABLEAU = TableauImex(
    explicit=ARK548L2SA_EXPLICIT,
    implicit=ARK548L2SA_IMPLICIT,
    short_name="ARK548L2SA",
    full_name="ARK5(4)8L[2]SA",
)
KENNEDY_CARPENTER54_TABLEAU = ARK548L2SA_TABLEAU


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records accepted/rejected adaptive-step monitor events.
# Provides: call_monitored.
@with_adaptive_step_monitoring
class SchemeKennedyCarpenter54:
    """Adaptive Kennedy-Carpenter ARK5(4)8L[2]SA IMEX method."""

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

    descriptor = SchemeDescriptor("KC54", "Kennedy-Carpenter 5(4)")
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

    tableau = KENNEDY_CARPENTER54_TABLEAU

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
    "ARK548L2SA_EXPLICIT",
    "ARK548L2SA_IMPLICIT",
    "ARK548L2SA_TABLEAU",
    "KENNEDY_CARPENTER54_TABLEAU",
    "SchemeKennedyCarpenter54",
]
