from __future__ import annotations

from stark.contracts import IntervalLike, State
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_adaptive_step_monitoring
from stark.methods.schemes.execution.step_control import (
    SchemeStepControl,
)
from stark.methods.schemes.imex._support import (
    imex_display_resolvent_problem,
    imex_snapshot_state,
)
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.method.tableau import ButcherTableau, ButcherTableauImex
from stark.methods.schemes.imex.adaptive._kennedy_carpenter import SchemeKennedyCarpenterAdaptive


ARK548L2SAB_EXPLICIT = ButcherTableau(
    c=(0.0, 4.0 / 9.0, 6456083330201.0 / 8509243623797.0, 1632083962415.0 / 14158861528103.0, 6365430648612.0 / 17842476412687.0, 18.0 / 25.0, 191.0 / 200.0, 1.0),
    a=(
        (),
        (4.0 / 9.0,),
        (2366667076620.0 / 8822750406821.0, 2366667076620.0 / 8822750406821.0),
        (-257962897183.0 / 4451812247028.0, -257962897183.0 / 4451812247028.0, 128530224461.0 / 14379561246022.0),
        (-486229321650.0 / 11227943450093.0, -486229321650.0 / 11227943450093.0, -225633144460.0 / 6633558740617.0, 1741320951451.0 / 6824444397158.0),
        (621307788657.0 / 4714163060173.0, 621307788657.0 / 4714163060173.0, -125196015625.0 / 3866852212004.0, 940440206406.0 / 7593089888465.0, 961109811699.0 / 6734810228204.0),
        (2036305566805.0 / 6583108094622.0, 2036305566805.0 / 6583108094622.0, -3039402635899.0 / 4450598839912.0, -1829510709469.0 / 31102090912115.0, -286320471013.0 / 6931253422520.0, 8651533662697.0 / 9642993110008.0),
        (0.0, 0.0, 3517720773327.0 / 20256071687669.0, 4569610470461.0 / 17934693873752.0, 2819471173109.0 / 11655438449929.0, 3296210113763.0 / 10722700128969.0, -1142099968913.0 / 5710983926999.0),
    ),
    b=(0.0, 0.0, 3517720773327.0 / 20256071687669.0, 4569610470461.0 / 17934693873752.0, 2819471173109.0 / 11655438449929.0, 3296210113763.0 / 10722700128969.0, -1142099968913.0 / 5710983926999.0, 2.0 / 9.0),
    order=5,
    b_embedded=(0.0, 0.0, 520639020421.0 / 8300446712847.0, 4550235134915.0 / 17827758688493.0, 1482366381361.0 / 6201654941325.0, 5551607622171.0 / 13911031047899.0, -5266607656330.0 / 36788968843917.0, 1074053359553.0 / 5740751784926.0),
    embedded_order=4,
)
ARK548L2SAB_IMPLICIT = ButcherTableau(
    c=ARK548L2SAB_EXPLICIT.c,
    a=(
        (),
        (2.0 / 9.0, 2.0 / 9.0),
        (2366667076620.0 / 8822750406821.0, 2366667076620.0 / 8822750406821.0, 2.0 / 9.0),
        (-257962897183.0 / 4451812247028.0, -257962897183.0 / 4451812247028.0, 128530224461.0 / 14379561246022.0, 2.0 / 9.0),
        (-486229321650.0 / 11227943450093.0, -486229321650.0 / 11227943450093.0, -225633144460.0 / 6633558740617.0, 1741320951451.0 / 6824444397158.0, 2.0 / 9.0),
        (621307788657.0 / 4714163060173.0, 621307788657.0 / 4714163060173.0, -125196015625.0 / 3866852212004.0, 940440206406.0 / 7593089888465.0, 961109811699.0 / 6734810228204.0, 2.0 / 9.0),
        (2036305566805.0 / 6583108094622.0, 2036305566805.0 / 6583108094622.0, -3039402635899.0 / 4450598839912.0, -1829510709469.0 / 31102090912115.0, -286320471013.0 / 6931253422520.0, 8651533662697.0 / 9642993110008.0, 2.0 / 9.0),
        (0.0, 0.0, 3517720773327.0 / 20256071687669.0, 4569610470461.0 / 17934693873752.0, 2819471173109.0 / 11655438449929.0, 3296210113763.0 / 10722700128969.0, -1142099968913.0 / 5710983926999.0, 2.0 / 9.0),
    ),
    b=ARK548L2SAB_EXPLICIT.b,
    order=5,
    b_embedded=ARK548L2SAB_EXPLICIT.b_embedded,
    embedded_order=4,
)
ARK548L2SAB_TABLEAU = ButcherTableauImex(
    explicit=ARK548L2SAB_EXPLICIT,
    implicit=ARK548L2SAB_IMPLICIT,
    short_name="ARK548L2SAb",
    full_name="ARK5(4)8L[2]SA(b)",
)
KENNEDY_CARPENTER54B_TABLEAU = ARK548L2SAB_TABLEAU


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, short_name, full_name, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records accepted/rejected adaptive-step monitor events.
# Provides: call_monitored.
@with_adaptive_step_monitoring
class SchemeKennedyCarpenter54b(SchemeKennedyCarpenterAdaptive):
    """Adaptive Kennedy-Carpenter ARK5(4)8L[2]SA(b) IMEX method."""

    step_control: SchemeStepControl
    descriptor = SchemeDescriptor("KC54b", "Kennedy-Carpenter 5(4) b")
    display_resolvent_problem = classmethod(imex_display_resolvent_problem)
    snapshot_state = imex_snapshot_state
    tableau = KENNEDY_CARPENTER54B_TABLEAU

    def __call__(self, interval: IntervalLike, state: State) -> float:
        return self.redirect_call(interval, state)


__all__ = [
    "ARK548L2SAB_EXPLICIT",
    "ARK548L2SAB_IMPLICIT",
    "ARK548L2SAB_TABLEAU",
    "KENNEDY_CARPENTER54B_TABLEAU",
    "SchemeKennedyCarpenter54b",
]
