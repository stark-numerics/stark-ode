from __future__ import annotations

"""Use the high-level interface with the native Python engine."""

from array import array

from stark import Configuration, Interval, Layout, Method, System
from stark.engines import EngineNative
from stark.methods.schemes import SchemeCashKarp


def exponential_decay(t, state, out):
    del t
    out.dy[0] = -0.5 * state.y[0]


system = System(
    derivative=exponential_decay,
    layout=Layout({"y": {"translation": "dy", "shape": (1,)}}),
)
ivp = system.ivp(
    initial={"y": array("d", [2.0])},
    interval=Interval(present=0.0, step=0.1, stop=2.0),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNative,
    configuration=Configuration(check_progress=False),
)

for interval, state in ivp.integrate():
    print(f"{interval.present:.3f}", state.y[0])
