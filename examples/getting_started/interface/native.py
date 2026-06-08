from __future__ import annotations

"""Use the high-level interface with the native Python engine."""

from array import array

from stark import Configuration, Interval, StarkLayout, StarkMethod, StarkSystem
from stark.engines import StarkEngineNative
from stark.schemes import SchemeCashKarp


def exponential_decay(t, state, out):
    del t
    out.dy[0] = -0.5 * state.y[0]


system = StarkSystem(
    derivative=exponential_decay,
    layout=StarkLayout({"y": {"translation": "dy", "shape": (1,)}}),
)
ivp = system.ivp(
    initial={"y": array("d", [2.0])},
    interval=Interval(present=0.0, step=0.1, stop=2.0),
    method=StarkMethod(scheme=SchemeCashKarp),
    engine=StarkEngineNative,
    configuration=Configuration(check_progress=False),
)

for interval, state in ivp.integrate():
    print(f"{interval.present:.3f}", state.y[0])
