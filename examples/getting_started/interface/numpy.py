from __future__ import annotations

"""Use the high-level interface with the standard NumPy engine."""

import numpy as np

from stark import Configuration, Interval, StarkLayout, StarkMethod, StarkSystem
from stark.engines import StarkEngineNumpy
from stark.schemes import SchemeCashKarp


def harmonic_oscillator(t, state, out):
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


system = StarkSystem(
    derivative=harmonic_oscillator,
    layout=StarkLayout({"y": {"translation": "dy", "shape": (2,)}}),
)
ivp = system.ivp(
    initial={"y": np.array([1.0, 0.0])},
    interval=Interval(present=0.0, step=0.05, stop=6.283185307179586),
    method=StarkMethod(scheme=SchemeCashKarp),
    engine=StarkEngineNumpy,
    configuration=Configuration(check_progress=False),
)

for interval, state in ivp.integrate():
    print(f"{interval.present:.3f}", state.y)
