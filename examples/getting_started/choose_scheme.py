from __future__ import annotations

"""Choose a built-in scheme through `Method`."""

import numpy as np

from stark import (
    Configuration,
    Interval,
    Layout,
    Method,
    System,
)
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeDormandPrince


def oscillator_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


system = System(
    derivative=oscillator_rhs,
    layout=Layout({"y": {"translation": "dy", "shape": (2,)}}),
)
ivp = system.ivp(
    initial={"y": np.array([1.0, 0.0])},
    interval=Interval(present=0.0, step=0.05, stop=0.5),
    method=Method(scheme=SchemeDormandPrince),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)

print("Dormand-Prince harmonic oscillator")
for interval, state in ivp.integrate():
    position, velocity = state.y
    print(f"t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")
