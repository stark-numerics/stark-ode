from __future__ import annotations

"""High-level STARK solve with one NumPy vector field."""

import numpy as np

from stark import (
    Configuration,
    Interval,
    DerivativeStyle,
    Layout,
    Method,
    System,
)
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp


def harmonic_oscillator(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


layout = Layout({"y": {"translation": "dy", "shape": (2,)}})
system = System(
    derivative=DerivativeStyle.in_place(harmonic_oscillator),
    layout=layout,
)
ivp = system.ivp(
    initial={"y": np.array([1.0, 0.0])},
    interval=Interval(present=0.0, step=0.05, stop=0.5),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)

print("NumPy harmonic oscillator")
for interval, state in ivp.integrate():
    position, velocity = state.y
    print(f"t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")
