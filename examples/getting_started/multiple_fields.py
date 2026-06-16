from __future__ import annotations

"""Solve a structured two-field oscillator without flattening user state."""

import numpy as np

from stark import Configuration, DerivativeStyle, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp


@DerivativeStyle.in_place
def oscillator(t: float, state, out) -> None:
    del t
    out.dx[:] = state.v
    out.dv[:] = -state.x


frame = Frame(
    {
        "x": {"translation": "dx", "shape": (1,)},
        "v": {"translation": "dv", "shape": (1,)},
    }
)
system = System(derivative=oscillator, frame=frame)
ivp = system.ivp(
    initial={"x": np.array([1.0]), "v": np.array([0.0])},
    interval=Interval(present=0.0, step=0.05, stop=0.25),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)

print("Structured two-field oscillator")
for interval, state in ivp.integrate():
    print(f"t={interval.present:.2f}, x={state.x[0]:.6f}, v={state.v[0]:.6f}")
