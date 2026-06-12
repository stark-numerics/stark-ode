from __future__ import annotations

"""Use an in-place derivative signature with a structured system state."""

import numpy as np

from stark import (
    Configuration,
    Interval,
    DerivativeStyle,
    Frame,
    Method,
    System,
)
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp


@DerivativeStyle.in_place
def oscillator_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


frame = Frame({"y": {"translation": "dy", "shape": (2,)}})
system = System(derivative=oscillator_rhs, frame=frame)
ivp = system.ivp(
    initial={"y": np.array([1.0, 0.0])},
    interval=Interval(present=0.0, step=0.05, stop=0.5),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)

print("In-place NumPy derivative")
for interval, state in ivp.integrate():
    position, velocity = state.y
    print(f"t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")
