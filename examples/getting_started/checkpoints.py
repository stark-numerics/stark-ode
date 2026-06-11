from __future__ import annotations

"""Ask STARK for evenly spaced output without forcing fixed solver steps."""

import numpy as np

from stark import (
    Configuration,
    Interval,
    Layout,
    Method,
    System,
)
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp


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
    interval=Interval(present=0.0, step=0.05, stop=1.0),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)

print("Checkpointed harmonic oscillator")
for interval, state in ivp.integrate(checkpoints=4):
    position, velocity = state.y
    print(f"checkpoint t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")
