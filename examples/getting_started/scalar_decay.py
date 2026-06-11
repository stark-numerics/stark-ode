from __future__ import annotations

"""Smallest high-level STARK solve: one scalar-like field."""

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


def exponential_decay(t: float, state, out) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


layout = Layout({"y": {"translation": "dy", "shape": (1,)}})
system = System(
    derivative=DerivativeStyle.in_place(exponential_decay),
    layout=layout,
)
ivp = system.ivp(
    initial={"y": np.array([2.0])},
    interval=Interval(present=0.0, step=0.1, stop=1.0),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)

print("Scalar exponential decay")
for interval, state in ivp.integrate():
    print(f"t={interval.present:.1f}, y={state.y[0]:.6f}")
