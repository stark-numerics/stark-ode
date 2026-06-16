from __future__ import annotations

"""Use a pure derivative that returns the translation instead of mutating it."""

import numpy as np

from stark import Configuration, DerivativeStyle, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp


@DerivativeStyle.returning
def exponential_decay(t: float, state):
    del t
    # Returning a mapping lets STARK copy the value into the translation field.
    # This style is convenient for array libraries where in-place mutation is
    # not natural.
    return {"dy": -0.5 * state.y}


system = System(
    derivative=exponential_decay,
    frame=Frame({"y": {"translation": "dy", "shape": (1,)}}),
)
ivp = system.ivp(
    initial={"y": np.array([2.0])},
    interval=Interval(present=0.0, step=0.1, stop=0.5),
    method=Method(scheme=SchemeCashKarp),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)

print("Return-style derivative")
for interval, state in ivp.integrate():
    print(f"t={interval.present:.1f}, y={state.y[0]:.6f}")
