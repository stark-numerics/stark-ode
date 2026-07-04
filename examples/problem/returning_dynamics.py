"""Use a pure dynamics that returns the translation instead of mutating it."""

from __future__ import annotations

import numpy as np

from stark import DynamicsStyle, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


@DynamicsStyle.accepts_instant_returns
def exponential_decay(t: float, state):
    del t
    # Returning a mapping lets STARK copy the value into the translation field.
    # This style is convenient for array libraries where in-place mutation is
    # not natural.
    return {"dy": -0.5 * state.y}


if __name__ == "__main__":
    system = System(
        dynamics=exponential_decay,
        frame=Frame.scalar("y", translation="dy"),
    )
    ivp = system.ivp(
        initial={"y": np.array([2.0])},
        interval=Interval(present=0.0, step=0.1, stop=0.5),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )

    print("Return-style dynamics")
    for interval, state in ivp.stable_trajectory():
        print(f"t={interval.present:.1f}, y={state.y[0]:.6f}")
