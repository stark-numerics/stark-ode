"""High-level STARK solve with one NumPy vector field."""

from __future__ import annotations

import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


def harmonic_oscillator(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


if __name__ == "__main__":
    system = System(
        dynamics=harmonic_oscillator,
        frame=Frame.vector("y", translation="dy", length=2),
    )
    ivp = system.ivp(
        initial={"y": np.array([1.0, 0.0])},
        interval=Interval(present=0.0, step=0.05, stop=0.5),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )

    print("NumPy harmonic oscillator")
    for interval, state in ivp.stable_trajectory():
        position, velocity = state.y
        print(f"t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")
