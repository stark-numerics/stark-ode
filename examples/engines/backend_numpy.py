"""Use the high-level interface with the standard NumPy engine."""

from __future__ import annotations

import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


def harmonic_oscillator(t, state, out):
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


if __name__ == "__main__":
    system = System(
        derivative=harmonic_oscillator,
        frame=Frame.vector("y", translation="dy", length=2),
    )
    ivp = system.ivp(
        initial={"y": np.array([1.0, 0.0])},
        interval=Interval(present=0.0, step=0.05, stop=6.283185307179586),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )

    print("NumPy backend")
    for interval, state in ivp.integrate():
        print(f"{interval.present:.3f}", state.y)
