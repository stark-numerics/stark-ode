"""Ask STARK for evenly spaced output without forcing fixed solver steps."""

from __future__ import annotations

import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


def oscillator_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


if __name__ == "__main__":
    system = System(
        derivative=oscillator_rhs,
        frame=Frame.vector("y", translation="dy", length=2),
    )
    ivp = system.ivp(
        initial={"y": np.array([1.0, 0.0])},
        interval=Interval(present=0.0, step=0.05, stop=1.0),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )

    print("Checkpointed harmonic oscillator")
    for interval, state in ivp.integrate(checkpoints=4):
        position, velocity = state.y
        print(f"checkpoint t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")
