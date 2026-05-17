from __future__ import annotations

"""Ask STARK for evenly spaced output without forcing fixed solver steps."""

import numpy as np

from stark import Interval
from stark.interface import StarkIVP


def oscillator_rhs(t: float, y: np.ndarray) -> np.ndarray:
    del t
    return np.array([y[1], -y[0]])


ivp = StarkIVP(
    derivative=oscillator_rhs,
    initial=np.array([1.0, 0.0]),
    interval=Interval(present=0.0, step=0.05, stop=1.0),
)
build = ivp.build()

print("Checkpointed harmonic oscillator")
for interval, state in build.integrator(
    build.marcher,
    build.interval,
    build.initial,
    checkpoints=4,
):
    position, velocity = state.value
    print(f"checkpoint t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")

