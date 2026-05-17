from __future__ import annotations

"""High-level STARK solve with a NumPy vector state."""

import numpy as np

from stark import Interval
from stark.interface import StarkIVP


def harmonic_oscillator(t: float, y: np.ndarray) -> np.ndarray:
    del t
    return np.array([y[1], -y[0]])


ivp = StarkIVP(
    derivative=harmonic_oscillator,
    initial=np.array([1.0, 0.0]),
    interval=Interval(present=0.0, step=0.05, stop=0.5),
)

print("NumPy harmonic oscillator")
for interval, state in ivp.integrate():
    position, velocity = state.value
    print(f"t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")

