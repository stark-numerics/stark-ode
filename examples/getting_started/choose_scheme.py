from __future__ import annotations

"""Choose a built-in scheme through the high-level StarkIVP interface."""

import numpy as np

from stark import Interval
from stark.interface import StarkIVP
from stark.schemes import SchemeDormandPrince


def oscillator_rhs(t: float, y: np.ndarray) -> np.ndarray:
    del t
    return np.array([y[1], -y[0]])


ivp = StarkIVP(
    derivative=oscillator_rhs,
    initial=np.array([1.0, 0.0]),
    interval=Interval(present=0.0, step=0.05, stop=0.5),
    scheme=SchemeDormandPrince,
)

print("Dormand-Prince harmonic oscillator")
for interval, state in ivp.integrate():
    position, velocity = state.value
    print(f"t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")

