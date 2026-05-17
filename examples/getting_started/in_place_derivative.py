from __future__ import annotations

"""Use an in-place derivative to avoid per-call array allocation."""

import numpy as np

from stark import Interval
from stark.interface import StarkDerivative, StarkIVP


def oscillator_rhs(t: float, y: np.ndarray, out: np.ndarray) -> None:
    del t
    out[0] = y[1]
    out[1] = -y[0]


ivp = StarkIVP(
    derivative=StarkDerivative.in_place(oscillator_rhs),
    initial=np.array([1.0, 0.0]),
    interval=Interval(present=0.0, step=0.05, stop=0.5),
)

print("In-place NumPy derivative")
for interval, state in ivp.integrate():
    position, velocity = state.value
    print(f"t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")

