from __future__ import annotations

"""Smallest high-level STARK solve: one scalar initial value problem."""

from stark import Interval
from stark.interface import StarkIVP


def exponential_decay(t: float, y: float) -> float:
    del t
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=2.0,
    interval=Interval(present=0.0, step=0.1, stop=1.0),
)

print("Scalar exponential decay")
for interval, state in ivp.integrate():
    print(f"t={interval.present:.1f}, y={state.value:.6f}")

