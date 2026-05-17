import numpy as np

from stark import Interval
from stark.interface import StarkIVP


def harmonic_oscillator(t, y):
    # y[0] = position
    # y[1] = velocity
    return np.array([y[1], -y[0]])


ivp = StarkIVP(
    derivative=harmonic_oscillator,
    initial=np.array([1.0, 0.0]),
    interval=Interval(present=0.0, step=0.05, stop=6.283185307179586),
)

for interval, state in ivp.integrate():
    print(f"{interval.present:.3f}", state.value)