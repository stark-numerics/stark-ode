try:
    import cupy as cp
except ImportError:
    print("CuPy is not installed; skipping example.")
    raise SystemExit(0)

from stark import Interval
from stark.interface import StarkIVP


def exponential_decay(t, y):
    return -0.5 * y


try:
    initial = cp.array([2.0, 4.0, 8.0])
except Exception as error:
    print(f"CuPy is installed but not usable here; skipping example: {error}")
    raise SystemExit(0)


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=initial,
    interval=Interval(present=0.0, step=0.1, stop=2.0),
)

for interval, state in ivp.integrate():
    print(f"{interval.present:.3f}", cp.asnumpy(state.value))