try:
    import jax.numpy as jnp
except ImportError:
    print("JAX is not installed; skipping example.")
    raise SystemExit(0)

from stark import Interval
from stark.interface import StarkIVP


def exponential_decay(t, y):
    return -0.5 * y


ivp = StarkIVP(
    derivative=exponential_decay,
    initial=jnp.array([2.0, 4.0, 8.0]),
    interval=Interval(present=0.0, step=0.1, stop=2.0),
)

for interval, state in ivp.integrate():
    print(f"{interval.present:.3f}", state.value)
