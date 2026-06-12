"""Use a custom fixed-step explicit scheme through `System`.

Custom schemes do not need a separate runner path. If the scheme accepts the
same constructor ingredients as the built-in schemes, `Method` can select
it and `System` can build the IVP in the usual way.

The essential scheme surface is still small:

- `__call__(interval, state) -> float`
- `snapshot_state(state)`
"""

from __future__ import annotations

import numpy as np

from stark import Configuration, Interval, Frame, Method, System
from stark.engines import EngineNumpy


def derivative(t: float, state, out) -> None:
    del t
    out.dy[:] = state.y


class ForwardEuler:
    """Minimal custom fixed-step explicit scheme.

    STARK already includes a production `SchemeEuler`; this class exists only to
    show how a user scheme can be constructed by the high-level method stack.
    """

    def __init__(self, derivative, allocator) -> None:
        self.derivative = derivative
        self.allocator = allocator
        self.delta = allocator.allocate_translation()

    def __call__(self, interval, state) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        self.derivative(interval, state, self.delta)
        (dt * self.delta)(state, state)
        return dt

    def snapshot_state(self, state):
        snapshot = self.allocator.allocate_state()
        self.allocator.copy_state(state, snapshot)
        return snapshot


system = System(
    derivative=derivative,
    frame=Frame({"y": {"translation": "dy", "shape": (1,)}}),
)
ivp = system.ivp(
    initial={"y": np.array([1.0])},
    interval=Interval(present=0.0, step=0.1, stop=0.3),
    method=Method(scheme=ForwardEuler),
    engine=EngineNumpy,
    configuration=Configuration(check_progress=False),
)

for interval, state in ivp.stable_trajectory():
    print(f"t={interval.present:.1f}, y={state.y[0]:.6f}")
