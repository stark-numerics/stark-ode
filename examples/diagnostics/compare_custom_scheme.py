"""Compare a user-defined scheme against a built-in method.

`ComparisonRunner` works at the same level as a normal STARK solve: each entry
is a `Method`. That means a contributor can write a new scheme class, wrap it
in `Method(...)`, and compare it with existing methods before committing to a
longer run.
"""

from __future__ import annotations

import numpy as np

from stark import Frame, Interval, Method, System
from stark.diagnostics.comparison import ComparisonEntry, ComparisonProblem, ComparisonRunner
from stark.engines import EngineNumpy
from stark.methods import SchemeRK4


def decay_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


class ForwardEuler:
    """Small fixed-step scheme written in the public scheme shape."""

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


def diagnostics(state) -> dict[str, float]:
    return {"y": float(state.y[0])}


system = System(
    derivative=decay_rhs,
    frame=Frame.scalar("y", translation="dy"),
)

template = system.ivp(
    initial={"y": np.array([2.0])},
    interval=Interval(present=0.0, step=0.1, stop=1.0),
    method=Method(SchemeRK4),
    engine=EngineNumpy,
)

problem = ComparisonProblem(
    "scalar decay",
    template,
    diagnostics=diagnostics,
    description="A custom ForwardEuler scheme is compared through the normal Method API.",
)

entries = [
    ComparisonEntry("custom ForwardEuler", Method(ForwardEuler)),
    ComparisonEntry("built-in RK4", Method(SchemeRK4)),
]

if __name__ == "__main__":
    print(ComparisonRunner(problem, entries, repeats=1)())
