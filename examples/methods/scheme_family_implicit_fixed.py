"""Use a fixed-step implicit scheme.

Implicit schemes solve a stage equation on every step. Picard is used here so
the example can focus on the scheme family without introducing Jacobians or
linear correction solvers.
"""

from __future__ import annotations

import numpy as np

from stark import Configuration, Frame, Interval, Method, System, Tolerance
from stark.engines import EngineNumpy
from stark.methods import ResolventPicard, SchemeBackwardEuler


def decay_rhs(t, state, out) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


if __name__ == "__main__":
    system = System(
        derivative=decay_rhs,
        frame=Frame.scalar("y", translation="dy"),
    )
    configuration = Configuration(
        resolvent_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-8),
        resolvent_maximum_steps=12,
    )
    ivp = system.ivp(
        initial={"y": np.array([2.0])},
        interval=Interval(present=0.0, step=0.05, stop=0.5),
        method=Method(SchemeBackwardEuler, resolvent=ResolventPicard),
        engine=EngineNumpy,
        configuration=configuration,
    )
    result = ivp.final_result()

    print("Implicit fixed scheme: Backward Euler + Picard")
    print(f"accepted steps: {result.steps}")
    print(f"final t:        {result.interval.present:.6f}")
    print(f"final y:        {result.state.y[0]:.8f}")
