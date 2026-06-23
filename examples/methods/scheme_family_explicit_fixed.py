"""Use a fixed-step explicit scheme.

Fixed explicit schemes are the simplest method family: they do no implicit
solve and no adaptive error control. The caller chooses the step size, and the
scheme advances by that size until the interval stop is reached.
"""

from __future__ import annotations

import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeRK4


def decay_rhs(t, state, out) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


if __name__ == "__main__":
    system = System(
        derivative=decay_rhs,
        frame=Frame.scalar("y", translation="dy"),
    )
    ivp = system.ivp(
        initial={"y": np.array([2.0])},
        interval=Interval(present=0.0, step=0.05, stop=0.5),
        method=Method(SchemeRK4),
        engine=EngineNumpy,
    )
    result = ivp.final_result()

    print("Explicit fixed scheme: RK4")
    print(f"accepted steps: {result.steps}")
    print(f"final t:        {result.interval.present:.6f}")
    print(f"final y:        {result.state.y[0]:.8f}")
