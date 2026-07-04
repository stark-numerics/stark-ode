"""Use an adaptive explicit scheme.

Adaptive explicit schemes estimate local error and choose accepted step sizes
for non-stiff problems. They keep the problem declaration unchanged while the
method stack controls accuracy.
"""

from __future__ import annotations

import numpy as np

from stark import Configuration, Frame, Interval, Method, System, Tolerance
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


def decay_rhs(t, state, out) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


if __name__ == "__main__":
    system = System(
        dynamics=decay_rhs,
        frame=Frame.scalar("y", translation="dy"),
    )
    configuration = Configuration(
        scheme_tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
    )
    ivp = system.ivp(
        initial={"y": np.array([2.0])},
        interval=Interval(present=0.0, step=0.2, stop=0.5),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
        configuration=configuration,
    )
    result = ivp.final_result()

    print("Explicit adaptive scheme: Cash-Karp")
    print(f"accepted steps: {result.steps}")
    print(f"final t:        {result.interval.present:.6f}")
    print(f"final y:        {result.state.y[0]:.8f}")
