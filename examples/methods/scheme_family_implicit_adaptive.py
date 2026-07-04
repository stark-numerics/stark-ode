"""Use an adaptive implicit scheme.

Adaptive implicit schemes combine an error-controlled time step with an
implicit stage solve. This is useful when stability, rather than only
accuracy, influences the practical step size.
"""

from __future__ import annotations

import numpy as np

from stark import Configuration, Frame, Interval, Method, System, Tolerance
from stark.engines import EngineNumpy
from stark.methods import ResolventPicard, SchemeKvaerno5


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
        resolvent_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-8),
        resolvent_maximum_steps=12,
    )
    ivp = system.ivp(
        initial={"y": np.array([2.0])},
        interval=Interval(present=0.0, step=0.15, stop=0.5),
        method=Method(SchemeKvaerno5, resolvent=ResolventPicard),
        engine=EngineNumpy,
        configuration=configuration,
    )
    result = ivp.final_result()

    print("Implicit adaptive scheme: Kvaerno5 + Picard")
    print(f"accepted steps: {result.steps}")
    print(f"final t:        {result.interval.present:.6f}")
    print(f"final y:        {result.state.y[0]:.8f}")
