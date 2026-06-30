"""Use an adaptive IMEX scheme.

Adaptive IMEX methods are the natural high-level choice when a problem has a
cheap explicit part and a stability-limited implicit part.
"""

from __future__ import annotations

import numpy as np

from stark import Configuration, Derivative, Frame, Interval, Method, System, Tolerance
from stark.engines import EngineNumpy
from stark.methods import ResolventPicard, SchemeKennedyCarpenter32


def implicit_rhs(t, state, out) -> None:
    del t
    out.dy[0] = -0.35 * state.y[0]


def explicit_rhs(t, state, out) -> None:
    del t
    out.dy[0] = -0.15 * state.y[0]


if __name__ == "__main__":
    system = System(
        derivative=Derivative.split(
            implicit=Derivative(implicit_rhs),
            explicit=Derivative(explicit_rhs),
        ),
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
        method=Method(SchemeKennedyCarpenter32, resolvent=ResolventPicard),
        engine=EngineNumpy,
        configuration=configuration,
    )
    result = ivp.final_result()

    print("IMEX adaptive scheme: Kennedy-Carpenter 3(2) + Picard")
    print(f"accepted steps: {result.steps}")
    print(f"final t:        {result.interval.present:.6f}")
    print(f"final y:        {result.state.y[0]:.8f}")
