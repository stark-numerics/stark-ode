"""Use a fixed-step IMEX scheme.

IMEX schemes split the derivative into explicit and implicit parts. The
explicit part is advanced directly; the implicit part is handed to the chosen
resolvent.
"""

from __future__ import annotations

import numpy as np

from stark import Configuration, Derivative, Frame, Interval, Method, System, Tolerance
from stark.engines import EngineNumpy
from stark.methods import ResolventPicard, SchemeIMEXEuler


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
        resolvent_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-8),
        resolvent_maximum_steps=12,
    )
    ivp = system.ivp(
        initial={"y": np.array([2.0])},
        interval=Interval(present=0.0, step=0.05, stop=0.5),
        method=Method(SchemeIMEXEuler, resolvent=ResolventPicard),
        engine=EngineNumpy,
        configuration=configuration,
    )
    result = ivp.final_result()

    print("IMEX fixed scheme: IMEX Euler + Picard")
    print(f"accepted steps: {result.steps}")
    print(f"final t:        {result.interval.present:.6f}")
    print(f"final y:        {result.state.y[0]:.8f}")
