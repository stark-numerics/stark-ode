"""Solve a structured two-field oscillator without flattening user state."""

from __future__ import annotations

import numpy as np

from stark import DerivativeStyle, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


@DerivativeStyle.accepts_instant_writes
def oscillator(t: float, state, out) -> None:
    del t
    out.dx[:] = state.v
    out.dv[:] = -state.x


if __name__ == "__main__":
    frame = Frame.from_fields(
        ("x", "dx", (1,)),
        ("v", "dv", (1,)),
    )
    system = System(derivative=oscillator, frame=frame)
    ivp = system.ivp(
        initial={"x": np.array([1.0]), "v": np.array([0.0])},
        interval=Interval(present=0.0, step=0.05, stop=0.25),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )

    print("Structured two-field oscillator")
    for interval, state in ivp.stable_trajectory():
        print(f"t={interval.present:.2f}, x={state.x[0]:.6f}, v={state.v[0]:.6f}")
