"""Use an in-place derivative signature with a structured system state."""

from __future__ import annotations

import numpy as np

from stark import DerivativeStyle, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


@DerivativeStyle.accepts_instant_writes
def oscillator_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


def build_ivp():
    system = System(
        derivative=oscillator_rhs,
        frame=Frame.vector("y", translation="dy", length=2),
    )
    return system.ivp(
        initial={"y": np.array([1.0, 0.0])},
        interval=Interval(present=0.0, step=0.05, stop=0.5),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )


def main() -> None:
    print("In-place NumPy derivative")
    for interval, state in build_ivp().integrate():
        position, velocity = state.y
        print(f"t={interval.present:.2f}, x={position:.6f}, v={velocity:.6f}")


if __name__ == "__main__":
    main()
