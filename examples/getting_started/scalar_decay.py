"""Smallest high-level STARK solve: one scalar-like field."""

from __future__ import annotations

import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


def exponential_decay(t: float, state, out) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


def build_ivp():
    system = System(
        derivative=exponential_decay,
        frame=Frame.scalar("y", translation="dy"),
    )
    return system.ivp(
        initial={"y": np.array([2.0])},
        interval=Interval(present=0.0, step=0.1, stop=1.0),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )


def main() -> None:
    print("Scalar exponential decay")
    for interval, state in build_ivp().integrate():
        print(f"t={interval.present:.1f}, y={state.y[0]:.6f}")


if __name__ == "__main__":
    main()
