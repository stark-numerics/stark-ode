"""Represent nested structured state with named `Frame` paths.

The minimal structured-state route is to name nested fields in a `Frame`.
STARK then allocates matching state and translation objects through the
selected engine. Nesting alone is not a reason to write a custom allocator;
reserve that path for existing foreign model objects with their own storage,
constructors, or invariants.
"""

from __future__ import annotations

import numpy as np

from stark import DynamicsStyle, Field, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeRK4


@DynamicsStyle.accepts_instant_writes
def harmonic_motion(_time: float, state, out) -> None:
    out.delta.particle.position[:] = state.model.particle.velocity
    out.delta.particle.velocity[:] = -state.model.particle.position


def build_system() -> System:
    frame = Frame(
        (
            Field(
                "model.particle.position",
                translation="delta.particle.position",
                shape=(1,),
            ),
            Field(
                "model.particle.velocity",
                translation="delta.particle.velocity",
                shape=(1,),
            ),
        )
    )
    return System(dynamics=harmonic_motion, frame=frame)


if __name__ == "__main__":
    ivp = build_system().ivp(
        initial={
            "model": {
                "particle": {
                    "position": np.array([1.0]),
                    "velocity": np.array([0.0]),
                }
            }
        },
        interval=Interval(present=0.0, step=0.1, stop=0.5),
        method=Method(SchemeRK4),
        engine=EngineNumpy,
    )

    print("Nested structured state through Frame paths")
    for interval, state in ivp.stable_trajectory():
        position = state.model.particle.position[0]
        velocity = state.model.particle.velocity[0]
        print(f"t={interval.present:.1f}, x={position:.6f}, v={velocity:.6f}")
