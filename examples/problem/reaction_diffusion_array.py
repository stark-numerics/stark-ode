"""Solve a spatial field as one array-valued state.

Use this pattern when a model is naturally stored as a single NumPy array:
declare one vector field in the `Frame`, write the array derivative in-place,
and let the ordinary high-level `System` path handle the integration.
"""

from __future__ import annotations

import numpy as np

from stark import Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


GRID_SIZE = 64
LENGTH = 6.0
DIFFUSIVITY = 0.08


def initial_profile() -> np.ndarray:
    x = np.linspace(0.0, LENGTH, GRID_SIZE, endpoint=False)
    wave_number = 2.0 * np.pi / LENGTH
    return 0.5 * np.sin(wave_number * x) + 0.5 * np.sin(3.0 * wave_number * x)


def reaction_diffusion_rhs(t: float, state, out) -> None:
    del t
    dx = LENGTH / GRID_SIZE
    laplacian = (np.roll(state.u, 1) - 2.0 * state.u + np.roll(state.u, -1)) / (dx * dx)
    out.du[:] = DIFFUSIVITY * laplacian + state.u - state.u * state.u * state.u


if __name__ == "__main__":
    system = System(
        derivative=reaction_diffusion_rhs,
        frame=Frame.array("u", translation="du", shape=(GRID_SIZE,)),
    )
    ivp = system.ivp(
        initial={"u": initial_profile()},
        interval=Interval(present=0.0, step=1.0e-3, stop=0.1),
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )

    print("Array-valued reaction-diffusion state")
    for interval, state in ivp.integrate(checkpoints=4):
        print(
            f"t={interval.present:.3f}, "
            f"mean={state.u.mean(): .6f}, "
            f"min={state.u.min(): .6f}, "
            f"max={state.u.max(): .6f}"
        )
