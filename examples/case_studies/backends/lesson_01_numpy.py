"""Lesson 1: use the NumPy backend as the baseline."""

from __future__ import annotations

# This case study compares STARK's array backends on the same small PDE-like
# problem.  The first lesson is deliberately ordinary: NumPy arrays, an
# in-place derivative, and a Cash-Karp explicit scheme.
#
# Read this file as the reference syntax for a named-field System solve:
#
# - Frame describes the user state field ``u`` and solver translation field
#   ``du``.
# - DerivativeStyle.kernel adapts an in-place array kernel to STARK's derivative
#   contract.
# - EngineNumpy chooses NumPy-backed carriers and Algebraist-generated field
#   operations.
#
# Lesson 4 compares timings.  The important baseline there is the explicit
# ``accelerated=False`` path, which disables the Numba accelerator so that the
# table can separate plain NumPy from NumPy+Numba.
#
# In a source checkout, run from the ``stark-ode`` directory with:
#
#     python -m examples.case_studies.backends.lesson_01_numpy

import numpy as np

from stark import Configuration, DerivativeStyle, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.engines.accelerators import AcceleratorNone
from stark.methods.schemes import SchemeCashKarp


DIFFUSION = 0.01
REACTION = 0.25
SIZE = 256


def initial_numpy(size: int = SIZE):
    # The same smooth initial condition is used in every backend lesson.  It is
    # large enough to exercise vector operations, but simple enough that the
    # final min/max/mean diagnostics should match across backends.
    x = np.linspace(0.0, 1.0, size, endpoint=False)
    return 0.55 + 0.15 * np.sin(2.0 * np.pi * x) + 0.05 * np.cos(6.0 * np.pi * x)


@DerivativeStyle.kernel(state=("u",), translation=("du",), parameters=(DIFFUSION, REACTION))
def rhs_numpy(u, du, diffusion: float, reaction: float) -> None:
    """In-place NumPy derivative.

    NumPy arrays are mutable, so the kernel writes into the translation field
    that STARK provides.  This is the most direct style for NumPy and CuPy.
    """

    laplacian = np.roll(u, -1) - 2.0 * u + np.roll(u, 1)
    du[...] = diffusion * laplacian + reaction * u * (1.0 - u)


def build_numpy_ivp(*, accelerated: bool = False, size: int = SIZE):
    # Frame is the bridge from user field names to solver translation names.
    # STARK owns the translation storage for this high-level path.
    frame = Frame({"u": {"translation": "du", "shape": (size,)}})
    system = System(derivative=rhs_numpy, frame=frame)

    if accelerated:
        # EngineNumpy's ordinary path may use the configured Numba accelerator
        # for generated Algebraist kernels.  This is useful for repeated
        # same-shaped solves, but the first solve may include compilation.
        engine = EngineNumpy
    else:
        # The explicit baseline disables acceleration.  Lesson 4 uses this row
        # as the plain NumPy reference for speed factors.
        engine = lambda frame: EngineNumpy(frame, accelerator=AcceleratorNone())

    return system.ivp(
        initial={"u": initial_numpy(size)},
        interval=Interval(present=0.0, step=0.005, stop=0.015),
        method=Method(scheme=SchemeCashKarp),
        engine=engine,
        configuration=Configuration(check_progress=False),
    )


def main() -> None:
    print("Lesson 1: NumPy backend")
    print("=======================")
    print("This is the reference backend lesson: named field, in-place derivative,")
    print("Frame-backed System, and EngineNumpy.")
    print()

    ivp = build_numpy_ivp(accelerated=False)
    result = ivp.final_result()
    u = result.state.u

    print(f"steps={result.steps}, final t={result.interval.present:.4f}")
    print(f"min={u.min():.5f}, max={u.max():.5f}, mean={u.mean():.5f}")
    print()
    print("What to notice:")
    print("- The derivative mutates du because NumPy arrays are mutable.")
    print("- The Frame names the state field u and the translation field du.")
    print("- Later lessons should produce the same final diagnostics, within dtype noise.")


if __name__ == "__main__":
    main()
