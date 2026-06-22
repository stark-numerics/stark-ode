"""Lesson 3: use the CuPy backend with GPU-backed arrays."""

from __future__ import annotations

# CuPy looks very close to NumPy at the derivative level.  The arrays are
# mutable and support familiar vector operations, so the derivative can remain
# in-place.  The difference is operational: work is queued on the GPU, so timing
# code must synchronize before stopping the clock or reading diagnostics.
#
# Whether CuPy is faster depends heavily on the machine, GPU, driver, array
# size, and launch overhead.  Lesson 4 treats this as an observation rather
# than a promise.
#
# In a source checkout, run from the ``stark-ode`` directory with:
#
#     python -m examples.case_studies.backends.lesson_03_cupy

import numpy as np

from stark import Configuration, DerivativeStyle, Frame, Interval, Method, System
from stark.methods.schemes import SchemeCashKarp

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None


DIFFUSION = 0.01
REACTION = 0.25
SIZE = 256


def initial_cupy(size: int = SIZE):
    if cp is None:
        raise RuntimeError("CuPy is not installed.")
    x = cp.linspace(0.0, 1.0, size, endpoint=False)
    return 0.55 + 0.15 * cp.sin(2.0 * cp.pi * x) + 0.05 * cp.cos(6.0 * cp.pi * x)


if cp is not None:

    @DerivativeStyle.kernel_accepts_instant_writes(state=("u",), translation=("du",), parameters=(DIFFUSION, REACTION))
    def rhs_cupy(t, u, du, diffusion: float, reaction: float) -> None:
        """In-place CuPy derivative.

        The syntax mirrors NumPy because CuPy arrays are mutable.  Generated
        Algebraist kernels and CuPy array operations run on GPU-backed arrays.
        """

        laplacian = cp.roll(u, -1) - 2.0 * u + cp.roll(u, 1)
        du[...] = diffusion * laplacian + reaction * u * (1.0 - u)
else:
    rhs_cupy = None


def build_cupy_ivp(*, size: int = SIZE):
    if cp is None or rhs_cupy is None:
        raise RuntimeError("CuPy is not installed.")
    from stark.engines import EngineCupy

    frame = Frame({"u": {"translation": "du", "shape": (size,)}})
    system = System(derivative=rhs_cupy, frame=frame)
    return system.ivp(
        initial={"u": initial_cupy(size)},
        interval=Interval(present=0.0, step=0.005, stop=0.015),
        method=Method(scheme=SchemeCashKarp),
        engine=EngineCupy,
        configuration=Configuration(check_progress=False),
    )


def synchronize_cupy() -> None:
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def main() -> None:
    print("Lesson 3: CuPy backend")
    print("======================")
    print("CuPy keeps the in-place derivative style, but the arrays live on the GPU.")
    print("Synchronize before timing or converting the final state back to NumPy.")
    print()

    if cp is None:
        print("CuPy is not installed; skipping this lesson.")
        return

    ivp = build_cupy_ivp()
    result = ivp.final_result()
    synchronize_cupy()
    u = cp.asnumpy(result.state.u)

    print(f"steps={result.steps}, final t={result.interval.present:.4f}")
    print(f"min={u.min():.5f}, max={u.max():.5f}, mean={u.mean():.5f}")
    print()
    print("What to notice:")
    print("- The derivative is in-place, like NumPy, because CuPy arrays are mutable.")
    print("- Timing GPU work without synchronization usually under-reports cost.")
    print("- A GPU may win only once the array work is large enough to amortize overhead.")


if __name__ == "__main__":
    main()
