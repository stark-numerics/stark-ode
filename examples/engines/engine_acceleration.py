"""Inspect whether the NumPy engine is using compiled acceleration."""

from __future__ import annotations

from stark import Frame
from stark.engines import AcceleratorNone, EngineNumpy


if __name__ == "__main__":
    frame = Frame.vector("y", translation="dy", length=2)

    default_engine = EngineNumpy(frame)
    unaccelerated_engine = EngineNumpy(
        frame,
        accelerator=AcceleratorNone(),
    )

    print("Default NumPy engine")
    print(default_engine)
    print()
    print("Simulated default fallback when Numba is not installed")
    print(unaccelerated_engine)
    print()
    print("EngineNumpy tries to use Numba by default.")
    print("If Numba is not installed, the default engine falls back to this unaccelerated form.")
    print("A slow run is easiest to diagnose by inspecting the engine repr.")
    print("When accelerator='none', generated CPU kernels are not compiled.")
