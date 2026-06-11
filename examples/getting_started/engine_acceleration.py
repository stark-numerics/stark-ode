from __future__ import annotations

"""Inspect whether the NumPy engine is using compiled acceleration."""

import numpy as np

from stark import Layout
from stark.accelerators import AcceleratorNone
from stark.engines import EngineNumpy


layout = Layout({"y": {"translation": "dy", "shape": (2,)}})

default_engine = EngineNumpy(layout)
unaccelerated_engine = EngineNumpy(
    layout,
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

# Keep a visible array nearby so the layout shape reads like the state it supports.
initial_y = np.array([1.0, 0.0])
print(f"Example initial field shape: y={initial_y.shape}")
