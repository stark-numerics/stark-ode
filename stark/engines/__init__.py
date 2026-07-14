"""Engine catalogue for backend runtime bundles.

Engines decide where state lives and which algebra kernels a prepared IVP uses.
Most examples can import `EngineNumpy`, `EngineNative`, `EngineCupy`, or
`EngineJax` from this package. Deeper engine modules are primarily for backend
authors and advanced users inspecting storage, carrier, or Algebraist details.
"""

from stark.core.contracts.engine import Engine
from stark.engines.allocator import Allocator
from stark.engines.accelerators import Accelerator, AcceleratorNone
from stark.engines.engine_native import EngineNative
from stark.engines.engine_numpy import EngineNumpy

has_engine_cupy = False
try:
    from stark.engines.engine_cupy import EngineCupy
except ImportError:
    pass
else:
    has_engine_cupy = True

has_engine_jax = False
try:
    from stark.engines.engine_jax import EngineJax
except ImportError:
    pass
else:
    has_engine_jax = True

if has_engine_cupy and has_engine_jax:
    __all__ = (
        "Accelerator",
        "AcceleratorNone",
        "Allocator",
        "Engine",
        "EngineCupy",
        "EngineJax",
        "EngineNative",
        "EngineNumpy",
    )
elif has_engine_cupy:
    __all__ = (
        "Accelerator",
        "AcceleratorNone",
        "Allocator",
        "Engine",
        "EngineCupy",
        "EngineNative",
        "EngineNumpy",
    )
elif has_engine_jax:
    __all__ = (
        "Accelerator",
        "AcceleratorNone",
        "Allocator",
        "Engine",
        "EngineJax",
        "EngineNative",
        "EngineNumpy",
    )
else:
    __all__ = (
        "Accelerator",
        "AcceleratorNone",
        "Allocator",
        "Engine",
        "EngineNative",
        "EngineNumpy",
    )
