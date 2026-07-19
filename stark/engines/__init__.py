"""Engine catalogue for backend runtime bundles.

Engines decide where state lives and which algebra kernels a prepared IVP uses.
Most examples can import `EngineNumpy`, `EngineNative`, `EngineCupy`, or
`EngineJax` from this package. Deeper engine modules are primarily for backend
authors and advanced users inspecting storage, carriers, allocators, or
generated kernels.
"""

from stark.engines.allocator import (
    Allocator,
    AllocatorGeneratedHooks,
    AllocatorRuntimeHooks,
)
from stark.engines.accelerators import Accelerator, AcceleratorNone
from stark.engines.engine import Engine, EngineFactory, EngineNative, EngineNumpy
from stark.engines.engine_allocator import EngineAllocator
from stark.engines.engine_translation import EngineTranslation

has_engine_cupy = False
try:
    from stark.engines.engine import EngineCupy
except ImportError:
    pass
else:
    has_engine_cupy = True

has_engine_jax = False
try:
    from stark.engines.engine import EngineJax
except ImportError:
    pass
else:
    has_engine_jax = True

if has_engine_cupy and has_engine_jax:
    __all__ = (
        "Accelerator",
        "AcceleratorNone",
        "Allocator",
        "AllocatorGeneratedHooks",
        "AllocatorRuntimeHooks",
        "Engine",
        "EngineAllocator",
        "EngineCupy",
        "EngineFactory",
        "EngineTranslation",
        "EngineJax",
        "EngineNative",
        "EngineNumpy",
    )
elif has_engine_cupy:
    __all__ = (
        "Accelerator",
        "AcceleratorNone",
        "Allocator",
        "AllocatorGeneratedHooks",
        "AllocatorRuntimeHooks",
        "Engine",
        "EngineAllocator",
        "EngineCupy",
        "EngineFactory",
        "EngineTranslation",
        "EngineNative",
        "EngineNumpy",
    )
elif has_engine_jax:
    __all__ = (
        "Accelerator",
        "AcceleratorNone",
        "Allocator",
        "AllocatorGeneratedHooks",
        "AllocatorRuntimeHooks",
        "Engine",
        "EngineAllocator",
        "EngineFactory",
        "EngineTranslation",
        "EngineJax",
        "EngineNative",
        "EngineNumpy",
    )
else:
    __all__ = (
        "Accelerator",
        "AcceleratorNone",
        "Allocator",
        "AllocatorGeneratedHooks",
        "AllocatorRuntimeHooks",
        "Engine",
        "EngineAllocator",
        "EngineFactory",
        "EngineTranslation",
        "EngineNative",
        "EngineNumpy",
    )
