"""Engine catalogue for backend runtime bundles."""

from stark.core.contracts.engine import Engine
from stark.engines.native import EngineNative
from stark.engines.numpy import EngineNumpy
from stark.engines.shared.accelerators import Accelerator, AcceleratorNone

__all__ = [
    "Accelerator",
    "AcceleratorNone",
    "Engine",
    "EngineNative",
    "EngineNumpy",
]

try:
    from stark.engines.cupy import EngineCupy
except ImportError:
    pass
else:
    __all__ += ["EngineCupy"]

try:
    from stark.engines.jax import EngineJax
except ImportError:
    pass
else:
    __all__ += ["EngineJax"]
