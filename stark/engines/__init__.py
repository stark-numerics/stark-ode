"""Engine catalogue for backend runtime bundles."""

from stark.core.contracts.engine import Engine
from stark.engines.backends import (
    EngineNative,
    EngineNumpy,
)

__all__ = [
    "Engine",
    "EngineNative",
    "EngineNumpy",
]

try:
    from stark.engines.backends import EngineCupy
except ImportError:
    pass
else:
    __all__ += ["EngineCupy"]

try:
    from stark.engines.backends import EngineJax
except ImportError:
    pass
else:
    __all__ += ["EngineJax"]