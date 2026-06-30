"""Native Python engine parts.

`EngineNative` stores one-dimensional frame fields in Python `array.array`
objects and uses generated Algebraist kernels with Numba when available. It is
useful as a dependency-light accelerated CPU backend and as a reference for
backend authors who need to support Python-owned storage.
"""

from stark.engines.native.allocator import EngineAllocatorNative
from stark.engines.native.engine import EngineNative
from stark.engines.native.translation import EngineTranslationNative

__all__ = [
    "EngineAllocatorNative",
    "EngineNative",
    "EngineTranslationNative",
]
