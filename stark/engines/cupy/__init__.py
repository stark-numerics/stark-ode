"""CuPy engine parts.

`EngineCupy` stores frame fields in CuPy arrays for GPU-backed array algebra.
It is intended for large enough array problems to amortize GPU launch and
transfer costs. Tests and examples should make host transfers explicit.
"""

from stark.engines.cupy.allocator import EngineAllocatorCupy
from stark.engines.cupy.engine import EngineCupy
from stark.engines.cupy.translation import EngineTranslationCupy

__all__ = [
    "EngineAllocatorCupy",
    "EngineCupy",
    "EngineTranslationCupy",
]
