"""NumPy engine parts.

`EngineNumpy` stores frame fields in NumPy arrays and is the most direct CPU
backend for array-valued problems. It is the usual first backend for numerical
work because NumPy is widely available and easy to inspect.
"""

from stark.engines.numpy.allocator import EngineAllocatorNumpy
from stark.engines.numpy.engine import EngineNumpy
from stark.engines.numpy.translation import EngineTranslationNumpy

__all__ = [
    "EngineAllocatorNumpy",
    "EngineNumpy",
    "EngineTranslationNumpy",
]
