"""Concrete engine backend catalogue."""

__all__: list[str] = []

from stark.engines.backends.native import (
    EngineAllocatorNative,
    EngineNative,
    EngineTranslationNative,
)
from stark.engines.backends.numpy import (
    EngineAllocatorNumpy,
    EngineNumpy,
    EngineTranslationNumpy,
)

__all__ += [
    "EngineAllocatorNative",
    "EngineAllocatorNumpy",
    "EngineNative",
    "EngineNumpy",
    "EngineTranslationNative",
    "EngineTranslationNumpy",
]

try:
    from stark.engines.backends.cupy import (
        EngineAllocatorCupy,
        EngineCupy,
        EngineTranslationCupy,
    )
except ImportError:
    pass
else:
    __all__ += [
        "EngineAllocatorCupy",
        "EngineCupy",
        "EngineTranslationCupy",
    ]

try:
    from stark.engines.backends.jax import (
        EngineAllocatorJax,
        EngineJax,
        EngineTranslationJax,
    )
except ImportError:
    pass
else:
    __all__ += [
        "EngineAllocatorJax",
        "EngineJax",
        "EngineTranslationJax",
    ]