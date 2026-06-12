"""Engine catalogue for backend runtime bundles."""

from stark.core.contracts.engine import Engine
from stark.engines.backends import (
    EngineAllocatorNative,
    EngineAllocatorNumpy,
    EngineNative,
    EngineNumpy,
    EngineTranslationNative,
    EngineTranslationNumpy,
)

__all__ = [
    "Engine",
    "EngineAllocatorNative",
    "EngineAllocatorNumpy",
    "EngineNative",
    "EngineNumpy",
    "EngineTranslationNative",
    "EngineTranslationNumpy",
]

try:
    from stark.engines.backends import (
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
    from stark.engines.backends import (
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