"""Backend engines for layout-backed STARK systems."""

__all__: list[str] = []

from stark.contracts.engine import Engine
from stark.engines.native import (
    EngineAllocatorNative,
    EngineNative,
    EngineTranslationNative,
)
from stark.engines.numpy import (
    EngineAllocatorNumpy,
    EngineNumpy,
    EngineTranslationNumpy,
)

__all__ += [
    "Engine",
    "EngineAllocatorNative",
    "EngineAllocatorNumpy",
    "EngineNative",
    "EngineNumpy",
    "EngineTranslationNative",
    "EngineTranslationNumpy",
]

try:
    from stark.engines.cupy import (
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
    from stark.engines.jax import (
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
