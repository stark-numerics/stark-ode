"""Backend engines for layout-backed STARK systems."""

__all__: list[str] = []

from stark.contracts.engine import StarkEngine
from stark.engines.native import (
    StarkEngineAllocatorNative,
    StarkEngineNative,
    StarkEngineTranslationNative,
)
from stark.engines.numpy import (
    StarkEngineAllocatorNumpy,
    StarkEngineNumpy,
    StarkEngineTranslationNumpy,
)

__all__ += [
    "StarkEngine",
    "StarkEngineAllocatorNative",
    "StarkEngineAllocatorNumpy",
    "StarkEngineNative",
    "StarkEngineNumpy",
    "StarkEngineTranslationNative",
    "StarkEngineTranslationNumpy",
]

try:
    from stark.engines.cupy import (
        StarkEngineAllocatorCupy,
        StarkEngineCupy,
        StarkEngineTranslationCupy,
    )
except ImportError:
    pass
else:
    __all__ += [
        "StarkEngineAllocatorCupy",
        "StarkEngineCupy",
        "StarkEngineTranslationCupy",
    ]

try:
    from stark.engines.jax import (
        StarkEngineAllocatorJax,
        StarkEngineJax,
        StarkEngineTranslationJax,
    )
except ImportError:
    pass
else:
    __all__ += [
        "StarkEngineAllocatorJax",
        "StarkEngineJax",
        "StarkEngineTranslationJax",
    ]
