from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from stark.schemes.support.stencil import SchemeStencil


StateType = TypeVar("StateType")
TranslationType = TypeVar("TranslationType")


class SchemeDeltaKernel(Protocol[TranslationType]):
    """Kernel that writes a weighted translation delta.

    Semantics:

        out = step * stencil.scale * sum(c_i * translation_i)
    """

    def __call__(
        self,
        step: float,
        out: TranslationType,
        *translations: TranslationType,
    ) -> TranslationType:
        ...


class SchemeApplyKernel(Protocol[StateType, TranslationType]):
    """Kernel that applies a weighted translation delta to an origin state.

    Semantics:

        result = origin + step * stencil.scale * sum(c_i * translation_i)
    """

    def __call__(
        self,
        step: float,
        result: StateType,
        origin: StateType,
        *translations: TranslationType,
    ) -> StateType:
        ...


SchemeKernel = Callable[..., object]


class SchemeSpecialist(Protocol[StateType, TranslationType]):
    """Provider of scheme kernels from fixed-coefficient stencils.

    ``stencil.apply`` selects the produced kernel semantics:

    - ``False`` -> delta kernel
    - ``True`` -> apply/update kernel
    """

    def provide(self, stencil: SchemeStencil) -> SchemeKernel:
        ...


__all__ = [
    "SchemeApplyKernel",
    "SchemeDeltaKernel",
    "SchemeKernel",
    "SchemeSpecialist",
]
