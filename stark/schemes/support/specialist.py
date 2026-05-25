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

    The final positional argument is the output translation.
    """

    def __call__(
        self,
        step: float,
        *terms: TranslationType,
    ) -> TranslationType:
        ...


class SchemeApplyKernel(Protocol[StateType, TranslationType]):
    """Kernel that applies a weighted translation delta to an origin state.

    Semantics:

        result = origin + step * stencil.scale * sum(c_i * translation_i)

    The first argument after ``step`` is the origin state. The final positional
    argument is the result state.
    """

    def __call__(
        self,
        step: float,
        origin: StateType,
        *terms: TranslationType | StateType,
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
