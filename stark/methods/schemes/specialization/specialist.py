from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from stark.methods.schemes.specialization.stencil import SchemeStencil
from stark.core.contracts.state import State, StateType, StateTypeCovariant
from stark.core.contracts.translation import Translation, TranslationType, TranslationTypeCovariant

SchemeSpecialistKernel = Callable[..., object]

class SchemeSpecialist(Protocol[StateTypeCovariant, TranslationTypeCovariant]):
    """Provider of scheme kernels from fixed-coefficient stencils.

    ``stencil.apply`` selects the produced kernel semantics:

    - ``False`` -> delta kernel
    - ``True`` -> apply/update kernel
    """

    def provide(self, stencil: SchemeStencil) -> SchemeSpecialistKernel:
        ...

class SchemeSpecialistKernelDelta(Protocol[TranslationType]):
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


class SchemeSpecialistKernelApply(Protocol[StateTypeCovariant, TranslationTypeCovariant]):
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
        *terms: Translation | State,
    ) -> StateType:
        ...



__all__ = [
    "SchemeSpecialistKernelApply",
    "SchemeSpecialistKernelDelta",
    "SchemeSpecialistKernel",
    "SchemeSpecialist",
]
