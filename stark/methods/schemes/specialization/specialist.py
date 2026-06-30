from __future__ import annotations

from typing import Protocol

from stark.methods.schemes.specialization.stencil import SchemeStencil
from stark.core.contracts.state import StateType
from stark.core.contracts.translation import TranslationType, TranslationTypeContravariant


class SchemeSpecialist(Protocol[StateType, TranslationType]):
    """Provider of scheme kernels from fixed-coefficient stencils.

    Use the method that matches the desired callable shape. This keeps the
    generated path explicit at the call site and avoids making readers infer
    semantics from ``stencil.apply``.
    """

    def provide_delta(
        self,
        stencil: SchemeStencil,
    ) -> SchemeSpecialistKernelDelta[TranslationType]:
        ...

    def provide_apply(
        self,
        stencil: SchemeStencil,
    ) -> SchemeSpecialistKernelApply[StateType, TranslationType]:
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


class SchemeSpecialistKernelApply(Protocol[StateType, TranslationTypeContravariant]):
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
        *terms: TranslationTypeContravariant | StateType,
    ) -> StateType:
        ...



__all__ = [
    "SchemeSpecialistKernelApply",
    "SchemeSpecialistKernelDelta",
    "SchemeSpecialist",
]
