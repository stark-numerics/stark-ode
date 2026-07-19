from __future__ import annotations

from typing import Any, Protocol

from stark.core.contracts.problem.state import StateType, StateTypeCovariant
from stark.core.contracts.problem.translation import (
    TranslationType,
    TranslationTypeContravariant,
    TranslationTypeCovariant,
)
from stark.methods.schemes.linear_fixed_generation.stencil import SchemeStencil


class SchemeLinearFixedLike(Protocol[StateTypeCovariant, TranslationTypeCovariant]):
    """Callable source of scheme kernels from fixed-coefficient stencils.

    `SchemeStencil` is a generator request: its `operation` identifies the
    fixed-linear operation, while `apply` selects whether the returned kernel
    writes a translation delta or applies that delta to an origin state.
    """

    def __call__(
        self,
        stencil: SchemeStencil,
    ) -> Any:
        ...


class SchemeLinearFixedKernelDeltaLike(Protocol[TranslationType]):
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


class SchemeLinearFixedKernelApplyLike(Protocol[StateType, TranslationTypeContravariant]):
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
    "SchemeLinearFixedKernelApplyLike",
    "SchemeLinearFixedKernelDeltaLike",
    "SchemeLinearFixedLike",
]
