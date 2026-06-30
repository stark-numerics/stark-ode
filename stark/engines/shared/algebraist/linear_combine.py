from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from stark.engines.shared.algebraist.algebraist import Algebraist
from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.core.contracts.translation import Translation, TranslationTypeCovariant

TranslationType = TypeVar("TranslationType", bound=Translation)
AlgebraistLinearCombineKernel = Callable[..., TranslationType]


class AlgebraistLinearCombine(
    Algebraist[AlgebraistArity, AlgebraistLinearCombineKernel[TranslationTypeCovariant]],
    Protocol[TranslationTypeCovariant],
):
    """Provider of general arity-based linear-combination kernels."""

    def provide(self, request: AlgebraistArity) -> AlgebraistLinearCombineKernel[TranslationTypeCovariant]:
        ...


__all__ = ["AlgebraistLinearCombine", "AlgebraistLinearCombineKernel"]
