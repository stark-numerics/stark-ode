from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from stark.engines.algebraist.algebraist import Algebraist
from stark.engines.algebraist.arity import AlgebraistArity
from stark.contracts.translation import Translation

TranslationType = TypeVar("TranslationType", bound=Translation)
AlgebraistLinearCombineKernel = Callable[..., TranslationType]


class AlgebraistLinearCombine(Algebraist[AlgebraistArity, AlgebraistLinearCombineKernel[TranslationType]], Protocol[TranslationType]):
    """Provider of general arity-based linear-combination kernels."""

    def provide(self, request: AlgebraistArity) -> AlgebraistLinearCombineKernel[TranslationType]:
        ...


__all__ = ["AlgebraistLinearCombine", "AlgebraistLinearCombineKernel"]
