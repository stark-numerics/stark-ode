from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from stark.algebraist.algebraist import Algebraist
from stark.algebraist.arity import AlgebraistArity
from stark.contracts.translations import Translation

TranslationType = TypeVar("TranslationType", bound=Translation)
AlgebraistGeneralKernel = Callable[..., TranslationType]


class AlgebraistGeneral(Algebraist[AlgebraistArity, AlgebraistGeneralKernel[TranslationType]], Protocol[TranslationType]):
    """Provider of general arity-based linear-combination kernels."""

    def provide(self, request: AlgebraistArity) -> AlgebraistGeneralKernel[TranslationType]:
        ...


__all__ = ["AlgebraistGeneral", "AlgebraistGeneralKernel"]
