from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from stark.engines.algebraist.algebraist import Algebraist

TranslationType = TypeVar("TranslationType")

AlgebraistNormKernel = Callable[[TranslationType], float]


class AlgebraistNorm(
    Algebraist[None, AlgebraistNormKernel[TranslationType]],
    Protocol[TranslationType],
):
    """Provider of frame-aware translation norm kernels."""

    def provide(self, request: None = None) -> AlgebraistNormKernel[TranslationType]:
        ...


__all__ = ["AlgebraistNorm", "AlgebraistNormKernel"]
