from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from stark.engines.shared.algebraist.algebraist import Algebraist
from stark.core.contracts.translation import TranslationTypeContravariant


AlgebraistNormKernel = Callable[[TranslationTypeContravariant], float]


class AlgebraistNorm(
    Algebraist[None, AlgebraistNormKernel[TranslationTypeContravariant]],
    Protocol[TranslationTypeContravariant],
):
    """Provider of frame-aware translation norm kernels."""

    def provide(self, request: None = None) -> AlgebraistNormKernel[TranslationTypeContravariant]:
        ...


__all__ = ["AlgebraistNorm", "AlgebraistNormKernel"]
