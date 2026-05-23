from __future__ import annotations

from typing import Protocol, TypeVar

from stark.algebraist.specialist import AlgebraistSpecialist
from stark.algebraist.stencil import AlgebraistStencil
from stark.contracts.translations import Translation

TranslationType = TypeVar("TranslationType", bound=Translation)


class AlgebraistDeltaStencil(AlgebraistStencil, Protocol):
    """Stencil for producing a weighted translation delta."""


class AlgebraistDeltaKernel(Protocol[TranslationType]):
    """Kernel computing `out = step * scale * sum(c_i * translation_i)`."""

    def __call__(
        self,
        step: float,
        out: TranslationType,
        *translations: TranslationType,
    ) -> TranslationType:
        ...


class AlgebraistDeltaSpecialist(
    AlgebraistSpecialist[AlgebraistDeltaStencil, AlgebraistDeltaKernel[TranslationType]],
    Protocol[TranslationType],
):
    """Provider of fixed-coefficient translation-delta kernels."""

    def provide(self, request: AlgebraistDeltaStencil) -> AlgebraistDeltaKernel[TranslationType]:
        ...


__all__ = [
    "AlgebraistDeltaKernel",
    "AlgebraistDeltaSpecialist",
    "AlgebraistDeltaStencil",
]
