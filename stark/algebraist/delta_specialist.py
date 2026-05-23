from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from stark.algebraist.specialist import AlgebraistSpecialist

TranslationType = TypeVar("TranslationType")


class AlgebraistDeltaStencil(Protocol):
    """Fixed-coefficient request for producing a translation/delta."""

    @property
    def scale(self) -> float:
        ...

    @property
    def coefficients(self) -> Sequence[float]:
        ...


class AlgebraistDeltaKernel(Protocol[TranslationType]):
    """Kernel that fills a translation/delta result."""

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
    """Provider of fixed-coefficient delta kernels."""

    def provide(self, stencil: AlgebraistDeltaStencil) -> AlgebraistDeltaKernel[TranslationType]:
        ...
