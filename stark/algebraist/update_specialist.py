from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from stark.algebraist.specialist import AlgebraistSpecialist

StateType = TypeVar("StateType")
TranslationType = TypeVar("TranslationType")


class AlgebraistUpdateStencil(Protocol):
    """Fixed-coefficient request for updating a state from translations."""

    @property
    def scale(self) -> float:
        ...

    @property
    def coefficients(self) -> Sequence[float]:
        ...


class AlgebraistUpdateKernel(Protocol[StateType, TranslationType]):
    """Kernel that fills a state result from an origin and translations."""

    def __call__(
        self,
        step: float,
        result: StateType,
        origin: StateType,
        *translations: TranslationType,
    ) -> StateType:
        ...


class AlgebraistUpdateSpecialist(
    AlgebraistSpecialist[
        AlgebraistUpdateStencil,
        AlgebraistUpdateKernel[StateType, TranslationType],
    ],
    Protocol[StateType, TranslationType],
):
    """Provider of fixed-coefficient state-update kernels."""

    def provide(self, stencil: AlgebraistUpdateStencil) -> AlgebraistUpdateKernel[StateType, TranslationType]:
        ...
