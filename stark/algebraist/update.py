from __future__ import annotations

from typing import Protocol, TypeVar

from stark.algebraist.specialist import AlgebraistSpecialist
from stark.algebraist.stencil import AlgebraistStencil
from stark.contracts.translations import State, Translation

StateType = TypeVar("StateType", bound=State)
TranslationType = TypeVar("TranslationType", bound=Translation)


class AlgebraistUpdateStencil(AlgebraistStencil, Protocol):
    """Stencil for producing a state update from weighted translations."""


class AlgebraistUpdateKernel(Protocol[StateType, TranslationType]):
    """Kernel computing `result = origin + step * scale * sum(c_i * translation_i)`."""

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

    def provide(self, request: AlgebraistUpdateStencil) -> AlgebraistUpdateKernel[StateType, TranslationType]:
        ...


__all__ = [
    "AlgebraistUpdateKernel",
    "AlgebraistUpdateSpecialist",
    "AlgebraistUpdateStencil",
]
