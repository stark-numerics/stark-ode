from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from stark.core.contracts.state import State
from stark.core.contracts.translation import Translation
from stark.engines._algebraist.stencil import AlgebraistStencil


StateType = TypeVar("StateType", covariant=True, bound=State)
TranslationType = TypeVar("TranslationType", covariant=True, bound=Translation)


class AlgebraistLinearFixed(Protocol[StateType, TranslationType]):
    """Provider of fixed-coefficient scheme kernels."""

    def __call__(self, stencil: AlgebraistStencil) -> Callable[..., object]:
        ...

    def provide_delta(self, stencil: AlgebraistStencil) -> Callable[..., TranslationType]:
        ...

    def provide_apply(self, stencil: AlgebraistStencil) -> Callable[..., StateType]:
        ...

    def provide_unit_apply(self) -> Callable[..., object]:
        ...


__all__ = ["AlgebraistLinearFixed"]
