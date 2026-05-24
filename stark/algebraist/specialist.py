from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from stark.algebraist.algebraist import Algebraist
from stark.algebraist.stencil import AlgebraistStencil


KernelType = TypeVar("KernelType", covariant=True)


AlgebraistKernel = Callable[..., object]


class AlgebraistSpecialist(Algebraist[AlgebraistStencil], Protocol):
    """Provider of the best available kernel for a stencil request.

    ``stencil.apply`` selects the produced kernel semantics.
    """

    def provide(self, request: AlgebraistStencil) -> AlgebraistKernel:
        ...


__all__ = ["AlgebraistKernel", "AlgebraistSpecialist"]
