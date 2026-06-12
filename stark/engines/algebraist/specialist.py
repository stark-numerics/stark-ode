from __future__ import annotations

from typing import Protocol, TypeVar

from stark.engines.algebraist.algebraist import Algebraist
from stark.engines.algebraist.stencil import AlgebraistStencil


RequestType = TypeVar("RequestType", contravariant=True, bound=AlgebraistStencil)
KernelType = TypeVar("KernelType", covariant=True)


class AlgebraistSpecialist(Algebraist[RequestType, KernelType], Protocol[RequestType, KernelType]):
    """Provider of the best available kernel for a stencil request.

    ``stencil.apply`` selects the produced kernel semantics.
    """

    def provide(self, request: RequestType) -> KernelType:
        ...


__all__ = ["AlgebraistSpecialist"]
