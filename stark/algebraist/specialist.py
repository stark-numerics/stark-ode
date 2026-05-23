from __future__ import annotations

from typing import Protocol, TypeVar

StencilType = TypeVar("StencilType", contravariant=True)
KernelType = TypeVar("KernelType", covariant=True)


class AlgebraistSpecialist(Protocol[StencilType, KernelType]):
    """Provider of a specialized algebra kernel from a stencil request."""

    def provide(self, stencil: StencilType) -> KernelType:
        ...
