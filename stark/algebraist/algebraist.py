from __future__ import annotations

from typing import Protocol, TypeVar

AlgebraistRequest = TypeVar("AlgebraistRequest", contravariant=True)
AlgebraistKernel = TypeVar("AlgebraistKernel", covariant=True)


class Algebraist(Protocol[AlgebraistRequest, AlgebraistKernel]):
    """Protocol for objects that provide algebra kernels from requests."""

    def provide(self, request: AlgebraistRequest) -> AlgebraistKernel:
        ...


__all__ = ["Algebraist", "AlgebraistKernel", "AlgebraistRequest"]
