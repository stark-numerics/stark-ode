from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar


AlgebraistRequest = TypeVar("AlgebraistRequest", contravariant=True)
AlgebraistKernel = Callable[..., object]


class Algebraist(Protocol[AlgebraistRequest]):
    """Provider of the best available kernel for an algebra request."""

    def provide(self, request: AlgebraistRequest) -> AlgebraistKernel:
        ...