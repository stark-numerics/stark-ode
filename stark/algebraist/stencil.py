from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class AlgebraistStencil(Protocol):
    """Fixed-coefficient stencil request shape."""

    @property
    def scale(self) -> float:
        ...

    @property
    def coefficients(self) -> Sequence[float]:
        ...


__all__ = ["AlgebraistStencil"]
