from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class AlgebraistStencil(Protocol):
    """Fixed-coefficient request consumed by an Algebraist specialist."""

    @property
    def scale(self) -> float:
        ...

    @property
    def coefficients(self) -> Sequence[float]:
        ...

    @property
    def apply(self) -> bool:
        ...


__all__ = ["AlgebraistStencil"]
