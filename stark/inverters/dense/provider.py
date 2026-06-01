from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from typing import Protocol


class InverterProviderDense(Protocol):
    """
    Dense linear inversion backend for dense direct inverters.

    A dense provider owns the backend-specific operation that solves the
    materialised coordinate system

        matrix * result = image

    for a prepared dimension. Matrix storage is flat row-major. Implementations
    may redirect to exact small kernels, backend linalg routines, or fallback
    algorithms during ``prepare``. The hot ``invert`` method assumes preparation
    has already selected the appropriate path.
    """

    dimension: int | None

    def prepare(self, dimension: int) -> None:
        """Prepare this provider for square systems of the given dimension."""
        ...

    def invert(
        self,
        matrix: Sequence[float],
        image: Sequence[float],
        result: MutableSequence[float],
    ) -> MutableSequence[float]:
        """Solve the prepared dense system into ``result``."""
        ...


__all__ = ["InverterProviderDense"]
