from __future__ import annotations

from typing import Protocol

from stark.algebraist.algebraist import Algebraist, AlgebraistKernel

class AlgebraistGeneral(Algebraist[int], Protocol):
    """Provides general arity-based linear-combination kernels."""

    def provide(self, request: int) -> AlgebraistKernel:
        ...