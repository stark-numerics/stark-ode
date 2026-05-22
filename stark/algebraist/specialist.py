from __future__ import annotations

from typing import Protocol

from stark.algebraist.algebraist import Algebraist, AlgebraistKernel
from stark.algebraist.stencil import AlgebraistStencil


class AlgebraistSpecialist(Algebraist[AlgebraistStencil], Protocol):
    """Provides fixed-coefficient kernels from stencil requests."""

    def provide(self, request: AlgebraistStencil) -> AlgebraistKernel:
        ...