from __future__ import annotations

from typing import Protocol

from stark.core.block import BlockKernel
from stark.core.contracts import Translation, TranslationType
from stark.methods.inverters.relaxation.stencil import InverterRelaxationStencilUpdate


class InverterRelaxationLinearFixed(Protocol[TranslationType]):
    """Provider of fixed-coefficient Block kernels for relaxation inverters."""

    def __call__(
        self,
        stencil: InverterRelaxationStencilUpdate,
    ) -> BlockKernel[TranslationType]:
        ...


InverterRelaxationBlockKernel = BlockKernel[Translation]


__all__ = ["InverterRelaxationBlockKernel", "InverterRelaxationLinearFixed"]
