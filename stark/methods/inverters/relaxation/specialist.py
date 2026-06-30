from __future__ import annotations

from typing import Protocol

from stark.core.block import BlockKernel
from stark.core.contracts import Translation, TranslationType
from stark.methods.inverters.relaxation.stencil import InverterRelaxationStencil, InverterRelaxationStencilUpdate


class InverterRelaxationSpecialist(Protocol[TranslationType]):
    """Provider of fixed-coefficient Block kernels for relaxation inverters."""

    def provide(
        self,
        stencil: InverterRelaxationStencil | InverterRelaxationStencilUpdate,
    ) -> BlockKernel[TranslationType]:
        ...


InverterRelaxationBlockKernel = BlockKernel[Translation]


__all__ = ["InverterRelaxationBlockKernel", "InverterRelaxationSpecialist"]
