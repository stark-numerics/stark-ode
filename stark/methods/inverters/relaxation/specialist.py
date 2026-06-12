from __future__ import annotations

from typing import Protocol

from stark.core.block import BlockKernel
from stark.core.contracts import Translation, TranslationTypeCovariant
from stark.methods.inverters.relaxation.stencil import InverterRelaxationStencil, InverterRelaxationStencilUpdate


class InverterRelaxationSpecialist(Protocol[TranslationTypeCovariant]):
    """Provider of fixed-coefficient Block kernels for relaxation inverters."""

    def provide(
        self,
        stencil: InverterRelaxationStencil | InverterRelaxationStencilUpdate,
    ) -> BlockKernel[TranslationTypeCovariant]:
        ...


InverterRelaxationBlockKernel = BlockKernel[Translation]


__all__ = ["InverterRelaxationBlockKernel", "InverterRelaxationSpecialist"]
