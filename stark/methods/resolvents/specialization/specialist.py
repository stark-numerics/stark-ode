from __future__ import annotations

from typing import Protocol, TypeVar

from stark.block import Block, BlockKernel
from stark.contracts import Translation
from stark.methods.resolvents.specialization.stencil import ResolventStencilBlock


TranslationType = TypeVar("TranslationType", bound=Translation)


class ResolventSpecialist(Protocol[TranslationType]):
    """Provider of fixed-coefficient Block kernels for resolvents."""

    def provide(
        self,
        stencil: ResolventStencilBlock,
    ) -> BlockKernel[TranslationType]:
        ...


ResolventBlockKernel = BlockKernel[TranslationType]


__all__ = ["ResolventBlockKernel", "ResolventSpecialist"]
