from __future__ import annotations

from typing import Protocol, TypeVar

from stark.core.block import Block, BlockKernel
from stark.core.contracts import Translation
from stark.methods.resolvents.linear_fixed_generation.stencil import ResolventStencilBlock


TranslationType = TypeVar("TranslationType", bound=Translation)


class ResolventLinearFixed(Protocol[TranslationType]):
    """Provider of fixed-coefficient Block kernels for resolvents."""

    def __call__(
        self,
        stencil: ResolventStencilBlock,
    ) -> BlockKernel[TranslationType]:
        ...


ResolventBlockKernel = BlockKernel[TranslationType]


__all__ = ["ResolventBlockKernel", "ResolventLinearFixed"]
