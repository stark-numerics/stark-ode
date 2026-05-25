from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from stark.block.block import Block


StencilType = TypeVar("StencilType")
TranslationType = TypeVar("TranslationType")
ItemKernel = Callable[..., object]
BlockKernel = Callable[..., Block[TranslationType]]


class BlockItemSpecialist(Protocol[StencilType]):
    """Provider of entry-level kernels that can be lifted to Blocks."""

    def provide(self, stencil: StencilType) -> ItemKernel:
        ...


@dataclass(frozen=True, slots=True)
class BlockSpecialist(Generic[StencilType, TranslationType]):
    """Lift an entry-level specialist over Block entries."""

    specialist: BlockItemSpecialist[StencilType]

    def provide(self, stencil: StencilType) -> BlockKernel[TranslationType]:
        item_kernel = self.specialist.provide(stencil)

        def kernel(
            step: float,
            out: Block[TranslationType],
            *blocks: Block[TranslationType],
        ) -> Block[TranslationType]:
            if not blocks:
                raise TypeError("Block specialist kernels need source blocks.")

            Block._require_same_size(out, *blocks)

            for index, out_item in enumerate(out):
                item_kernel(
                    step,
                    out_item,
                    *(block[index] for block in blocks),
                )

            return out

        return kernel


__all__ = ["BlockItemSpecialist", "BlockKernel", "BlockSpecialist"]
