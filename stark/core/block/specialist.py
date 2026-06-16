from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from stark.core.block.block import Block


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

        if getattr(stencil, "apply", False):

            def apply_kernel(
                step: float,
                origin: Block[TranslationType],
                *terms: Block[TranslationType],
            ) -> Block[TranslationType]:
                if not terms:
                    raise TypeError("Block apply kernels need a result block.")

                sources = terms[:-1]
                result = terms[-1]
                for index, result_item in enumerate(result):
                    item_kernel(
                        step,
                        origin[index],
                        *(source[index] for source in sources),
                        result_item,
                    )

                return result

            return apply_kernel

        def delta_kernel(
            step: float,
            *terms: Block[TranslationType],
        ) -> Block[TranslationType]:
            if not terms:
                raise TypeError("Block delta kernels need an output block.")

            sources = terms[:-1]
            out = terms[-1]
            for index, out_item in enumerate(out):
                item_kernel(
                    step,
                    *(source[index] for source in sources),
                    out_item,
                )

            return out

        return delta_kernel


__all__ = ["BlockItemSpecialist", "BlockKernel", "BlockSpecialist"]
