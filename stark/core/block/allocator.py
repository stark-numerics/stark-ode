from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol

from stark.core.block.block import Block
from stark.core.contracts.problem.translation import TranslationType, TranslationTypeCovariant


class BlockTranslationAllocatorLike(Protocol[TranslationTypeCovariant]):
    """Allocator subset required to allocate translation block entries."""

    def allocate_translation(self) -> TranslationTypeCovariant:
        ...


@dataclass(frozen=True, slots=True)
class BlockAllocator(Generic[TranslationType]):
    """Allocate blocks whose entries are compatible translations."""

    allocator: BlockTranslationAllocatorLike[TranslationType]

    def allocate(self, size: int) -> Block[TranslationType]:
        if type(size) is not int:
            raise TypeError("Block size must be an integer.")
        if size < 0:
            raise ValueError("Block size must be non-negative.")

        return Block([self.allocator.allocate_translation() for _ in range(size)])

    def allocate_like(self, block: Block[TranslationType]) -> Block[TranslationType]:
        return self.allocate(len(block))


__all__ = ["BlockTranslationAllocatorLike", "BlockAllocator"]
