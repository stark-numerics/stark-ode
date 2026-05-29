from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from stark.block.block import Block


TranslationType = TypeVar("TranslationType")


class BlockAllocationAllocator(Protocol[TranslationType]):
    """Allocator subset required for block allocation."""

    def allocate_translation(self) -> TranslationType:
        ...


@dataclass(frozen=True, slots=True)
class BlockAllocator(Generic[TranslationType]):
    """Allocate blocks whose entries are compatible translations."""

    allocator: BlockAllocationAllocator[TranslationType]

    def allocate(self, size: int) -> Block[TranslationType]:
        if type(size) is not int:
            raise TypeError("Block size must be an integer.")
        if size < 0:
            raise ValueError("Block size must be non-negative.")

        return Block([self.allocator.allocate_translation() for _ in range(size)])

    def allocate_like(self, block: Block[object]) -> Block[TranslationType]:
        return self.allocate(len(block))


__all__ = ["BlockAllocationAllocator", "BlockAllocator"]
