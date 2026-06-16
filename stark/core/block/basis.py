"""Coordinate bases for block product spaces."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Generic

from stark.core.contracts.block import BlockLike
from stark.core.contracts.translation import TranslationType
from stark.core.contracts.translation_basis import TranslationBasis


@dataclass(frozen=True, slots=True)
class BlockBasis(Generic[TranslationType]):
    """
    Product-space basis for a block of translations.

    A block basis lifts one translation basis per block entry into a basis for
    the whole block. It owns the offset arithmetic that maps a flat coordinate
    index to the corresponding block entry and local translation coordinate.
    """

    bases: Sequence[TranslationBasis[TranslationType]]
    offsets: tuple[int, ...] = field(init=False)
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        bases = tuple(self.bases)
        offsets = tuple(accumulate((basis.dimension for basis in bases), initial=0))
        object.__setattr__(self, "bases", bases)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "dimension", offsets[-1])

    def vector(self, index: int, output: BlockLike[TranslationType]) -> BlockLike[TranslationType]:
        """Write or return the selected block basis vector."""

        block_index, local_index = self.local_index(index)
        for basis_index, basis in enumerate(self.bases):
            item = output[basis_index]
            if basis_index == block_index:
                output[basis_index] = basis.vector(local_index, item)
            else:
                output[basis_index] = basis.synthesize([0.0] * basis.dimension, item)

        return output

    def coordinate(self, index: int, block: BlockLike[TranslationType]) -> float:
        """Apply the selected block coordinate form to a block."""

        block_index, local_index = self.local_index(index)
        return self.bases[block_index].coordinate(local_index, block[block_index])

    def coordinates(
        self,
        block: BlockLike[TranslationType],
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        """Analyse a block into flat coordinates in this block basis."""

        for basis_index, basis in enumerate(self.bases):
            start = self.offsets[basis_index]
            local = [0.0] * basis.dimension
            basis.coordinates(block[basis_index], local)
            for local_index, coordinate in enumerate(local):
                output[start + local_index] = coordinate
        return output

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: BlockLike[TranslationType],
    ) -> BlockLike[TranslationType]:
        """Reconstruct a block from flat coordinates in this block basis."""

        for basis_index, basis in enumerate(self.bases):
            start = self.offsets[basis_index]
            stop = self.offsets[basis_index + 1]
            output[basis_index] = basis.synthesize(coordinates[start:stop], output[basis_index])

        return output

    def local_index(self, index: int) -> tuple[int, int]:
        """Return the block entry and local coordinate for a flat index."""

        if index < 0 or index >= self.dimension:
            raise IndexError("Block basis index out of range.")

        for block_index in range(len(self.bases)):
            start = self.offsets[block_index]
            stop = self.offsets[block_index + 1]
            if index < stop:
                return block_index, index - start

        raise IndexError("Block basis index out of range.")  # pragma: no cover - defensive.


__all__ = ["BlockBasis"]
