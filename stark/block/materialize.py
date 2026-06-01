from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic

from stark.contracts.block import BlockLike, BlockOperatorDiagonalLike
from stark.contracts.translation import TranslationType
from stark.contracts.translation_basis import TranslationBasis


@dataclass(slots=True)
class OperatorMaterialize(Generic[TranslationType]):
    """
    Dense materialisation of a translation operator in a translation basis.

    For a translation operator ``operator`` and basis ``basis``, the matrix is
    built column by column using

        matrix[row, column] = basis.coordinate(row, operator(basis.vector(column)))

    Matrix storage is a flat row-major coordinate buffer. ``source`` and
    ``image`` are reusable scratch translations supplied by the caller.
    ``basis.vector`` returns the written source so immutable-return carriers,
    such as JAX, can be handled by the same materialisation loop.
    """

    operator: object
    basis: TranslationBasis[TranslationType]
    source: TranslationType
    image: TranslationType
    matrix: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.matrix = self._allocate_matrix()
        self.refresh()

    @property
    def dimension(self) -> int:
        return self.basis.dimension

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    def _allocate_matrix(self) -> list[float]:
        dimension = self.basis.dimension
        return [0.0 for _ in range(dimension * dimension)]

    def refresh(self) -> list[float]:
        """Rebuild and return the dense row-major matrix for the current operator."""

        dimension = self.basis.dimension
        if len(self.matrix) != dimension * dimension:
            self.matrix = self._allocate_matrix()

        for column in range(dimension):
            self.source = self.basis.vector(column, self.source)
            result = self.operator(self.source, self.image)  # type: ignore[operator]
            if result is not None:
                self.image = result

            for row in range(dimension):
                self.matrix[row * dimension + column] = self.basis.coordinate(row, self.image)

        return self.matrix


@dataclass(slots=True)
class BlockOperatorDiagonalMaterialize(Generic[TranslationType]):
    """
    Fast dense materialisation of an inspectable diagonal block operator.

    A diagonal block operator acts entrywise,

        target[i] <- operator[i](source[i])

    so its dense matrix is block diagonal. This materialiser builds each entry
    matrix with the corresponding translation basis and places it on the dense
    block diagonal. Matrix storage is flat row-major to support accelerator
    backends that do not handle nested Python lists efficiently.
    """

    operator: BlockOperatorDiagonalLike[TranslationType]
    bases: TranslationBasis[TranslationType] | Sequence[TranslationBasis[TranslationType]]
    source: BlockLike[TranslationType]
    image: BlockLike[TranslationType]
    matrix: list[float] = field(init=False)
    offsets: list[int] = field(init=False)
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        self.bases = self._normalize_bases(self.bases)
        self.offsets = self._build_offsets(self.bases)
        self.dimension = self.offsets[-1]
        self.matrix = self._allocate_matrix()
        self.refresh()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    def _normalize_bases(
        self,
        bases: TranslationBasis[TranslationType] | Sequence[TranslationBasis[TranslationType]],
    ) -> list[TranslationBasis[TranslationType]]:
        block_size = len(self.source)
        if len(self.image) != block_size:
            raise ValueError("Source and image blocks must have the same size.")
        if len(self.operator) != block_size:
            raise ValueError("Operator size must match the source block size.")

        if isinstance(bases, Sequence):
            normalized = list(bases)
        else:
            normalized = [bases for _ in range(block_size)]

        if len(normalized) != block_size:
            raise ValueError("Basis count must match the source block size.")

        return normalized

    @staticmethod
    def _build_offsets(bases: Sequence[TranslationBasis[TranslationType]]) -> list[int]:
        offsets = [0]
        for basis in bases:
            offsets.append(offsets[-1] + basis.dimension)
        return offsets

    def _allocate_matrix(self) -> list[float]:
        return [0.0 for _ in range(self.dimension * self.dimension)]

    def refresh(self, operator: BlockOperatorDiagonalLike[TranslationType] | None = None) -> list[float]:
        """Rebuild and return the dense row-major block-diagonal matrix."""

        if operator is not None:
            self.operator = operator
            if len(self.operator) != len(self.source):
                raise ValueError("Operator size must match the source block size.")

        if len(self.matrix) != self.dimension * self.dimension:
            self.matrix = self._allocate_matrix()
        else:
            for index in range(len(self.matrix)):
                self.matrix[index] = 0.0

        dimension = self.dimension
        for block_index, basis in enumerate(self.bases):
            entry_operator = self.operator[block_index]
            if entry_operator is None:
                raise RuntimeError(f"Block operator diagonal entry {block_index} is not configured.")

            offset = self.offsets[block_index]
            for local_column in range(basis.dimension):
                source_item = basis.vector(local_column, self.source[block_index])
                self.source[block_index] = source_item
                result = entry_operator(source_item, self.image[block_index])
                if result is not None:
                    self.image[block_index] = result

                image_item = self.image[block_index]
                for local_row in range(basis.dimension):
                    row = offset + local_row
                    column = offset + local_column
                    self.matrix[row * dimension + column] = basis.coordinate(local_row, image_item)

        return self.matrix


__all__ = [
    "BlockOperatorDiagonalMaterialize",
    "OperatorMaterialize",
]
