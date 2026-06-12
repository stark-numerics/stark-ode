from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import Generic

from stark.core.contracts.block import BlockLike, BlockOperatorDiagonalLike
from stark.core.contracts.translation import TranslationType
from stark.core.contracts.translation_basis import TranslationBasis


OperatorDenseFill = Callable[
    [
        TranslationBasis[TranslationType],
        MutableSequence[float],
        int,
        int,
        int,
    ],
    None,
]


class OperatorDenseEntryMaterialize(Generic[TranslationType]):
    """Prepared dense materializer for one translation-level operator."""

    def refresh(
        self,
        matrix: MutableSequence[float],
        *,
        row_offset: int,
        column_offset: int,
        stride: int,
        source: TranslationType,
        image: TranslationType,
    ) -> tuple[TranslationType, TranslationType]:
        raise NotImplementedError


@dataclass(slots=True)
class OperatorDenseEntryMaterializeDirect(Generic[TranslationType]):
    """Use an operator-provided dense-fill capability."""

    dense_fill: OperatorDenseFill[TranslationType]
    basis: TranslationBasis[TranslationType]

    def refresh(
        self,
        matrix: MutableSequence[float],
        *,
        row_offset: int,
        column_offset: int,
        stride: int,
        source: TranslationType,
        image: TranslationType,
    ) -> tuple[TranslationType, TranslationType]:
        self.dense_fill(self.basis, matrix, row_offset, column_offset, stride)
        return source, image


@dataclass(slots=True)
class OperatorDenseEntryMaterializeProbe(Generic[TranslationType]):
    """Materialize an operator by applying it to basis vectors."""

    operator: object
    basis: TranslationBasis[TranslationType]

    def refresh(
        self,
        matrix: MutableSequence[float],
        *,
        row_offset: int,
        column_offset: int,
        stride: int,
        source: TranslationType,
        image: TranslationType,
    ) -> tuple[TranslationType, TranslationType]:
        dimension = self.basis.dimension
        for local_column in range(dimension):
            source = self.basis.vector(local_column, source)
            result = self.operator(source, image)  # type: ignore[operator]
            if result is not None:
                image = result

            column = column_offset + local_column
            for local_row in range(dimension):
                row = row_offset + local_row
                matrix[row * stride + column] = self.basis.coordinate(local_row, image)

        return source, image


def dense_entry_materializer(
    operator: object,
    basis: TranslationBasis[TranslationType],
    dense_fill: object | None = None,
) -> OperatorDenseEntryMaterialize[TranslationType]:
    if dense_fill is None:
        dense_fill = getattr(operator, "dense_fill", None)
    if callable(dense_fill):
        return OperatorDenseEntryMaterializeDirect(dense_fill, basis)
    return OperatorDenseEntryMaterializeProbe(operator, basis)


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
    entry_materializer: OperatorDenseEntryMaterialize[TranslationType] = field(init=False)
    refresh_initial: bool = True

    def __post_init__(self) -> None:
        self.matrix = self._allocate_matrix()
        self.entry_materializer = dense_entry_materializer(self.operator, self.basis)
        if self.refresh_initial:
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
        else:
            for index in range(len(self.matrix)):
                self.matrix[index] = 0.0

        self.source, self.image = self.entry_materializer.refresh(
            self.matrix,
            row_offset=0,
            column_offset=0,
            stride=dimension,
            source=self.source,
            image=self.image,
        )

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
    entry_materializers: list[OperatorDenseEntryMaterialize[TranslationType] | None] = field(init=False)
    entry_operators: list[object | None] = field(init=False)
    entry_dense_fills: list[object | None] = field(init=False)
    refresh_initial: bool = True

    def __post_init__(self) -> None:
        self.bases = self._normalize_bases(self.bases)
        self.offsets = self._build_offsets(self.bases)
        self.dimension = self.offsets[-1]
        self.matrix = self._allocate_matrix()
        self.entry_materializers = [None for _ in self.bases]
        self.entry_operators = [None for _ in self.bases]
        self.entry_dense_fills = [None for _ in self.bases]
        if self.refresh_initial:
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

            entry_materializer = self.entry_materializers[block_index]
            entry_dense_fill = getattr(entry_operator, "dense_fill", None)
            if (
                entry_materializer is None
                or self.entry_operators[block_index] is not entry_operator
                or self.entry_dense_fills[block_index] is not entry_dense_fill
            ):
                entry_materializer = dense_entry_materializer(
                    entry_operator,
                    basis,
                    entry_dense_fill,
                )
                self.entry_materializers[block_index] = entry_materializer
                self.entry_operators[block_index] = entry_operator
                self.entry_dense_fills[block_index] = entry_dense_fill

            offset = self.offsets[block_index]
            source_item, image_item = entry_materializer.refresh(
                self.matrix,
                row_offset=offset,
                column_offset=offset,
                stride=dimension,
                source=self.source[block_index],
                image=self.image[block_index],
            )
            self.source[block_index] = source_item
            self.image[block_index] = image_item

        return self.matrix


__all__ = [
    "BlockOperatorDiagonalMaterialize",
    "OperatorDenseEntryMaterialize",
    "OperatorDenseEntryMaterializeDirect",
    "OperatorDenseEntryMaterializeProbe",
    "OperatorDenseFill",
    "OperatorMaterialize",
    "dense_entry_materializer",
]
