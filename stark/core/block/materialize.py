from __future__ import annotations

from collections.abc import Callable, MutableSequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeAlias, cast

from stark.core.block.basis import BlockBasis
from stark.core.contracts.methods.block import BlockLike, BlockOperatorDiagonalLike, BlockOperatorLike
from stark.core.contracts.problem.translation import TranslationType
from stark.core.contracts.engines.translation_basis import TranslationBasisLike


DenseOperatorMatrix: TypeAlias = MutableSequence[float]
"""Flat row-major dense matrix storage."""


TranslationOperatorDenseFill = Callable[
    [
        TranslationBasisLike[TranslationType],
        DenseOperatorMatrix,
        int,
        int,
        int,
    ],
    None,
]


class TranslationOperatorDenseEntryMaterializeLike(Protocol[TranslationType]):
    """Prepared dense materializer for one translation-level operator."""

    def refresh(
        self,
        matrix: DenseOperatorMatrix,
        *,
        row_offset: int,
        column_offset: int,
        stride: int,
        source: TranslationType,
        image: TranslationType,
    ) -> tuple[TranslationType, TranslationType]:
        ...


@dataclass(slots=True)
class TranslationOperatorDenseEntryMaterializeDirect(Generic[TranslationType]):
    """Use an operator-provided dense-fill capability."""

    dense_fill: TranslationOperatorDenseFill[TranslationType]
    basis: TranslationBasisLike[TranslationType]

    def refresh(
        self,
        matrix: DenseOperatorMatrix,
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
class TranslationOperatorDenseEntryMaterializeProbe(Generic[TranslationType]):
    """Materialize an operator by applying it to basis vectors."""

    operator: object
    basis: TranslationBasisLike[TranslationType]

    def refresh(
        self,
        matrix: DenseOperatorMatrix,
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
    basis: TranslationBasisLike[TranslationType],
    dense_fill: object | None = None,
) -> TranslationOperatorDenseEntryMaterializeLike[TranslationType]:
    if dense_fill is None:
        dense_fill = getattr(operator, "dense_fill", None)
    if callable(dense_fill):
        return TranslationOperatorDenseEntryMaterializeDirect(
            cast(TranslationOperatorDenseFill[TranslationType], dense_fill),
            basis,
        )
    return TranslationOperatorDenseEntryMaterializeProbe(operator, basis)


@dataclass(slots=True)
class TranslationOperatorMaterialize(Generic[TranslationType]):
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
    basis: TranslationBasisLike[TranslationType]
    source: TranslationType
    image: TranslationType
    matrix: DenseOperatorMatrix | None = None
    entry_materializer: TranslationOperatorDenseEntryMaterializeLike[TranslationType] = field(init=False)
    refresh_initial: bool = True

    def __post_init__(self) -> None:
        self.matrix = self.matrix if self.matrix is not None else self._allocate_matrix()
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

    def _ensure_matrix(self) -> DenseOperatorMatrix:
        matrix = self.matrix
        if matrix is None:
            matrix = self._allocate_matrix()
            self.matrix = matrix
        return matrix

    def refresh(self) -> DenseOperatorMatrix:
        """Rebuild and return the dense row-major matrix for the current operator."""

        matrix = self._ensure_matrix()
        dimension = self.basis.dimension
        if len(matrix) != dimension * dimension:
            matrix = self._allocate_matrix()
            self.matrix = matrix
        else:
            for index in range(len(matrix)):
                matrix[index] = 0.0

        self.source, self.image = self.entry_materializer.refresh(
            matrix,
            row_offset=0,
            column_offset=0,
            stride=dimension,
            source=self.source,
            image=self.image,
        )

        return matrix


@dataclass(slots=True)
class BlockOperatorMaterialize(Generic[TranslationType]):
    """
    Dense materialisation of a block operator in a block basis.

    The materialiser works in the product-space basis owned by ``BlockBasis``.
    Inspectable diagonal block operators use the block-diagonal fast path

        target[i] <- operator[i](source[i])

    by materialising each entry operator into the corresponding translation
    basis. General block operators fall back to probing the full block basis
    column by column. Matrix storage is flat row-major to support accelerator
    backends that do not handle nested Python lists efficiently.
    """

    operator: BlockOperatorLike[TranslationType]
    basis: BlockBasis[TranslationType]
    source: BlockLike[TranslationType]
    image: BlockLike[TranslationType]
    matrix: DenseOperatorMatrix | None = None
    dimension: int = field(init=False)
    entry_materializers: list[TranslationOperatorDenseEntryMaterializeLike[TranslationType] | None] = field(init=False)
    entry_operators: list[object | None] = field(init=False)
    entry_dense_fills: list[object | None] = field(init=False)
    refresh_initial: bool = True

    def __post_init__(self) -> None:
        self.dimension = self.basis.dimension
        if self.matrix is None and self.refresh_initial:
            self.matrix = self._allocate_matrix()
        self.entry_materializers = [None for _basis in self.basis.bases]
        self.entry_operators = [None for _basis in self.basis.bases]
        self.entry_dense_fills = [None for _basis in self.basis.bases]
        if self.refresh_initial:
            self.refresh()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    def _allocate_matrix(self) -> list[float]:
        return [0.0 for _ in range(self.dimension * self.dimension)]

    def _ensure_matrix(self) -> DenseOperatorMatrix:
        matrix = self.matrix
        if matrix is None:
            matrix = self._allocate_matrix()
            self.matrix = matrix
        return matrix

    def prepare_entries(self) -> None:
        """Prepare entry materializers for the currently configured operator."""

        diagonal_operator = self._require_diagonal_operator(self.operator)
        for block_index, basis in enumerate(self.basis.bases):
            entry_operator = diagonal_operator[block_index]
            entry_dense_fill = getattr(entry_operator, "dense_fill", None)
            self.entry_materializers[block_index] = dense_entry_materializer(
                entry_operator,
                basis,
                entry_dense_fill,
            )
            self.entry_operators[block_index] = entry_operator
            self.entry_dense_fills[block_index] = entry_dense_fill

    def refresh(self, operator: BlockOperatorLike[TranslationType] | None = None) -> DenseOperatorMatrix:
        """
        Rebuild and return the dense row-major block matrix.

        Inspectable diagonal operators use the compact block-diagonal
        materialisation path. General operators are probed in the full block
        basis.
        """

        matrix = self._ensure_matrix()
        if operator is not None:
            self.operator = operator

        if len(matrix) != self.dimension * self.dimension:
            matrix = self._allocate_matrix()
            self.matrix = matrix

        diagonal_operator = self._diagonal_operator(self.operator)
        if diagonal_operator is None:
            return self._refresh_probe_current()

        self._require_configured_entries(diagonal_operator)
        return self._refresh_diagonal_current_prepared(diagonal_operator)

    def _diagonal_operator(
        self,
        operator: BlockOperatorLike[TranslationType],
    ) -> BlockOperatorDiagonalLike[TranslationType] | None:
        try:
            len(operator)  # type: ignore[arg-type]
        except TypeError:
            return None

        return cast(BlockOperatorDiagonalLike[TranslationType], operator)

    def _require_diagonal_operator(
        self,
        operator: BlockOperatorLike[TranslationType],
    ) -> BlockOperatorDiagonalLike[TranslationType]:
        diagonal_operator = self._diagonal_operator(operator)
        if diagonal_operator is None:
            raise TypeError("Prepared block materialisation requires an inspectable diagonal block operator.")
        return diagonal_operator

    def _require_configured_entries(self, operator: BlockOperatorDiagonalLike[TranslationType]) -> None:
        """Reject incomplete block operators on the public materialisation path."""

        if len(operator) != len(self.source):
            raise ValueError("Block operator size must match the source block size.")
        if len(self.image) != len(self.source):
            raise ValueError("Source and image blocks must have the same size.")
        if len(self.basis.bases) != len(self.source):
            raise ValueError("Block basis size must match the source block size.")

        for block_index in range(len(self.basis.bases)):
            if operator[block_index] is None:
                raise RuntimeError(f"Block operator diagonal entry {block_index} is not configured.")

    def refresh_block_prepared(
        self,
        block_index: int,
        operator: BlockOperatorDiagonalLike[TranslationType],
        matrix: DenseOperatorMatrix,
    ) -> DenseOperatorMatrix:
        """Rebuild one prepared diagonal block into a compact row-major matrix."""

        self.operator = operator
        basis = self.basis.bases[block_index]
        dimension = basis.dimension

        for index in range(len(matrix)):
            matrix[index] = 0.0

        entry_operator = operator[block_index]
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

        source_item, image_item = entry_materializer.refresh(
            matrix,
            row_offset=0,
            column_offset=0,
            stride=dimension,
            source=self.source[block_index],
            image=self.image[block_index],
        )
        self.source[block_index] = source_item
        self.image[block_index] = image_item
        return matrix

    def refresh_one_block_prepared(self, operator: BlockOperatorDiagonalLike[TranslationType]) -> DenseOperatorMatrix:
        """Rebuild a prepared one-block dense matrix without shape checks."""

        matrix = self.matrix
        if matrix is None:
            basis = self.basis.bases[0]
            dimension = basis.dimension
            matrix = [0.0 for _ in range(dimension * dimension)]
            self.matrix = matrix
        return self.refresh_block_prepared(0, operator, matrix)

    def refresh_prepared(self, operator: BlockOperatorDiagonalLike[TranslationType]) -> DenseOperatorMatrix:
        """Rebuild a previously checked dense block-diagonal matrix for ``operator``."""

        self.operator = operator
        return self._refresh_diagonal_current_prepared(operator)

    def _refresh_diagonal_current_prepared(
        self,
        operator: BlockOperatorDiagonalLike[TranslationType],
    ) -> DenseOperatorMatrix:
        """Rebuild the current dense block-diagonal matrix without validation."""

        matrix = cast(DenseOperatorMatrix, self.matrix)

        for index in range(len(matrix)):
            matrix[index] = 0.0

        dimension = self.dimension
        for block_index, basis in enumerate(self.basis.bases):
            entry_operator = operator[block_index]

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

            offset = self.basis.offsets[block_index]
            source_item, image_item = entry_materializer.refresh(
                matrix,
                row_offset=offset,
                column_offset=offset,
                stride=dimension,
                source=self.source[block_index],
                image=self.image[block_index],
            )
            self.source[block_index] = source_item
            self.image[block_index] = image_item

        return matrix

    def _refresh_probe_current(self) -> DenseOperatorMatrix:
        """Rebuild the current dense matrix by probing the full block basis."""

        matrix = self._ensure_matrix()
        dimension = self.dimension
        for index in range(len(matrix)):
            matrix[index] = 0.0

        for column in range(dimension):
            self.source = self.basis.vector(column, self.source)
            result = self.operator(self.source, self.image)
            if result is not None:
                self.image = result

            for row in range(dimension):
                matrix[row * dimension + column] = self.basis.coordinate(row, self.image)

        return matrix


__all__ = [
    "BlockOperatorMaterialize",
    "DenseOperatorMatrix",
    "TranslationOperatorDenseEntryMaterializeLike",
    "TranslationOperatorDenseEntryMaterializeDirect",
    "TranslationOperatorDenseEntryMaterializeProbe",
    "TranslationOperatorDenseFill",
    "TranslationOperatorMaterialize",
    "dense_entry_materializer",
]
