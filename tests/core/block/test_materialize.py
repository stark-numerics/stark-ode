from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isclose

import pytest

from stark.core.block import Block, BlockBasis
from stark.core.block.materialize import BlockOperatorMaterialize, TranslationOperatorMaterialize
from stark.core.contracts import BlockLike
from tests.support import DummyVectorBasis, DummyVectorTranslation


DummyVectorOperator = Callable[[DummyVectorTranslation, DummyVectorTranslation], None]


class DummyVectorReturningBasis(DummyVectorBasis):
    """Basis variant that returns a fresh vector instead of mutating `output`."""

    def __init__(self) -> None:
        super().__init__(2)

    def vector(
        self,
        index: int,
        output: DummyVectorTranslation,
    ) -> DummyVectorTranslation:
        del output
        values = [0.0, 0.0]
        values[index] = 1.0
        return DummyVectorTranslation(values[0], values[1])


def apply_matrix(matrix: list[list[float]]) -> DummyVectorOperator:
    """Return a two-by-two matrix action over `DummyVectorTranslation`."""

    def operator(
        source: DummyVectorTranslation,
        target: DummyVectorTranslation,
    ) -> None:
        target.values[0] = matrix[0][0] * source.values[0] + matrix[0][1] * source.values[1]
        target.values[1] = matrix[1][0] * source.values[0] + matrix[1][1] * source.values[1]

    return operator


class DummyBlockOperatorDiagonal:
    """Block-diagonal operator assembled from entry callables."""

    def __init__(self, entries: Sequence[DummyVectorOperator | None]) -> None:
        self.entries = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> DummyVectorOperator | None:
        return self.entries[index]

    def __call__(
        self,
        source: BlockLike[DummyVectorTranslation],
        target: BlockLike[DummyVectorTranslation],
    ) -> BlockLike[DummyVectorTranslation]:
        for index, entry in enumerate(self.entries):
            if entry is None:
                raise RuntimeError("missing entry")
            entry(source[index], target[index])
        return target


class DummyBlockOperatorFull:
    """Coupled block operator with no inspectable diagonal entries."""

    def __call__(
        self,
        source: BlockLike[DummyVectorTranslation],
        target: BlockLike[DummyVectorTranslation],
    ) -> BlockLike[DummyVectorTranslation]:
        target[0].values[0] = 2.0 * source[0].values[0] + 3.0 * source[1].values[0]
        target[1].values[0] = 5.0 * source[0].values[0] + 7.0 * source[1].values[0]
        return target


def flatten_matrix(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def assert_matrix_close(left: Sequence[float], right: list[list[float]]) -> None:
    expected = flatten_matrix(right)
    assert len(left) == len(expected)
    for left_value, right_value in zip(left, expected, strict=True):
        assert isclose(left_value, right_value, rel_tol=0.0, abs_tol=1.0e-12)


def test_operator_materialize_builds_matrix_from_basis_vectors_and_coordinates() -> None:
    materialized = TranslationOperatorMaterialize(
        operator=apply_matrix([[2.0, 3.0], [5.0, 7.0]]),
        basis=DummyVectorBasis(2),
        source=DummyVectorTranslation(0.0, 0.0),
        image=DummyVectorTranslation(0.0, 0.0),
    )

    assert materialized.matrix is not None
    assert_matrix_close(materialized.matrix, [[2.0, 3.0], [5.0, 7.0]])


def test_operator_materialize_uses_vector_return_value() -> None:
    materialized = TranslationOperatorMaterialize(
        operator=apply_matrix([[11.0, 13.0], [17.0, 19.0]]),
        basis=DummyVectorReturningBasis(),
        source=DummyVectorTranslation(99.0, 99.0),
        image=DummyVectorTranslation(0.0, 0.0),
    )

    assert materialized.matrix is not None
    assert_matrix_close(materialized.matrix, [[11.0, 13.0], [17.0, 19.0]])


def test_block_operator_diagonal_materialize_builds_block_diagonal_matrix() -> None:
    operator = DummyBlockOperatorDiagonal(
        [
            apply_matrix([[2.0, 3.0], [5.0, 7.0]]),
            apply_matrix([[11.0, 13.0], [17.0, 19.0]]),
        ]
    )
    source = Block([DummyVectorTranslation(0.0, 0.0), DummyVectorTranslation(0.0, 0.0)])
    image = Block([DummyVectorTranslation(0.0, 0.0), DummyVectorTranslation(0.0, 0.0)])

    materialized = BlockOperatorMaterialize(
        operator=operator,
        basis=BlockBasis([DummyVectorBasis(2), DummyVectorBasis(2)]),
        source=source,
        image=image,
    )

    assert materialized.matrix is not None
    assert_matrix_close(
        materialized.matrix,
        [
            [2.0, 3.0, 0.0, 0.0],
            [5.0, 7.0, 0.0, 0.0],
            [0.0, 0.0, 11.0, 13.0],
            [0.0, 0.0, 17.0, 19.0],
        ],
    )


def test_block_operator_materialize_uses_block_basis_for_product_space() -> None:
    operator = DummyBlockOperatorDiagonal(
        [
            apply_matrix([[1.0, 0.0], [0.0, 2.0]]),
            apply_matrix([[3.0, 0.0], [0.0, 4.0]]),
        ]
    )

    materialized = BlockOperatorMaterialize(
        operator=operator,
        basis=BlockBasis([DummyVectorBasis(2), DummyVectorBasis(2)]),
        source=Block([DummyVectorTranslation(0.0, 0.0), DummyVectorTranslation(0.0, 0.0)]),
        image=Block([DummyVectorTranslation(0.0, 0.0), DummyVectorTranslation(0.0, 0.0)]),
    )

    assert materialized.dimension == 4
    assert materialized.matrix is not None
    assert_matrix_close(
        materialized.matrix,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
    )


def test_block_operator_materialize_probes_coupled_block_operator() -> None:
    materialized = BlockOperatorMaterialize(
        operator=DummyBlockOperatorFull(),
        basis=BlockBasis([DummyVectorBasis(1), DummyVectorBasis(1)]),
        source=Block([DummyVectorTranslation(0.0), DummyVectorTranslation(0.0)]),
        image=Block([DummyVectorTranslation(0.0), DummyVectorTranslation(0.0)]),
    )

    assert materialized.matrix is not None
    assert_matrix_close(materialized.matrix, [[2.0, 3.0], [5.0, 7.0]])


def test_block_operator_diagonal_materialize_rejects_size_mismatch() -> None:
    operator = DummyBlockOperatorDiagonal([apply_matrix([[1.0, 0.0], [0.0, 1.0]])])

    with pytest.raises(ValueError, match="Block operator size"):
        BlockOperatorMaterialize(
            operator=operator,
            basis=BlockBasis([DummyVectorBasis(2), DummyVectorBasis(2)]),
            source=Block([DummyVectorTranslation(0.0, 0.0), DummyVectorTranslation(0.0, 0.0)]),
            image=Block([DummyVectorTranslation(0.0, 0.0), DummyVectorTranslation(0.0, 0.0)]),
        )


def test_block_operator_diagonal_materialize_rejects_unconfigured_entry() -> None:
    operator = DummyBlockOperatorDiagonal([None])

    with pytest.raises(RuntimeError, match="entry 0"):
        BlockOperatorMaterialize(
            operator=operator,
            basis=BlockBasis([DummyVectorBasis(2)]),
            source=Block([DummyVectorTranslation(0.0, 0.0)]),
            image=Block([DummyVectorTranslation(0.0, 0.0)]),
        )


def test_block_operator_diagonal_materialize_refreshes_compact_block_matrix() -> None:
    operator = DummyBlockOperatorDiagonal(
        [
            apply_matrix([[2.0, 3.0], [5.0, 7.0]]),
            apply_matrix([[11.0, 13.0], [17.0, 19.0]]),
        ]
    )
    materialized = BlockOperatorMaterialize(
        operator=operator,
        basis=BlockBasis([DummyVectorBasis(2), DummyVectorBasis(2)]),
        source=Block([DummyVectorTranslation(0.0, 0.0), DummyVectorTranslation(0.0, 0.0)]),
        image=Block([DummyVectorTranslation(0.0, 0.0), DummyVectorTranslation(0.0, 0.0)]),
        refresh_initial=False,
    )
    compact = [99.0, 99.0, 99.0, 99.0]

    materialized.refresh_block_prepared(1, operator, compact)

    assert_matrix_close(compact, [[11.0, 13.0], [17.0, 19.0]])
