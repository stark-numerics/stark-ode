from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt

import pytest

from stark.core.block import Block
from stark.core.block.materialize import BlockOperatorDiagonalMaterialize, OperatorMaterialize


@dataclass
class VectorTwo:
    values: list[float]

    def __init__(self, first: float = 0.0, second: float = 0.0) -> None:
        self.values = [float(first), float(second)]

    def norm(self) -> float:
        return sqrt(sum(value * value for value in self.values))

    def __add__(self, other: VectorTwo) -> VectorTwo:
        return VectorTwo(self.values[0] + other.values[0], self.values[1] + other.values[1])

    def __rmul__(self, scalar: float) -> VectorTwo:
        return VectorTwo(scalar * self.values[0], scalar * self.values[1])


class VectorTwoBasis:
    dimension = 2

    def vector(self, index: int, output: VectorTwo) -> VectorTwo:
        output.values[:] = [0.0, 0.0]
        output.values[index] = 1.0
        return output

    def coordinate(self, index: int, value: VectorTwo) -> float:
        return value.values[index]


class VectorTwoReturningBasis(VectorTwoBasis):
    def vector(self, index: int, output: VectorTwo) -> VectorTwo:
        del output
        values = [0.0, 0.0]
        values[index] = 1.0
        return VectorTwo(values[0], values[1])


def apply_matrix(matrix: list[list[float]]):
    def operator(source: VectorTwo, target: VectorTwo) -> None:
        target.values[0] = matrix[0][0] * source.values[0] + matrix[0][1] * source.values[1]
        target.values[1] = matrix[1][0] * source.values[0] + matrix[1][1] * source.values[1]

    return operator


class FakeBlockOperatorDiagonal:
    def __init__(self, entries):
        self.entries = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        return self.entries[index]

    def __call__(self, source: Block[VectorTwo], target: Block[VectorTwo]) -> Block[VectorTwo]:
        for index, entry in enumerate(self.entries):
            if entry is None:
                raise RuntimeError("missing entry")
            entry(source[index], target[index])
        return target


def flatten_matrix(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def assert_matrix_close(left: list[float], right: list[list[float]]) -> None:
    expected = flatten_matrix(right)
    assert len(left) == len(expected)
    for left_value, right_value in zip(left, expected, strict=True):
        assert isclose(left_value, right_value, rel_tol=0.0, abs_tol=1.0e-12)


def test_operator_materialize_builds_matrix_from_basis_vectors_and_coordinates() -> None:
    materialized = OperatorMaterialize(
        operator=apply_matrix([[2.0, 3.0], [5.0, 7.0]]),
        basis=VectorTwoBasis(),
        source=VectorTwo(),
        image=VectorTwo(),
    )

    assert_matrix_close(materialized.matrix, [[2.0, 3.0], [5.0, 7.0]])


def test_operator_materialize_uses_vector_return_value() -> None:
    materialized = OperatorMaterialize(
        operator=apply_matrix([[11.0, 13.0], [17.0, 19.0]]),
        basis=VectorTwoReturningBasis(),
        source=VectorTwo(99.0, 99.0),
        image=VectorTwo(),
    )

    assert_matrix_close(materialized.matrix, [[11.0, 13.0], [17.0, 19.0]])


def test_block_operator_diagonal_materialize_builds_block_diagonal_matrix() -> None:
    operator = FakeBlockOperatorDiagonal(
        [
            apply_matrix([[2.0, 3.0], [5.0, 7.0]]),
            apply_matrix([[11.0, 13.0], [17.0, 19.0]]),
        ]
    )
    source = Block([VectorTwo(), VectorTwo()])
    image = Block([VectorTwo(), VectorTwo()])

    materialized = BlockOperatorDiagonalMaterialize(
        operator=operator,
        bases=[VectorTwoBasis(), VectorTwoBasis()],
        source=source,
        image=image,
    )

    assert_matrix_close(
        materialized.matrix,
        [
            [2.0, 3.0, 0.0, 0.0],
            [5.0, 7.0, 0.0, 0.0],
            [0.0, 0.0, 11.0, 13.0],
            [0.0, 0.0, 17.0, 19.0],
        ],
    )


def test_block_operator_diagonal_materialize_accepts_one_basis_for_all_entries() -> None:
    operator = FakeBlockOperatorDiagonal(
        [
            apply_matrix([[1.0, 0.0], [0.0, 2.0]]),
            apply_matrix([[3.0, 0.0], [0.0, 4.0]]),
        ]
    )

    materialized = BlockOperatorDiagonalMaterialize(
        operator=operator,
        bases=VectorTwoBasis(),
        source=Block([VectorTwo(), VectorTwo()]),
        image=Block([VectorTwo(), VectorTwo()]),
    )

    assert materialized.dimension == 4
    assert_matrix_close(
        materialized.matrix,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
    )


def test_block_operator_diagonal_materialize_rejects_size_mismatch() -> None:
    operator = FakeBlockOperatorDiagonal([apply_matrix([[1.0, 0.0], [0.0, 1.0]])])

    with pytest.raises(ValueError, match="Operator size"):
        BlockOperatorDiagonalMaterialize(
            operator=operator,
            bases=[VectorTwoBasis(), VectorTwoBasis()],
            source=Block([VectorTwo(), VectorTwo()]),
            image=Block([VectorTwo(), VectorTwo()]),
        )


def test_block_operator_diagonal_materialize_rejects_unconfigured_entry() -> None:
    operator = FakeBlockOperatorDiagonal([None])

    with pytest.raises(RuntimeError, match="entry 0"):
        BlockOperatorDiagonalMaterialize(
            operator=operator,
            bases=VectorTwoBasis(),
            source=Block([VectorTwo()]),
            image=Block([VectorTwo()]),
        )


def test_block_operator_diagonal_materialize_refreshes_compact_block_matrix() -> None:
    operator = FakeBlockOperatorDiagonal(
        [
            apply_matrix([[2.0, 3.0], [5.0, 7.0]]),
            apply_matrix([[11.0, 13.0], [17.0, 19.0]]),
        ]
    )
    materialized = BlockOperatorDiagonalMaterialize(
        operator=operator,
        bases=[VectorTwoBasis(), VectorTwoBasis()],
        source=Block([VectorTwo(), VectorTwo()]),
        image=Block([VectorTwo(), VectorTwo()]),
        refresh_initial=False,
    )
    compact = [99.0, 99.0, 99.0, 99.0]

    materialized.refresh_block_prepared(1, operator, compact)

    assert_matrix_close(compact, [[11.0, 13.0], [17.0, 19.0]])
