from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt

import pytest

from stark.core.block import Block, BlockBasis
from stark.methods.inverters.dense import InverterDense, InverterDenseInstanceSingle


@dataclass
class Vector:
    values: list[float]

    def __init__(self, *values: float) -> None:
        self.values = [float(value) for value in values]

    def norm(self) -> float:
        return sqrt(sum(value * value for value in self.values))

    def __add__(self, other: Vector) -> Vector:
        return Vector(*(left + right for left, right in zip(self.values, other.values, strict=True)))

    def __rmul__(self, scalar: float) -> Vector:
        return Vector(*(scalar * value for value in self.values))


class VectorBasis:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def vector(self, index: int, output: Vector) -> Vector:
        output.values[:] = [0.0] * self.dimension
        output.values[index] = 1.0
        return output

    def coordinate(self, index: int, translation: Vector) -> float:
        return translation.values[index]

    def coordinates(self, translation: Vector, output: list[float]) -> list[float]:
        output[:] = translation.values[:]
        return output

    def synthesize(self, coordinates: list[float], output: Vector) -> Vector:
        output.values[:] = list(coordinates)
        return output


class OperatorDiagonalFake:
    def __init__(self, entries):
        self.entries = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        return self.entries[index]

    def __call__(self, source: Block[Vector], target: Block[Vector]) -> Block[Vector]:
        for index, entry in enumerate(self.entries):
            if entry is None:
                raise RuntimeError("missing entry")
            entry(source[index], target[index])
        return target


class RequestFake:
    def __init__(self, operator, residual):
        self.operator = operator
        self.residual = residual


def matrix_operator(matrix: list[list[float]]):
    def apply(source: Vector, target: Vector) -> Vector:
        target.values[:] = [
            sum(coefficient * source_value for coefficient, source_value in zip(row, source.values, strict=True))
            for row in matrix
        ]
        return target

    return apply


def assert_vector_close(actual: Vector, expected: list[float]) -> None:
    assert len(actual.values) == len(expected)
    for actual_value, expected_value in zip(actual.values, expected, strict=True):
        assert isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1.0e-12)


def test_dense_inverter_solves_scalar_problem() -> None:
    basis = BlockBasis([VectorBasis(1)])
    inverter = InverterDense(basis=basis)
    request = RequestFake(
        operator=OperatorDiagonalFake([matrix_operator([[4.0]])]),
        residual=Block([Vector(8.0)]),
    )
    output = Block([Vector(0.0)])

    inverter(request, output)

    assert_vector_close(output[0], [2.0])


def test_dense_inverter_solves_two_by_two_problem() -> None:
    basis = BlockBasis([VectorBasis(2)])
    inverter = InverterDense(basis=basis)
    request = RequestFake(
        operator=OperatorDiagonalFake([matrix_operator([[2.0, 1.0], [1.0, 3.0]])]),
        residual=Block([Vector(1.0, 2.0)]),
    )
    output = Block([Vector(0.0, 0.0)])

    inverter(request, output)

    assert_vector_close(output[0], [0.2, 0.6])


def test_dense_inverter_solves_three_by_three_problem() -> None:
    basis = BlockBasis([VectorBasis(3)])
    inverter = InverterDense(basis=basis)
    request = RequestFake(
        operator=OperatorDiagonalFake([matrix_operator([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])]),
        residual=Block([Vector(14.0, 14.0, 17.0)]),
    )
    output = Block([Vector(0.0, 0.0, 0.0)])

    inverter(request, output)

    assert_vector_close(output[0], [1.0, 2.0, 3.0])


def test_dense_inverter_solves_blockwise_larger_system() -> None:
    basis = BlockBasis([VectorBasis(2), VectorBasis(2)])
    inverter = InverterDense(basis=basis)
    request = RequestFake(
        operator=OperatorDiagonalFake([
            matrix_operator([[2.0, 0.0], [0.0, 4.0]]),
            matrix_operator([[5.0, 0.0], [0.0, 10.0]]),
        ]),
        residual=Block([Vector(6.0, 8.0), Vector(15.0, 40.0)]),
    )
    output = Block([Vector(0.0, 0.0), Vector(0.0, 0.0)])

    inverter(request, output)

    assert_vector_close(output[0], [3.0, 2.0])
    assert_vector_close(output[1], [3.0, 4.0])


def test_dense_inverter_rejects_singular_exact_system() -> None:
    basis = BlockBasis([VectorBasis(2)])
    inverter = InverterDense(basis=basis)
    request = RequestFake(
        operator=OperatorDiagonalFake([matrix_operator([[1.0, 2.0], [2.0, 4.0]])]),
        residual=Block([Vector(1.0, 1.0)]),
    )

    with pytest.raises(ZeroDivisionError):
        inverter(request, Block([Vector(0.0, 0.0)]))


def dense_fill_operator(matrix: list[list[float]]):
    def apply(source: Vector, target: Vector) -> Vector:
        target.values[:] = [
            sum(coefficient * source_value for coefficient, source_value in zip(row, source.values, strict=True))
            for row in matrix
        ]
        return target

    def dense_fill(_basis, target: list[float], row_offset: int, column_offset: int, stride: int) -> None:
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                target[(row_offset + row) * stride + column_offset + column] = value

    apply.dense_fill = dense_fill  # type: ignore[attr-defined]
    return apply


def test_dense_inverter_instance_reuses_materialized_block_matrix() -> None:
    basis = BlockBasis([VectorBasis(3)])
    inverter = InverterDense(basis=basis)
    operator = OperatorDiagonalFake([
        dense_fill_operator([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
    ])
    instance = inverter.instance(operator)
    output = Block([Vector(0.0, 0.0, 0.0)])

    assert isinstance(instance, InverterDenseInstanceSingle)
    instance(Block([Vector(14.0, 14.0, 17.0)]), output)

    assert_vector_close(output[0], [1.0, 2.0, 3.0])


def test_dense_inverter_instance_requires_dense_fill_entries() -> None:
    basis = BlockBasis([VectorBasis(2)])
    inverter = InverterDense(basis=basis)
    operator = OperatorDiagonalFake([matrix_operator([[2.0, 1.0], [1.0, 3.0]])])

    with pytest.raises(TypeError, match="dense_fill"):
        inverter.instance(operator)



class MonitorFake:
    def __init__(self) -> None:
        self.solves = []

    def record_solve(
        self,
        inverter,
        converged,
        iteration_count,
        initial_residual,
        final_residual,
        failure_reason,
    ) -> None:
        self.solves.append(
            (inverter, converged, iteration_count, initial_residual, final_residual, failure_reason)
        )


def test_dense_inverter_unmonitored_path_does_not_call_record(monkeypatch) -> None:
    def fail_record(*_args, **_kwargs):
        raise AssertionError("unmonitored dense path should not call record_solve")

    monkeypatch.setattr(InverterDense, "record_solve", fail_record)
    basis = BlockBasis([VectorBasis(2)])
    inverter = InverterDense(basis=basis)
    request = RequestFake(
        operator=OperatorDiagonalFake([matrix_operator([[2.0, 0.0], [0.0, 4.0]])]),
        residual=Block([Vector(6.0, 8.0)]),
    )
    output = Block([Vector(0.0, 0.0)])

    inverter(request, output)

    assert_vector_close(output[0], [3.0, 2.0])


def test_dense_inverter_monitored_path_records_once() -> None:
    monitor = MonitorFake()
    basis = BlockBasis([VectorBasis(2)])
    inverter = InverterDense(basis=basis, monitor=monitor)
    request = RequestFake(
        operator=OperatorDiagonalFake([matrix_operator([[2.0, 0.0], [0.0, 4.0]])]),
        residual=Block([Vector(6.0, 8.0)]),
    )
    output = Block([Vector(0.0, 0.0)])

    inverter(request, output)

    assert_vector_close(output[0], [3.0, 2.0])
    assert monitor.solves == [("Dense", True, None, None, None, None)]


def test_dense_inverter_instance_unmonitored_path_does_not_call_record(monkeypatch) -> None:
    def fail_record(*_args, **_kwargs):
        raise AssertionError("unmonitored dense instance path should not call record_solve")

    monkeypatch.setattr(InverterDense, "record_solve", fail_record)
    basis = BlockBasis([VectorBasis(2)])
    inverter = InverterDense(basis=basis)
    operator = OperatorDiagonalFake([dense_fill_operator([[2.0, 0.0], [0.0, 4.0]])])
    instance = inverter.instance(operator)
    output = Block([Vector(0.0, 0.0)])

    instance(Block([Vector(6.0, 8.0)]), output)

    assert_vector_close(output[0], [3.0, 2.0])


def test_dense_inverter_instance_monitored_path_records_once() -> None:
    monitor = MonitorFake()
    basis = BlockBasis([VectorBasis(2)])
    inverter = InverterDense(basis=basis, monitor=monitor)
    operator = OperatorDiagonalFake([dense_fill_operator([[2.0, 0.0], [0.0, 4.0]])])
    instance = inverter.instance(operator)
    output = Block([Vector(0.0, 0.0)])

    instance(Block([Vector(6.0, 8.0)]), output)

    assert_vector_close(output[0], [3.0, 2.0])
    assert monitor.solves == [("Dense", True, None, None, None, None)]
