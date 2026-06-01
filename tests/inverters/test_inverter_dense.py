from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt

import pytest

from stark.block import Block, BlockBasis
from stark.inverters.dense import InverterDense, InverterProviderDenseNative


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
    assert inverter.provider.dimension == 1


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
    assert inverter.provider.dimension == 2


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
    assert inverter.provider.dimension == 3


def test_dense_inverter_uses_native_general_fallback_for_larger_system() -> None:
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
    assert inverter.provider.dimension == 4


def test_dense_native_provider_rejects_singular_exact_system() -> None:
    basis = BlockBasis([VectorBasis(2)])
    inverter = InverterDense(basis=basis, provider=InverterProviderDenseNative())
    request = RequestFake(
        operator=OperatorDiagonalFake([matrix_operator([[1.0, 2.0], [2.0, 4.0]])]),
        residual=Block([Vector(1.0, 1.0)]),
    )

    with pytest.raises(ZeroDivisionError, match="singular"):
        inverter(request, Block([Vector(0.0, 0.0)]))


class AcceleratorFake:
    name = "fake"

    def __init__(self) -> None:
        self.labels: list[str | None] = []

    def compile(self, function=None, /, *, label=None, cache=None, **options):
        del cache, options

        def compile_function(target):
            self.labels.append(label)
            return target

        if function is None:
            return compile_function
        return compile_function(function)

    def compile_examples(self, function, *examples):
        del examples
        return function


def test_dense_native_provider_compiles_selected_kernel_when_accelerator_is_supplied() -> None:
    accelerator = AcceleratorFake()
    basis = BlockBasis([VectorBasis(3)])
    inverter = InverterDense(
        basis=basis,
        provider=InverterProviderDenseNative(accelerator=accelerator),
    )
    request = RequestFake(
        operator=OperatorDiagonalFake([matrix_operator([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])]),
        residual=Block([Vector(2.0, 8.0, 32.0)]),
    )
    output = Block([Vector(0.0, 0.0, 0.0)])

    inverter(request, output)

    assert_vector_close(output[0], [2.0, 4.0, 8.0])
    assert accelerator.labels == ["inverter-provider-dense-native-three"]
