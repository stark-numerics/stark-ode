from __future__ import annotations

"""Use the current request-shaped dense inverter on a tiny block system."""

from dataclasses import dataclass
from math import sqrt

from stark.core.block import Block, BlockBasis
from stark.methods.inverters.dense import InverterDense


@dataclass
class Vector:
    values: list[float]

    def __init__(self, *values: float) -> None:
        self.values = [float(value) for value in values]

    def norm(self) -> float:
        return sqrt(sum(value * value for value in self.values))

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(*(left + right for left, right in zip(self.values, other.values, strict=True)))

    def __rmul__(self, scalar: float) -> "Vector":
        return Vector(*(scalar * value for value in self.values))


class VectorBasis:
    dimension = 2

    def vector(self, index: int, output: Vector) -> Vector:
        output.values[:] = [0.0, 0.0]
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


class OperatorDiagonal:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.apply_one

    def __call__(self, source: Block[Vector], target: Block[Vector]) -> Block[Vector]:
        self.apply_one(source[0], target[0])
        return target

    @staticmethod
    def apply_one(source: Vector, target: Vector) -> Vector:
        x, y = source.values
        target.values[:] = [2.0 * x + y, x + 3.0 * y]
        return target


@dataclass
class Request:
    operator: OperatorDiagonal
    residual: Block[Vector]


inverter = InverterDense(BlockBasis([VectorBasis()]))
request = Request(OperatorDiagonal(), Block([Vector(1.0, 2.0)]))
output = Block([Vector(0.0, 0.0)])

inverter(request, output)  # solves [[2, 1], [1, 3]] x = [1, 2]

print("Dense inverter")
print("system: [[2, 1], [1, 3]] x = [1, 2]")
print(f"solution: {output[0].values}")
