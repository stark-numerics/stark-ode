"""Use the dense inverter with engine-owned translations.

Dense inverters solve small coordinate systems internally. The engine can
provide the translation basis used for that coordinate view, so user examples
do not need to define custom basis classes just to inspect the inverter path.
"""

from __future__ import annotations

from typing import Any

from stark import Frame
from stark.core.block import Block, BlockBasis
from stark.core.block.operator import BlockOperatorDiagonal
from stark.core.contracts import TranslationBasisLike
from stark.engines import EngineNumpy
from stark.methods import InverterDense
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


class MatrixOperator:
    """Apply and materialise ``[[2, 1], [1, 3]]`` on one translation field."""

    def __init__(self, basis: TranslationBasisLike[Any]) -> None:
        self.basis = basis
        self.image = [0.0, 0.0]
        self.result = [0.0, 0.0]

    def __call__(self, source: Any, target: Any) -> None:
        self.basis.coordinates(source, self.image)
        x, y = self.image
        self.result[0] = 2.0 * x + y
        self.result[1] = x + 3.0 * y
        self.basis.synthesize(self.result, target)

    def dense_fill(self, _basis: object, matrix: list[float], row: int, column: int, stride: int) -> None:
        matrix[(row + 0) * stride + column + 0] = 2.0
        matrix[(row + 0) * stride + column + 1] = 1.0
        matrix[(row + 1) * stride + column + 0] = 1.0
        matrix[(row + 1) * stride + column + 1] = 3.0


if __name__ == "__main__":
    engine = EngineNumpy(Frame.vector("x", translation="dx", length=2))
    basis = engine.translation_basis()

    residual = engine.allocator.allocate_translation()
    basis.synthesize([1.0, 2.0], residual)
    output_delta = engine.allocator.allocate_translation()

    operator: BlockOperatorDiagonal[Any] = BlockOperatorDiagonal([MatrixOperator(basis)])
    residual_block: Block[Any] = Block([residual])
    request = ResolventInverterRequest(operator=operator, residual=residual_block)
    output: Block[Any] = Block([output_delta])
    inverter = InverterDense(BlockBasis([basis]))

    inverter(request, output)
    coordinates = [0.0, 0.0]
    basis.coordinates(output[0], coordinates)

    print("Dense inverter")
    print("==============")
    print("system:   [[2, 1], [1, 3]] x = [1, 2]")
    print(f"solution: [{coordinates[0]:.6g}, {coordinates[1]:.6g}]")
