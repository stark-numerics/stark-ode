"""Use the built-in Jacobi relaxation inverter.

Jacobi relaxation uses the diagonal entries of the request operator. The caller
supplies the local inverse action for each diagonal entry, while the inverter
owns the block-level relaxation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark import Configuration, Frame, Tolerance
from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.core.contracts import BlockOperatorEntryLike, TranslationBasisLike
from stark.engines import EngineNumpy
from stark.methods import InverterRelaxationJacobi
from stark.methods.inverters.support import InverterDefect
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


@dataclass(slots=True)
class ScaleEntryOperator:
    basis: TranslationBasisLike[Any]
    scale: float
    coordinates: list[float]
    image: list[float]

    def __call__(self, source, target) -> None:
        self.basis.coordinates(source, self.coordinates)
        self.image[0] = self.scale * self.coordinates[0]
        self.basis.synthesize(self.image, target)

    def inverse(self, source, target) -> None:
        self.basis.coordinates(source, self.coordinates)
        self.image[0] = self.coordinates[0] / self.scale
        self.basis.synthesize(self.image, target)


def invert_entry(
    operator: BlockOperatorEntryLike[Any],
    source: Any,
    target: Any,
) -> None:
    inverse = getattr(operator, "inverse")
    inverse(source, target)


if __name__ == "__main__":
    engine = EngineNumpy(Frame.scalar("x", translation="dx"))
    basis = engine.translation_basis()

    residual_left = engine.allocator.allocate_translation()
    basis.synthesize([6.0], residual_left)
    residual_right = engine.allocator.allocate_translation()
    basis.synthesize([20.0], residual_right)
    output_left = engine.allocator.allocate_translation()
    output_right = engine.allocator.allocate_translation()

    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal(
            [
                ScaleEntryOperator(basis, 2.0, [0.0], [0.0]),
                ScaleEntryOperator(basis, 4.0, [0.0], [0.0]),
            ]
        ),
        residual=Block([residual_left, residual_right]),
    )
    output = Block([output_left, output_right])
    defect = InverterDefect()
    inverter = InverterRelaxationJacobi(
        invert_entry,
        configuration=Configuration(
            inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0),
            inverter_maximum_steps=2,
        ),
    )

    initial_defect = defect(request, output)
    inverter(request, output)
    final_defect = defect(request, output)
    output_coordinates = [0.0]
    basis.coordinates(output[0], output_coordinates)
    left = output_coordinates[0]
    basis.coordinates(output[1], output_coordinates)
    right = output_coordinates[0]

    print("Jacobi relaxation inverter")
    print("==========================")
    print("problem:        [2, 4] * output = [6, 20]")
    print("initial output: [0, 0]")
    print(f"initial defect: {initial_defect:.6g}")
    print(f"final output:   [{left:.6g}, {right:.6g}]")
    print(f"final defect:   {final_defect:.6g}")
