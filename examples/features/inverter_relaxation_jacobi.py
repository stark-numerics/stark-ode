"""Use the built-in Jacobi relaxation inverter.

Jacobi relaxation uses the diagonal entries of the request operator. The caller
supplies the local inverse action for each diagonal entry, while the inverter
owns the block-level relaxation loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from stark import Configuration, Layout, Tolerance
from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.engines import EngineNumpy
from stark.methods.inverters.relaxation import InverterRelaxationJacobi
from stark.methods.inverters.support import InverterDefect
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


@dataclass(slots=True)
class ScaleEntryOperator:
    scale: float

    def __call__(self, source, target) -> None:
        target.dx[:] = self.scale * source.dx

    def inverse(self, source, target) -> None:
        target.dx[:] = source.dx / self.scale


def invert_entry(operator: ScaleEntryOperator, source, target) -> None:
    operator.inverse(source, target)


def main() -> None:
    engine = EngineNumpy(Layout({"x": {"translation": "dx", "shape": (1,)}}))

    residual_left = engine.allocator.allocate_translation()
    residual_left.dx[0] = 6.0
    residual_right = engine.allocator.allocate_translation()
    residual_right.dx[0] = 20.0
    output_left = engine.allocator.allocate_translation()
    output_right = engine.allocator.allocate_translation()

    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([ScaleEntryOperator(2.0), ScaleEntryOperator(4.0)]),
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

    print("Jacobi relaxation inverter")
    print("==========================")
    print("problem:        [2, 4] * output = [6, 20]")
    print("initial output: [0, 0]")
    print(f"initial defect: {initial_defect:.6g}")
    print(f"final output:   [{output[0].dx[0]:.6g}, {output[1].dx[0]:.6g}]")
    print(f"final defect:   {final_defect:.6g}")


if __name__ == "__main__":
    main()
